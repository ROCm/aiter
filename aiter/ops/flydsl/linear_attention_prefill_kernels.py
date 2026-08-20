# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Linear Attention Prefill K5 host wrapper (gated delta rule).

This module hosts ``chunk_gated_delta_rule_fwd_h_flydsl`` -- the host
wrapper around the K5 hidden-state recurrence FlyDSL kernel
(``compile_chunk_gated_delta_h``). It performs PyTorch tensor
preparation, chooses ``BV`` with a rule-based grid/CU heuristic, manages
the compiled kernel cache, and handles the launch stream. The kernel-
compile module ``kernels.chunk_gated_delta_h`` is kept ``torch``-free,
mirroring the layering used by ``kernels.gdr_decode``.

For an end-to-end GDN forward that uses this K5 wrapper, call
``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` with
``use_chunk_flydsl=True``.
"""

from __future__ import annotations

import math
import os

import torch
import triton

from ..triton._triton_kernels.gated_delta_rule.utils import (
    GatedDeltaRulePrefillMetadata,
    K5K6Fusion,  # noqa: F401  re-exported: the fused-vs-separate selection enum
    prepare_chunk_offsets,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)
from flydsl.runtime.device import get_rocm_arch as _get_rocm_arch

from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from .kernels.chunk_gated_delta_h_gfx942 import (
    compile_chunk_gated_delta_h_gfx942,
    select_variant as _gfx942_select_variant,
    select_fused_variant as _gfx942_select_fused_variant,
)

from .kernels.tensor_shim import _run_compiled

# Arch-agnostic K5 variant tag grammar + legality (shared with the gfx942 kernel
# module -- kept in its own module to avoid an import cycle). Re-exported here so
# existing callers (bench, AOT, tests) keep importing them from this wrapper.
from .kernels.k5_variants import (  # noqa: F401  (re-exported)
    _BV_CANDIDATES,
    _DEFAULT_BV,
    _bv_of_variant,
    _bv_waves_of_variant,
    _legal_bv_candidates,
    K5_DEFAULT_VARIANT,
    K5_VARIANTS,
)

# Arch detected once at import time; used by _get_or_compile to select the
# right compile backend.
_ARCH: str = _get_rocm_arch()

# log2(e); g pre-scaled by this constant lets the kernel use exp2(g) in
# place of exp(g) (matches the Triton VK / HIP K5 convention).
_RCP_LN2 = math.log2(math.e)


__all__ = [
    "K5K6Fusion",
    "chunk_gated_delta_rule_fwd_h_flydsl",
    "chunk_gated_delta_rule_fwd_h_o_flydsl",
    "chunk_gated_delta_rule_fwd_h_o_auto",
    "should_use_fused_gfx942",
]


# -- K5 host wrapper (FlyDSL kernel + rule-based BV selection) ------------

_compiled_kernels = {}


def _gate_of(use_g: bool, use_gk: bool) -> str:
    """Gate rank as used by the tuned table ("gk" dominates when both set)."""
    return "gk" if use_gk else "g"


def _grid_ctas(*, H: int, V: int, N: int, BV: int) -> int:
    return max(1, N) * H * ((V + BV - 1) // BV)


_cu_count_cache: int | None = None


def _device_cu_count() -> int:
    """Number of CUs on the current device (cached). Falls back to 304 (MI300X/
    MI325X gfx942) if the query fails."""
    global _cu_count_cache
    if _cu_count_cache is None:
        try:
            import torch

            _cu_count_cache = int(
                torch.cuda.get_device_properties(
                    torch.cuda.current_device()
                ).multi_processor_count
            )
        except Exception:
            _cu_count_cache = 304
    return _cu_count_cache


# gfx942: minimum grid fill (CTAs / CU_count) a tile size must achieve to be
# chosen. _heuristic_bv takes the LARGEST legal BV clearing this bar.
#

_GFX942_MIN_FILL = 0.37


# The env override for the variant selector (host-wrapper concern; the tag
# grammar itself lives in kernels/k5_variants.py).
_K5_VARIANT_ENV = "FLYDSL_GDN_K5_VARIANT"


def _auto_variant(*, gate: str | None = None, **kw) -> str:
    """The shape-adaptive choice, as a variant tag.

    Arch dispatcher: on gfx942, defer to the gfx942 kernel module's measured
    ``H*N`` rule (``select_variant``). On any other arch (or
    when the rule's BV is illegal for this V) fall back to ``_heuristic_bv``,
    the cross-arch grid-fill heuristic. 
    """
    if _ARCH == "gfx942":
        tuned = _gfx942_select_variant(
            gate=gate, H=kw["H"], N=kw["N"], V=kw["V"], is_varlen=kw["is_varlen"]
        )
        if tuned is not None:
            return tuned
    return f"bv{_heuristic_bv(**kw)}"


def _resolve_variant(variant: str | None, *, gate: str | None = None, **kw) -> str:
    """Effective variant tag: explicit arg > env var > shape-adaptive.

    Mirrors the resolution chain used by the fp8_mqa_logits kernel.
    """
    tag = variant or os.environ.get(_K5_VARIANT_ENV) or _auto_variant(gate=gate, **kw)
    if tag == K5_DEFAULT_VARIANT:  # env var may legitimately say "auto"
        tag = _auto_variant(gate=gate, **kw)
    bv = _bv_of_variant(tag)
    legal = _legal_bv_candidates(kw["V"])
    if bv not in legal:
        raise ValueError(
            f"GDN K5 variant {tag!r} is not legal for V={kw['V']} "
            f"(needs BV <= V and V % BV == 0); legal here: "
            f"{[f'bv{b}' for b in legal]}"
        )
    return tag


def _select_bv_for_grid(*, H: int, V: int, N: int, target_ctas: int) -> int:
    """Choose the largest legal BV whose grid still covers target_ctas."""
    legal = sorted(_legal_bv_candidates(V), reverse=True)
    if not legal:
        return _DEFAULT_BV
    for bv in legal:
        if _grid_ctas(H=H, V=V, N=N, BV=bv) >= target_ctas:
            return bv
    # If even BV=16 cannot reach the target, use it to maximize grid size.
    return legal[-1]


def _target_bv_for_shape(
    *, H: int, Hg: int, T_flat: int, N: int, is_varlen: bool
) -> int | None:
    """Return the calibrated BV regime before legality/grid adjustment."""
    if is_varlen and H == 32 and Hg == 16:
        if N == 2 and 11000 <= T_flat < 15000:
            return 16
        if N == 3 and not (10000 <= T_flat < 12000 or 20000 <= T_flat < 25000):
            return 64
    if is_varlen and H == 16 and T_flat >= 32768 and N >= 7:
        return 64
    return None


def _lookup_tuned_bv(
    dtype_str,
    K,
    V,
    BT,
    H,
    Hg,
    T_flat,
    N,
    use_g,
    use_gk,
    use_h0,
    store_fs,
    save_vn,
    is_varlen,
    wu_contig,
):
    """Select ``BV`` with the rule-based grid/CU heuristic.

    Kept as the stable signature hook for a future tuned lookup table (and
    referenced from ``aiter/aot/flydsl/chunk_gdn_h.py``). The live selection path
    is now ``_resolve_variant`` -> ``_auto_variant`` -> ``_heuristic_bv``, which
    additionally honours an explicit ``variant=`` and the env override.
    """
    del (
        dtype_str,
        K,
        BT,
        use_g,
        use_gk,
        use_h0,
        store_fs,
        save_vn,
        wu_contig,
    )
    return _heuristic_bv(
        H=H,
        Hg=Hg,
        V=V,
        T_flat=T_flat,
        N=N,
        is_varlen=is_varlen,
    )


def _heuristic_bv(
    *,
    H: int,
    Hg: int,
    V: int,
    T_flat: int,
    N: int,
    is_varlen: bool,
) -> int:
    """Pick a sensible BV for the requested shape. Pure function: no IO, no state.

    Rules calibrated against a 27-point sweep matrix on gfx950 (20 in-csv
    shapes + 7 csv-uncovered probes). The 27 points span H in
    {8,16,24,32,48,64,128} and T_local in [256, 128000]; see
    flydsl_bv_sweep.log + flydsl_heuristic_verify.log.

      * First pick a target CTA count, then choose the largest legal BV whose
        grid ``N * H * ceil(V / BV)`` still reaches that target. Larger BV
        reduces per-CTA overhead; smaller BV exposes more CTAs for CU
        utilization.

      * ``is_varlen=False`` -- target one wave of CTAs over gfx950's 256 CUs.

      * ``is_varlen=True`` -- the target grid depends on (H, T_local) jointly:
          H <= 8:
            short chunks target the BV=64 grid; medium chunks target BV=32;
            long chunks target BV=16.
          H in (8, 16]:
            long chunks target BV=32; shorter chunks target BV=64.
          H == 32, Hg == 16:
            target grid follows the bench333/407 production trace: single
            sequence needs BV=16 grid; N=2/3 use total-T windows; N>=4 has
            enough grid at BV=64.
          H > 16:
            target the BV=64 grid unless a more specific regime above applies.

    Coverage: the rule matches the AOT seed CSV plus the measured bench333 /
    bench407 probes used during calibration. Shapes far outside the sampled
    (H, T_local) grid may still be suboptimal; extend the calibration sweep
    when production reports new shape families.

    Args:
        H: number of v-heads (per TP rank).
        V: head_v_dim.
        T_flat: flat token count fed to the kernel (sum of context lens
            in varlen, ``B*T`` otherwise).
        N: number of sequences in the batch (varlen) or batch size.
        is_varlen: whether the kernel runs in variable-length mode.
        Hg: number of k-heads (per TP rank). Currently only used to scope
            trace-calibrated rules to the K5 H=32/Hg=16 family.

    Returns:
        A BV from ``_BV_CANDIDATES`` that satisfies ``BV <= V`` and
        ``V % BV == 0``. If the rule's first choice is illegal for this
        V (rare: V<16 or V not divisible by 16), falls back to the
        largest legal candidate, then finally to ``_DEFAULT_BV``.
    """
    # gfx942 BV=64 rule (profile-driven): after the lds_vnt reclaim, BV=64 fits in
    # ~58 KiB (< 64 KiB/CU) and matches the HIP kernel's grid. BV=64 uses fatter
    # tiles / fewer CTAs, which beats the small-BV heuristic ONLY when the BV=64
    # grid still fills the CUs (measured cutoff: fill >= ~0.30). Below that the
    # grid starves; the small-BV heuristic (more CTAs) wins. Prefer BV=64 when it
    # is legal for this V and clears the fill bar; otherwise fall through.
    if _ARCH == "gfx942":
        # Largest tile whose own grid still keeps at least _GFX942_MIN_FILL of the
        # CUs busy. Bigger BV = fatter tiles and less redundant k/w traffic, but
        # fewer CTAs; this trades the two off with one physical quantity instead
        # of special-casing BV=64.
        cus = max(_device_cu_count(), 1)
        legal = sorted(_legal_bv_candidates(V), reverse=True)
        for cand in legal:
            if _grid_ctas(H=H, V=V, N=N, BV=cand) / cus >= _GFX942_MIN_FILL:
                return cand
        # Nothing clears the bar: the shape cannot fill the device at any tile
        # size (e.g. N=1, few heads). Take the smallest legal tile, which yields
        # the most CTAs and so salvages what parallelism there is.
        if legal:
            return min(legal)

    target_bv = _target_bv_for_shape(
        H=H, Hg=Hg, T_flat=T_flat, N=N, is_varlen=is_varlen
    )
    target_ctas = (
        _grid_ctas(H=H, V=V, N=N, BV=target_bv) if target_bv is not None else 256
    )
    bv = _select_bv_for_grid(H=H, V=V, N=N, target_ctas=target_ctas)
    # Safety: the gfx942 kernel asserts BV <= 64.
    if _ARCH == "gfx942":
        bv = min(bv, 64)
    return bv


def _get_or_compile(
    K,
    V,
    BT,
    BV,
    H,
    Hg,
    use_g,
    use_gk,
    use_h0,
    store_fs,
    save_vn,
    is_varlen,
    wu_contig,
    state_bf16=False,
    g_log2_scaled=False,
    num_waves=4,
):
    cache_key = (
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        use_g,
        use_gk,
        use_h0,
        store_fs,
        save_vn,
        is_varlen,
        wu_contig,
        state_bf16,
        g_log2_scaled,
        num_waves,
    )
    if cache_key not in _compiled_kernels:
        _compile_kwargs = dict(
            K=K,
            V=V,
            BT=BT,
            BV=BV,
            H=H,
            Hg=Hg,
            USE_G=use_g,
            USE_GK=use_gk,
            USE_INITIAL_STATE=use_h0,
            STORE_FINAL_STATE=store_fs,
            SAVE_NEW_VALUE=save_vn,
            IS_VARLEN=is_varlen,
            WU_CONTIGUOUS=wu_contig,
            STATE_DTYPE_BF16=state_bf16,
            G_IS_LOG2_SCALED=g_log2_scaled,
        )
        if _ARCH == "gfx950":
            if num_waves != 4:
                raise ValueError(
                    "the wave-widening variant axis (w8/w16) is gfx942-only; "
                    f"got num_waves={num_waves} on {_ARCH}"
                )
            _compiled_kernels[cache_key] = compile_chunk_gated_delta_h(
                **_compile_kwargs
            )
        elif _ARCH == "gfx942":
            # COMPUTE_OUTPUT=False selects the K5-only build of the shared
            # gfx942 builder (the fused K5+K6 build is reached via
            # _run_fused_gfx942).
            _compiled_kernels[cache_key] = compile_chunk_gated_delta_h_gfx942(
                **_compile_kwargs,
                NR_SPLIT=num_waves // (BT // 16),
                COMPUTE_OUTPUT=False,
                STORE_H=True,
            )
        else:
            raise ValueError(
                f"FlyDSL GDN K5 is not supported on arch '{_ARCH}'. "
                f"Supported arches: gfx942, gfx950."
            )
    return _compiled_kernels[cache_key]


def _launch_kernel(
    launch_fn,
    BV,
    V,
    N,
    H,
    k,
    u,
    w,
    vn_arg,
    g_arg,
    gk_arg,
    h,
    h0_arg,
    ht_arg,
    cu_arg,
    co_arg,
    q_arg,
    o_arg,
    T,
    T_flat,
    stream,
):
    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H
    _run_compiled(
        launch_fn,
        k,
        u,
        w,
        vn_arg,
        g_arg,
        gk_arg,
        h,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        q_arg,
        o_arg,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
    )


def chunk_gated_delta_rule_fwd_h_flydsl(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
    variant: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FlyDSL K5 host wrapper.

    Signature is API-compatible with
    ``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h.chunk_gated_delta_rule_fwd_h_opt_vk``:

    Args:
        k: [B, T, Hg, K] bf16.
        w: [B, H, T_flat, K] bf16, head-major contiguous layout.
        u: [B, H, T_flat, V] bf16, head-major contiguous layout.
        g: [B, H, T_total] f32 cumulative gate, head-major contiguous
            (matches Triton VK / HIP K5), or None. Must be a
            ``contiguous()`` tensor with stride-1 along the T dimension.
            Caller passes ``g`` in natural-log space; when
            ``use_exp2=True`` the K1+K2 producer is expected to have
            already pre-scaled ``g`` by ``log2(e)`` (i.e. ``g`` is in
            log2 space) -- this matches the Triton VK convention and is
            NOT re-scaled by this wrapper.
        gk: [T_total, H, K] f32 per-K cumulative gate (natural-log
            space), or None. Pre-scaled to log2 space inside the wrapper
            when ``use_exp2=True``, mirroring
            ``chunk_gated_delta_rule_fwd_h_opt_vk``.
        initial_state: [N, H, V, K] f32, or None.
        output_final_state: whether to return the final hidden state.
        chunk_size: chunk size BT (default 64).
        save_new_value: whether to materialize ``v_new``.
        cu_seqlens: [N+1] LongTensor for variable-length batching, or None.
        state_dtype: optional initial/final state dtype (float32 or bfloat16).
        use_exp2: whether ``g`` is in log2 space. Standalone K5 callers pass
            natural-log ``g`` by default; end-to-end prefill passes the Triton
            K1 ``use_exp2`` setting through explicitly.
        num_decodes: number of leading decode-only sequences to skip in
            ``cu_seqlens``. When nonzero, ``cu_seqlens`` is the ORIGINAL,
            cache-stable metadata tensor (decode prefix included) and the
            data tensors (``k/w/u/g/...``) are expected to be pre-sliced to
            the prefill region; the offsets are rebased internally via the
            cached ``prepare_rebased_cu_seqlens``.
        num_decode_tokens: number of leading decode tokens stripped from the
            data tensors; subtracted from the rebased offsets so they index
            from token 0 of the prefill region.

    Returns:
        (h, v_new, final_state) in VK-ordered layout (``[..., V, K]`` on the
        last two dims).

    BV-tile selection is rule-based. ``chunk_gdn_h_tuned.csv`` remains an AOT
    seed list for pre-compilation, but runtime BV selection does not read it.
    """
    # Layout is fixed to head-major contiguous (matches Triton VK wrapper).
    wu_contiguous = True

    g_log2_scaled = bool(use_exp2)

    # SSM state dtype: derived from ``initial_state.dtype`` when provided,
    # otherwise from ``state_dtype`` kwarg, otherwise default f32 (matches
    # the legacy behaviour). Only ``torch.float32`` and ``torch.bfloat16``
    # are supported by the kernel.
    if initial_state is not None:
        resolved_state_dtype = initial_state.dtype
        if state_dtype is not None and state_dtype != resolved_state_dtype:
            raise ValueError(
                f"state_dtype={state_dtype} conflicts with "
                f"initial_state.dtype={initial_state.dtype}; pass them consistently "
                f"or omit state_dtype."
            )
    elif state_dtype is not None:
        resolved_state_dtype = state_dtype
    else:
        resolved_state_dtype = torch.float32
    if resolved_state_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"SSM state dtype must be float32 or bfloat16, got {resolved_state_dtype}."
        )
    state_bf16 = resolved_state_dtype == torch.bfloat16

    B, T, Hg, K = k.shape
    BT = chunk_size

    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
        kernel_cu_seqlens = None
    elif prefill_metadata is not None:
        prefill_metadata.validate(
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            total_prefill_tokens=T,
            num_sequences=len(cu_seqlens) - 1,
        )
        schedule = prefill_metadata.get_chunk_schedule(
            BT,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )
        chunk_offsets = schedule.chunk_offsets
        NT = schedule.total_chunks
        kernel_cu_seqlens = schedule.kernel_cu_seqlens
        N = schedule.n_prefill
    else:
        # Pass the ORIGINAL (cache-stable) cu_seqlens + the decode ints into
        # the cached prologue helpers. They all key on the original tensor's
        # identity, so chunk_offsets / NT / the rebased kernel cu_seqlens are
        # computed ONCE per (cu_seqlens_id, BT, num_decodes, num_decode_tokens)
        # tuple and every subsequent forward is a pure cache hit -> no
        # per-forward D2H. (Passing a freshly-rebased tensor instead would key
        # the offset/num-chunk caches on an unstable identity and re-fire the
        # .tolist()/int() syncs every call.)
        chunk_offsets = prepare_chunk_offsets(
            cu_seqlens, BT, num_decodes, num_decode_tokens
        )
        NT = prepare_num_chunks(cu_seqlens, BT, num_decodes, num_decode_tokens)
        # Rebased kernel-facing cu_seqlens (matches the pre-sliced prefill
        # data). N is the prefill sequence count (len() is a shape read, no
        # sync).
        kernel_cu_seqlens = prepare_rebased_cu_seqlens(
            cu_seqlens, num_decodes, num_decode_tokens
        )
        N = len(kernel_cu_seqlens) - 1

    assert K <= 256

    h = k.new_empty(B, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=resolved_state_dtype)
        if output_final_state
        else None
    )
    v_new_buf = k.new_empty(B, H, T_flat, V, dtype=u.dtype)
    v_new = v_new_buf if save_new_value else None

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)

    # G layout is fixed to head-major [B, H, T_flat] (matches Triton VK /
    # HIP K5). The kernel reads ``g`` with stride-1 along the T dim; require
    # the caller to provide a contiguous head-major tensor.
    if g is not None:
        assert g.is_contiguous(), (
            "FlyDSL K5: ``g`` must be contiguous (head-major [B, H, T_flat] "
            f"or [H, T_flat]); got strides={g.stride()}, shape={tuple(g.shape)}."
        )
        assert g.shape[-1] == T_flat, (
            f"FlyDSL K5: ``g.shape[-1]`` must equal T_flat={T_flat}, "
            f"got g.shape={tuple(g.shape)}."
        )
        assert g.shape[-2] == H, (
            f"FlyDSL K5: ``g.shape[-2]`` must equal H={H}, "
            f"got g.shape={tuple(g.shape)}."
        )
    g_arg = g if g is not None else dummy

    # Mirror the Triton VK wrapper: when ``use_exp2=True`` the K5 kernel
    # interprets ``gk`` in log2 space, so pre-scale by log2(e) here. The
    # kernel-side ``_fast_exp`` for ``gk`` is shared with the ``g`` path;
    # ``g`` itself must already be log2-scaled by the K1+K2 producer when
    # use_exp2 is on.
    if gk is not None:
        gk = gk.contiguous()
        if g_log2_scaled:
            gk = gk * _RCP_LN2
    gk_arg = gk if gk is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy
    vn_arg = v_new_buf
    # cu_arg / co_arg are the kernel-facing (rebased) offsets, narrowed to
    # int32. `.to(torch.int32)` is a device-to-device cast (no host sync); the
    # resulting fresh objects are consumed only by the kernel launch, so their
    # identity does not matter for the @tensor_cache helpers above.
    cu_arg = (
        kernel_cu_seqlens.to(torch.int32)
        if kernel_cu_seqlens is not None
        else dummy.to(torch.int32)
    )
    co_arg = (
        chunk_offsets.to(torch.int32)
        if chunk_offsets is not None
        else dummy.to(torch.int32)
    )
    stream = torch.cuda.current_stream()

    use_g = g is not None
    use_gk = gk is not None
    use_h0 = initial_state is not None
    is_varlen = cu_seqlens is not None

    # Resolve BV. Priority: explicit ``variant=`` > FLYDSL_GDN_K5_VARIANT env >
    # the rule-based grid/CU heuristic. Passing variant=None (the default) keeps
    # the historical behaviour exactly.
    BV, num_waves = _bv_waves_of_variant(
        _resolve_variant(
            variant,
            H=H,
            Hg=Hg,
            V=V,
            T_flat=T_flat,
            N=N,
            is_varlen=is_varlen,
            gate=_gate_of(use_g, use_gk),
        )
    )

    launch_fn = _get_or_compile(
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        use_g,
        use_gk,
        use_h0,
        output_final_state,
        save_new_value,
        is_varlen,
        wu_contiguous,
        state_bf16=state_bf16,
        g_log2_scaled=g_log2_scaled,
        num_waves=num_waves,
    )
    _launch_kernel(
        launch_fn,
        BV,
        V,
        N,
        H,
        k,
        u,
        w,
        vn_arg,
        g_arg,
        gk_arg,
        h,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        # q/o belong to the fused K5+K6 build (COMPUTE_OUTPUT=True); the K5
        # kernel declares but never dereferences them, so dummies suffice.
        dummy,
        dummy,
        T,
        T_flat,
        stream,
    )

    return h, v_new, final_state


# -- Fused K5+K6 compile cache + launch (gfx942) -------------------------

_compiled_fused_kernels: dict = {}


# Fused-vs-unfused selection threshold. Fuse when the fused kernel's actual 
# grid over-fills the device by at least this fraction.
_FUSED_MIN_FILL = 0.45

def should_use_fused_gfx942(
    *, H: int, N: int, V: int, gate: str | None = None, is_varlen: bool | None = None
) -> bool:
    """Heuristic: is the fused K5+K6 kernel the faster choice for this shape?

    Rule (gfx942): fuse iff the fused kernel's ACTUAL grid fill clears
    ``_FUSED_MIN_FILL``:

        fill = ceil(V / BV_run) * N * H / CU_count  >=  _FUSED_MIN_FILL

    where ``BV_run`` comes from ``_fused_bv_for_shape``. ``gate`` and
    ``is_varlen`` are accepted for signature compatibility but do not affect ``BV_run`.
    Off-arch always returns False (the fused kernel is tested only on gfx942).
    """
    if _ARCH != "gfx942":
        return False
    # Hg/T_flat/gate/is_varlen are unused by the fused BV rule; pass placeholders.
    bv_run, _ = _fused_bv_for_shape(
        H=H, Hg=H, V=V, T_flat=0, N=N, is_varlen=bool(is_varlen),
        gate=gate, variant=None,
    )
    fill = _grid_ctas(H=H, V=V, N=N, BV=bv_run) / max(_device_cu_count(), 1)
    return fill >= _FUSED_MIN_FILL


# Fused wave-widening: auto uses num_waves=8 (NR_SPLIT=2) when it selects BV=64
# AND the bv64 grid fills the device well enough that the extra resident waves
# have room to help.
_FUSED_W8_MIN_FILL = 0.55


def _fused_bv_for_shape(*, H, Hg, V, T_flat, N, is_varlen, gate, variant):
    """Return ``(BV, num_waves)`` for the fused kernel.

    An explicit ``variant`` is honoured (incl. ``bv64w8``); 
    otherwise the largest legal BV whose grid clears
    the fill bar is chosen, with num_waves=8 when that BV is 64 and the grid is
    high-fill. Wave-widening needs BT/16 (=4) divisible by NR_SPLIT and
    N_REPEAT=BV/16 divisible by NR_SPLIT, so it only applies at BV=64 here.
    """
    _FUSED_MAX_BV = 64
    legal = sorted(
        (b for b in _legal_bv_candidates(V) if b <= _FUSED_MAX_BV), reverse=True
    )
    if not legal:
        raise ValueError(
            f"fused GDN K5+K6: no legal BV <= {_FUSED_MAX_BV} for V={V}."
        )
    if variant is not None:
        bv, num_waves = _bv_waves_of_variant(variant)
        if bv not in legal:
            raise ValueError(
                f"fused GDN K5+K6 variant {variant!r} illegal for V={V}; "
                f"legal: {[f'bv{b}' for b in legal]}"
            )
        # NR_SPLIT = num_waves / (BT//16); the kernel asserts it divides
        # N_REPEAT=BV/16 and BT//16, so w8 is only legal at BV=64.
        # The fused kernel's b_A key-split needs N_REPEAT>=NR_SPLIT.
        if num_waves > 4 and bv < 64:
            raise NotImplementedError(
                f"fused GDN K5+K6 variant {variant!r}: wave-widening is only "
                f"supported at BV=64 (needs N_REPEAT >= NR_SPLIT); use bv64w8."
            )
        return bv, num_waves
    # Measured-best fused variant for this signature (gfx942 only); a miss falls
    # through to the grid-fill heuristic below.
    if _ARCH == "gfx942":
        tuned = _gfx942_select_fused_variant(
            gate=gate, H=H, N=N, V=V, is_varlen=is_varlen
        )
        if tuned is not None:
            return _bv_waves_of_variant(tuned)
    bv = min(
        _select_bv_for_grid(
            H=H, V=V, N=N, target_ctas=int(_GFX942_MIN_FILL * _device_cu_count())
        ),
        _FUSED_MAX_BV,
    )
    num_waves = 4
    if bv == 64:
        fill64 = _grid_ctas(H=H, V=V, N=N, BV=64) / _device_cu_count()
        if fill64 >= _FUSED_W8_MIN_FILL:
            num_waves = 8
    return bv, num_waves


def _run_fused_gfx942(
    *, q, k, w, u, g, gk, o, scale, initial_state, output_final_state,
    chunk_size, cu_seqlens, state_dtype, use_exp2, num_decodes,
    num_decode_tokens, prefill_metadata, variant,
):
    """gfx942 fused K5+K6 launch. Mirrors the K5 wrapper's input prep."""
    g_log2_scaled = bool(use_exp2)

    if initial_state is not None:
        resolved_state_dtype = initial_state.dtype
        if state_dtype is not None and state_dtype != resolved_state_dtype:
            raise ValueError(
                f"state_dtype={state_dtype} conflicts with "
                f"initial_state.dtype={initial_state.dtype}."
            )
    elif state_dtype is not None:
        resolved_state_dtype = state_dtype
    else:
        resolved_state_dtype = torch.float32
    if resolved_state_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"SSM state dtype must be float32 or bfloat16, got {resolved_state_dtype}."
        )
    state_bf16 = resolved_state_dtype == torch.bfloat16

    B, T, Hg, K = k.shape
    BT = chunk_size
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
        kernel_cu_seqlens = None
    elif prefill_metadata is not None:
        prefill_metadata.validate(
            cu_seqlens=cu_seqlens, chunk_size=BT, num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens, total_prefill_tokens=T,
            num_sequences=len(cu_seqlens) - 1,
        )
        schedule = prefill_metadata.get_chunk_schedule(
            BT, num_decodes=num_decodes, num_decode_tokens=num_decode_tokens,
        )
        chunk_offsets = schedule.chunk_offsets
        NT = schedule.total_chunks
        kernel_cu_seqlens = schedule.kernel_cu_seqlens
        N = schedule.n_prefill
    else:
        chunk_offsets = prepare_chunk_offsets(
            cu_seqlens, BT, num_decodes, num_decode_tokens
        )
        NT = prepare_num_chunks(cu_seqlens, BT, num_decodes, num_decode_tokens)
        kernel_cu_seqlens = prepare_rebased_cu_seqlens(
            cu_seqlens, num_decodes, num_decode_tokens
        )
        N = len(kernel_cu_seqlens) - 1

    assert K <= 256

    use_g = g is not None
    use_gk = gk is not None
    use_h0 = initial_state is not None
    is_varlen = cu_seqlens is not None

    final_state = (
        k.new_empty(N, H, V, K, dtype=resolved_state_dtype)
        if output_final_state
        else None
    )

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)

    if g is not None:
        assert g.is_contiguous(), "fused K5+K6: g must be contiguous head-major."
        assert g.shape[-1] == T_flat and g.shape[-2] == H, (
            f"fused K5+K6: g must be [.., H={H}, T_flat={T_flat}]; got {tuple(g.shape)}."
        )
    g_arg = g if g is not None else dummy

    if gk is not None:
        gk = gk.contiguous()
        if g_log2_scaled:
            gk = gk * _RCP_LN2
    gk_arg = gk if gk is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy

    cu_arg = (
        kernel_cu_seqlens.to(torch.int32)
        if kernel_cu_seqlens is not None
        else dummy.to(torch.int32)
    )
    co_arg = (
        chunk_offsets.to(torch.int32) if chunk_offsets is not None
        else dummy.to(torch.int32)
    )
    stream = torch.cuda.current_stream()

    BV, num_waves = _fused_bv_for_shape(
        H=H, Hg=Hg, V=V, T_flat=T_flat, N=N, is_varlen=is_varlen,
        gate=_gate_of(use_g, use_gk), variant=variant,
    )
    NR_SPLIT = num_waves // (BT // 16)

    cache_key = (
        K, V, BT, BV, H, Hg, float(scale), use_g, use_gk, use_h0,
        output_final_state, is_varlen, state_bf16, g_log2_scaled, NR_SPLIT,
    )
    if cache_key not in _compiled_fused_kernels:
        # Fused build of the shared gfx942 builder: COMPUTE_OUTPUT appends the
        # K6 stage, while STORE_H / SAVE_NEW_VALUE drop the two HBM drains that
        # only the separate K6 kernel consumes -- that elision is the whole
        # point of fusing.
        _compiled_fused_kernels[cache_key] = compile_chunk_gated_delta_h_gfx942(
            K=K, V=V, BT=BT, BV=BV, H=H, Hg=Hg,
            USE_G=use_g, USE_GK=use_gk, USE_INITIAL_STATE=use_h0,
            STORE_FINAL_STATE=output_final_state, SAVE_NEW_VALUE=False,
            IS_VARLEN=is_varlen,
            WU_CONTIGUOUS=True, STATE_DTYPE_BF16=state_bf16,
            G_IS_LOG2_SCALED=g_log2_scaled, NR_SPLIT=NR_SPLIT,
            COMPUTE_OUTPUT=True, STORE_H=False, SCALE=float(scale),
        )
    launch_fn = _compiled_fused_kernels[cache_key]

    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H
    # Unified kernel arg order; v_new and h are unused on the fused build.
    _run_compiled(
        launch_fn,
        k, u, w, dummy, g_arg, gk_arg, dummy, h0_arg, ht_arg, cu_arg, co_arg,
        q, o,
        T, T_flat, N, grid_v, grid_nh, stream,
    )

    return o, final_state


def chunk_gated_delta_rule_fwd_h_o_flydsl(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
    variant: str | None = None,
    o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FlyDSL fused K5+K6 host wrapper (GDN inter-chunk scan + output).

    Fuses the inter-chunk hidden-state recurrence (K5,
    ``chunk_gated_delta_rule_fwd_h_flydsl``) with the inter/intra-chunk output
    (K6, ``chunk_fwd_o_opt_vk``) into a single call. The fused kernel eliminates
    the ``h`` snapshot and ``v_new`` HBM round-trips between the two stages.

    Args:
        q: [B, T, Hg, K] bf16. Query; K6 scales it by ``scale``. Token-major,
            ``Hg``-strided (GQA: ``Hg`` may be < ``H``).
        k: [B, T, Hg, K] bf16. Shared by K5 (state scan) and K6 (q @ k^T).
        w: [B, H, T_flat, K] bf16, head-major contiguous (K5 input).
        u: [B, H, T_flat, V] bf16, head-major contiguous (K5 input).
        g: [B, H, T_flat] or [H, T_flat] f32 cumulative scalar gate, head-major
            contiguous, or None. Same convention as
            ``chunk_gated_delta_rule_fwd_h_flydsl``: when ``use_exp2=True`` the
            caller must have pre-scaled ``g`` to log2 space. Drives both the K5
            decay and the K6 output gate.
        gk: [T_flat, H, K] f32 per-channel cumulative gate (KDA), or None.
            Pre-scaled internally when ``use_exp2=True``. The per-channel decay
            is folded into ``h``/``v_new`` by K5; the K6 output path is ungated
            for the ``gk`` case (matches the Triton K6, which takes only scalar
            ``g``).
        scale: query scale for K6. Defaults to ``K ** -0.5`` when None.
        initial_state: [N, H, V, K] f32, or None.
        output_final_state: whether to return the final hidden state.
        chunk_size: chunk size BT (default 64).
        cu_seqlens: [N+1] LongTensor for variable-length batching, or None.
        state_dtype: optional initial/final state dtype (float32 or bfloat16).
        use_exp2: whether ``g``/``gk`` are in log2 space (see K5 wrapper).
        num_decodes / num_decode_tokens: leading decode prefix skipped in
            ``cu_seqlens`` (data tensors pre-sliced); see K5 wrapper.
        prefill_metadata: prebuilt reusable schedule; see K5 wrapper.
        variant: explicit K5 BV-variant tag, or None for the auto heuristic.
        o: optional pre-allocated [B, T, H, V] output buffer, written in place.
            A fresh buffer is allocated when None.

    Returns:
        (o, final_state) with ``o`` in token-major ``[B, T, H, V]`` (bf16) and
        ``final_state`` in ``[N, H, V, K]`` (``state_dtype``) or None.
    """
    exactly_one_gate = (g is None) != (gk is None)
    if not exactly_one_gate:
        raise ValueError(
            "chunk_gated_delta_rule_fwd_h_o_flydsl: exactly one of g, gk must be "
            "provided."
        )

    B, T, Hg, K = q.shape
    V = u.shape[-1]
    if scale is None:
        scale = K ** -0.5

    if o is None:
        # Token-major [B, T, H, V], matching the Triton K6 output layout.
        H = w.shape[1]
        o = u.new_empty(B, T, H, V, dtype=u.dtype)

    # The fused kernel is gfx942-only: it is the sole implementation of this
    # entry point (there is no separate-K5/K6 fallback here -- callers that need
    # one route through the pipeline's K5K6Fusion.NEVER path instead).
    if _ARCH != "gfx942":
        raise NotImplementedError(
            f"chunk_gated_delta_rule_fwd_h_o_flydsl: the fused GDN K5+K6 kernel "
            f"is implemented for gfx942 only; got arch '{_ARCH}'. Use the "
            f"separate K5 + Triton K6 path on this arch."
        )

    return _run_fused_gfx942(
        q=q, k=k, w=w, u=u, g=g, gk=gk, o=o, scale=scale,
        initial_state=initial_state, output_final_state=output_final_state,
        chunk_size=chunk_size, cu_seqlens=cu_seqlens, state_dtype=state_dtype,
        use_exp2=use_exp2, num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens, prefill_metadata=prefill_metadata,
        variant=variant,
    )


def chunk_gated_delta_rule_fwd_h_o_auto(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata=None,
    variant: str | None = None,
    o: torch.Tensor | None = None,
    fusion=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Combined K5+K6 wrapper that applies the fused-vs-separate heuristic.

    Routes to ``chunk_gated_delta_rule_fwd_h_o_flydsl`` (fused, gfx942 only)
    when ``should_use_fused_gfx942`` returns True, otherwise falls back to
    FlyDSL K5 + Triton K6 run sequentially.

    Args:
        fusion: ``K5K6Fusion`` enum value (or string / None). None / AUTO lets
            the shape heuristic decide. ALWAYS forces the fused path; NEVER
            forces the separate path. On non-gfx942 archs, AUTO and NEVER both
            use the separate path (fused is gfx942-only).
        All other args: same as ``chunk_gated_delta_rule_fwd_h_o_flydsl``.

    Returns:
        (o, final_state) as ``chunk_gated_delta_rule_fwd_h_o_flydsl``.
    """
    resolved_fusion = K5K6Fusion.coerce(fusion)
    H = w.shape[1]
    V = u.shape[-1]
    N = (cu_seqlens.shape[0] - 1) if cu_seqlens is not None else q.shape[0]
    gate = _gate_of(g is not None, gk is not None)
    is_varlen = cu_seqlens is not None

    use_fused = (
        resolved_fusion == K5K6Fusion.ALWAYS
        or (
            resolved_fusion == K5K6Fusion.AUTO
            and should_use_fused_gfx942(
                H=H, N=N, V=V, gate=gate, is_varlen=is_varlen
            )
        )
    )

    if use_fused:
        return chunk_gated_delta_rule_fwd_h_o_flydsl(
            q=q, k=k, w=w, u=u, g=g, gk=gk, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            chunk_size=chunk_size, cu_seqlens=cu_seqlens, state_dtype=state_dtype,
            use_exp2=use_exp2, num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens, prefill_metadata=prefill_metadata,
            variant=variant, o=o,
        )

    # Separate path: FlyDSL K5 + Triton K6.
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl(
        k=k, w=w, u=u, g=g, gk=gk, initial_state=initial_state,
        output_final_state=output_final_state, chunk_size=chunk_size,
        save_new_value=True, cu_seqlens=cu_seqlens, state_dtype=state_dtype,
        use_exp2=use_exp2, num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens, prefill_metadata=prefill_metadata,
        variant=variant,
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if o is None:
        B, T = q.shape[:2]
        o = u.new_empty(B, T, H, V)
    from ..triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
        chunk_fwd_o_opt_vk,
    )
    chunk_fwd_o_opt_vk(
        q=q, k=k, v=v_new, o=o, h=h, g=g, scale=scale,
        cu_seqlens=cu_seqlens, use_exp2=use_exp2,
        num_decodes=num_decodes, num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )
    return o, final_state
