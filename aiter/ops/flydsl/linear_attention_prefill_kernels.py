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

import functools
import math
import os

# NOTE (mfma16_hip fork): ``flydsl.compiler`` / ``flydsl.expr`` /
# ``get_rocm_arch`` are imported here for the additive HIP-aligned fork below.
# They are side-effect-free (``flydsl`` is already a hard dependency of the
# baseline ``compile_chunk_gated_delta_h``) and do NOT raise on flydsl <0.2.0 --
# the mfma16_hip-only ``>=0.2.0`` requirement (fx.Pointer ABI) is enforced
# lazily in ``_get_or_compile_mfma16_hip`` so the baseline path keeps its
# original ``>=0.1.8`` compatibility.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
from flydsl.runtime.device import get_rocm_arch

from ..triton._triton_kernels.gated_delta_rule.utils import (
    prepare_chunk_offsets,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)
from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from .kernels.tensor_shim import _run_compiled

# log2(e); g pre-scaled by this constant lets the kernel use exp2(g) in
# place of exp(g) (matches the Triton VK / HIP K5 convention).
_RCP_LN2 = math.log2(math.e)


__all__ = [
    "chunk_gated_delta_rule_fwd_h_flydsl",
    "chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip",
]


# -- K5 host wrapper (FlyDSL kernel + rule-based BV selection) ------------

_compiled_kernels = {}
_BV_CANDIDATES = [16, 32, 64]
_DEFAULT_BV = 16


def _legal_bv_candidates(V: int) -> list[int]:
    return [c for c in _BV_CANDIDATES if c <= V and V % c == 0]


def _grid_ctas(*, H: int, V: int, N: int, BV: int) -> int:
    return max(1, N) * H * ((V + BV - 1) // BV)


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
    """Select ``BV`` with the rule-based grid/CU heuristic."""
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
    target_bv = _target_bv_for_shape(
        H=H, Hg=Hg, T_flat=T_flat, N=N, is_varlen=is_varlen
    )
    target_ctas = (
        _grid_ctas(H=H, V=V, N=N, BV=target_bv) if target_bv is not None else 256
    )
    return _select_bv_for_grid(H=H, V=V, N=N, target_ctas=target_ctas)


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
    )
    if cache_key not in _compiled_kernels:
        _compiled_kernels[cache_key] = compile_chunk_gated_delta_h(
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

    # Resolve BV from the rule-based grid/CU heuristic.
    BV = _lookup_tuned_bv(
        dtype_str=str(k.dtype),
        K=K,
        V=V,
        BT=BT,
        H=H,
        Hg=Hg,
        T_flat=T_flat,
        N=N,
        use_g=use_g,
        use_gk=use_gk,
        use_h0=use_h0,
        store_fs=bool(output_final_state),
        save_vn=bool(save_new_value),
        is_varlen=is_varlen,
        wu_contig=wu_contiguous,
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
        T,
        T_flat,
        stream,
    )

    return h, v_new, final_state


# ==========================================================================
# mfma16_hip fork (additive) -- HIP-aligned FlyDSL K5 implementation.
#
# Everything below is self-contained and does NOT touch the baseline wrapper
# above: it has its own compiled-kernel cache, BV selection (reusing the hip
# K5 selector), pointer-ABI launch, and public entry point
# ``chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip``. The baseline path keeps
# its original behaviour / flydsl>=0.1.8 compatibility; the mfma16_hip fork's
# fx.Pointer ABI requires flydsl>=0.2.0, enforced lazily below.
# ==========================================================================

# mfma16_hip fork requires the fx.Pointer argument ABI (flyc.from_c_void_p +
# PointerAdaptor fast dispatch), which only exists from flydsl 0.2.0. Enforced
# lazily (in ``_get_or_compile_mfma16_hip``) so importing this module and using
# the baseline wrapper keeps working on flydsl>=0.1.8.
_MFMA16_HIP_MIN_FLYDSL_VERSION = "0.2.0"

# gfx942 gate: only the mfma16_hip fork toggles the gfx942 GEMM1 ds-scheduling
# (SCHED_GFX942). ``get_rocm_arch()`` may return a feature-suffixed string like
# ``gfx942:sramecc+:xnack-``; normalize before matching.
_IS_GFX942 = get_rocm_arch().split(":")[0].startswith("gfx942")

_INT32_ATTR = "_flydsl_int32_view"
_PROLOGUE_ATTR = "_flydsl_prologue_cache"


def _as_ptr(t: torch.Tensor):
    """Wrap a torch tensor as a flydsl ``Pointer`` argument (raw data ptr).

    Uses ``fx.Uint8`` element type: the mfma16_hip kernel re-types every slot
    inside the body via ``GTensor(ptr, dtype=...)``, so the host-side element
    type is irrelevant to codegen and only needs to be a valid 1-byte unit.
    """
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


def _as_int32(t: torch.Tensor) -> torch.Tensor:
    """Return an int32 narrowing of ``t``, cached on the tensor itself.

    ``t`` is expected to come from one of the ``@tensor_cache``-decorated
    prologue helpers (so its identity is stable across forwards). The cached
    int32 result lives as an attribute on ``t`` itself, keeping cache
    invalidation trivially correct.
    """
    if t.dtype == torch.int32:
        return t
    cached = getattr(t, _INT32_ATTR, None)
    if cached is None:
        cached = t.to(torch.int32)
        try:
            object.__setattr__(t, _INT32_ATTR, cached)
        except (AttributeError, TypeError):
            pass
    return cached


def _resolve_prologue(
    cu_seqlens: torch.Tensor,
    BT: int,
    num_decodes: int,
    num_decode_tokens: int,
):
    """Resolve the per-shape varlen prologue in one cached lookup.

    Collapses the three ``@tensor_cache``-decorated prologue helpers into a
    single tuple attached to ``cu_seqlens`` (keyed by ``(BT, num_decodes,
    num_decode_tokens)``), so repeat forwards on the same ``cu_seqlens`` tensor
    are one ``getattr`` + one dict get.

    Returns ``(NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen)``.
    """
    cache_key = (BT, num_decodes, num_decode_tokens)
    cache = getattr(cu_seqlens, _PROLOGUE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            object.__setattr__(cu_seqlens, _PROLOGUE_ATTR, cache)
        except (AttributeError, TypeError):
            cache = None
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    chunk_offsets = prepare_chunk_offsets(
        cu_seqlens, BT, num_decodes, num_decode_tokens
    )
    NT = prepare_num_chunks(cu_seqlens, BT, num_decodes, num_decode_tokens)
    kernel_cu_seqlens = prepare_rebased_cu_seqlens(
        cu_seqlens, num_decodes, num_decode_tokens
    )
    N = len(kernel_cu_seqlens) - 1
    if N >= 1:
        seg_lens = kernel_cu_seqlens[1:] - kernel_cu_seqlens[:-1]
        min_seqlen = int(seg_lens.min().item())
    else:
        min_seqlen = None
    result = (NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen)
    if cache is not None:
        cache[cache_key] = result
    return result


def _resolve_state_dtype(initial_state, state_dtype):
    """Resolve/validate the SSM state dtype (float32 or bfloat16)."""
    if initial_state is not None:
        resolved = initial_state.dtype
        if state_dtype is not None and state_dtype != resolved:
            raise ValueError(
                f"state_dtype={state_dtype} conflicts with "
                f"initial_state.dtype={initial_state.dtype}; pass them "
                f"consistently or omit state_dtype."
            )
    elif state_dtype is not None:
        resolved = state_dtype
    else:
        resolved = torch.float32
    if resolved not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"SSM state dtype must be float32 or bfloat16, got {resolved}."
        )
    return resolved


@functools.cache
def _get_or_compile_mfma16_hip(
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
    use_state_indices=False,
    sched_gfx942=False,
):
    """Compile (and cache) the mfma16 / HIP-aligned K5 kernel: 16x16x16 bf16
    MFMA + HIP-matching warp partition, writing the public VK layout [..., V, K].

    ``use_state_indices`` compiles the indexed state-pool variant: the SSM
    ``initial_state`` is a pool ``[pool_size, H, V, K]`` and each sequence's slot
    is gathered from an ``initial_state_indices[N]`` int32 array (with in-place
    final-state write-back into the same pool slot), mirroring the HIP kernel.

    The hip compile module + its flydsl>=0.2.0 requirement are imported lazily
    here so the baseline path is unaffected.
    """
    import flydsl
    from packaging.version import Version

    installed = Version(getattr(flydsl, "__version__", "0").split("+")[0])
    if installed < Version(_MFMA16_HIP_MIN_FLYDSL_VERSION):
        raise ImportError(
            "FlyDSL K5 mfma16_hip fork requires `flydsl` "
            f">=`{_MFMA16_HIP_MIN_FLYDSL_VERSION}` (for the fx.Pointer argument "
            f"ABI), but got `{getattr(flydsl, '__version__', 'unknown')}`."
        )

    from .kernels.chunk_gated_delta_h_mfma16_hip import (
        compile_chunk_gated_delta_h_mfma16_hip,
    )

    return compile_chunk_gated_delta_h_mfma16_hip(
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
        USE_STATE_INDICES=use_state_indices,
        SCHED_GFX942=sched_gfx942,
    )


def chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
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
    initial_state_indices: torch.Tensor | None = None,
    inplace_final_state: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """mfma16 / HIP-aligned K5 implementation: NON-VWARP only -- uses the
    16x16x16 bf16 MFMA and the SAME split-M warp partition (BT split-M, K split
    across waves, V not split across warps) as the hand-tuned HIP/C++ K5 kernel,
    writing the public VK layout [..., V, K]. API-compatible with
    ``chunk_gated_delta_rule_fwd_h_flydsl`` (plus the indexed state-pool
    contract via ``initial_state_indices`` / ``inplace_final_state``, matching
    ``chunk_gated_delta_rule_fwd_h_hip_fn``).

    Unlike the baseline wrapper, BV is chosen by the hip K5 selector
    (``_select_bv_for_varlen`` / ``_select_bv_for_dense``) so it matches
    ``chunk_gated_delta_rule_fwd_h_hip_fn`` exactly; ``FLYDSL_K5_MFMA16HIP_BV``
    (in {16,32,64}) overrides it for A/B sweeps.
    """
    use_g = g is not None
    use_gk = gk is not None
    use_h0 = initial_state is not None
    g_log2_scaled = bool(use_exp2)

    # Indexed state-pool support: when ``initial_state_indices`` is given,
    # ``initial_state`` is a pool ``[pool_size, H, V, K]`` and each sequence
    # gathers its slot from the index array; the final state is written back
    # in place into that same pool. ``inplace_final_state`` defaults to True
    # whenever indices are given.
    use_state_indices = initial_state_indices is not None
    inplace = use_state_indices if inplace_final_state is None else inplace_final_state
    if use_state_indices:
        if initial_state is None:
            raise ValueError(
                "FlyDSL K5: initial_state_indices requires initial_state (the "
                "state pool)."
            )
        if not inplace:
            raise ValueError(
                "FlyDSL K5: initial_state_indices requires in-place final-state "
                "write-back; leave inplace_final_state unset or set it to True."
            )
        if not output_final_state:
            raise ValueError(
                "FlyDSL K5: initial_state_indices requires output_final_state=True "
                "(the indexed path writes the final state back into the pool)."
            )
    elif inplace and initial_state is None:
        raise ValueError("FlyDSL K5: inplace_final_state requires initial_state.")

    resolved_state_dtype = _resolve_state_dtype(initial_state, state_dtype)
    state_bf16 = resolved_state_dtype is torch.bfloat16

    # mfma16_hip keeps the token-major [B, T_flat, Hg, K] k layout (no
    # host-side pre-transpose), matching the Triton VK convention.
    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size
    assert K <= 256

    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T, BT)
        chunk_offsets = None
        kernel_cu_seqlens = None
        is_varlen = False
    else:
        NT, chunk_offsets, kernel_cu_seqlens, N, _min_seqlen = _resolve_prologue(
            cu_seqlens, BT, num_decodes, num_decode_tokens
        )
        is_varlen = True

    # BV selection: reuse the hip K5 analytic selector so the fork picks the
    # same BV as ``chunk_gated_delta_rule_fwd_h_hip_fn`` (lazy import avoids a
    # potential circular import; both sides key on the same chunk_offsets).
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        _select_bv_for_dense as _hip_select_bv_for_dense,
    )
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        _select_bv_for_varlen as _hip_select_bv_for_varlen,
    )

    if is_varlen:
        BV = _hip_select_bv_for_varlen(chunk_offsets, H)
    else:
        BV = _hip_select_bv_for_dense(B, T_flat, chunk_size, H, k.device)

    # Env override for A/B BV sweeps; the hand-tuned HIP K5 reference is fixed
    # at BV=16 (FLYDSL_K5_MFMA16HIP_BV=16 reproduces it).
    _bv_env = os.environ.get("FLYDSL_K5_MFMA16HIP_BV")
    if _bv_env:
        BV = int(_bv_env)
    assert BV in (16, 32, 64), "mfma16_hip BV must be in {16,32,64}"
    if V % BV != 0:
        raise ValueError(
            f"FlyDSL K5 mfma16_hip: requires V % BV == 0; got V={V}, BV={BV}."
        )

    # SCHED_GFX942 is only enabled on gfx942; other arches (incl. gfx950) pass
    # False, keeping their emitted code byte-identical, and it joins the
    # lru_cache key as a distinct compiled product.
    launch_fn = _get_or_compile_mfma16_hip(
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
        True,
        state_bf16=state_bf16,
        g_log2_scaled=g_log2_scaled,
        use_state_indices=use_state_indices,
        sched_gfx942=_IS_GFX942,
    )

    # Null-arg placeholder for the @flyc.jit slots ignored on this path. Sized
    # 1 (not 0) so its ``data_ptr()`` is always a valid non-null device address.
    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    int32_dummy = dummy.to(torch.int32) if not is_varlen else None
    cu_arg = (
        _as_int32(kernel_cu_seqlens) if kernel_cu_seqlens is not None else int32_dummy
    )
    co_arg = _as_int32(chunk_offsets) if chunk_offsets is not None else int32_dummy
    stream = torch.cuda.current_stream(k.device)

    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H

    # mfma16_hip writes the public VK layout ([..., V, K]) directly.
    h_shape = (B, NT, H, V, K)
    vn_shape = (B, H, T_flat, V)
    vn_dtype = u.dtype
    fs_shape = (N, H, V, K) if output_final_state else None
    fs_dtype = resolved_state_dtype if output_final_state else None
    save_vn = save_new_value

    # G contiguity guard (stride-1 along T; head-major [B, H, T_flat]).
    if g is not None and not g.is_contiguous():
        raise AssertionError(
            "FlyDSL K5: ``g`` must be contiguous (head-major [B, H, T_flat] "
            f"or [H, T_flat]); got strides={g.stride()}, shape={tuple(g.shape)}."
        )

    # gk pre-scaling to log2 space (mirrors the Triton VK wrapper).
    if gk is not None:
        gk = gk.contiguous()
        if g_log2_scaled:
            gk = gk * _RCP_LN2

    h = k.new_empty(h_shape)
    v_new_buf = k.new_empty(vn_shape, dtype=vn_dtype)
    if fs_shape is None:
        final_state = None
    elif inplace:
        # In-place write-back: the final state aliases the ``initial_state``
        # buffer (the pool when indexed, or the dense [N,H,V,K] state
        # otherwise), so no separate output tensor is allocated.
        final_state = initial_state
    else:
        final_state = k.new_empty(fs_shape, dtype=fs_dtype)

    # The 11 tensor slots, wrapped as raw fx.Pointer args. Keep the torch
    # tensors referenced as locals so the storage stays alive across the
    # (synchronous) launch -- ``from_c_void_p`` only captures the data pointer.
    tensor_args = tuple(
        _as_ptr(t)
        for t in (
            k,
            u,
            w,
            v_new_buf,
            g if g is not None else dummy,
            gk if gk is not None else dummy,
            h,
            initial_state if initial_state is not None else dummy,
            final_state if final_state is not None else dummy,
            cu_arg,
            co_arg,
        )
    )

    # The mfma16_hip kernel carries an extra ``state_indices`` slot (12th tensor
    # arg): a real int32 [N] index array when indexed, else a 1-elem int32 dummy.
    if use_state_indices:
        si_i32 = initial_state_indices.to(torch.int32).contiguous()
        if si_i32.numel() != N:
            raise ValueError(
                "FlyDSL K5: initial_state_indices length "
                f"({si_i32.numel()}) must equal the number of sequences N={N}."
            )
    else:
        si_i32 = dummy.to(torch.int32)
    tensor_args = tensor_args + (_as_ptr(si_i32),)

    launch_fn(
        *tensor_args,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
    )

    return h, (v_new_buf if save_vn else None), final_state
