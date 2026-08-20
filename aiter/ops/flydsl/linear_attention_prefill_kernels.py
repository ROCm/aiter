# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL host wrappers for GDN prefill.

The module exposes hidden-state and prepare kernels. The prepare wrapper matches
the Triton prepare layout and exponent-domain contract, so both paths can be
selected independently through ``chunk_gated_delta_rule_opt_vk``. Call
``gdn_prepare_flydsl_supported`` before selecting the fused prepare path.
"""

from __future__ import annotations

import functools
import math
import os

# NOTE (mfma16_hip fork): ``get_rocm_arch`` is imported here for the additive
# HIP-aligned fork below. It is side-effect-free (``flydsl`` is already a hard
# dependency of the baseline ``compile_chunk_gated_delta_h``) and does NOT raise
# on flydsl <0.2.0 -- the mfma16_hip-only ``>=0.2.0`` requirement is enforced
# lazily in ``_get_or_compile_mfma16_hip`` so the baseline path keeps its
# original ``>=0.1.8`` compatibility.
import torch
import triton
from flydsl.runtime.device import get_rocm_arch as _get_rocm_arch

from ..triton._triton_kernels.gated_delta_rule.utils import (
    GatedDeltaRulePrefillMetadata,
    K5K6Fusion,
    prepare_chunk_offsets,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)
from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from .kernels.gdn_prepare import compile_gdn_prepare
from .kernels.chunk_gated_delta_h_gfx942 import (
    compile_chunk_gated_delta_h_gfx942,
)
from .kernels.chunk_gated_delta_h_gfx942 import (
    select_fused_variant as _gfx942_select_fused_variant,
)
from .kernels.chunk_gated_delta_h_gfx942 import (
    select_variant as _gfx942_select_variant,
)

# Arch-agnostic K5 variant tag grammar + legality (shared with the gfx942 kernel
# module -- kept in its own module to avoid an import cycle). Re-exported here so
# existing callers (bench, AOT, tests) keep importing them from this wrapper.
from .kernels.k5_variants import (  # noqa: F401  (re-exported)
    _BV_CANDIDATES,
    _DEFAULT_BV,
    K5_DEFAULT_VARIANT,
    K5_VARIANTS,
    _bv_of_variant,
    _bv_waves_of_variant,
    _legal_bv_candidates,
)
from .kernels.tensor_shim import _run_compiled

# Arch detected once at import time; used by _get_or_compile to select the
# right compile backend.
_ARCH: str = _get_rocm_arch()

# log2(e); g pre-scaled by this constant lets the kernel use exp2(g) in
# place of exp(g) (matches the Triton VK / HIP convention).
_RCP_LN2 = math.log2(math.e)


__all__ = [
    "K5K6Fusion",
    "chunk_gated_delta_rule_fwd_h_flydsl",
    "chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip",
    "gdn_prepare_flydsl_supported",
    "gdn_prepare_fwd_flydsl",
    "chunk_gated_delta_rule_fwd_h_o_auto",
    "chunk_gated_delta_rule_fwd_h_o_flydsl",
    "should_use_fused_gfx942",
]


# -- Hidden-state host wrapper (FlyDSL kernel + rule-based BV selection) ---

_compiled_kernels = {}


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


def _auto_variant(**kw) -> str:
    """The shape-adaptive choice, as a variant tag.

    Arch dispatcher: on gfx942, defer to the gfx942 kernel module's measured
    ``H*N`` rule (``select_variant``). On any other arch (or when the rule's BV
    is illegal for this V) fall back to ``_heuristic_bv``.
    """
    if _ARCH == "gfx942":
        tuned = _gfx942_select_variant(H=kw["H"], N=kw["N"], V=kw["V"])
        if tuned is not None:
            return tuned
    return f"bv{_heuristic_bv(**kw)}"


def _resolve_variant(variant: str | None, **kw) -> str:
    """Effective variant tag: explicit arg > env var > shape-adaptive.

    Mirrors the resolution chain used by the fp8_mqa_logits kernel.
    """
    tag = variant or os.environ.get(_K5_VARIANT_ENV) or _auto_variant(**kw)
    if tag == K5_DEFAULT_VARIANT:  # env var may legitimately say "auto"
        tag = _auto_variant(**kw)
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
            trace-calibrated rules to the hidden-state H=32/Hg=16 family.

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


# -- HIP-equivalent BV selector (frozen, self-contained copy) --------------
# The mfma16_hip fork below picks BV to match the hand-tuned HIP K5 kernel
# (``aiter.ops.chunk_gated_delta_rule_fwd_h``) point-for-point. Rather than
# importing that module's private ``_select_bv`` -- whose name/signature drift
# with mainline HIP retunes and have already broken this fork once -- we keep a
# frozen copy of its LDS/CU-threshold algorithm here. This intentionally does
# NOT track future mainline HIP changes; re-sync deliberately if the HIP
# heuristic is retuned and parity is still desired.
_HIPEQ_BV_FIXED_LDS_BYTES = 32 * 1024
_HIPEQ_BV_LDS_BYTES_PER_BV = 512
_HIPEQ_BV_RESIDENT_WGS_CAP = 2
_HIPEQ_BV_CANDIDATES = (64, 32, 16)
_HIPEQ_BV_CACHE: dict[tuple[int, int, int, int], int] = {}


def _hipeq_device_idx(device: torch.device) -> int:
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _hipeq_shared_memory_per_cu(props: object) -> int:
    """Per-CU shared memory with architecture-based fallback."""
    shared_per_cu = getattr(props, "shared_memory_per_multiprocessor", None)
    if shared_per_cu is not None:
        return int(shared_per_cu)
    arch = getattr(props, "gcnArchName", "")
    if arch:
        arch = arch.split(":")[0]
    _arch_lds = {"gfx95": 128 * 1024, "gfx94": 64 * 1024}
    for prefix, size in _arch_lds.items():
        if arch.startswith(prefix):
            return size
    shared_per_block = getattr(props, "shared_memory_per_block", None)
    if shared_per_block is not None:
        return int(shared_per_block)
    raise RuntimeError("Unable to determine shared memory per CU.")


def _hipeq_compute_bv(
    device: torch.device, total_chunks: int, max_seq_chunks: int, num_heads: int
) -> int:
    props = torch.cuda.get_device_properties(device)
    num_cus = props.multi_processor_count
    lds_per_cu = _hipeq_shared_memory_per_cu(props)
    for bv in _HIPEQ_BV_CANDIDATES:
        lds_per_wg = _HIPEQ_BV_FIXED_LDS_BYTES + _HIPEQ_BV_LDS_BYTES_PER_BV * bv
        resident = min(max(1, lds_per_cu // lds_per_wg), _HIPEQ_BV_RESIDENT_WGS_CAP)
        total_wgs = (128 // bv) * num_heads * total_chunks
        threshold = max(1, (num_cus * resident) // 2) * max_seq_chunks
        if total_wgs >= threshold:
            return bv
    return 16


def _hipeq_select_bv(
    device: torch.device, num_heads: int, total_chunks: int, max_seq_chunks: int
) -> int:
    key = (_hipeq_device_idx(device), num_heads, total_chunks, max_seq_chunks)
    cached = _HIPEQ_BV_CACHE.get(key)
    if cached is not None:
        return cached
    bv = _hipeq_compute_bv(device, total_chunks, max_seq_chunks, num_heads)
    _HIPEQ_BV_CACHE[key] = bv
    return bv


def _hipeq_varlen_host_metadata(chunk_offsets: torch.Tensor) -> tuple[int, int]:
    """Total and maximum per-sequence chunk counts (one D2H transfer)."""
    offsets = chunk_offsets.tolist()
    total_chunks = offsets[-1]
    max_seq_chunks = max(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1))
    return total_chunks, max_seq_chunks


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
    """FlyDSL hidden-state recurrence host wrapper.

    Signature is API-compatible with
    ``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h.chunk_gated_delta_rule_fwd_h_opt_vk``:

    Args:
        k: [B, T, Hg, K] bf16.
        w: [B, H, T_flat, K] bf16, head-major contiguous layout.
        u: [B, H, T_flat, V] bf16, head-major contiguous layout.
        g: [B, H, T_total] f32 cumulative gate, head-major contiguous
            (matches Triton VK / HIP), or None. Must be a
            ``contiguous()`` tensor with stride-1 along the T dimension.
            Caller passes ``g`` in natural-log space; when
            ``use_exp2=True`` the prepare stage is expected to have
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
        use_exp2: whether ``g`` is in log2 space. Standalone callers pass
            natural-log ``g`` by default; end-to-end prefill passes the Triton
            prepare stage's ``use_exp2`` setting through explicitly.
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
    # HIP). The kernel reads ``g`` with stride-1 along the T dim; require
    # the caller to provide a contiguous head-major tensor.
    if g is not None:
        assert g.is_contiguous(), (
            "FlyDSL hidden state: ``g`` must be contiguous (head-major "
            f"[B, H, T_flat] or [H, T_flat]); got strides={g.stride()}, "
            f"shape={tuple(g.shape)}."
        )
        assert g.shape[-1] == T_flat, (
            f"FlyDSL hidden state: ``g.shape[-1]`` must equal T_flat={T_flat}, "
            f"got g.shape={tuple(g.shape)}."
        )
        assert g.shape[-2] == H, (
            f"FlyDSL hidden state: ``g.shape[-2]`` must equal H={H}, "
            f"got g.shape={tuple(g.shape)}."
        )
    g_arg = g if g is not None else dummy

    # Mirror the Triton VK wrapper: when ``use_exp2=True`` the hidden-state
    # kernel interprets ``gk`` in log2 space, so pre-scale by log2(e) here. The
    # kernel-side ``_fast_exp`` for ``gk`` is shared with the ``g`` path;
    # ``g`` itself must already be log2-scaled by the prepare stage when
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


# ==========================================================================
# mfma16_hip fork (additive) -- HIP-aligned FlyDSL K5 implementation.
#
# Everything below is self-contained and does NOT touch the baseline wrapper
# above: it has its own compiled-kernel cache, BV selection (reusing the hip
# K5 selector), launch path, and public entry point
# ``chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip``. The baseline path keeps
# its original behaviour / flydsl>=0.1.8 compatibility; the mfma16_hip fork
# requires flydsl>=0.2.0, enforced lazily below.
# ==========================================================================

# mfma16_hip fork is written against the fx layout / tiled-copy / tiled-MMA API
# surface (``make_buffer_tensor``, ``fx.copy``, ``fx.gemm``) that only exists
# from flydsl 0.2.0. Enforced lazily (in ``_get_or_compile_mfma16_hip``) so
# importing this module and using the baseline wrapper keeps working on
# flydsl>=0.1.8.
_MFMA16_HIP_MIN_FLYDSL_VERSION = "0.2.0"

# gfx942 gate: only the mfma16_hip fork toggles the gfx942 GEMM1 ds-scheduling
# (SCHED_GFX942). ``get_rocm_arch()`` may return a feature-suffixed string like
# ``gfx942:sramecc+:xnack-``; normalize before matching.
_IS_GFX942 = get_rocm_arch().split(":")[0].startswith("gfx942")

_INT32_ATTR = "_flydsl_int32_view"
_PROLOGUE_ATTR = "_flydsl_prologue_cache"


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
    T_flat: int,
):
    """Resolve the per-shape varlen prologue in one cached lookup.

    Collapses the three ``@tensor_cache``-decorated prologue helpers into a
    single tuple attached to ``cu_seqlens`` (keyed by ``(BT, num_decodes,
    num_decode_tokens)``), so repeat forwards on the same ``cu_seqlens`` tensor
    are one ``getattr`` + one dict get.

    Returns ``(NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen)``.
    """
    cache_key = (BT, num_decodes, num_decode_tokens, T_flat)
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
        first = int(kernel_cu_seqlens[0].item())
        last = int(kernel_cu_seqlens[-1].item())
        if first != 0 or last != T_flat or min_seqlen < 0:
            raise ValueError(
                "FlyDSL K5 mfma16_hip: rebased cu_seqlens must start at 0, "
                f"end at T_flat={T_flat}, and be nondecreasing; got "
                f"first={first}, last={last}, min_seqlen={min_seqlen}."
            )
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
    g_head_major=False,
    bf16_convert_trunc=True,
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
            f">=`{_MFMA16_HIP_MIN_FLYDSL_VERSION}` (for the fx layout / "
            f"tiled-copy API), but got `{getattr(flydsl, '__version__', 'unknown')}`."
        )

    from .kernels.chunk_gated_delta_h_mfma16x16x16 import (
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
        G_HEAD_MAJOR=g_head_major,
        BF16_CONVERT_TRUNC=bf16_convert_trunc,
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
    g_head_major: bool = False,
    bf16_convert_trunc: bool = True,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """mfma16 / HIP-aligned K5 implementation: NON-VWARP only -- uses the
    16x16x16 bf16 MFMA and the SAME split-M warp partition (BT split-M, K split
    across waves, V not split across warps) as the hand-tuned HIP/C++ K5 kernel,
    writing the public VK layout [..., V, K]. API-compatible with
    ``chunk_gated_delta_rule_fwd_h_flydsl`` (plus the indexed state-pool
    contract via ``initial_state_indices`` / ``inplace_final_state``, matching
    ``chunk_gated_delta_rule_fwd_h_hip_fn``).

    Unlike the baseline wrapper, BV is chosen by a frozen, self-contained copy
    of the hip K5 LDS/CU selector (``_hipeq_select_bv``) so it matches the
    hand-tuned HIP kernel today without importing its private API;
    ``FLYDSL_K5_MFMA16HIP_BV`` (in {16,32,64}) overrides it for A/B sweeps.
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
    elif inplace and not output_final_state:
        raise ValueError(
            "FlyDSL K5: inplace_final_state requires output_final_state=True."
        )

    resolved_state_dtype = _resolve_state_dtype(initial_state, state_dtype)
    state_bf16 = resolved_state_dtype is torch.bfloat16

    # mfma16_hip keeps the token-major [B, T_flat, Hg, K] k layout (no
    # host-side pre-transpose), matching the Triton VK convention.
    if k.dim() != 4 or w.dim() != 4 or u.dim() != 4:
        raise ValueError(
            "FlyDSL K5 mfma16_hip: k/w/u must be 4-D (k=[B,T,Hg,K], "
            f"w=[B,H,T,K], u=[B,H,T,V]); got k={tuple(k.shape)}, "
            f"w={tuple(w.shape)}, u={tuple(u.shape)}."
        )
    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size

    # -- Input validation (k/w/u/gk). These feed the kernel's raw buffer loads
    # with no further checks, so a dtype / layout / shape mismatch would
    # silently read OOB or return wrong results. Fail early with a clear error.
    if not (k.dtype == w.dtype == u.dtype):
        raise ValueError(
            f"FlyDSL K5 mfma16_hip: k/w/u dtype must match; got k={k.dtype}, "
            f"w={w.dtype}, u={u.dtype}."
        )
    if k.dtype != torch.bfloat16:
        raise ValueError(
            "FlyDSL K5 mfma16_hip: k/w/u must be bfloat16 (the 16x16x16 bf16 "
            f"MFMA path), got {k.dtype}."
        )
    if not (w.device == k.device and u.device == k.device):
        raise ValueError(
            "FlyDSL K5 mfma16_hip: k/w/u must be on the same device; got "
            f"k={k.device}, w={w.device}, u={u.device}."
        )
    if not (k.is_contiguous() and w.is_contiguous() and u.is_contiguous()):
        raise ValueError(
            "FlyDSL K5 mfma16_hip: k/w/u must be contiguous; got strides "
            f"k={k.stride()}, w={w.stride()}, u={u.stride()}."
        )
    if k.shape[1] != T_flat:
        raise ValueError(
            f"FlyDSL K5 mfma16_hip: k T dim ({k.shape[1]}) must equal w/u T ({T_flat})."
        )
    if w.shape != (B, H, T_flat, K):
        raise ValueError(
            f"FlyDSL K5 mfma16_hip: expected w=[B,H,T,K]=({B},{H},{T_flat},{K}), "
            f"got {tuple(w.shape)}."
        )
    if u.shape != (B, H, T_flat, V):
        raise ValueError(
            f"FlyDSL K5 mfma16_hip: expected u=[B,H,T,V]=({B},{H},{T_flat},{V}), "
            f"got {tuple(u.shape)}."
        )
    if H % Hg != 0:
        raise ValueError(
            f"FlyDSL K5 mfma16_hip: H ({H}) must be a multiple of Hg ({Hg})."
        )
    if gk is not None:
        if gk.device != k.device:
            raise ValueError(
                f"FlyDSL K5 mfma16_hip: gk must be on k's device ({k.device}); "
                f"got {gk.device}."
            )
        if gk.dtype != torch.float32:
            raise ValueError(
                f"FlyDSL K5 mfma16_hip: gk must be float32, got {gk.dtype}."
            )
        expected_gk_shape = (B, T_flat, H, K)
        if tuple(gk.shape) != expected_gk_shape:
            raise ValueError(
                "FlyDSL K5 mfma16_hip: gk must use token-major [B,T,H,K] "
                f"layout with shape {expected_gk_shape}, got {tuple(gk.shape)}."
            )

    # Explicitly reject unvalidated configs: this kernel's wave mapping
    # (wid*16, 4 waves cover 64 rows), the gated_v alias-reuse of h_state
    # panel1 (needs NUM_K_BLOCKS>=2), and the LDS layout are only validated
    # for K=128, BT=64 (see the asserts inside the kernel). Other values would
    # trigger LDS aliasing OOB, out-of-bounds stores, or excessive LDS usage,
    # so fail early with a clear error instead of silently producing wrong
    # results.
    if BT != 64:
        raise ValueError(
            f"FlyDSL K5 mfma16_hip: only chunk_size=64 is supported, got "
            f"chunk_size={BT}."
        )
    if K != 128:
        raise ValueError(f"FlyDSL K5 mfma16_hip: only K=128 is supported, got K={K}.")
    if V != 128:
        raise ValueError(f"FlyDSL K5 mfma16_hip: only V=128 is supported, got V={V}.")

    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T, BT)
        chunk_offsets = None
        kernel_cu_seqlens = None
        is_varlen = False
    else:
        if B != 1:
            raise ValueError(
                f"FlyDSL K5 mfma16_hip: varlen mode requires B=1, got B={B}."
            )
        if cu_seqlens.device != k.device:
            raise ValueError(
                "FlyDSL K5 mfma16_hip: cu_seqlens must be on k's device "
                f"({k.device}), got {cu_seqlens.device}."
            )
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "FlyDSL K5 mfma16_hip: cu_seqlens must be int32 or int64, "
                f"got {cu_seqlens.dtype}."
            )
        if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                "FlyDSL K5 mfma16_hip: cu_seqlens must be a 1-D tensor with "
                f"at least two elements, got shape {tuple(cu_seqlens.shape)}."
            )
        if not cu_seqlens.is_contiguous():
            raise ValueError("FlyDSL K5 mfma16_hip: cu_seqlens must be contiguous.")
        if prefill_metadata is not None:
            prefill_metadata.validate(
                cu_seqlens=cu_seqlens,
                chunk_size=BT,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
                total_prefill_tokens=T_flat,
                num_sequences=len(cu_seqlens) - 1,
            )
            schedule = prefill_metadata.get_chunk_schedule(
                BT,
                num_decodes=num_decodes,
                num_decode_tokens=num_decode_tokens,
            )
            NT = schedule.total_chunks
            chunk_offsets = schedule.chunk_offsets
            kernel_cu_seqlens = schedule.kernel_cu_seqlens
            N = schedule.n_prefill
        else:
            NT, chunk_offsets, kernel_cu_seqlens, N, _min_seqlen = _resolve_prologue(
                cu_seqlens, BT, num_decodes, num_decode_tokens, T_flat
            )
        is_varlen = True

    if initial_state is not None:
        if initial_state.device != k.device:
            raise ValueError(
                "FlyDSL K5 mfma16_hip: initial_state must be on k's device "
                f"({k.device}), got {initial_state.device}."
            )
        if not initial_state.is_contiguous():
            raise ValueError("FlyDSL K5 mfma16_hip: initial_state must be contiguous.")
        if initial_state.dim() != 4 or tuple(initial_state.shape[1:]) != (H, V, K):
            raise ValueError(
                "FlyDSL K5 mfma16_hip: initial_state must have shape "
                f"[N,H,V,K] or [pool_size,H,V,K] with trailing shape "
                f"({H},{V},{K}), got {tuple(initial_state.shape)}."
            )
        if not use_state_indices and initial_state.shape[0] != N:
            raise ValueError(
                "FlyDSL K5 mfma16_hip: dense initial_state first dimension "
                f"must equal N={N}, got {initial_state.shape[0]}."
            )

    # Validate indexed pool access before selecting/compiling a kernel. Indices
    # gather from and scatter into ``initial_state[pool_size, H, V, K]``:
    # out-of-range values access OOB, while duplicates race on in-place write-back.
    if use_state_indices:
        indices = initial_state_indices
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "FlyDSL K5: initial_state_indices must be int32 or int64, "
                f"got {indices.dtype}."
            )
        if indices.dim() != 1:
            raise ValueError(
                "FlyDSL K5: initial_state_indices must be 1-D, "
                f"got shape {tuple(indices.shape)}."
            )
        if initial_state.device != k.device:
            raise ValueError(
                "FlyDSL K5: initial_state must be on the same device as k; "
                f"got initial_state={initial_state.device}, k={k.device}."
            )
        if indices.device != k.device:
            raise ValueError(
                "FlyDSL K5: initial_state_indices must be on the same device as "
                f"k and initial_state; got indices={indices.device}, k={k.device}."
            )
        if indices.numel() != N:
            raise ValueError(
                "FlyDSL K5: initial_state_indices length "
                f"({indices.numel()}) must equal the number of sequences N={N}."
            )
        pool_size = initial_state.shape[0]
        if indices.numel():
            # Validate in the ORIGINAL integer dtype. Narrowing first would let
            # int64 values such as 2**32 wrap to a valid-looking int32 zero.
            idx_min = int(indices.min())
            idx_max = int(indices.max())
            if idx_min < 0 or idx_max >= pool_size:
                raise ValueError(
                    "FlyDSL K5: initial_state_indices out of range for a state pool "
                    f"of size {pool_size}; got [{idx_min}, {idx_max}], expected "
                    f"values in [0, {pool_size})."
                )
            if idx_max > torch.iinfo(torch.int32).max:
                raise ValueError(
                    "FlyDSL K5: initial_state_indices values must fit in int32; "
                    f"got maximum {idx_max}."
                )
            if inplace and torch.unique(indices).numel() != indices.numel():
                raise ValueError(
                    "FlyDSL K5: duplicate initial_state_indices with in-place "
                    "final-state write-back race on the shared pool slot; indices "
                    "must be unique."
                )
        # The kernel ABI is int32; narrow only after all checks pass.
        si_i32 = indices.to(torch.int32).contiguous()
    else:
        si_i32 = None

    # BV selection: use the frozen, self-contained copy of the hip K5 LDS/CU
    # selector (``_hipeq_select_bv`` above) so this fork picks the same BV as
    # the hand-tuned HIP kernel today, without importing its private API.
    # dense: total_chunks = B*NT, max_seq_chunks = NT (NT = cdiv(T, BT));
    # varlen: both come from chunk_offsets (one D2H transfer, like hip).
    if is_varlen:
        _total_chunks, _max_seq_chunks = _hipeq_varlen_host_metadata(chunk_offsets)
    else:
        _total_chunks, _max_seq_chunks = B * NT, NT
    BV = _hipeq_select_bv(k.device, H, _total_chunks, _max_seq_chunks)

    # Env override for A/B BV sweeps; the hand-tuned HIP K5 reference is fixed
    # at BV=16 (FLYDSL_K5_MFMA16HIP_BV=16 reproduces it).
    _bv_env = os.environ.get("FLYDSL_K5_MFMA16HIP_BV")
    if _bv_env:
        try:
            BV = int(_bv_env)
        except ValueError as exc:
            raise ValueError(
                f"FLYDSL_K5_MFMA16HIP_BV must be one of 16, 32, or 64, got {_bv_env!r}."
            ) from exc
    if BV not in (16, 32, 64):
        raise ValueError(f"mfma16_hip BV must be in {{16,32,64}}, got {BV}.")
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
        g_head_major=g_head_major,
        bf16_convert_trunc=bf16_convert_trunc,
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

    # g layout validation, strictly matching the HIP kernel's contract
    # (aiter.ops.chunk_gated_delta_rule_fwd_h._normalize_g_tensor): g must be a
    # 3-D tensor whose shape exactly matches the selected layout --
    #   g_head_major=True  -> head-major  [B, H, T_flat]
    #   g_head_major=False -> token-major [B, T_flat, H]   (default, == HIP)
    # In varlen mode the batch dim is 1 (flattened input, N segments live in
    # cu_seqlens), so B is k.shape[0] (==1). g=None keeps the USE_G=False path.
    if g is not None:
        if g.device != k.device:
            raise ValueError(
                f"FlyDSL K5 mfma16_hip: g must be on k's device ({k.device}), "
                f"got {g.device}."
            )
        if g.dtype != torch.float32:
            g = g.to(torch.float32)
        if g.dim() != 3:
            raise ValueError(
                f"FlyDSL K5 mfma16_hip: `g` must be 3-D, got shape {tuple(g.shape)}."
            )
        expected_g_shape = (B, H, T_flat) if g_head_major else (B, T_flat, H)
        if tuple(g.shape) != expected_g_shape:
            layout = "head-major [B, H, T]" if g_head_major else "token-major [B, T, H]"
            raise ValueError(
                f"FlyDSL K5 mfma16_hip: `g` shape mismatch, expected "
                f"{expected_g_shape} for {layout} layout, got {tuple(g.shape)}."
            )
        g = g.contiguous()

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

    # The 11 tensor slots, passed as fx.Tensor args. The kernel body only reads
    # each slot's base pointer and element type, so the placeholder ``dummy``
    # stands in for the slots this configuration disables -- its float32 dtype
    # matches the only such slot the body still views unconditionally (g).
    tensor_args = (
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

    # The mfma16_hip kernel carries an extra ``state_indices`` slot (12th tensor
    # arg): a real int32 [N] index array when indexed, else a 1-elem int32 dummy.
    if not use_state_indices:
        si_i32 = dummy.to(torch.int32)
    tensor_args = tensor_args + (si_i32,)

    _run_compiled(
        launch_fn,
        *tensor_args,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
    )

    return h, (v_new_buf if save_vn else None), final_state


# -- GDN prepare host wrapper (single fused FlyDSL kernel) -----------------


def _device_index(t: torch.Tensor) -> int:
    """Concrete ordinal of a CUDA tensor's device (``None`` = the current one)."""
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _pad_grid_x_odd(grid_x: int) -> int:
    """Round grid_x up to odd; only upward, since NT columns are required."""
    return grid_x | 1


@functools.cache
def _is_cdna_mfma_arch() -> bool:
    """Whether the current device supports the fused prepare kernel."""
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime().startswith(("gfx94", "gfx95"))
    except Exception:  # noqa: BLE001
        return False


# The launch ABI uses 32-bit element counts; v also bounds the widest outputs.
_MAX_FLAT_ELEMS = 2**31


def gdn_prepare_flydsl_supported(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    BT: int = 64,
) -> bool:
    """Whether ``gdn_prepare_fwd_flydsl`` supports this problem shape."""
    return (
        BT == 64
        and k.shape[-1] == 128
        and v.shape[-1] == 128
        and k.dtype is torch.bfloat16
        and v.dtype is torch.bfloat16
        and k.is_cuda
        and v.is_cuda
        and _device_index(k) == _device_index(v)
        and v.numel() < _MAX_FLAT_ELEMS
        and _is_cdna_mfma_arch()
    )


def gdn_prepare_fwd_flydsl(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    BT: int = 64,
    Hg: int | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused GDN prepare wrapper compatible with the Triton prepare pair.

    It preserves input/output layouts and exponent semantics without
    materializing ``A_raw``.

    Args:
        k: [B, T, Hg, K] bf16.
        v: [B, T, H, V] bf16.
        g: [B, T, H] f32 raw forget-gate increments (natural-log space).
        beta: [B, T, H] f32, post-sigmoid.
        cu_seqlens: [N+1] for variable-length batching, or None for dense.
            When given, ``B`` must be 1 and ``T`` is the packed token count.
        BT: chunk size (must be 64).
        Hg: number of K/V heads; defaults to ``k.shape[2]``. ``H % Hg == 0``.
        use_exp2: publish ``g_cumsum`` in log2 space when True.
            ``w_bar`` and ``u_bar`` are unchanged.
        num_decodes / num_decode_tokens: skip a leading decode-only prefix in
            ``cu_seqlens``; data tensors contain only prefill tokens.
        prefill_metadata: reusable schedule required for variable-length input.
        stream: launch stream; defaults to the current stream.

    Returns:
        Head-major contiguous ``w_bar [B,H,T,K]`` bf16,
        ``u_bar [B,H,T,V]`` bf16, and ``g_cumsum [B,H,T]`` fp32.

    ``T`` need not be a multiple of ``BT``.
    """
    B, T, Hg_in, K = k.shape
    H = v.shape[2]
    V = v.shape[3]
    if Hg is None:
        Hg = Hg_in
    assert H % Hg == 0

    # Reject mixed precision before it reaches downstream kernels.
    if k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        raise TypeError(
            "gdn_prepare_fwd_flydsl emits bf16 `w_bar`/`u_bar` and therefore "
            f"requires bf16 `k` and `v`; got k={k.dtype}, v={v.dtype}."
        )
    if cu_seqlens is None and (num_decodes or num_decode_tokens):
        raise ValueError(
            "`num_decodes` / `num_decode_tokens` describe a packed varlen batch "
            "and require `cu_seqlens`."
        )
    # Validate the supported slice before compilation and launch.
    if not gdn_prepare_flydsl_supported(k, v, BT=BT):
        raise ValueError(
            "gdn_prepare_fwd_flydsl serves bf16 `k`/`v` with K=V=128 and BT=64, "
            "co-resident on one CDNA device, under 2**31 flattened elements; "
            f"got k={tuple(k.shape)} {k.dtype} on {k.device}, "
            f"v={tuple(v.shape)} {v.dtype} on {v.device}, BT={BT}. Gate on "
            "`gdn_prepare_flydsl_supported` and use the Triton prepare pair "
            "wherever it returns False."
        )

    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous().float()
    beta = beta.contiguous().float()

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1
        # The schedule supplies the maximum per-sequence chunk count.
        if prefill_metadata is None:
            raise ValueError(
                "gdn_prepare_fwd_flydsl needs `prefill_metadata` for a varlen "
                "batch: its launch grid is sized by the longest sequence's chunk "
                "count, which is only available on the host from the prefill "
                "schedule. Build one with "
                "`build_gated_delta_rule_prefill_metadata`, or use the Triton "
                "prepare pair."
            )
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
        num_seqs = schedule.n_prefill
        NT = schedule.max_seq_chunks
        cu = schedule.kernel_cu_seqlens.to(
            device=k.device, dtype=torch.int32
        ).contiguous()
    else:
        # Dense mode does not dereference ``cu``.
        cu = k
        num_seqs = B
        # Tail accesses are bounded, so no padding is required.
        NT = (T + BT - 1) // BT

    # Every output position is written once.
    w_bar = torch.empty(B, H, T, K, dtype=torch.bfloat16, device=k.device)
    u_bar = torch.empty(B, H, T, V, dtype=torch.bfloat16, device=k.device)
    g_cumsum = torch.empty(B, H, T, dtype=torch.float32, device=k.device)

    grid_x = NT
    grid_y = num_seqs * H

    # Odd width distributes skewed varlen chunks; the extra column exits early.
    if is_varlen and NT >= 2:
        grid_x = _pad_grid_x_odd(NT)

    if stream is None:
        stream = torch.cuda.current_stream()

    exe = compile_gdn_prepare(
        BT=BT,
        K=K,
        V=V,
        is_varlen=is_varlen,
        g_scale=_RCP_LN2 if use_exp2 else 1.0,
    )
    _run_compiled(
        exe,
        k.view(-1),
        v.view(-1),
        g.view(-1),
        beta.view(-1),
        cu.view(-1),
        w_bar.view(-1),
        u_bar.view(-1),
        g_cumsum.view(-1),
        T,
        H,
        Hg,
        grid_x,
        grid_y,
        stream,
    )

    return w_bar, u_bar, g_cumsum


# -- Fused K5+K6 compile cache + launch (gfx942) -------------------------

_compiled_fused_kernels: dict = {}


# Fused-vs-unfused selection threshold. Fuse when the fused kernel's actual
# grid over-fills the device by at least this fraction.
_FUSED_MIN_FILL = 0.45


def should_use_fused_gfx942(*, H: int, N: int, V: int) -> bool:
    """Heuristic: is the fused K5+K6 kernel the faster choice for this shape?

    Rule (gfx942): fuse iff the fused kernel's ACTUAL grid fill clears
    ``_FUSED_MIN_FILL``:

        fill = ceil(V / BV_run) * N * H / CU_count  >=  _FUSED_MIN_FILL

    where ``BV_run`` comes from ``_fused_bv_for_shape``.
    """
    if _ARCH != "gfx942":
        return False
    bv_run, _ = _fused_bv_for_shape(H=H, V=V, N=N, variant=None)
    fill = _grid_ctas(H=H, V=V, N=N, BV=bv_run) / max(_device_cu_count(), 1)
    return fill >= _FUSED_MIN_FILL


# Fused wave-widening: auto uses num_waves=8 (NR_SPLIT=2) when it selects BV=64
# AND the bv64 grid fills the device well enough that the extra resident waves
# have room to help.
_FUSED_W8_MIN_FILL = 0.55


def _fused_bv_for_shape(*, H, V, N, variant):
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
        raise ValueError(f"fused GDN K5+K6: no legal BV <= {_FUSED_MAX_BV} for V={V}.")
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

    if _ARCH == "gfx942":
        tuned = _gfx942_select_fused_variant(H=H, N=N, V=V)
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
    *,
    q,
    k,
    w,
    u,
    g,
    gk,
    o,
    scale,
    initial_state,
    output_final_state,
    chunk_size,
    cu_seqlens,
    state_dtype,
    use_exp2,
    num_decodes,
    num_decode_tokens,
    prefill_metadata,
    variant,
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
        assert (
            g.shape[-1] == T_flat and g.shape[-2] == H
        ), f"fused K5+K6: g must be [.., H={H}, T_flat={T_flat}]; got {tuple(g.shape)}."
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
        chunk_offsets.to(torch.int32)
        if chunk_offsets is not None
        else dummy.to(torch.int32)
    )
    stream = torch.cuda.current_stream()

    BV, num_waves = _fused_bv_for_shape(H=H, V=V, N=N, variant=variant)
    NR_SPLIT = num_waves // (BT // 16)

    cache_key = (
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        float(scale),
        use_g,
        use_gk,
        use_h0,
        output_final_state,
        is_varlen,
        state_bf16,
        g_log2_scaled,
        NR_SPLIT,
    )
    if cache_key not in _compiled_fused_kernels:
        # Fused build of the shared gfx942 builder: COMPUTE_OUTPUT appends the
        # K6 stage, while STORE_H / SAVE_NEW_VALUE drop the two HBM drains that
        # only the separate K6 kernel consumes -- that elision is the whole
        # point of fusing.
        _compiled_fused_kernels[cache_key] = compile_chunk_gated_delta_h_gfx942(
            K=K,
            V=V,
            BT=BT,
            BV=BV,
            H=H,
            Hg=Hg,
            USE_G=use_g,
            USE_GK=use_gk,
            USE_INITIAL_STATE=use_h0,
            STORE_FINAL_STATE=output_final_state,
            SAVE_NEW_VALUE=False,
            IS_VARLEN=is_varlen,
            WU_CONTIGUOUS=True,
            STATE_DTYPE_BF16=state_bf16,
            G_IS_LOG2_SCALED=g_log2_scaled,
            NR_SPLIT=NR_SPLIT,
            COMPUTE_OUTPUT=True,
            STORE_H=False,
            SCALE=float(scale),
        )
    launch_fn = _compiled_fused_kernels[cache_key]

    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H
    # Unified kernel arg order; v_new and h are unused on the fused build.
    _run_compiled(
        launch_fn,
        k,
        u,
        w,
        dummy,
        g_arg,
        gk_arg,
        dummy,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        q,
        o,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
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
        scale = K**-0.5

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
        q=q,
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        o=o,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
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

    use_fused = resolved_fusion == K5K6Fusion.ALWAYS or (
        resolved_fusion == K5K6Fusion.AUTO and should_use_fused_gfx942(H=H, N=N, V=V)
    )

    if use_fused:
        return chunk_gated_delta_rule_fwd_h_o_flydsl(
            q=q,
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            state_dtype=state_dtype,
            use_exp2=use_exp2,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=prefill_metadata,
            variant=variant,
            o=o,
        )

    # Separate path: FlyDSL K5 + Triton K6.
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
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
        q=q,
        k=k,
        v=v_new,
        o=o,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )
    return o, final_state
