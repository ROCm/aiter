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
    GatedDeltaRulePrefillMetadata,
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
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
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
            f">=`{_MFMA16_HIP_MIN_FLYDSL_VERSION}` (for the fx.Pointer argument "
            f"ABI), but got `{getattr(flydsl, '__version__', 'unknown')}`."
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
            f"FlyDSL K5 mfma16_hip: k T dim ({k.shape[1]}) must equal w/u T "
            f"({T_flat})."
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
                f"FlyDSL K5 mfma16_hip: gk must be float32, got " f"{gk.dtype}."
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
                "FLYDSL_K5_MFMA16HIP_BV must be one of 16, 32, or 64, "
                f"got {_bv_env!r}."
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
                f"FlyDSL K5 mfma16_hip: `g` must be 3-D, got shape "
                f"{tuple(g.shape)}."
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
    if not use_state_indices:
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
