# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL host wrappers for GDN prefill.

The module exposes hidden-state and prepare kernels. The prepare wrapper matches
the Triton prepare layout and exponent-domain contract, so both paths can be
selected independently through ``chunk_gated_delta_rule_opt_vk``. Call
``gdn_prepare_flydsl_supported`` before selecting the fused prepare path.
"""

from __future__ import annotations

import csv
import functools
import math
import os
import warnings
from collections.abc import Sequence

# NOTE (opt fork): ``get_rocm_arch`` is imported here for the additive
# HIP-aligned fork below. It is side-effect-free (``flydsl`` is already a hard
# dependency of the baseline ``compile_chunk_gated_delta_h``) and does NOT raise
# on flydsl <0.2.0 -- the opt-only ``>=0.2.0`` requirement is enforced
# lazily in ``_get_or_compile_opt`` so the baseline path keeps its
# original ``>=0.1.8`` compatibility.
import torch
import triton
from flydsl.runtime.device import get_rocm_arch

from aiter.jit.core import AITER_CONFIGS

from ..triton._triton_kernels.gated_delta_rule.utils import (
    GatedDeltaRulePrefillMetadata,
    build_gated_delta_rule_prefill_metadata,
    prepare_chunk_offsets,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)
from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from .kernels.gdn_prepare import compile_gdn_prepare
from .kernels.tensor_shim import _run_compiled

# log2(e); g pre-scaled by this constant lets the kernel use exp2(g) in
# place of exp(g) (matches the Triton VK / HIP convention).
_RCP_LN2 = math.log2(math.e)


__all__ = [
    "chunk_gated_delta_rule_fwd_h_flydsl",
    "chunk_gated_delta_rule_fwd_h_flydsl_opt",
    "gdn_prepare_flydsl_supported",
    "gdn_prepare_fwd_flydsl",
]


# -- Hidden-state host wrapper (FlyDSL kernel + rule-based BV selection) ---

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
            trace-calibrated rules to the hidden-state H=32/Hg=16 family.

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
# The opt fork below picks BV to match the hand-tuned HIP K5 kernel
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


_HOST_CHUNK_META_ATTR = "_flydsl_host_chunk_meta"


def _hipeq_varlen_host_metadata(chunk_offsets: torch.Tensor) -> tuple[int, int]:
    """Total/max per-sequence chunk counts, cached on ``chunk_offsets``."""
    cached = getattr(chunk_offsets, _HOST_CHUNK_META_ATTR, None)
    if cached is not None:
        return cached
    offsets = chunk_offsets.tolist()
    total_chunks = offsets[-1]
    max_seq_chunks = max(offsets[i + 1] - offsets[i] for i in range(len(offsets) - 1))
    result = (total_chunks, max_seq_chunks)
    try:
        object.__setattr__(chunk_offsets, _HOST_CHUNK_META_ATTR, result)
    except (AttributeError, TypeError):
        pass
    return result


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

    BV-tile selection is rule-based on this entry; the opt entry reads
    ``AITER_CONFIG_GDN_K5_OPT`` first.
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
    int32_dummy = torch.empty(1, device=k.device, dtype=torch.int32)

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
    # int32. The narrowing is cached on the source tensor, whose identity is
    # stable across forwards, so a steady-state forward launches no copy for it.
    cu_arg = (
        _as_int32(kernel_cu_seqlens) if kernel_cu_seqlens is not None else int32_dummy
    )
    co_arg = _as_int32(chunk_offsets) if chunk_offsets is not None else int32_dummy
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


# opt fork below (flydsl>=0.2.0, lazy-checked); baseline above unchanged.
_OPT_MIN_FLYDSL_VERSION = "0.2.0"

# gfx942 gate for SCHED_GFX942; normalize feature-suffixed arch strings first.
_GFX_ARCH = get_rocm_arch().split(":")[0]
_IS_GFX942 = _GFX_ARCH.startswith("gfx942")


def _load_tuned_bv_table() -> dict[tuple, int]:
    """Load tuned BV rows at import; miss or error falls back to the rule."""
    table: dict[tuple, int] = {}
    try:
        with open(
            AITER_CONFIGS.AITER_CONFIG_GDN_K5_OPT_FILE,
            encoding="utf-8",
            newline="",
        ) as f:
            rows = csv.DictReader(
                line for line in f if not line.lstrip().startswith("#")
            )
            for row in rows:
                bv = (row.get("BV") or "").strip()
                if not bv:
                    continue  # tuned rows must carry a measured BV
                gfx = (row.get("gfx") or "").strip()
                cu_raw = (row.get("cu_num") or "").strip()
                cu_num = int(cu_raw) if cu_raw else None
                shape_tail = (
                    int(row["H"]),
                    int(row["Hg"]),
                    int(row["V"]),
                    row["is_varlen"].strip() == "True",
                    row["use_h0"].strip() == "True",
                    row["store_fs"].strip() == "True",
                    row["snapshot_bf16"].strip() == "True",
                    row["state_bf16"].strip() == "True",
                    int(row["total_chunks"]),
                    int(row["max_seq_chunks"]),
                )
                key = (
                    (gfx, cu_num, *shape_tail)
                    if cu_num is not None
                    else (gfx, *shape_tail)
                )
                bv_int = int(bv)
                if table.get(key, bv_int) != bv_int:
                    warnings.warn(
                        f"chunk_gdn_h_opt tuned table disagrees on "
                        f"{key}: BV {table[key]} vs {bv_int}; keeping "
                        f"{table[key]}.",
                        stacklevel=2,
                    )
                    continue
                table[key] = bv_int
    except (OSError, KeyError, ValueError):
        return {}
    return table


def reload_tuned_bv_table() -> None:
    """Reload BV lookup after ``AITER_CONFIG_GDN_K5_OPT`` changes."""
    global _BV_TUNED_TABLE
    _BV_TUNED_TABLE = _load_tuned_bv_table()


_BV_TUNED_TABLE = _load_tuned_bv_table()


def _tuned_bv(
    *,
    H: int,
    Hg: int,
    V: int,
    is_varlen: bool,
    use_h0: bool,
    store_fs: bool,
    snapshot_bf16: bool,
    state_bf16: bool,
    total_chunks: int,
    max_seq_chunks: int,
) -> int | None:
    """Measured BV for this batch shape, or None to use the rule.

    Lookup requires an exact ``(gfx, cu_num, shape)`` match on the current GPU,
    same as MoE tuned-config lookup. Binned SKUs (e.g. MI308X cu_num=80) never
    reuse rows tuned on a sibling card (e.g. MI300X cu_num=304).
    """
    if not _BV_TUNED_TABLE:
        return None
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

    shape_key = (
        H,
        Hg,
        V,
        is_varlen,
        use_h0,
        store_fs,
        snapshot_bf16,
        state_bf16,
        total_chunks,
        max_seq_chunks,
    )
    return _BV_TUNED_TABLE.get((get_gfx_runtime(), get_cu_num(), *shape_key))


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


def _as_state_indices(indices: torch.Tensor) -> torch.Tensor:
    """Narrow pool indices to the int32 contiguous ABI the K5 kernel expects.

    Value range and uniqueness are the caller's responsibility, matching the
    HIP/Triton K5 wrappers (``initial_state_indices.to(torch.int32).contiguous()``).
    """
    return indices.to(torch.int32).contiguous()


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
                "FlyDSL K5 opt: rebased cu_seqlens must start at 0, "
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


def _resolve_snapshot_dtype(snapshot_dtype, input_dtype):
    """Per-chunk snapshot dtype: defaults to ``k.dtype`` and is independent of
    the state dtype, matching ``aiter.ops.chunk_gated_delta_rule_fwd_h``."""
    resolved = input_dtype if snapshot_dtype is None else snapshot_dtype
    if resolved not in (torch.float32, torch.bfloat16):
        raise ValueError(f"`snapshot_dtype` must be fp32 or bf16, got {resolved}.")
    return resolved


@functools.cache
def _get_or_compile_opt(
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
    snapshot_bf16=True,
):
    """Compile (and cache) the K5 opt kernel: 16x16x16 bf16
    MFMA + HIP-matching warp partition, writing the public VK layout [..., V, K].

    ``snapshot_bf16`` selects the per-chunk ``h`` snapshot specialization and
    joins the cache key, so the bf16 and fp32 snapshot variants are separate
    compiled products and the bf16 one keeps its emitted code.

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
    if installed < Version(_OPT_MIN_FLYDSL_VERSION):
        raise ImportError(
            "FlyDSL K5 opt fork requires `flydsl` "
            f">=`{_OPT_MIN_FLYDSL_VERSION}` (for the fx layout / "
            f"tiled-copy API), but got `{getattr(flydsl, '__version__', 'unknown')}`."
        )

    from .kernels.chunk_gated_delta_h_opt import (
        compile_chunk_gated_delta_h_opt,
    )

    return compile_chunk_gated_delta_h_opt(
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
        SNAPSHOT_DTYPE_BF16=snapshot_bf16,
        G_IS_LOG2_SCALED=g_log2_scaled,
        USE_STATE_INDICES=use_state_indices,
        SCHED_GFX942=sched_gfx942,
        G_HEAD_MAJOR=g_head_major,
        BF16_CONVERT_TRUNC=bf16_convert_trunc,
    )


def chunk_gated_delta_rule_fwd_h_flydsl_opt(
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
    seq_lens_cpu: Sequence[int] | None = None,
    snapshot_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """K5 opt implementation: NON-VWARP only -- uses the
    16x16x16 bf16 MFMA and the SAME split-M warp partition (BT split-M, K split
    across waves, V not split across warps) as the hand-tuned HIP/C++ K5 kernel,
    writing the public VK layout [..., V, K]. API-compatible with
    ``chunk_gated_delta_rule_fwd_h_flydsl`` (plus the indexed state-pool
    contract via ``initial_state_indices`` / ``inplace_final_state``, matching
    ``chunk_gated_delta_rule_fwd_h_hip_fn``).

    Unlike the baseline wrapper, BV is ``_tuned_bv`` then ``_hipeq_select_bv``;
    ``FLYDSL_K5_OPT_BV`` (in {16,32,64}) overrides both for A/B sweeps.

    ``state_dtype`` controls the persistent initial/final state, while
    ``snapshot_dtype`` independently controls the per-chunk ``h`` snapshots and
    defaults to ``k.dtype`` (bf16 here), mirroring
    ``chunk_gated_delta_rule_fwd_h_hip_fn``. fp32 snapshots take two half-BV
    transpose rounds, since the f32 tile does not fit the LDS budget whole.
    In varlen mode, pass ``prefill_metadata`` (preferred for reuse) or
    ``seq_lens_cpu`` to avoid reading chunk scheduling values back from the GPU.
    ``gk`` accepts token-major ``[B, T, H, K]`` or the HIP flat layout
    ``[T, H, K]`` (varlen) / ``[B * T, H, K]`` (dense).
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
    resolved_snapshot_dtype = _resolve_snapshot_dtype(snapshot_dtype, k.dtype)
    snapshot_bf16 = resolved_snapshot_dtype is torch.bfloat16

    # opt keeps the token-major [B, T_flat, Hg, K] k layout (no
    # host-side pre-transpose), matching the Triton VK convention.
    if k.dim() != 4 or w.dim() != 4 or u.dim() != 4:
        raise ValueError(
            "FlyDSL K5 opt: k/w/u must be 4-D (k=[B,T,Hg,K], "
            f"w=[B,H,T,K], u=[B,H,T,V]); got k={tuple(k.shape)}, "
            f"w={tuple(w.shape)}, u={tuple(u.shape)}."
        )
    B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size
    is_varlen = cu_seqlens is not None

    if is_varlen and prefill_metadata is None and seq_lens_cpu is not None:
        prefill_metadata = build_gated_delta_rule_prefill_metadata(
            seq_lens_cpu,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )

    # -- Input validation (k/w/u/gk). These feed the kernel's raw buffer loads
    # with no further checks, so a dtype / layout / shape mismatch would
    # silently read OOB or return wrong results. Fail early with a clear error.
    if not (k.dtype == w.dtype == u.dtype):
        raise ValueError(
            f"FlyDSL K5 opt: k/w/u dtype must match; got k={k.dtype}, "
            f"w={w.dtype}, u={u.dtype}."
        )
    if k.dtype != torch.bfloat16:
        raise ValueError(
            "FlyDSL K5 opt: k/w/u must be bfloat16 (the 16x16x16 bf16 "
            f"MFMA path), got {k.dtype}."
        )
    if not (w.device == k.device and u.device == k.device):
        raise ValueError(
            "FlyDSL K5 opt: k/w/u must be on the same device; got "
            f"k={k.device}, w={w.device}, u={u.device}."
        )
    k = k.contiguous()
    w = w.contiguous()
    u = u.contiguous()
    if k.shape[1] != T_flat:
        raise ValueError(
            f"FlyDSL K5 opt: k T dim ({k.shape[1]}) must equal w/u T ({T_flat})."
        )
    if w.shape != (B, H, T_flat, K):
        raise ValueError(
            f"FlyDSL K5 opt: expected w=[B,H,T,K]=({B},{H},{T_flat},{K}), "
            f"got {tuple(w.shape)}."
        )
    if u.shape != (B, H, T_flat, V):
        raise ValueError(
            f"FlyDSL K5 opt: expected u=[B,H,T,V]=({B},{H},{T_flat},{V}), "
            f"got {tuple(u.shape)}."
        )
    if H % Hg != 0:
        raise ValueError(f"FlyDSL K5 opt: H ({H}) must be a multiple of Hg ({Hg}).")
    if gk is not None:
        if gk.device != k.device:
            raise ValueError(
                f"FlyDSL K5 opt: gk must be on k's device ({k.device}); "
                f"got {gk.device}."
            )
        if gk.dtype != torch.float32:
            raise ValueError(f"FlyDSL K5 opt: gk must be float32, got {gk.dtype}.")
        token_major_shape = (B, T_flat, H, K)
        flat_shape = (T_flat if is_varlen else B * T_flat, H, K)
        if tuple(gk.shape) == token_major_shape:
            pass
        elif tuple(gk.shape) == flat_shape:
            gk = gk.reshape(token_major_shape)
        else:
            raise ValueError(
                "FlyDSL K5 opt: gk shape mismatch; expected "
                f"{token_major_shape} (token-major) or {flat_shape} (HIP flat), "
                f"got {tuple(gk.shape)}."
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
            f"FlyDSL K5 opt: only chunk_size=64 is supported, got " f"chunk_size={BT}."
        )
    if K != 128:
        raise ValueError(f"FlyDSL K5 opt: only K=128 is supported, got K={K}.")
    if V != 128:
        raise ValueError(f"FlyDSL K5 opt: only V=128 is supported, got V={V}.")

    # BV selector counts from the caller's metadata; None = read off chunk_offsets.
    host_chunk_meta = None
    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T, BT)
        chunk_offsets = None
        kernel_cu_seqlens = None
        is_varlen = False
    else:
        if B != 1:
            raise ValueError(f"FlyDSL K5 opt: varlen mode requires B=1, got B={B}.")
        if cu_seqlens.device != k.device:
            raise ValueError(
                "FlyDSL K5 opt: cu_seqlens must be on k's device "
                f"({k.device}), got {cu_seqlens.device}."
            )
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "FlyDSL K5 opt: cu_seqlens must be int32 or int64, "
                f"got {cu_seqlens.dtype}."
            )
        if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
            raise ValueError(
                "FlyDSL K5 opt: cu_seqlens must be a 1-D tensor with "
                f"at least two elements, got shape {tuple(cu_seqlens.shape)}."
            )
        if not cu_seqlens.is_contiguous():
            raise ValueError("FlyDSL K5 opt: cu_seqlens must be contiguous.")
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
            host_chunk_meta = (schedule.total_chunks, schedule.max_seq_chunks)
        else:
            NT, chunk_offsets, kernel_cu_seqlens, N, _min_seqlen = _resolve_prologue(
                cu_seqlens, BT, num_decodes, num_decode_tokens, T_flat
            )
        is_varlen = True

    if initial_state is not None:
        if initial_state.device != k.device:
            raise ValueError(
                "FlyDSL K5 opt: initial_state must be on k's device "
                f"({k.device}), got {initial_state.device}."
            )
        if not initial_state.is_contiguous():
            raise ValueError("FlyDSL K5 opt: initial_state must be contiguous.")
        if initial_state.dim() != 4 or tuple(initial_state.shape[1:]) != (H, V, K):
            raise ValueError(
                "FlyDSL K5 opt: initial_state must have shape "
                f"[N,H,V,K] or [pool_size,H,V,K] with trailing shape "
                f"({H},{V},{K}), got {tuple(initial_state.shape)}."
            )
        if not use_state_indices and initial_state.shape[0] != N:
            raise ValueError(
                "FlyDSL K5 opt: dense initial_state first dimension "
                f"must equal N={N}, got {initial_state.shape[0]}."
            )

    # Indexed pool: gather/scatter through ``initial_state[pool_size, H, V, K]``.
    # Layout checks fail early; index *values* are the caller's responsibility,
    # matching the HIP/Triton wrappers (int32 narrow + contiguous only).
    if use_state_indices:
        indices = initial_state_indices
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
        si_i32 = _as_state_indices(indices)
    else:
        si_i32 = None

    # BV: tuned table, then hip LDS/CU rule; env override wins last.
    if is_varlen:
        _total_chunks, _max_seq_chunks = (
            host_chunk_meta
            if host_chunk_meta is not None
            else _hipeq_varlen_host_metadata(chunk_offsets)
        )
    else:
        _total_chunks, _max_seq_chunks = B * NT, NT
    BV = _tuned_bv(
        H=H,
        Hg=Hg,
        V=V,
        is_varlen=is_varlen,
        use_h0=use_h0,
        store_fs=bool(output_final_state),
        snapshot_bf16=snapshot_bf16,
        state_bf16=state_bf16,
        total_chunks=_total_chunks,
        max_seq_chunks=_max_seq_chunks,
    )
    if BV is None:
        BV = _hipeq_select_bv(k.device, H, _total_chunks, _max_seq_chunks)

    # Env override for A/B BV sweeps; the hand-tuned HIP K5 reference is fixed
    # at BV=16 (FLYDSL_K5_OPT_BV=16 reproduces it).
    _bv_env = os.environ.get("FLYDSL_K5_OPT_BV")
    if _bv_env:
        try:
            BV = int(_bv_env)
        except ValueError as exc:
            raise ValueError(
                f"FLYDSL_K5_OPT_BV must be one of 16, 32, or 64, got {_bv_env!r}."
            ) from exc
    if BV not in (16, 32, 64):
        raise ValueError(f"opt BV must be in {{16,32,64}}, got {BV}.")
    if V % BV != 0:
        raise ValueError(f"FlyDSL K5 opt: requires V % BV == 0; got V={V}, BV={BV}.")

    # SCHED_GFX942 is only enabled on gfx942; other arches (incl. gfx950) pass
    # False, keeping their emitted code byte-identical, and it joins the
    # lru_cache key as a distinct compiled product.
    launch_fn = _get_or_compile_opt(
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
        snapshot_bf16=snapshot_bf16,
    )

    # Null-arg placeholders for the @flyc.jit slots ignored on this path. Sized
    # 1 (not 0) for a non-null ``data_ptr()``; allocated, not cast, to avoid a copy.
    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    int32_dummy = torch.empty(1, device=k.device, dtype=torch.int32)
    cu_arg = (
        _as_int32(kernel_cu_seqlens) if kernel_cu_seqlens is not None else int32_dummy
    )
    co_arg = _as_int32(chunk_offsets) if chunk_offsets is not None else int32_dummy
    stream = torch.cuda.current_stream(k.device)

    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H

    # opt writes the public VK layout ([..., V, K]) directly.
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
                f"FlyDSL K5 opt: g must be on k's device ({k.device}), "
                f"got {g.device}."
            )
        if g.dtype != torch.float32:
            g = g.to(torch.float32)
        if g.dim() != 3:
            raise ValueError(
                f"FlyDSL K5 opt: `g` must be 3-D, got shape {tuple(g.shape)}."
            )
        expected_g_shape = (B, H, T_flat) if g_head_major else (B, T_flat, H)
        if tuple(g.shape) != expected_g_shape:
            layout = "head-major [B, H, T]" if g_head_major else "token-major [B, T, H]"
            raise ValueError(
                f"FlyDSL K5 opt: `g` shape mismatch, expected "
                f"{expected_g_shape} for {layout} layout, got {tuple(g.shape)}."
            )
        g = g.contiguous()

    # gk pre-scaling to log2 space (mirrors the Triton VK wrapper).
    if gk is not None:
        gk = gk.contiguous()
        if g_log2_scaled:
            gk = gk * _RCP_LN2

    h = k.new_empty(h_shape, dtype=resolved_snapshot_dtype)
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

    # The opt kernel carries an extra ``state_indices`` slot (12th tensor
    # arg): a real int32 [N] index array when indexed, else a 1-elem int32 dummy.
    if not use_state_indices:
        si_i32 = int32_dummy
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
