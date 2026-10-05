# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fp8 paged unified attention on gfx950.

Main's fmha_gfx950 builder serves single-pass prefill. Shuffled decode uses
pa_decode; causal calls outside the served routing region fall back to the Triton-wrapper path (Gluon on gfx950).
"""

from __future__ import annotations

import importlib.util
import os
from collections import OrderedDict
from functools import cache, lru_cache

import torch

from .kernels.flash_attn_func_fp8_gfx950 import (
    _fp8_auto_block_m,
    _fp8_rescale_threshold,
)
from .kernels.fmha_gfx950.flash_attn_fp8_gfx950 import (
    build_flash_attn_dualwave_swp_fp8_module,
)

__all__ = ["flydsl_unified_attention"]


@lru_cache(maxsize=1)
def _is_flydsl_installed() -> bool:
    return importlib.util.find_spec("flydsl") is not None


@cache
def is_flydsl_available(device_index: int) -> bool:
    if not _is_flydsl_installed():
        return False
    with torch.cuda.device(device_index):
        props = torch.cuda.get_device_properties(device_index)
    arch = props.gcnArchName.split(":", 1)[0]
    return arch == "gfx950"


# Page size is structural, not a builder parameter: the paged path addresses KV
# in BLOCK_N-sized pages and BLOCK_N is pinned at 64 by the MFMA tile.
_PAGE_SIZE = 64

# Head dim is fixed by the kernel (it raises on anything else).
_HEAD_DIM = 128

# Vectorization width of the shuffled 5D KV cache: 16 fp8 elements = one
# 128-bit dwordx4. Backs the _strides_ok 5D validation branch and
# _get_kernel's layout selection; _dispatch_mode_ok accepts shuffled_kv_cache.
_KV_VEC_SIZE = 16


@cache
def _target_num_prgms(device_index: int) -> int:
    """Split-K fill target: a launch with num_2d_prgms base workgroups is "full"
    at num_2d_prgms >= this value.

    A device CU-count query (not a hardcoded constant), so it also handles
    CU-partitioned modes (CPX/NPS) where fewer CUs are exposed; falls back to
    256 (full-chip gfx950) if the query fails. Keyed on device.index and
    resolved at call time: in a multi-GPU process the import-time device can
    differ from the run device, and a heterogeneous host exposes different CU
    counts per device -- either would lock in a wrong fill target.
    """
    try:
        from aiter.ops.triton.utils.device_info import get_num_sms

        with torch.cuda.device(device_index):
            return get_num_sms()
    except Exception:  # noqa: BLE001
        return 256


# The block table is staged through a fixed LDS window of PAGED_BT_LDS_SIZE=2048
# entries. Past that the stager writes only `local_tile < segment_tiles` slots
# and silently drops the remaining page ids -- wrong output, not a fault -- so
# the KV length must be capped here: 2048 pages * 64 tokens = 131072 tokens.
_MAX_KV_TILES = 2048

_FP8_DTYPE = torch.float8_e4m3fn

# Minimum decode-half context depth (KV length) at which the mixed-batch
# dispatch split is taken. Below this the decode half's split-K does not
# amortize the split's two-launch + partition-sync overhead, so the split
# regresses vs the single call; 6144 (96 pages) is the crossover.
_SPLIT_MIN_DECODE_KV = 6144

# Memoizes only DECLINES of the mixed-batch split probe, to avoid re-paying
# _partition_mixed's host sync for a batch shape that recurs each decoder
# layer. The asymmetry is a correctness invariant: a declined split falls
# through to the always-correct single call, so a stale decline costs at
# most a missed split, never a wrong result -- whereas a TAKE re-slices on
# the probed split_point and must always re-probe. Bounded LRU; eviction
# only forces a re-probe.
_SPLIT_DECLINE_MEMO: OrderedDict[tuple, bool] = OrderedDict()
_SPLIT_DECLINE_MEMO_MAX = 128


def _split_declined_before(key) -> bool:
    """True if this batch signature already probed and declined the split."""
    if key in _SPLIT_DECLINE_MEMO:
        _SPLIT_DECLINE_MEMO.move_to_end(key)
        return True
    return False


def _remember_split_decline(key) -> None:
    """Record that this batch signature's split probe declined."""
    _SPLIT_DECLINE_MEMO[key] = True
    _SPLIT_DECLINE_MEMO.move_to_end(key)
    if len(_SPLIT_DECLINE_MEMO) > _SPLIT_DECLINE_MEMO_MAX:
        _SPLIT_DECLINE_MEMO.popitem(last=False)


# GQA group sizes routed to the packed BN64 body (group 1 packing is the identity).
_PACKED_BN64_GROUPS = (4, 8, 16)


@lru_cache(maxsize=64)
def _get_kernel(
    num_heads: int,
    num_kv_heads: int,
    causal: bool,
    out_dtype_str: str,
    shuffled_kv_cache: bool,
    rescale_threshold: float,
    block_m: int,
    packed_bn64: bool = False,
):
    """Cache only build-time specializations, not runtime batch/sequence sizes.

    ``packed_bn64`` selects the conventional BN64 body with the GQA group packed
    into M; that body fixes block_m=128 itself, so ``block_m`` is ignored for it.
    """
    variant = (
        {"body_variant": "conventional_bn64", "gqa_pack_m": True} if packed_bn64 else {}
    )
    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        num_kv_heads=num_kv_heads,
        causal=causal,
        varlen=True,
        cross_seqlen=True,
        paged=True,
        kv_cache_layout="shuffled" if shuffled_kv_cache else "linear",
        out_dtype=out_dtype_str,
        rescale_threshold=rescale_threshold,
        block_m=block_m,
        batch_interleave_group=1,
        num_kv_splits=1,
        **variant,
    )


# FlyDSL pa_decode route for all-decode calls on the shuffled fp8 KV cache. The
# partition count is a host-only static rule (no planner, no device sync).
_PA_DECODE_TILE = 256  # pa_decode's fixed context_partition_size
_PA_DECODE_MAX_NP = 32
# Target workgroups per CU for one launch. Measured best on gfx950 for both
# pa_decode tile variants: 144 VGPRs with wide KV addressing (3 WGs/CU resident)
# and 114 VGPRs without (4 resident). More WGs per CU adds partitions and reduce
# work without raising in-flight loads.
_PA_DECODE_WGS_PER_CU = 3


def _env_use_pa_decode(default: bool = True) -> bool:
    raw = os.environ.get("AITER_UNIFIED_ATTN_PA_DECODE")
    return default if raw is None else raw == "1"


_USE_PA_DECODE = _env_use_pa_decode()


def _pa_decode_num_partitions(
    num_seqs: int, num_kv_heads: int, max_seqlen_k: int, num_cus: int
) -> int:
    """Static partition count NP: split each sequence into about
    three workgroups per CU across the GPU, but never less than one 256-token
    tile per partition. Python ints only."""
    fit = _PA_DECODE_WGS_PER_CU * num_cus // (num_seqs * num_kv_heads)
    hi = min(-(-max_seqlen_k // _PA_DECODE_TILE), _PA_DECODE_MAX_NP)
    return max(1, min(fit, hi))


def _pa_decode_ok(
    max_seqlen_q, num_queries_per_kv, sinks, shuffled_kv_cache, out, num_seqs
) -> bool:
    """Layout/dtype gate on top of _supported (which already pins fp8 QKV,
    5D shuffled strides, page 64, D128, int32 index tensors, no softcap/alibi)."""
    return (
        _USE_PA_DECODE
        and max_seqlen_q == 1
        and shuffled_kv_cache
        and out.dtype in (torch.bfloat16, torch.float16)
        and num_seqs > 0
    )


def _route_pa_decode(
    q,
    k,
    v,
    out,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    block_table,
    q_descale,
    k_descale,
    v_descale,
    *,
    num_kv_heads,
    num_seqs,
    sinks=None,
):
    """All-decode call -> FlyDSL pa_decode with fp8 Q. seqused_k is already the
    per-sequence context length, so no cumulative-offset conversion is needed."""
    from .pa_decode import pa_decode

    hq, d = q.shape[1], q.shape[2]
    np_ = _pa_decode_num_partitions(
        num_seqs, num_kv_heads, int(max_seqlen_k), _target_num_prgms(q.device.index)
    )
    shape = (num_seqs, num_kv_heads, np_, hq // num_kv_heads)
    dev = q.device
    exp_sums = torch.empty(shape, dtype=torch.float32, device=dev)
    max_logits = torch.empty(shape, dtype=torch.float32, device=dev)
    tmp_out = torch.empty((*shape, d), dtype=out.dtype, device=dev)
    with torch.cuda.device(dev.index):
        pa_decode(
            out,
            q,
            k,
            v,
            seqused_k,
            block_table.contiguous(),
            float(softmax_scale),
            1,
            np_,
            _PA_DECODE_TILE,
            _FP8_DTYPE,
            q_descale.reshape(-1),
            k_descale.reshape(-1),
            v_descale.reshape(-1),
            exp_sums,
            max_logits,
            tmp_out,
            sinks=sinks,
        )
    return out


def _kv_strides_ok_5d(k, v, num_kv_heads, head_size) -> bool:
    """Validate the shuffled 5D KV-cache shape/strides.

    Layout: K = ``[num_blocks, kv_heads, head_size//x, block_size, x]``, V =
    ``[num_blocks, kv_heads, block_size//x, head_size, x]``, x = _KV_VEC_SIZE.
    Requires the trailing (vectorized) dim to be exactly ``x`` elements,
    contiguous, and the whole tensor row-major in that 5D shape -- the
    vectorized loaders assume the fixed byte-offset formula that only holds
    under that layout.
    """
    if k.dim() != 5 or v.dim() != 5:
        return False
    x = _KV_VEC_SIZE
    if head_size % x != 0 or _PAGE_SIZE % x != 0:
        return False
    k_shape = (k.shape[0], num_kv_heads, head_size // x, _PAGE_SIZE, x)
    v_shape = (v.shape[0], num_kv_heads, _PAGE_SIZE // x, head_size, x)
    if tuple(k.shape) != k_shape or tuple(v.shape) != v_shape:
        return False
    for t, shape in ((k, k_shape), (v, v_shape)):
        stride = 1
        for dim in reversed(range(5)):
            if t.stride(dim) != stride:
                return False
            stride *= shape[dim]
    return True


def _strides_ok(
    q,
    out,
    k,
    v,
    block_table,
    num_query_heads,
    num_kv_heads,
    head_size,
    shuffled_kv_cache,
) -> bool:
    """Check the exact layout the kernel's two scalar strides imply.

    The launcher takes one ``stride_q_n`` and one ``stride_kv_n`` and derives
    every offset from them, so anything else must be declined rather than
    silently mis-addressed.
    """
    if q.stride(2) != 1 or q.stride(1) != head_size:
        return False
    if out.stride(2) != 1 or out.stride(1) != head_size:
        return False
    # One stride_q_n argument serves the Q read, the O write, and BOTH buffer
    # num_records bounds (init_descriptors), so Q and O must agree on it.
    if q.stride(0) != out.stride(0):
        return False
    # Flattening must remain a view with the row stride passed to the kernel.
    if block_table.stride(1) != 1 or block_table.stride(0) != block_table.shape[1]:
        return False
    # `_run_compiled` launches on `q.reshape(-1)` / `out.reshape(-1)`, baking
    # the tensor's full memref into the kernel cache signature. A padded
    # (non-flattenable) layout would make reshape return a silent COPY --
    # the kernel writes the copy, the caller's real `out` stays untouched.
    # Requiring stride(0) == flattened row size restricts acceptance to
    # layouts where reshape(-1) is a view; padded layouts decline and fall
    # through to the Triton-wrapper path (Gluon on gfx950).
    if q.stride(0) != num_query_heads * head_size:
        return False
    if out.stride(0) != num_query_heads * head_size:
        return False
    # Shuffled 5D KV cache: the production gate (_dispatch_mode_ok) accepts
    # shuffled_kv_cache, so this branch is live on the real dispatch path. The
    # flag and the tensor layout MUST agree: a shuffled flag on a 4D linear
    # tensor (or vice versa) would run the vectorized loader's byte-offset
    # formula against the wrong memory, so decline the mismatch rather than
    # mis-address it. shuffled -> require 5D.
    if shuffled_kv_cache:
        if k.dim() != 5 or v.dim() != 5:
            return False
        return _kv_strides_ok_5d(k, v, num_kv_heads, head_size)
    if k.dim() != 4 or v.dim() != 4:
        return False
    page_row = num_kv_heads * head_size
    for t in (k, v):
        if t.stride(3) != 1 or t.stride(2) != head_size:
            return False
        if t.stride(1) != page_row or t.stride(0) != _PAGE_SIZE * page_row:
            return False
    return True


def _dispatch_mode_ok(window_size, block_table, skip_reduce) -> bool:
    """Paged, full-window, non-reduce. The reduce flag belongs to a Triton
    layout this kernel does not read; the paged path is the only one wired
    here. (Causal and non-causal are both built.)

    ``shuffled_kv_cache`` is accepted unconditionally by design, not gated
    here: the vectorized K and V loaders are both correctness-validated
    against a torch reference, and ``_get_kernel``/``_strides_ok`` route a
    shuffled call to the vectorized builder and validate its 5D K/V shape.
    """
    return window_size == (-1, -1) and block_table is not None and not skip_reduce


def _page_geometry_ok(block_size, max_seqlen_k) -> bool:
    """Structural page shape, not tunable: fixed page size, KV within the staged
    block-table window."""
    return (
        block_size == _PAGE_SIZE
        and (max_seqlen_k + _PAGE_SIZE - 1) // _PAGE_SIZE <= _MAX_KV_TILES
    )


def _dtypes_ok(q, k, v, out, cu_seqlens_q, seqused_k, block_table) -> bool:
    """fp8 QKV, bf16/f16 output (both pack to 2 bytes), int32 index tensors."""
    return (
        q.dtype == _FP8_DTYPE
        and k.dtype == _FP8_DTYPE
        and v.dtype == _FP8_DTYPE
        and out.dtype in (torch.bfloat16, torch.float16)
        and cu_seqlens_q.dtype == torch.int32
        and seqused_k.dtype == torch.int32
        and block_table.dtype == torch.int32
    )


def _descales_ok(q_descale, k_descale, v_descale) -> bool:
    """Per-tensor fp8 descales are mandatory: the kernel reads all three to form
    c_logit_scale and the V dequant. numel()==1 admits both a 1-D [1] tensor and
    a 0-dim scalar; _as_1d_descale normalizes the latter before it reaches the
    kernel (see flydsl_unified_attention)."""
    return all(
        d is not None and d.dtype == torch.float32 and d.numel() == 1
        for d in (q_descale, k_descale, v_descale)
    )


def _as_1d_descale(d):
    """Normalize a 0-dim scalar descale to a 1-element 1-D tensor.

    vLLM passes per-tensor fp8 descales as 0-dim scalars (shape ()). FlyDSL's
    from_dlpack rejects those -- a scalar has no stride-1 axis to auto-mark
    layout-dynamic -- so widen to [1] here. reshape on a 0-dim tensor is a view,
    not a copy, and numel is unchanged so _descales_ok still passes. Passes None
    and already-1-D descales through untouched."""
    if d is not None and d.ndim == 0:
        return d.reshape(1)
    return d


def _geometry_ok(
    head_size, num_query_heads, num_kv_heads, num_queries_per_kv, cu_seqlens_q, num_seqs
) -> bool:
    """Fixed head dim, integral GQA, and cu_seqlens covering every sequence."""
    return (
        head_size == _HEAD_DIM
        and num_query_heads % num_kv_heads == 0
        and cu_seqlens_q.numel() == num_seqs + 1
    )


def _no_unsupported_features(
    softcap, alibi_slopes, qq_bias, q_scales, output_scale
) -> bool:
    """Features the kernel has no path for; declining beats silently dropping."""
    return (
        softcap == 0
        and alibi_slopes is None
        and qq_bias is None
        and q_scales is None
        and output_scale is None
    )


def _sinks_ok(sinks, num_query_heads) -> bool:
    """Sinks are not served by FlyDSL; causal auto calls fall back to the Triton-wrapper path (Gluon on gfx950).

    Non-causal and explicit-FlyDSL sinks calls cannot fall back.
    """
    return sinks is None


def _supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_k,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    num_kv_heads,
    block_size,
    num_queries_per_kv,
    num_seqs,
    q_scales,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    shuffled_kv_cache,
    skip_reduce,
) -> bool:
    """Whether this exact configuration can be served. Kept separate from the
    marshalling so it can be unit-tested against meta tensors, with no GPU.
    Causal and non-causal are both built, so this gate does not branch on it."""
    if not is_flydsl_available(q.device.index):
        return False

    head_size = q.shape[-1]
    num_query_heads = q.shape[1]

    return (
        _dispatch_mode_ok(window_size, block_table, skip_reduce)
        and _page_geometry_ok(block_size, max_seqlen_k)
        and _dtypes_ok(q, k, v, out, cu_seqlens_q, seqused_k, block_table)
        and _descales_ok(q_descale, k_descale, v_descale)
        and _geometry_ok(
            head_size,
            num_query_heads,
            num_kv_heads,
            num_queries_per_kv,
            cu_seqlens_q,
            num_seqs,
        )
        and _no_unsupported_features(
            softcap, alibi_slopes, qq_bias, q_scales, output_scale
        )
        and _sinks_ok(sinks, num_query_heads)
        and _strides_ok(
            q,
            out,
            k,
            v,
            block_table,
            num_query_heads,
            num_kv_heads,
            head_size,
            shuffled_kv_cache,
        )
    )


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """fp8 buffers are passed to flydsl as int8 views (the kernel builds i8-typed
    descriptors so DMA and register loads share one byte view)."""
    return t.view(torch.int8) if t.dtype == _FP8_DTYPE else t


def _partition_mixed(cu_seqlens_q, seqused_k, num_seqs):
    """Partition a mixed batch into a contiguous prefill block and a contiguous
    decode block, or return None when the layout isn't a clean two-block split.

    A mixed batch is independent sequences, but the single launch below picks one
    ``num_kv_splits`` for all of them -- wrong for one half. This finds the split
    so each half can run on its own optimal path. Returns None (caller runs the
    single unified call, still correct) unless every prefill seq (query_len > 1)
    forms one contiguous run and every decode seq (query_len == 1) forms another,
    both non-empty. Only that shape lets each half be a zero-copy view slice of
    q/out. Interleaved layouts decline here and fall through to the single call.

    On success returns (split_point, prefill_first, n_pre, n_dec, pre_max_q,
    dec_max_kv):
      split_point   -- seq index where the second block begins
      prefill_first -- True if prefills lead, False if decodes lead
      n_pre, n_dec  -- sequence counts per block (both > 0)
      pre_max_q     -- max query_len over the prefill block (grid.y for the 2d call)
      dec_max_kv    -- max KV length over the decode block; the caller gates the
                       split on this (shallow decodes don't amortize split-K)
    Costs one device->host sync (all scalars pulled in a single transfer).
    """
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]  # [num_seqs], device
    is_dec = seqlens_q == 1
    n_dec_t = is_dec.sum()
    di = is_dec.to(torch.int32)
    # Clean two-block <=> is_dec is monotonic: all-False-then-True (prefill-first)
    # or all-True-then-False (decode-first). Both non-empty => exactly one holds.
    asc = torch.all(di[1:] >= di[:-1])
    desc = torch.all(di[1:] <= di[:-1])
    pre_max_q_t = torch.where(is_dec, torch.zeros_like(seqlens_q), seqlens_q).max()
    # Max KV depth over the decode seqs only (prefill KV is larger but irrelevant
    # to whether the decode half wants split-K).
    dec_max_kv_t = torch.where(is_dec, seqused_k, torch.zeros_like(seqused_k)).max()
    stats = torch.stack(
        [
            n_dec_t.to(torch.int64),
            asc.to(torch.int64),
            desc.to(torch.int64),
            pre_max_q_t.to(torch.int64),
            dec_max_kv_t.to(torch.int64),
        ]
    ).tolist()
    n_dec, asc_b, desc_b, pre_max_q, dec_max_kv = (
        stats[0],
        bool(stats[1]),
        bool(stats[2]),
        stats[3],
        stats[4],
    )
    n_pre = num_seqs - n_dec
    if n_pre == 0 or n_dec == 0:
        return None  # pure prefill or pure decode: nothing to split
    if asc_b:
        return (n_pre, True, n_pre, n_dec, pre_max_q, dec_max_kv)
    if desc_b:
        return (n_dec, False, n_pre, n_dec, pre_max_q, dec_max_kv)
    return None  # interleaved: decline (single call stays correct)


def flydsl_unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    *,
    num_kv_heads,
    block_size,
    num_queries_per_kv,
    num_seqs,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    shuffled_kv_cache=False,
    skip_reduce=False,
    _from_split=False,
):
    """Run unified attention on the FlyDSL fp8 gfx950 kernel.

    The positional parameters mirror ``unified_attention`` exactly so the hook is
    a mechanical forward. The keyword-only block is quantities the caller has
    already derived; recomputing them here would duplicate its layout unpacking.

    Returns ``out`` (written in place) if this configuration is supported, or
    ``None`` so the caller falls back to the Triton-wrapper path (Gluon on gfx950; real Triton only if Gluon is unsupported, e.g. softcap, alibi, qq_bias).
    """
    # Widen 0-dim scalar descales to [1] before the gate, the recursion, and the
    # kernel see them (vLLM passes per-tensor descales as scalars; FlyDSL's
    # from_dlpack rejects a shape-() tensor). A view, not a copy.
    q_descale = _as_1d_descale(q_descale)
    k_descale = _as_1d_descale(k_descale)
    v_descale = _as_1d_descale(v_descale)

    if not _supported(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
        max_seqlen_k,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        num_kv_heads,
        block_size,
        num_queries_per_kv,
        num_seqs,
        q_scales,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        shuffled_kv_cache,
        skip_reduce,
    ):
        return None

    if _pa_decode_ok(
        max_seqlen_q, num_queries_per_kv, sinks, shuffled_kv_cache, out, num_seqs
    ):
        return _route_pa_decode(
            q,
            k,
            v,
            out,
            seqused_k,
            max_seqlen_k,
            softmax_scale,
            block_table,
            q_descale,
            k_descale,
            v_descale,
            num_kv_heads=num_kv_heads,
            num_seqs=num_seqs,
            sinks=sinks,
        )

    # Decode cannot use the single-pass prefill body without wasting its M tile.
    # The caller owns fallback (or explicit-backend rejection), including recursion.
    if max_seqlen_q == 1:
        return None

    # Clean two-block mixed batches can use separate prefill and pa_decode calls.
    # Keep the measured depth threshold and decline memo until perf re-measure.
    if max_seqlen_q > 1 and num_seqs > 1 and max_seqlen_k >= _SPLIT_MIN_DECODE_KV:
        memo_key = (
            cu_seqlens_q.data_ptr(),
            seqused_k.data_ptr(),
            num_seqs,
            q.shape[0],
            max_seqlen_q,
            max_seqlen_k,
        )
        part = None
        if not _split_declined_before(memo_key):
            part = _partition_mixed(cu_seqlens_q, seqused_k, num_seqs)
            if part is not None and part[5] < _SPLIT_MIN_DECODE_KV:
                # Clean two-block batch whose decode half is too shallow to
                # amortize the split -- the residual case. Record it so the
                # recurrence on later layers skips the probe. part is None
                # (interleaved, or a pure prefill/decode sub-batch of a taken
                # split) is left unmemoized: its key can be an ephemeral sliced
                # tensor, and re-probing it is both correct and the pre-existing
                # behavior.
                _remember_split_decline(memo_key)
        if part is not None and part[5] >= _SPLIT_MIN_DECODE_KV:
            split_point, prefill_first, n_pre, n_dec, pre_max_q, dec_max_kv = part
            row_split = int(cu_seqlens_q[split_point])
            total_q = q.shape[0]
            if prefill_first:
                pre_rows, dec_rows = (0, row_split), (row_split, total_q)
                pre_seqs, dec_seqs = (0, split_point), (split_point, num_seqs)
            else:
                dec_rows, pre_rows = (0, row_split), (row_split, total_q)
                dec_seqs, pre_seqs = (0, split_point), (split_point, num_seqs)

            def _sub(rows, seqs, n_sub, max_q_sub, max_kv_sub):
                r0, r1 = rows
                s0, s1 = seqs
                return flydsl_unified_attention(
                    q[r0:r1],
                    k,
                    v,
                    out[r0:r1],
                    cu_seqlens_q[s0 : s1 + 1] - cu_seqlens_q[s0],
                    max_q_sub,
                    seqused_k[s0:s1],
                    max_kv_sub,
                    softmax_scale,
                    causal,
                    window_size,
                    block_table[s0:s1],
                    softcap,
                    q_descale,
                    k_descale,
                    v_descale,
                    num_kv_heads=num_kv_heads,
                    block_size=block_size,
                    num_queries_per_kv=num_queries_per_kv,
                    num_seqs=n_sub,
                    q_scales=q_scales,
                    alibi_slopes=alibi_slopes,
                    output_scale=output_scale,
                    qq_bias=qq_bias,
                    sinks=sinks,
                    shuffled_kv_cache=shuffled_kv_cache,
                    skip_reduce=skip_reduce,
                    _from_split=True,
                )

            # Prefill first. A small multi-chunk prefill half can underfill and
            # decline (return None from the tier-selection guard below); the split
            # has no wrapper fall-through, only the external caller does, so cede the
            # whole batch to the Triton-wrapper path rather than leaving its rows unwritten. The decode
            # half has max_q == 1 and never declines today, but a future cede or a
            # _supported failure on the sliced views could return None too -- if so,
            # its rows would be left unwritten while this function still returns
            # `out`, so check it the same way as the prefill half rather than
            # dropping the return value.
            if (
                _sub(pre_rows, pre_seqs, n_pre, pre_max_q, max_seqlen_k) is None
            ):  # prefill -> single-pass 2d
                return None
            if (
                _sub(dec_rows, dec_seqs, n_dec, 1, dec_max_kv) is None
            ):  # decode -> pa_decode
                return None
            return out

    num_query_heads = q.shape[1]
    out_dtype_str = "f16" if out.dtype == torch.float16 else "bf16"
    target_num_prgms = _target_num_prgms(q.device.index)
    block_m = _fp8_auto_block_m(
        num_seqs,
        num_query_heads,
        int(max_seqlen_q),
        int(max_seqlen_k),
        target_num_prgms,
    )
    # Packed BN64 (paged, single-pass, D128; pipeline.py allows groups 1/4/8/16,
    # only the packing groups 4/8/16 are routed) folds the GQA group into the
    # 128-row M tile: one workgroup per KV head and 128 // group query tokens.
    # Everything else keeps one query head per workgroup.
    group = num_query_heads // num_kv_heads
    packed_bn64 = group in _PACKED_BN64_GROUPS
    if packed_bn64:
        block_m = 128  # fixed by the body; keeps the cache key equal to the AOT job

    with torch.cuda.device(q.device.index):
        kernel = _get_kernel(
            num_query_heads,
            num_kv_heads,
            bool(causal),
            out_dtype_str,
            shuffled_kv_cache,
            _fp8_rescale_threshold(max_seqlen_k),
            block_m,
            packed_bn64=packed_bn64,
        )
        kernel(
            _as_i8(q).reshape(-1),
            # Rank two avoids the C-ABI signed-i32 flat-pool shape overflow.
            _as_i8(k).reshape(k.shape[0], -1),
            _as_i8(v).reshape(v.shape[0], -1),
            out.reshape(-1),
            num_seqs,
            int(max_seqlen_q),
            num_kv_heads * _HEAD_DIM,
            q.stride(0),
            softmax_scale=float(softmax_scale),
            seq_len_kv=int(max_seqlen_k),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=seqused_k,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table.reshape(-1),
            block_table_stride=int(block_table.stride(0)),
            stream=torch.cuda.current_stream(q.device),
        )
    return out
