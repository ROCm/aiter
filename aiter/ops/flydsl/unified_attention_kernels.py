# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fp8 unified-attention backend for gfx950.

Adapts the vendored ``flash_attn_dualwave_swp`` fp8 kernel
(``kernels/flash_attn_fp8_gfx950.py``) to the ``unified_attention`` calling
convention vLLM/SGLang import directly, so a supported gfx950 fp8 paged call
routes here instead of Triton.

Dispatch is a pure predicate returning ``None`` when it can't serve the
config, falling through to Triton unchanged (matches
``flydsl_flash_attn_varlen_func`` / the flydsl branch in
``ops/gemm_op_a8w8.py``) -- deliberately no dispatch env var, since a code gate
states an arch/shape-scoped choice more honestly than another undocumented
backend flag. Force Triton by setting ``_FLYDSL_UNIFIED_ATTN_ARCH`` to ``False`` in
the Triton module. The one tuning knob is the split-count cap,
``AITER_UNIFIED_ATTN_MAX_KV_SPLITS`` (see ``_MAX_SEGMENTS``).

Two tiers chosen at call time from the machine-fill deficit (``_split_count``
and the gate in ``flydsl_unified_attention``), not from all-decode-ness: an
underfilled launch (few workgroups each serially scanning full KV depth)
takes packed + split-K, partitioning each sequence's KV across split
workgroups and combining partials; a full-pass batch stays on the
single-pass packed kernel. Split-K also serves the low-chunk mixed case
because the combine rebases its O write on ``cu_seqlens_q``
(``DualwaveSplitKCombineContext.init_descriptors``), so unequal query
lengths combine correctly.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from functools import cache, lru_cache

import torch

from .kernels.flash_attn_dualwave_common import dualwave_splitk_workspace_elems
from .kernels.flash_attn_fp8_gfx950 import build_flash_attn_dualwave_swp_fp8_module
from .utils import is_flydsl_available

__all__ = ["flydsl_unified_attention"]


# Page size is structural, not a builder parameter: the paged path addresses KV
# in BLOCK_N-sized pages and BLOCK_N is pinned at 64 by the MFMA tile.
_PAGE_SIZE = 64

# Head dim is fixed by the kernel (it raises on anything else).
_HEAD_DIM = 128

# Vectorization width of the shuffled 5D KV cache (stage 4 of the shuffled-
# cache-support-investigation port): 16 fp8 elements = one 128-bit dwordx4.
# Backs the _strides_ok 5D validation branch and _get_kernel's layout
# selection; _dispatch_mode_ok accepts shuffled_kv_cache as of Stage 3.
_KV_VEC_SIZE = 16


@cache
def _target_num_prgms(device_index: int) -> int:
    """Split-K fill target: a launch with num_2d_prgms base workgroups is "full"
    at num_2d_prgms >= this value.

    Both full-chip gfx950 parts (MI350X, MI355X) are 256 CUs, so this is a
    device CU-count query, not a hardcoded arch constant -- it also handles
    CU-partitioned modes (CPX/NPS) where fewer CUs are exposed. Falls back to
    256 (the full-chip count) if the query fails.

    Keyed on ``q.device`` (via ``device_index``) and resolved at call time, not
    once at import: in a multi-GPU process the active device at import can
    differ from the device a call actually runs on, and a heterogeneous host
    can expose different CU counts per device -- either would lock in a wrong
    fill target and mis-select split-K. ``@cache`` keeps it a one-time
    host-side query per device. The value comes from ``get_num_sms()`` under
    that device so the CU_NUM override and the tuning-dispatch CU count stay
    consistent.
    """
    try:
        from aiter.ops.triton.utils.device_info import get_num_sms

        with torch.cuda.device(device_index):
            return get_num_sms()
    except Exception:  # noqa: BLE001
        return 256


def _env_max_kv_splits(default: int = 16) -> int:
    """Auto-dispatch split-count cap, overridable by environment.

    Split-K selection cannot be tuned perfectly from shape alone (a lesson from
    CK), so the cap is an override knob rather than a derived guess.
    ``AITER_UNIFIED_ATTN_MAX_KV_SPLITS`` raises or lowers it; a non-integer or
    non-positive value is ignored and the default is used.
    """
    raw = os.environ.get("AITER_UNIFIED_ATTN_MAX_KV_SPLITS")
    if raw is None:
        return default
    try:
        n = int(raw)
    except ValueError:
        return default
    return n if n >= 1 else default


# Cap on the auto-dispatched split count. Default 16: no regression across
# tested decode shapes while capturing most of the long-context win. >16
# helps only very long contexts and regresses short ones, since _split_count
# keys on machine fill, not KV depth (counts are verified correct to 128).
# Raise via AITER_UNIFIED_ATTN_MAX_KV_SPLITS for long-context workloads.
_MAX_SEGMENTS = _env_max_kv_splits(16)

# block_m, and with it the largest GQA group that can pack into the M dimension.
# _make_dualwave_swp_fp8_traits asserts block_m % gqa_group_size == 0.
_BLOCK_M = 256

# The block table is staged through a fixed LDS window of PAGED_BT_LDS_SIZE=2048
# entries. Past that the stager writes only `local_tile < segment_tiles` slots
# and silently drops the remaining page ids -- wrong output, not a fault -- so
# the KV length must be capped here: 2048 pages * 64 tokens = 131072 tokens.
_MAX_KV_TILES = 2048

_FP8_DTYPE = torch.float8_e4m3fn

# Minimum decode-half context depth (KV length) at which the mixed-batch dispatch
# split (below) is taken. Below this the decode half's split-K does not amortize
# the split's two-launch + partition-sync overhead, so the split regresses vs the
# single call (chunk=256 was 0.87x at ctx=4096); at/above this every sampled
# chunk size wins (1.01-1.93x). Set from a gfx950 ctx sweep at chunks 256/512/4023
# (SILOTIGER-877): the crossover is chunk-independent and lands in (5632, 6144];
# 6144 (96 pages) is the lowest depth where all sampled chunks win noise-robustly.
_SPLIT_MIN_DECODE_KV = 6144


# Bounded memo of mixed-batch signatures whose dispatch-split probe DECLINED.
# Keyed on host-only quantities (the two device-tensor data_ptrs plus the shape
# scalars), so a lookup costs no device sync -- the point is to avoid re-paying
# _partition_mixed's ~29us host sync.
#
# The residual it removes: a large-chunk + shallow-decode mixed batch passes the
# host-scalar pre-check (max_seqlen_k reflects the deep PREFILL KV) but declines
# on the true, shallow decode depth. That batch recurs identically on every
# decoder layer of a forward pass, so without a memo each layer probes only to
# decline again. With it, the first layer probes and records the signature; the
# rest skip straight to the single call.
#
# ONLY declines are memoized, and that asymmetry is a correctness invariant, not
# an optimization: a decline falls through to the single unified call, which is
# correct for ANY batch composition, so a stale hit (a reused data_ptr whose
# contents now differ) costs at most a missed split -- never wrong output. A
# TAKE slices q/out at the probed split_point, so a stale partition would route
# the wrong rows to each half; takes therefore always re-probe (see the dispatch
# site). Bounded LRU because data_ptr keys churn across steps; eviction only ever
# forces a re-probe, never affects correctness.
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


@lru_cache(maxsize=64)
def _get_kernel(
    num_heads: int,
    num_kv_heads: int,
    causal: bool,
    out_dtype_str: str,
    use_sinks: bool,
    num_kv_splits: int = 1,
    shuffled_kv_cache: bool = False,
):
    """Build (and cache) the paged+varlen fp8 launcher.

    Keyed on head counts, mask mode, output dtype, split count, and the
    shuffled-cache flag; every other builder argument is pinned by the
    support gate or left at default.

    ``num_kv_splits`` selects the tier: at 1, the builder auto-enables
    M-dimension packing for GQA > 1:1 (``gqa_pack_m=None``) and builds the
    single-pass kernel; at > 1, ``gqa_pack_m`` defaults to unpacked, so the
    decode tier passes True explicitly to get the packed+split-K binary.
    ``GQA_PACK_M`` and ``NUM_KV_SPLITS`` both key the JIT cache so the two
    binaries cannot alias.

    ``causal``, ``out_dtype_str``, ``use_sinks`` key the cache too since each
    changes compiled code (mask path, store packer, sink init/epilogue).
    ``shuffled_kv_cache`` selects ``kv_cache_layout`` -- it must key the cache
    too, since it picks a different compiled binary (the traits factory's
    ``cache_tag`` already keys the *builder's own* JIT cache on it, but
    ``_get_kernel``'s ``lru_cache`` is a separate memo one layer up and would
    otherwise alias linear and vectorized closures under the same key).

    Do not key on batch/seqlen/device: ``_run_compiled`` memoizes the
    compiled function on the returned closure, so a finer key would defeat
    that cache and recompile per shape.
    """
    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        causal=causal,
        dtype_str="fp8",
        out_dtype_str=out_dtype_str,
        use_sinks=use_sinks,
        num_kv_heads=num_kv_heads,
        varlen=True,
        paged=True,
        num_kv_splits=num_kv_splits,
        gqa_pack_m=True if num_kv_splits > 1 else None,
        kv_cache_layout="vectorized" if shuffled_kv_cache else "linear",
    )


def _split_count(num_2d_prgms: int, target_num_prgms: int) -> int:
    """Split count from the machine-fill deficit; 1 means single-pass.

    The measured no-oversubscribe rule (NOT Triton's ceil/round-up/MIN=8): the
    largest power of two such that the launch does not oversubscribe the CU
    count, capped at _MAX_SEGMENTS. No MIN floor -- b=9 needs 4, below Triton's
    floor of 8. Below 2 there is no split to take, so the caller falls back to
    single-pass.

    Verified against the measured best split count: num_2d = 4*b for a
    b-sequence GQA-16 decode gives b=7 -> 8, b=8 -> 8, b=9 -> 4.
    """
    n = 1
    while n * 2 <= _MAX_SEGMENTS and num_2d_prgms * (n * 2) <= target_num_prgms:
        n *= 2
    return n if n >= 2 else 1


def _kv_strides_ok_5d(k, v, num_kv_heads, head_size) -> bool:
    """Validate the shuffled 5D KV-cache shape/strides (stage 4 gate scaffolding).

    Layout spec (shuffled-cache-support-investigation, "the layout spec"):
    K = ``[num_blocks, kv_heads, head_size//x, block_size, x]``, V =
    ``[num_blocks, kv_heads, block_size//x, head_size, x]``, x = _KV_VEC_SIZE.
    Requires the trailing (vectorized) dim to be exactly ``x`` elements,
    contiguous, and the whole tensor row-major in that 5D shape -- the
    vectorized loaders (Stage 2/3) assume the fixed byte-offset formula that
    only holds under that layout.
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
    # `_run_compiled` launches on `q.reshape(-1)` / `out.reshape(-1)`, baking
    # the tensor's full memref into the kernel cache signature. A padded
    # (non-flattenable) layout would make reshape return a silent COPY --
    # the kernel writes the copy, the caller's real `out` stays untouched.
    # Requiring stride(0) == flattened row size restricts acceptance to
    # layouts where reshape(-1) is a view; padded layouts decline and fall
    # through to Triton.
    if q.stride(0) != num_query_heads * head_size:
        return False
    if out.stride(0) != num_query_heads * head_size:
        return False
    # Shuffled 5D KV cache (stage 4): the production gate (_dispatch_mode_ok)
    # accepts shuffled_kv_cache as of the end of Stage 3, so this branch is
    # live on the real dispatch path. The flag and the tensor layout MUST agree:
    # a shuffled flag on a 4D linear tensor (or vice versa) would run the
    # vectorized loader's byte-offset formula against the wrong memory, so
    # decline the mismatch rather than mis-address it. shuffled -> require 5D.
    if shuffled_kv_cache:
        if k.dim() != 5 or v.dim() != 5:
            return False
        return (
            _kv_strides_ok_5d(k, v, num_kv_heads, head_size)
            and block_table.stride(1) == 1
        )
    if k.dim() != 4 or v.dim() != 4:
        return False
    page_row = num_kv_heads * head_size
    for t in (k, v):
        if t.stride(3) != 1 or t.stride(2) != head_size:
            return False
        if t.stride(1) != page_row or t.stride(0) != _PAGE_SIZE * page_row:
            return False
    return block_table.stride(1) == 1


def _dispatch_mode_ok(window_size, block_table, shuffled_kv_cache, skip_reduce) -> bool:
    """Paged, full-window, non-reduce. The reduce flag belongs to a Triton
    layout this kernel does not read; the paged path is the only one wired
    here. (Causal and non-causal are both built.)

    ``shuffled_kv_cache`` is accepted as of the end of Stage 3
    (shuffled-cache-support-investigation): the vectorized K and V loaders
    are both correctness-validated against a torch reference (Stage 2 K,
    Stage 3 V), and ``_get_kernel``/``_strides_ok`` route a shuffled call to
    the vectorized builder and validate its 5D K/V shape.
    """
    return window_size[0] < 0 and block_table is not None and not skip_reduce


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
    c_logit_scale and the V dequant."""
    return all(
        d is not None and d.dtype == torch.float32 and d.numel() == 1
        for d in (q_descale, k_descale, v_descale)
    )


def _geometry_ok(
    head_size, num_query_heads, num_kv_heads, num_queries_per_kv, cu_seqlens_q, num_seqs
) -> bool:
    """Fixed head dim; GQA group divides block_m for M-packing; cu_seqlens covers
    every sequence."""
    return (
        head_size == _HEAD_DIM
        and num_query_heads % num_kv_heads == 0
        and _BLOCK_M % num_queries_per_kv == 0
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
    """Sinks (per-head [num_query_heads] fp32) are served only single-split; the
    dispatch gate refuses split-K when sinks is set, so an accepted call always
    lands single-split."""
    return sinks is None or (
        sinks.dtype == torch.float32 and sinks.numel() == num_query_heads
    )


def _supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_k,
    causal,
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
    marshalling so it can be unit-tested against meta tensors, with no GPU."""
    if not is_flydsl_available():
        return False

    head_size = q.shape[-1]
    num_query_heads = q.shape[1]

    return (
        _dispatch_mode_ok(window_size, block_table, shuffled_kv_cache, skip_reduce)
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


def _cu_seqlens_kv(seqused_k: torch.Tensor, num_seqs: int) -> torch.Tensor:
    """Lengths -> cumulative offsets.

    ``unified_attention`` passes per-sequence KV *lengths*; the kernel wants
    cumulative offsets and recovers the length as ``cu[i+1] - cu[i]``. Under
    paging the absolute bases are unobservable -- init_gmem_offsets drops
    kv_tok_base from the KV offset and the paged path builds per-page
    descriptors instead of a whole-tensor view -- so a synthesized cumsum is
    exactly equivalent to the real one.
    """
    cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=seqused_k.device)
    torch.cumsum(seqused_k, 0, out=cu[1:])
    return cu


def _scaled_q_descale(q_descale: torch.Tensor, softmax_scale: float, head_size: int):
    """Fold an arbitrary softmax scale into the Q descale.

    The kernel takes no runtime softmax scale: init_descale bakes in
    ``rsqrt(head_dim) * log2e`` and multiplies it by ``q_descale * k_descale``
    to form ``c_logit_scale``. Since that's a product, scaling q_descale by
    ``softmax_scale / rsqrt(head_dim)`` yields the requested scale with no
    kernel change.

    Sound only while q_descale reaches c_logit_scale and nothing else (holds
    today: exactly two uses, load and multiply) and c_logit_scale is applied
    only to a logit DIFFERENCE, which is what a softmax scale is. A test
    guards the invariant.

    For the near-universal ``softmax_scale == 1/sqrt(head_size)`` the ratio
    is exactly 1 and the tensor passes through untouched.
    """
    ratio = float(softmax_scale) * math.sqrt(head_size)
    if abs(ratio - 1.0) <= 1e-6:
        return q_descale
    return q_descale * ratio


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
):
    """Run unified attention on the FlyDSL fp8 gfx950 kernel.

    The positional parameters mirror ``unified_attention`` exactly so the hook is
    a mechanical forward. The keyword-only block is quantities the caller has
    already derived; recomputing them here would duplicate its layout unpacking.

    Returns ``out`` (written in place) if this configuration is supported, or
    ``None`` so the caller falls through to Triton.
    """
    if not _supported(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
        max_seqlen_k,
        causal,
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

    # Mixed-batch dispatch split. The tier selection below picks ONE
    # num_kv_splits for the whole launch, which is wrong for one half of a mixed
    # batch: a machine-filling prefill chunk forced onto split-K, or riding
    # decodes denied it. A mixed batch is independent sequences, so route each
    # half to its own recursive call -- each re-runs tier selection and lands on
    # its optimal path (prefill -> single-pass 2d, decode -> split-K 3d). Both
    # halves are contiguous row-range VIEWS of q/out (no gather), writing
    # disjoint output rows in place. Declines to a single call (still correct)
    # unless the batch is a clean two-block prefill/decode split whose decode
    # half is deep enough for split-K to amortize the split's overhead. Recursion
    # terminates after one level: a pure-prefill sub-batch returns None from the
    # partition, and a pure-decode sub-batch has max_seqlen_q == 1 so the guard
    # below is false.
    #
    # max_seqlen_k >= _SPLIT_MIN_DECODE_KV is a cheap host-scalar pre-check: no
    # decode can be deeper than the batch max, so a shallow batch skips the
    # device partition entirely. Only when it could contain a deep decode do we
    # pay _partition_mixed's sync and check the decode half's own depth exactly.
    # The pre-check is host-only; the probe (_partition_mixed) costs a device
    # sync. A shallow-decode batch whose deep prefill chunk satisfies the
    # pre-check would probe and decline on every layer, so a decline is memoized
    # (host-only key) and its later recurrences skip the probe. Takes are never
    # memoized -- they re-slice on the probed split_point and must re-probe.
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
            split_point, prefill_first, n_pre, n_dec, pre_max_q, _dec_max_kv = part
            row_split = int(cu_seqlens_q[split_point])
            total_q = q.shape[0]
            if prefill_first:
                pre_rows, dec_rows = (0, row_split), (row_split, total_q)
                pre_seqs, dec_seqs = (0, split_point), (split_point, num_seqs)
            else:
                dec_rows, pre_rows = (0, row_split), (row_split, total_q)
                dec_seqs, pre_seqs = (0, split_point), (split_point, num_seqs)

            def _sub(rows, seqs, n_sub, max_q_sub):
                r0, r1 = rows
                s0, s1 = seqs
                flydsl_unified_attention(
                    q[r0:r1],
                    k,
                    v,
                    out[r0:r1],
                    cu_seqlens_q[s0 : s1 + 1] - cu_seqlens_q[s0],
                    max_q_sub,
                    seqused_k[s0:s1],
                    max_seqlen_k,
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
                )

            _sub(pre_rows, pre_seqs, n_pre, pre_max_q)  # prefill -> single-pass 2d
            _sub(dec_rows, dec_seqs, n_dec, 1)  # decode -> split-K 3d
            return out

    num_query_heads = q.shape[1]
    out_dtype_str = "f16" if out.dtype == torch.float16 else "bf16"

    # Tier selection. Split-K fires when ALL hold; otherwise single-pass.
    #  - max_seqlen_k > _PAGE_SIZE * 20 (1280): below this the combine pass
    #    isn't amortized (measured crossover).
    #  - num_2d_prgms < target: the machine is underfilled, the only regime
    #    split-K helps -- a high-chunk mixed batch (long prefill) fills the
    #    machine on its own and stays single-pass; only low-chunk mixes and
    #    all-decode route to split-K.
    #  - sinks is None: split-K + sinks is refused by the builder (exp(sink)
    #    would be double-counted across split combines).
    # The combine rebases its O write on cu_seqlens_q, so varlen batches with
    # unequal query lengths combine correctly under the same dense workspace.
    #
    # num_2d_prgms is the single-pass base workgroup count: num_kv_heads *
    # sum_i ceil(query_len_i / BLOCK_Q) (packed per-sequence work, not the
    # dense grid which over-counts padding). BLOCK_Q is block_m / GQA group.
    #
    # The exact sum needs per-sequence query lengths on cu_seqlens_q (device);
    # reading via .item() forces a ~29us host sync into the dispatch path, so
    # it's avoided wherever the branch can be decided from host scalars alone:
    #  - all-decode (max_seqlen_q == 1): sum is exactly num_seqs, no read
    #    needed -- this is the shallow split-K regime where the sync used to
    #    dominate kernel time.
    #  - otherwise, a host-side lower bound max(num_seqs, ceil(total_q /
    #    BLOCK_Q)) already fills the machine for a real prefill/high-chunk
    #    mix, so single-pass is decided without a read.
    #  - only a genuine low-chunk mix, where that lower bound underfills,
    #    needs the exact per-sequence sum and takes the device read.
    # Every branch matches the exact-sum decision; the read is skipped only
    # where it cannot change the outcome.
    num_kv_splits = 1
    if max_seqlen_k > _PAGE_SIZE * 20 and sinks is None:
        target_num_prgms = _target_num_prgms(q.device.index)
        block_q = _BLOCK_M // num_queries_per_kv
        if max_seqlen_q == 1:
            num_q_blocks = num_seqs
        else:
            total_q = q.shape[0]
            lower_bound = max(num_seqs, (total_q + block_q - 1) // block_q)
            if num_kv_heads * lower_bound >= target_num_prgms:
                # Provably full at single-pass; exact value only gates <target.
                num_q_blocks = lower_bound
            else:
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
                num_q_blocks = int(
                    ((seqlens_q + (block_q - 1)) // block_q).sum().item()
                )
        num_2d_prgms = num_kv_heads * num_q_blocks
        if num_2d_prgms < target_num_prgms:
            num_kv_splits = _split_count(num_2d_prgms, target_num_prgms)

    kernel = _get_kernel(
        num_query_heads,
        num_kv_heads,
        bool(causal),
        out_dtype_str,
        sinks is not None,
        num_kv_splits,
        shuffled_kv_cache,
    )

    workspace = None
    if num_kv_splits > 1:
        # fp32 partial workspace: O_partial + Mrow + Lrow, sized for the dense
        # [batch, max_seqlen_q(==1), heads, ...] layout the store/combine use.
        ws_elems = dualwave_splitk_workspace_elems(
            num_seqs, num_query_heads, int(max_seqlen_q), num_kv_splits, _HEAD_DIM
        )
        workspace = torch.empty(ws_elems, device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device.index):
        kernel(
            # .reshape(-1) relies on _strides_ok already restricting Q/O to
            # flattenable layouts (see _strides_ok), so this is always a
            # view here, never the silent copy that would occur otherwise.
            _as_i8(q).reshape(-1),
            # K/V go in as a rank-2 [num_blocks, page_row_elems] view, NOT
            # flattened. FlyDSL's C-ABI codec packs every memref shape dim as
            # signed int32 (only strides may be int64), so a flattened KV pool
            # of >= 2**31 elements -- e.g. num_blocks=32768, block_size=64,
            # 8 kv-heads, head_dim=128 fp8 = exactly 2**31 bytes -- overflows
            # the shape field and raises struct.error at launch. The kernel
            # only takes K/V's base pointer (get_iter; the paged path rebases
            # per page from the block table, dense num_records is a scalar
            # arg), so the shape is never read -- keeping num_blocks as its own
            # int32 dim removes the overflow with no addressing or perf change.
            _as_i8(k).reshape(k.shape[0], -1),
            _as_i8(v).reshape(v.shape[0], -1),
            out.reshape(-1),
            num_seqs,
            # grid.y is ceil(seq_len / BLOCK_Q), so this must be the MAX q len;
            # per-sequence trimming comes from cu_seqlens_q via the active guard.
            int(max_seqlen_q),
            # The KV stride is the within-page row stride, not the page stride.
            #
            # Shuffled 5D KV cache (stage 4): `k.stride(1)` is always the true
            # stride_kv_n = per-token element count across ALL kv_heads
            # (num_kv_heads * head_dim). This arg sizes the paged
            # buffer-descriptor: `page_elems = BLOCK_N * stride_kv_n_v *
            # ELEM_BYTES` (init_descriptors, flash_attn_dualwave_common.py
            # ~:1263) bounds one page and rebases the per-page-id pointer, so it
            # must be the true per-page geometry regardless of layout. For the
            # linear 4D cache [nb, block, kv_heads, head] that equals k.stride(1).
            # For the 5D shuffled cache [nb, kv_heads, head//x, block, x],
            # k.stride(1) is the kv_heads-dim stride (head_dim*block = 16x too
            # large here), which oversizes the descriptor and walks page rebases
            # out of bounds -- garbage single-pass, OOB fault on split-K. So for
            # 5D compute it from shape. (The vectorized loaders derive their own
            # fetch address from trait constants; this value is only the
            # descriptor geometry for them.)
            (num_kv_heads * _HEAD_DIM if k.dim() == 5 else k.stride(1)),
            q.stride(0),
            workspace=workspace,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=_cu_seqlens_kv(seqused_k, num_seqs),
            q_descale=_scaled_q_descale(q_descale, softmax_scale, q.shape[-1]),
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table.reshape(-1),
            block_table_stride=int(block_table.stride(0)),
            sink=None if sinks is None else sinks.reshape(-1),
            stream=torch.cuda.current_stream(q.device),
        )
    return out
