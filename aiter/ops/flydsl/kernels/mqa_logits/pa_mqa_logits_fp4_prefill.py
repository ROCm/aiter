# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

from functools import cache, lru_cache
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import triton
import triton.language as tl
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import Float4E2M1FN, Int32, T

from ..candidate_topk_common import (
    BLOCK_THREADS as TOPK_BLOCK_THREADS,
)
from ..candidate_topk_common import (
    NUM_HIST_BINS,
    NUM_KEY_BYTES,
    NUM_WAVES,
    RADIX_SIGN_BIT,
    block_exclusive_prefix_i32,
    f32_to_ordered_i32,
    key_is_better,
    key_matches_prefix,
    make_streaming_topk_storage,
    prefix_byte_mask,
    radix_byte,
)
from ..kernels_common import atomic_add_i32
from .pa_mqa_logits_fp4_common import (
    _i32_buffer,
    _load_vec4_i32,
)

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_WARPS = 4
MFMA_M = 16
MFMA_N = 16
WARP_SIZE = 64
DEFAULT_BLOCK_THREADS = DEFAULT_NUM_WARPS * WARP_SIZE  # 256
# Reduce once per 4K produced scores instead of once per 1K. With 16-byte
# aligned fields this consumes 42,064 bytes (K=512) or 50,256 bytes (K=1024),
# leaving both variants below gfx950's 64-KiB per-workgroup LDS limit.
DEFAULT_SCORE_BATCH_CHUNKS = 16
SUPPORTED_SCORE_BATCH_CHUNKS = (1, 2, 4, 8, 16)
_LDS_ALIGNMENT_BYTES = 16
_GFX950_MAX_WORKGROUP_LDS_BYTES = 64 * 1024

# cta_info packed fields per CTA.
CTA_INFO_WIDTH = 6

_STREAM_RETAINED = 0
_STREAM_PENDING = 1
_STREAM_SCORE_PREFIX = 2
_STREAM_SCORE_MASK = 3
_STREAM_INDEX_PREFIX = 4
_STREAM_INDEX_MASK = 5
_STREAM_REMAINING = 6
_STREAM_WRITE_CURSOR = 7
_STREAM_EQUAL_SEEN = 8
_STREAM_STATE_SIZE = 9
_PACK_SHIFT = 16
_PACK_MASK = (1 << _PACK_SHIFT) - 1


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _streaming_topk_lds_bytes(candidate_topk: int, score_batch_chunks: int) -> int:
    """Exact LDS footprint of ``make_streaming_topk_storage``."""

    pool_capacity = candidate_topk + score_batch_chunks * DEFAULT_BLOCK_THREADS
    field_bytes = (
        pool_capacity * 4,  # pool_values
        pool_capacity * 4,  # pool_indices
        candidate_topk * 4,  # selected_values
        candidate_topk * 4,  # selected_indices
        NUM_HIST_BINS * 4,
        (NUM_WAVES + 1) * 4,
        _STREAM_STATE_SIZE * 4,
    )
    offset = 0
    for size in field_bytes:
        offset = _align_up(offset, _LDS_ALIGNMENT_BYTES) + size
    return _align_up(offset, _LDS_ALIGNMENT_BYTES)


def compute_prefill_schedule(
    row_to_batch,
    local_starts,
    local_ends,
    block_k,
    parallel_unit_num,
    max_seq_len,
    cta_info_out=None,
    row_offsets_out=None,
    *,
    single_cta_per_row=False,
):
    """Compute the persistent-grid schedule for ragged-prefill MQA logits.

    Pass `cta_info_out` (a fixed [parallel_unit_num, CTA_INFO_WIDTH] int32 buffer)
    to write the schedule into a stable address (CUDAGraph decode: the captured
    kernel replays from this pointer while `build()` refreshes its contents).
    ``single_cta_per_row`` coalesces every nonempty row into one slot; this is
    used when the launch has exactly one available slot per row.

    Returns `(safe, cta_info, parallel_unit_num)`, `safe` being the [1] int32
    split factor the schedule was built with.

    The row plan is every-lane work over one [T] vector, so up to
    `_ROW_PLAN_MAX_ROWS` rows it is one block instead of the ~25 torch ops
    `_row_plan_torch` spells it as. What that buys is HOST latency, not device
    time -- the torch form hides its own kernels behind its own dispatch, and a
    decode step calls this once per forward, in the host gap between two.
    """
    device = local_ends.device
    P = parallel_unit_num
    T = local_ends.shape[0]  # fixed total_tokens (rows)

    assert P >= T, (
        f"compute_prefill_schedule: parallel_unit_num={P} < rows={T} would "
        f"silently drop rows past slot {P} (logits stay at the caller's "
        f"pre-fill -> wrong top-k). Pass parallel_unit_num >= number of rows."
    )

    # Asserted, not coerced. Every caller already holds int32, so the three
    # `.to()` calls this replaces were no-ops -- but on a host-bound path a
    # no-op still costs its dispatch, and a caller that stopped honouring this
    # would silently change what the kernels below compile to.
    assert (
        row_to_batch.dtype == local_starts.dtype == local_ends.dtype == torch.int32
    ), (
        f"compute_prefill_schedule: row_to_batch/local_starts/local_ends must be "
        f"int32, got {row_to_batch.dtype}/{local_starts.dtype}/{local_ends.dtype}."
    )
    if row_offsets_out is not None and (
        row_offsets_out.dtype != torch.int32
        or row_offsets_out.ndim != 1
        or row_offsets_out.shape[0] < T + 1
        or row_offsets_out.device != device
        or not row_offsets_out.is_contiguous()
    ):
        raise ValueError(
            "row_offsets_out must be a contiguous one-dimensional int32 buffer "
            "with at least rows + 1 entries on the schedule device"
        )

    s_max = max(1, (max_seq_len + block_k - 1) // block_k)
    plan = _row_plan(
        local_starts,
        local_ends,
        block_k,
        P,
        s_max,
        single_cta_per_row=single_cta_per_row,
    )

    # ── map each fixed slot → (row, split) + emit cta_info in ONE kernel ──
    if cta_info_out is None:
        cta_info = torch.empty(P, CTA_INFO_WIDTH, dtype=torch.int32, device=device)
    else:
        cta_info = cta_info_out
    BLOCK_P = 256
    grid = (triton.cdiv(P, BLOCK_P),)
    _prefill_cta_info_kernel[grid](
        plan.incl,
        plan.excl,
        plan.chunks,
        plan.first_chunks,
        row_to_batch,
        local_starts,
        local_ends,
        plan.safe,
        plan.total_splits,
        cta_info,
        T,
        P,
        BLOCK_P=BLOCK_P,
    )
    if row_offsets_out is not None:
        _prefill_row_offsets_kernel[(triton.cdiv(T + 1, BLOCK_P),)](
            plan.incl,
            row_offsets_out,
            T,
            BLOCK=BLOCK_P,
        )
    return plan.safe, cta_info, P


class _RowPlan(NamedTuple):
    """Everything the emit kernel needs about the rows, from either producer.

    One carrier, so a field can only be added where both have to answer for it.
    """

    incl: torch.Tensor  # [T] inclusive prefix sum of per-row CTA counts
    excl: torch.Tensor  # [T] exclusive prefix sum
    chunks: torch.Tensor  # [T] chunks intersecting [local_start, local_end)
    first_chunks: torch.Tensor  # [T] floor(max(local_start, 0) / block_k)
    safe: torch.Tensor  # [1] chunk-splits merged into one CTA
    total_splits: torch.Tensor  # [1] number of valid (row, split) slots


class FP4PrefillTopKWorkspace(NamedTuple):
    """Caller-owned scratch for fused FP4 prefill TopK."""

    cta_info: torch.Tensor
    row_offsets: torch.Tensor
    candidate_values: torch.Tensor
    candidate_indices: torch.Tensor
    candidate_counts: torch.Tensor
    merge_values: torch.Tensor
    merge_indices: torch.Tensor
    merge_physical_indices: torch.Tensor


class FP4PrefillTopKResult(NamedTuple):
    values: torch.Tensor
    raw_indices: torch.Tensor
    physical_indices: torch.Tensor
    counts: torch.Tensor


def allocate_fp4_prefill_topk_workspace(
    rows: int,
    parallel_unit_num: int,
    topk: int,
    device: torch.device | str,
) -> FP4PrefillTopKWorkspace:
    """Allocate all scratch needed by the no-logits fused path."""

    if rows <= 0:
        raise ValueError("rows must be positive")
    if parallel_unit_num < rows:
        raise ValueError("parallel_unit_num must be at least rows")
    if topk not in (512, 1024):
        raise ValueError("topk must be 512 or 1024")
    return FP4PrefillTopKWorkspace(
        cta_info=torch.empty(
            parallel_unit_num,
            CTA_INFO_WIDTH,
            dtype=torch.int32,
            device=device,
        ),
        row_offsets=torch.empty(rows + 1, dtype=torch.int32, device=device),
        candidate_values=torch.empty(
            parallel_unit_num,
            topk,
            dtype=torch.float32,
            device=device,
        ),
        candidate_indices=torch.empty(
            parallel_unit_num,
            topk,
            dtype=torch.int32,
            device=device,
        ),
        candidate_counts=torch.empty(
            parallel_unit_num,
            dtype=torch.int32,
            device=device,
        ),
        merge_values=torch.empty(rows, topk, dtype=torch.float32, device=device),
        merge_indices=torch.empty(rows, topk, dtype=torch.int32, device=device),
        merge_physical_indices=torch.empty(
            rows,
            topk,
            dtype=torch.int32,
            device=device,
        ),
    )


# Rows the fused arm plans, and the only two widths it plans them in. A block
# costs its WIDTH, not the rows that fill it -- 40.8us at 16384 lanes whether 300
# rows or 16384 arrive -- so the choice is which shapes share a width, and every
# distinct width is a kernel to compile:
#
#     block    device    cold compile   what reaches it
#      4096    13.3us          0.84s    every decode forward
#     16384    40.8us          1.86s    every prefill forward
#
# The FLOOR is what keeps decode off the wide block: `max_num_seqs * (1 + spec
# steps)` = 512 x 8 fits in one width, so decode compiles once and pays 13.3us.
# Sizing per shape below it saves 0.3us for nine more variants.
#
# NOTHING between them, though an 8192 rung measured 21.1us: on a 100k/10 conc-50
# trace 534 of 541 prefill forwards ran 8193-16384 rows and four ran 4097-8192,
# so that rung would cost a third of the ladder and a 1.13s compile to save 20us
# on 0.7% of forwards -- and a variant nothing exercises is one discovered
# mid-run.
#
# The CAP is where widening stops paying: 32768 is bit-exact and still beats the
# torch arm on both axes (87.1us device against 338.8us), but takes 4.98s to
# compile, and 65536 does not finish compiling at all.
_ROW_PLAN_BLOCK_FLOOR = 4096
_ROW_PLAN_MAX_ROWS = 16384

# int32s per 16 bytes -- the granularity Triton's pointer-alignment
# specialization works at.
_I32_PER_16B = 4


def _row_plan(
    ls,
    le,
    block_k,
    P,
    s_max,
    *,
    single_cta_per_row=False,
) -> _RowPlan:
    T = le.shape[0]
    if T > _ROW_PLAN_MAX_ROWS:
        return _row_plan_torch(
            ls,
            le,
            block_k,
            P,
            s_max,
            single_cta_per_row=single_cta_per_row,
        )
    # One allocation, sliced: separate `torch.empty` calls would add dispatches
    # dispatches back on the path this exists to shorten.
    #
    # Every region starts on a 16-byte boundary, which is not cosmetic: Triton
    # specializes a kernel on whether each pointer is 16-byte aligned, so a
    # stride of exactly T forks a variant per `T % 4` and the JIT never stops
    # finding new ones mid-run. Rounding the stride up pins every pointer to
    # one alignment signature.
    stride = (T + _I32_PER_16B - 1) // _I32_PER_16B * _I32_PER_16B
    tail = 4 * stride
    work = torch.empty(tail + 2 * _I32_PER_16B, dtype=torch.int32, device=le.device)
    plan = _RowPlan(
        incl=work[:T],
        excl=work[stride : stride + T],
        chunks=work[2 * stride : 2 * stride + T],
        first_chunks=work[3 * stride : 3 * stride + T],
        safe=work[tail : tail + 1],
        total_splits=work[tail + _I32_PER_16B : tail + _I32_PER_16B + 1],
    )
    # Named, not `*plan`: field ORDER should not become load-bearing.
    _prefill_row_plan_kernel[(1,)](
        ls,
        le,
        plan.incl,
        plan.excl,
        plan.chunks,
        plan.first_chunks,
        plan.safe,
        plan.total_splits,
        T,
        P,
        block_k,
        s_max,
        BLOCK_T=(
            _ROW_PLAN_BLOCK_FLOOR if T <= _ROW_PLAN_BLOCK_FLOOR else _ROW_PLAN_MAX_ROWS
        ),
        SEARCH_STEPS=max(1, (s_max - 1).bit_length() + 1),
        SINGLE_CTA_PER_ROW=single_cta_per_row,
    )
    return plan


def _row_plan_torch(
    ls,
    le,
    block_k,
    P,
    s_max,
    *,
    single_cta_per_row=False,
) -> _RowPlan:
    """The row plan as ~25 torch ops. Reference for `_prefill_row_plan_kernel`."""
    window_starts = torch.clamp(ls, min=0)
    first_chunks = window_starts // block_k
    end_chunks = torch.clamp((le + (block_k - 1)) // block_k, min=0)
    chunks_per_row = torch.where(
        le > window_starts,
        torch.clamp(end_chunks - first_chunks, min=0),
        0,
    )

    max_chunks = torch.clamp(chunks_per_row.max(), min=1).to(torch.int32)
    if single_cta_per_row:
        safe = max_chunks
    else:
        s_cand = torch.arange(
            1,
            s_max + 1,
            device=le.device,
            dtype=torch.int32,
        )  # [s_max]
        ctas_per_r_s = (chunks_per_row[None, :] + (s_cand[:, None] - 1)) // (
            s_cand[:, None]
        )  # [s_max, T]
        total_ctas_s = ctas_per_r_s.sum(dim=1)  # [s_max]
        feasible = total_ctas_s <= P  # [s_max] bool, monotonic False..True
        # smallest feasible s, via arithmetic (no tensor gather → no capture sync).
        first_feasible_s = torch.clamp(
            (~feasible).to(torch.int32).sum() + 1,
            max=s_max,
        )
        safe = torch.where(feasible.any(), first_feasible_s, max_chunks).to(
            torch.int32
        )

    # ── per-row number of CTAs (chunk-splits); 0 for empty rows ──
    ctas_r = (chunks_per_row + (safe - 1)) // safe  # [T]
    incl = torch.cumsum(ctas_r, dim=0, dtype=torch.int32)  # [T] inclusive prefix sum
    return _RowPlan(
        incl=incl,
        excl=incl - ctas_r,  # exclusive prefix sum
        chunks=chunks_per_row.to(torch.int32),
        first_chunks=first_chunks.to(torch.int32),
        safe=safe.reshape(1).to(torch.int32),
        total_splits=incl[-1].reshape(1).to(torch.int32),
    )


# `T` and `P` are batch shape, and a decode step's row count changes every
# forward, so specializing on whether they divide 16 keeps finding new variants
# to compile in the middle of a run. Both kernels here: over 17 shapes it takes
# this pair from 9 and 6 variants down to 3 and 1, and what is left of the 3 is
# exactly the BLOCK_T ladder -- a set small enough to be compiled through.
#
# `T` alone is the whole of that halving AND the whole of its cost: dropping it
# gives up the divisibility hint the wide blocks vectorize on, 41.1us -> 52.0us
# at 16384 lanes. That lands on prefill, one call per ~500ms forward; the widths
# decode runs stay hidden behind the call's own dispatch either way.
@triton.jit(do_not_specialize=["T", "P"])
def _prefill_row_plan_kernel(
    ls_ptr,  # [T] int32 local_starts
    le_ptr,  # [T] int32 local_ends
    incl_ptr,  # [T] int32 out
    excl_ptr,  # [T] int32 out
    chunks_ptr,  # [T] int32 out
    first_chunks_ptr,  # [T] int32 out
    safe_ptr,  # [1] int32 out
    total_splits_ptr,  # [1] int32 out
    T,
    P,
    block_k,
    s_max,
    BLOCK_T: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    SINGLE_CTA_PER_ROW: tl.constexpr,
):
    """Single-block row plan: chunk counts, split factor, and its prefix sums.

    `total_ctas(s) = sum ceil(chunks/s)` is non-increasing in s, so feasibility is
    monotone and the smallest feasible s is a binary search. `_row_plan_torch`
    instead materializes the whole [s_max, T] feasibility matrix and counts its
    False entries -- same answer, and the matrix is why that path's cost also
    tracks the model's context length.

    Masked-out lanes carry `chunks = 0`, which contributes 0 CTAs at every s, so
    no reduction below needs a second mask.

    `SEARCH_STEPS` is derived from `s_max`, not a generous constant: the range
    halves each step, so `(s_max - 1).bit_length()` converges it and the caller
    passes one more. Every surplus step is another reduction over all BLOCK_T
    lanes -- 0.74us each at 4096 rows -- and a fixed 32 was this kernel's ENTIRE
    growth with block width, 28.7us against 12.8us. The one spare step is not
    that: it is hidden behind the call's own dispatch, and coming up short
    returns a schedule that is merely wrong, with nothing downstream to fault.
    """
    t = tl.arange(0, BLOCK_T)
    mask = t < T
    ls = tl.load(ls_ptr + t, mask=mask, other=0)
    le = tl.load(le_ptr + t, mask=mask, other=0)
    window_starts = tl.maximum(ls, 0)
    first_chunks = tl.where(mask, window_starts // block_k, 0)
    end_chunks = tl.maximum((le + block_k - 1) // block_k, 0)
    chunks = tl.where(
        mask & (le > window_starts),
        tl.maximum(end_chunks - first_chunks, 0),
        0,
    )
    max_chunks = tl.maximum(tl.max(chunks, axis=0), 1)

    if SINGLE_CTA_PER_ROW:
        safe = max_chunks
    else:
        lo = 1
        hi = s_max
        for _ in tl.static_range(SEARCH_STEPS):
            mid = (lo + hi) // 2
            feasible = tl.sum((chunks + mid - 1) // mid, axis=0) <= P
            active = lo < hi
            hi = tl.where(active & feasible, mid, hi)
            lo = tl.where(active & (feasible == 0), mid + 1, lo)
        # No s in [1, s_max] fits: fall back to one CTA per row, as torch's
        # `feasible.any()` arm does.
        total_smax = tl.sum((chunks + s_max - 1) // s_max, axis=0)
        safe = tl.where(total_smax <= P, lo, max_chunks)

    ctas = tl.where(mask, (chunks + safe - 1) // safe, 0)
    incl = tl.cumsum(ctas, axis=0)
    tl.store(incl_ptr + t, incl, mask=mask)
    tl.store(excl_ptr + t, incl - ctas, mask=mask)
    tl.store(chunks_ptr + t, chunks, mask=mask)
    tl.store(first_chunks_ptr + t, first_chunks, mask=mask)
    tl.store(safe_ptr, safe)
    tl.store(total_splits_ptr, tl.sum(ctas, axis=0))


@triton.jit(do_not_specialize=["T", "P"])
def _prefill_cta_info_kernel(
    incl_ptr,  # [T] int32 inclusive prefix sum of per-row CTA counts
    excl_ptr,  # [T] int32 exclusive prefix sum
    chunks_ptr,  # [T] int32 chunks_per_row
    first_chunks_ptr,  # [T] first logical chunk
    rb_ptr,  # [T] int32 row_to_batch
    ls_ptr,  # [T] int32 local_starts
    le_ptr,  # [T] int32 local_ends
    safe_ptr,  # [1] int32
    total_splits_ptr,  # [1] int32
    cta_info_ptr,  # [P, 6] int32
    T,
    P,
    BLOCK_P: tl.constexpr,
):
    """Single-kernel slot->row mapping + cta_info emit for ragged prefill."""
    pid = tl.program_id(0)
    safe = tl.load(safe_ptr)
    total_splits = tl.load(total_splits_ptr)
    slot = pid * BLOCK_P + tl.arange(0, BLOCK_P)  # [BLOCK_P]
    smask = slot < P
    valid = slot < total_splits

    # searchsorted(incl, slot, right=True) = count(incl <= slot): per-slot
    # binary search over incl[T] in global memory (~log2(T) iters).
    lo = tl.zeros([BLOCK_P], tl.int32)
    hi = tl.full([BLOCK_P], T, tl.int32)
    for _ in tl.static_range(32):
        mid = (lo + hi) // 2
        incl_mid = tl.load(
            incl_ptr + tl.minimum(mid, T - 1), mask=(mid < T), other=2147483647
        )
        go_right = incl_mid <= slot
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
    safe_row = tl.minimum(lo, T - 1)  # clamp for gather

    excl_r = tl.load(excl_ptr + safe_row, mask=smask, other=0)
    chunks_r = tl.load(chunks_ptr + safe_row, mask=smask, other=0)
    first_chunk_r = tl.load(first_chunks_ptr + safe_row, mask=smask, other=0)
    rb_r = tl.load(rb_ptr + safe_row, mask=smask, other=0)
    ls_r = tl.load(ls_ptr + safe_row, mask=smask, other=0)
    le_r = tl.load(le_ptr + safe_row, mask=smask, other=0)

    vi = valid.to(tl.int32)
    split_within = slot - excl_r
    relative_start = split_within * safe  # pre-mask (count uses this)
    count = tl.maximum(tl.minimum(safe, chunks_r - relative_start), 0)
    row_id = safe_row * vi
    batch_id = rb_r * vi
    start = (first_chunk_r + relative_start) * vi
    # Zero is an inactive-CTA sentinel consumed uniformly by the score kernel.
    # In particular, padded schedule slots must not manufacture a chunk zero:
    # its page-table entry may legitimately be -1 for an empty request.
    count = tl.where(valid, count, 0)
    ls_out = ls_r * vi
    le_out = le_r * vi

    base = slot * 6
    tl.store(cta_info_ptr + base + 0, row_id, mask=smask)
    tl.store(cta_info_ptr + base + 1, batch_id, mask=smask)
    tl.store(cta_info_ptr + base + 2, start, mask=smask)
    tl.store(cta_info_ptr + base + 3, count, mask=smask)
    tl.store(cta_info_ptr + base + 4, ls_out, mask=smask)
    tl.store(cta_info_ptr + base + 5, le_out, mask=smask)


@triton.jit(do_not_specialize=["T"])
def _prefill_row_offsets_kernel(
    incl_ptr,  # [T] inclusive prefix sum
    row_offsets_ptr,  # [T + 1] output
    T,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset <= T
    previous_row = tl.maximum(offset - 1, 0)
    value = tl.load(incl_ptr + previous_row, mask=mask & (offset > 0), other=0)
    tl.store(row_offsets_ptr + offset, value, mask=mask)


@triton.jit(do_not_specialize=["rows"])
def _copy_single_cta_topk_candidates_kernel(
    candidate_values,
    candidate_indices,
    candidate_counts,
    row_offsets,
    out_values,
    out_indices,
    out_counts,
    rows,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy one exact CTA plane per row without repeating radix selection."""

    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    cta_begin = tl.load(row_offsets + row)
    cta_end = tl.load(row_offsets + row + 1)
    has_cta = cta_end > cta_begin
    count = tl.load(candidate_counts + cta_begin, mask=has_cta, other=0)
    count = tl.minimum(tl.maximum(count, 0), TOPK)

    for step in tl.static_range(0, TOPK, BLOCK):
        slot = step + lane
        live = has_cta & (slot < count)
        source = cta_begin * TOPK + slot
        values = tl.load(
            candidate_values + source,
            mask=live,
            other=-float("inf"),
        )
        indices = tl.load(candidate_indices + source, mask=live, other=-1)
        destination = row * TOPK + slot
        tl.store(out_values + destination, values)
        tl.store(out_indices + destination, indices)
    tl.store(out_counts + row + lane, count, mask=lane == 0)


def _copy_single_cta_topk_candidates(
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    row_offsets: torch.Tensor,
    out_values: torch.Tensor,
    out_indices: torch.Tensor,
    out_counts: torch.Tensor,
) -> None:
    """Launch the no-radix merge for a schedule with at most one CTA per row."""

    rows, topk = out_values.shape
    block = 256
    _copy_single_cta_topk_candidates_kernel[(rows,)](
        candidate_values,
        candidate_indices,
        candidate_counts,
        row_offsets,
        out_values,
        out_indices,
        out_counts,
        rows,
        TOPK=topk,
        BLOCK=block,
    )


def build_pa_mqa_logits_fp4_prefill_module(
    block_k=256,
    kv_block_size=64,
    max_blocks_per_seq=256,
    max_chunks_per_cta=16,
    num_warps=DEFAULT_NUM_WARPS,
    heads=DEFAULT_HEADS,
    head_dim=DEFAULT_HEAD_DIM,
    candidate_topk=None,
    score_batch_chunks=DEFAULT_SCORE_BATCH_CHUNKS,
):
    """Build the ragged-prefill FP4 scorer.

    With ``candidate_topk=None`` the score callback writes the legacy logits
    matrix. With 512 or 1024 it instead streams bounded score batches through
    LDS and keeps an exact per-CTA candidate set.
    """
    block_threads_k = num_warps * WARP_SIZE
    emit_candidates = candidate_topk is not None
    if emit_candidates:
        assert candidate_topk in (512, 1024), "candidate_topk must be 512 or 1024"
        assert block_k == 256, "streaming TopK currently requires block_k=256"
        assert (
            block_threads_k == TOPK_BLOCK_THREADS
        ), "streaming TopK requires the 256-thread/4-wave score body"
        assert score_batch_chunks in SUPPORTED_SCORE_BATCH_CHUNKS, (
            "score_batch_chunks must be one of "
            f"{SUPPORTED_SCORE_BATCH_CHUNKS}"
        )
        pool_capacity = candidate_topk + score_batch_chunks * block_k
        lds_bytes = _streaming_topk_lds_bytes(candidate_topk, score_batch_chunks)
        assert lds_bytes <= _GFX950_MAX_WORKGROUP_LDS_BYTES, (
            f"streaming TopK requires {lds_bytes} LDS bytes, exceeding the "
            f"gfx950 workgroup limit of {_GFX950_MAX_WORKGROUP_LDS_BYTES}"
        )
        pool_steps = (pool_capacity + TOPK_BLOCK_THREADS - 1) // TOPK_BLOCK_THREADS
        output_steps = (candidate_topk + TOPK_BLOCK_THREADS - 1) // TOPK_BLOCK_THREADS
        streaming_storage_type = make_streaming_topk_storage(
            candidate_topk,
            pool_capacity,
            _STREAM_STATE_SIZE,
        )
    m_tiles = heads // MFMA_M
    k_tiles = head_dim // 128  # outer K-loop iters (MFMA K=128)
    assert (
        head_dim % 128 == 0
    ), f"head_dim must be a multiple of 128 (MFMA K), got {head_dim}"
    assert heads % MFMA_M == 0, f"heads must be a multiple of {MFMA_M}, got {heads}"

    N_TILES = block_k // MFMA_N
    assert (
        N_TILES % num_warps == 0
    ), f"block_k={block_k} -> N_TILES={N_TILES} must be multiple of num_warps={num_warps}"
    N_TILES_PER_WARP = N_TILES // num_warps

    assert (
        kv_block_size % MFMA_N == 0
    ), f"kv_block_size={kv_block_size} must be a multiple of MFMA_N={MFMA_N}"
    assert (
        block_k % kv_block_size == 0
    ), f"block_k={block_k} must be a multiple of kv_block_size={kv_block_size}"
    TILES_PER_BLOCK = kv_block_size // MFMA_N
    N_PHYS = (N_TILES_PER_WARP + TILES_PER_BLOCK - 1) // TILES_PER_BLOCK

    # block_tables row stride (i32 elements).
    _stride_bt = max_blocks_per_seq

    # KV preshuffle layout: [block_id, K_TILES, K_chunk=4, kv_block_size, 16] uint8.
    _kv_chunk_bytes = 16
    _stride_kv_ktile = 4 * kv_block_size * _kv_chunk_bytes
    _stride_kv_block = k_tiles * _stride_kv_ktile
    # byte stride between consecutive nt tiles inside one kv block (one MFMA_N
    # row of tokens); used as the per-nt constant `soffset` immediate delta.
    _stride_kv_ntile = MFMA_N * _kv_chunk_bytes
    # KV_scale: [block_id, K_TILES, K_chunks=4, kv_block_size]
    _stride_kvs_ktile = 4 * kv_block_size
    _stride_kvs_block = k_tiles * _stride_kvs_ktile

    _kb_is_pow2 = kv_block_size & (kv_block_size - 1) == 0
    _kb_log2 = kv_block_size.bit_length() - 1
    _kb_mask = kv_block_size - 1

    def _floordiv_kb(x):
        return (x >> fx.Int32(_kb_log2)) if _kb_is_pow2 else (x // kv_block_size)

    def _mod_kb(x):
        return (x & fx.Int32(_kb_mask)) if _kb_is_pow2 else (x % kv_block_size)

    QS_DW = (m_tiles + 3) // 4
    qs_pad = QS_DW * 4
    qs_pad_bits = qs_pad * 8

    def _make_qs_buf_copy():
        if qs_pad_bits == 32:
            return fx.rocdl.BufferCopy32b()
        elif qs_pad_bits == 64:
            return fx.rocdl.BufferCopy64b()
        elif qs_pad_bits == 128:
            return fx.rocdl.BufferCopy128b()
        else:
            raise ValueError(f"unsupported QS_DW={QS_DW} (qs_pad_bits={qs_pad_bits})")

    if N_PHYS == 1:

        def _phys_to_list(phys_v):
            return [phys_v] * N_TILES_PER_WARP

    else:

        def _phys_to_list(phys_v):
            return [
                fx.Vector(phys_v)[nt // TILES_PER_BLOCK]
                for nt in range(N_TILES_PER_WARP)
            ]

    @flyc.kernel
    def pa_mqa_logits_fp4_prefill_kernel(
        out_logits_ptr: fx.Tensor,
        candidate_values_ptr: fx.Tensor,
        candidate_indices_ptr: fx.Tensor,
        candidate_counts_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        kv_scale_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
        weights_ptr: fx.Tensor,
        cta_info_ptr: fx.Tensor,  # [n_ctas, 6] i32
        stride_out_row: Int32,
        output_token_base: Int32,
        weight_scale: fx.Float32,
    ):
        tid = gpu.thread_idx.x
        pid = gpu.block_idx.x

        warp_id = tid >> 6
        lane_id = tid % WARP_SIZE
        lane_mod_16 = lane_id & 15
        lane_div_16 = (lane_id >> 4) & 3

        # Per-CTA assignment: pid*CTA_INFO_WIDTH (wave-uniform) folded into the V#
        # base pointer so the 4 fields load as a dwordx4 at row 0; the window
        # bounds (fields 4,5) read as scalars at row 1 cols 0,1.
        cta_src = fx.get_iter(cta_info_ptr)
        cta_it = fx.add_offset(
            fx.recast_iter(fx.PointerType.get(T.i32, cta_src.memspace, 4), cta_src),
            fx.Int64(pid) * fx.Int64(CTA_INFO_WIDTH),
        )
        cta_info_bt = fx.rocdl.make_buffer_tensor(
            fx.make_view(cta_it, fx.make_layout((1 << 28, 4), (4, 1)))
        )
        cta_info_vec = fx.Vector(_load_vec4_i32(cta_info_bt, fx.Int32(0)))
        local_start = cta_info_bt[(fx.Int32(1), fx.Int32(0))]
        local_end = cta_info_bt[(fx.Int32(1), fx.Int32(1))]

        kv_bt = _i32_buffer(kv_cache_ptr, width=4)
        kvs_bt = _i32_buffer(kv_scale_ptr, width=1)
        bt_bt = _i32_buffer(kv_indices_ptr, width=1)

        ZERO_F = fx.Float32(0.0)
        c0_i32 = fx.Int32(0)

        row_id = cta_info_vec[0]
        batch_id = cta_info_vec[1]
        chunk_start = cta_info_vec[2]
        chunk_count = cta_info_vec[3]

        if const_expr(emit_candidates):
            # Inactive padded CTAs skip the page-dependent pipeline below. Their
            # candidate planes are ignored, but the merge still needs an
            # explicit zero count instead of stale workspace contents.
            if tid == 0:
                candidate_counts_ptr[pid] = 0
            topk_storage = fx.SharedAllocator().allocate(streaming_storage_type)
            pool_values = topk_storage.pool_values.peek().view(
                fx.make_layout(pool_capacity, 1)
            )
            pool_indices = topk_storage.pool_indices.peek().view(
                fx.make_layout(pool_capacity, 1)
            )
            selected_values = topk_storage.selected_values.peek().view(
                fx.make_layout(candidate_topk, 1)
            )
            selected_indices = topk_storage.selected_indices.peek().view(
                fx.make_layout(candidate_topk, 1)
            )
            topk_histogram = topk_storage.histogram.peek().view(
                fx.make_layout(NUM_HIST_BINS, 1)
            )
            topk_scan = topk_storage.scan.peek().view(fx.make_layout(NUM_WAVES + 1, 1))
            topk_state = topk_storage.state.peek().view(
                fx.make_layout(_STREAM_STATE_SIZE, 1)
            )
            candidate_values_row = fx.slice(candidate_values_ptr, (pid, None))
            candidate_indices_row = fx.slice(candidate_indices_ptr, (pid, None))
            if tid == 0:
                topk_state[_STREAM_RETAINED] = 0
                topk_state[_STREAM_PENDING] = 0
            gpu.barrier()
        else:
            # Fold the row into the f32 global base pointer; the per-token store
            # offset below stays small (no i32 overflow).
            _row_elems = fx.Int64(row_id) * fx.Int64(stride_out_row)
            out_base = fx.add_offset(fx.get_iter(out_logits_ptr), _row_elems)

        # Q load (hoisted): per (k_tile, mi_idx) a thread loads its 16-byte FP4
        # chunk for head row mi_idx*16+lane_mod_16. Q: [total_tokens, H, D/2] uint8.
        # Scaled FP4 16x16x128 MMA; opsel_b selects the per-nt scale byte, so one
        # atom per nt (opsel_a stays 0 — Q scale is one byte per (k_tile, mi)).
        mfma_atoms = [
            fx.make_mma_atom(
                fx.rocdl.cdna4.MFMA_Scale(
                    16, 16, 128, Float4E2M1FN, Float4E2M1FN, opsel_a=0, opsel_b=nt
                )
            )
            for nt in range(N_TILES_PER_WARP)
        ]

        Q_buf = fx.rocdl.make_buffer_tensor(q_ptr)
        q_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 8)
        q_reg_ty = fx.MemRefType.get(
            T.i8, fx.LayoutType.get(16, 1), fx.AddressSpace.Register
        )
        q_reg_lay = fx.make_layout(16, 1)
        q_a_ops = []
        for k_tile in range_constexpr(k_tiles):
            q_a_ops_kt = []
            for mi_idx in range_constexpr(m_tiles):
                q_row = fx.Int32(mi_idx * MFMA_M) + lane_mod_16
                q_row_bytes = fx.slice(Q_buf, (row_id, q_row, None))
                q_row_div = fx.logical_divide(q_row_bytes, fx.make_layout(16, 1))
                col_idx = fx.Int32(k_tile * 4) + lane_div_16
                r = fx.memref_alloca(q_reg_ty, q_reg_lay)
                fx.copy(q_atom, fx.slice(q_row_div, (None, col_idx)), r)
                q_4xi32 = fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Int32)
                a_frag = fx.make_rmem_tensor(4, fx.Int32)
                a_frag.store(q_4xi32)
                q_a_ops_kt.append(a_frag)
            q_a_ops.append(q_a_ops_kt)

        # Q scale: host-preshuffled [total_tokens, K_TILES, 4, 16, QS_PAD].
        assert m_tiles <= 8, f"m_tiles={m_tiles} > 8 not supported. Use heads <= 128."
        QS_buf = fx.rocdl.make_buffer_tensor(q_scale_ptr)
        qs_atom = fx.make_copy_atom(_make_qs_buf_copy(), 8)
        qs_reg_ty = fx.MemRefType.get(
            T.i8, fx.LayoutType.get(qs_pad, 1), fx.AddressSpace.Register
        )
        qs_reg_lay = fx.make_layout(qs_pad, 1)
        q_scale_ops = []
        for k_tile in range_constexpr(k_tiles):
            row = fx.slice(
                QS_buf, (row_id, fx.Int32(k_tile), lane_div_16, lane_mod_16, None)
            )
            r = fx.memref_alloca(qs_reg_ty, qs_reg_lay)
            fx.copy(qs_atom, row, r)
            qs_dws_vec = fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Int32)
            qs_dws = [qs_dws_vec[i] for i in range(QS_DW)]
            q_scale_ops.append(
                [qs_dws[mi // 4] >> fx.Int32(8 * (mi % 4)) for mi in range(m_tiles)]
            )

        # Weights (hoisted): [total_tokens, H] bf16, addressed by row_id.
        # Loaded as bf16 then widened to f32 for the per-head weighting below.
        W_buf = fx.rocdl.make_buffer_tensor(weights_ptr)
        w_row = fx.slice(W_buf, (row_id, None))
        w_tiled_mi = fx.logical_divide(w_row, fx.make_layout(MFMA_M, 1))
        w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), 16)
        w_reg_ty = fx.MemRefType.get(
            T.bf16, fx.LayoutType.get(4, 1), fx.AddressSpace.Register
        )
        w_reg_lay = fx.make_layout(4, 1)
        ws_vec = fx.Vector.from_elements([weight_scale] * 4, dtype=fx.Float32)
        w_per_lane = []
        for mi_idx in range_constexpr(m_tiles):
            tile = fx.slice(w_tiled_mi, (None, fx.Int32(mi_idx)))
            tile_div = fx.logical_divide(tile, fx.make_layout(4, 1))
            r = fx.memref_alloca(w_reg_ty, w_reg_lay)
            fx.copy(w_atom, fx.slice(tile_div, (None, lane_div_16)), r)
            w_f32 = fx.Vector(fx.memref_load_vec(r).to(fx.Float32))
            w_per_lane.append(w_f32 * ws_vec)

        # ── prologue + N-1 prefetch loop + epilogue ──

        def _load_phys(c_i32_arg):
            ni_base = warp_id * fx.Int32(N_TILES_PER_WARP)
            token_local_base = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + ni_base * fx.Int32(MFMA_N)
                + lane_mod_16
            )
            bi_base = _floordiv_kb(token_local_base)
            # The software pipeline intentionally asks for one chunk beyond the
            # epilogue. Clamp that speculative page-table lookup to the exact
            # inclusive table bounds so an unpadded table remains safe.
            bi_base = (bi_base < fx.Int32(0)).select(fx.Int32(0), bi_base)
            bi_base = (bi_base < fx.Int32(max_blocks_per_seq)).select(
                bi_base, fx.Int32(max_blocks_per_seq - 1)
            )
            phys_vec = bt_bt[batch_id * _stride_bt + bi_base]
            return _phys_to_list(phys_vec)

        def _prefetch_chunk(c_i32_arg, phys_list):
            assert N_TILES_PER_WARP == 4, "packed kvs assumes NTPW=4"
            assert N_PHYS == 1, "packed kvs assumes N_PHYS=1 (NTPW nts share one phys)"

            kv_list = []
            kvs_packed_list = []

            phys_shared = phys_list[0]
            kvs_base_off_elems = (
                phys_shared * _stride_kvs_block
                + lane_div_16 * kv_block_size
                + lane_mod_16 * fx.Int32(N_TILES_PER_WARP)
            ) >> fx.Int32(2)
            for k_tile in range_constexpr(k_tiles):
                kvs_packed = kvs_bt[
                    kvs_base_off_elems + fx.Int32(k_tile * _stride_kvs_ktile // 4)
                ]
                kvs_packed_list.append(kvs_packed)

            ni0 = warp_id * fx.Int32(N_TILES_PER_WARP)
            token_local0 = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + ni0 * fx.Int32(MFMA_N)
                + lane_mod_16
            )
            token_in_block0 = _mod_kb(token_local0)
            kv_base_off_elems = (
                phys_shared * _stride_kv_block
                + lane_div_16 * kv_block_size * _kv_chunk_bytes
                + token_in_block0 * _kv_chunk_bytes
            ) >> fx.Int32(2)
            for nt in range_constexpr(N_TILES_PER_WARP):
                for k_tile in range_constexpr(k_tiles):
                    kv_soffset = k_tile * _stride_kv_ktile + nt * _stride_kv_ntile
                    kv_c = _load_vec4_i32(
                        kv_bt, kv_base_off_elems + fx.Int32(kv_soffset // 4)
                    )
                    kv_list.append(kv_c)

            return kv_list, kvs_packed_list

        def _issue_nt_mfmas(kv_list_in, kvs_packed_per_kt, nt):
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            accs = [zero] * m_tiles
            # opsel_b=nt is baked into the atom; scale_b is the packed 4-nt word.
            atom = mfma_atoms[nt]
            for k_tile in range_constexpr(k_tiles):
                b_frag = fx.make_rmem_tensor(4, fx.Int32)
                b_frag.store(fx.Vector(kv_list_in[nt * k_tiles + k_tile]))
                kv_scale_packed = kvs_packed_per_kt[k_tile]
                for mi_idx in range_constexpr(m_tiles):
                    c_frag = fx.make_rmem_tensor(4, fx.Float32)
                    c_frag.store(fx.Vector(accs[mi_idx]))
                    fx.gemm(
                        atom,
                        c_frag,
                        q_a_ops[k_tile][mi_idx],
                        b_frag,
                        c_frag,
                        scale_a=q_scale_ops[k_tile][mi_idx],
                        scale_b=kv_scale_packed,
                    )
                    accs[mi_idx] = c_frag.load()
            return accs

        def _select_score_pool(
            pool_count,
            topk_state,
            topk_histogram,
            topk_scan,
            pool_values,
            pool_indices,
            selected_values,
            selected_indices,
        ):
            """Reduce ``pool[0:pool_count]`` to exact ``candidate_topk`` in LDS."""
            if tid == 0:
                topk_state[_STREAM_SCORE_PREFIX] = 0
                topk_state[_STREAM_SCORE_MASK] = 0
                topk_state[_STREAM_INDEX_PREFIX] = 0
                topk_state[_STREAM_INDEX_MASK] = 0
                topk_state[_STREAM_REMAINING] = fx.Int32(candidate_topk)
            gpu.barrier()

            # Composite radix key: score ordinal descending, then raw index
            # ascending. This makes NaN and signed-zero behavior explicit and
            # does not rely on the order candidates happen to occupy in LDS.
            for key_byte in range_constexpr(NUM_KEY_BYTES):
                topk_histogram[tid] = 0
                gpu.barrier()

                score_prefix = topk_state[_STREAM_SCORE_PREFIX]
                score_mask = topk_state[_STREAM_SCORE_MASK]
                index_prefix = topk_state[_STREAM_INDEX_PREFIX]
                index_mask = topk_state[_STREAM_INDEX_MASK]
                if const_expr(key_byte < 4):
                    byte_position = key_byte
                    xor_value = RADIX_SIGN_BIT if key_byte == 0 else 0
                else:
                    byte_position = key_byte - 4
                    xor_value = 0

                for step in range_constexpr(pool_steps):
                    pool_pos = fx.Int32(step * TOPK_BLOCK_THREADS) + tid
                    live = pool_pos < pool_count
                    safe_pos = live.select(pool_pos, fx.Int32(0))
                    value = pool_values[safe_pos]
                    score_ord = f32_to_ordered_i32(value)
                    raw_index = pool_indices[safe_pos]
                    prefix_match = live & key_matches_prefix(
                        score_ord,
                        raw_index,
                        score_prefix,
                        score_mask,
                        index_prefix,
                        index_mask,
                    )
                    if prefix_match:
                        key = score_ord if const_expr(key_byte < 4) else raw_index
                        bucket = radix_byte(key, byte_position) ^ fx.Int32(xor_value)
                        atomic_add_i32(topk_histogram, 1, bucket, "workgroup")
                gpu.barrier()

                selected_bucket = (
                    fx.Int32(NUM_HIST_BINS - 1) - tid
                    if const_expr(key_byte < 4)
                    else tid
                )
                bucket_count = topk_histogram[selected_bucket]
                before, _ = block_exclusive_prefix_i32(
                    tid,
                    bucket_count,
                    topk_scan,
                )
                remaining = topk_state[_STREAM_REMAINING]
                gpu.barrier()
                if (before < remaining) & (before + bucket_count >= remaining):
                    actual_bucket = selected_bucket ^ fx.Int32(xor_value)
                    byte_mask, shift = prefix_byte_mask(byte_position)
                    if const_expr(key_byte < 4):
                        topk_state[_STREAM_SCORE_PREFIX] = score_prefix | (
                            actual_bucket << fx.Int32(shift)
                        )
                        topk_state[_STREAM_SCORE_MASK] = score_mask | byte_mask
                    else:
                        topk_state[_STREAM_INDEX_PREFIX] = index_prefix | (
                            actual_bucket << fx.Int32(shift)
                        )
                        topk_state[_STREAM_INDEX_MASK] = index_mask | byte_mask
                    topk_state[_STREAM_REMAINING] = remaining - before
                gpu.barrier()

            threshold_score = topk_state[_STREAM_SCORE_PREFIX]
            threshold_index = topk_state[_STREAM_INDEX_PREFIX]
            threshold_equal_needed = topk_state[_STREAM_REMAINING]
            if tid == 0:
                topk_state[_STREAM_WRITE_CURSOR] = 0
                topk_state[_STREAM_EQUAL_SEEN] = 0
            gpu.barrier()

            for step in range_constexpr(pool_steps):
                pool_pos = fx.Int32(step * TOPK_BLOCK_THREADS) + tid
                live = pool_pos < pool_count
                safe_pos = live.select(pool_pos, fx.Int32(0))
                value = pool_values[safe_pos]
                score_ord = f32_to_ordered_i32(value)
                raw_index = pool_indices[safe_pos]
                better = live & key_is_better(
                    score_ord,
                    raw_index,
                    threshold_score,
                    threshold_index,
                )
                equal = (
                    live
                    & (score_ord == threshold_score)
                    & (raw_index == threshold_index)
                )
                better_i32 = better.select(fx.Int32(1), fx.Int32(0))
                equal_i32 = equal.select(fx.Int32(1), fx.Int32(0))
                packed = (better_i32 << fx.Int32(_PACK_SHIFT)) + equal_i32
                packed_before, packed_total = block_exclusive_prefix_i32(
                    tid,
                    packed,
                    topk_scan,
                )
                better_before = packed_before >> fx.Int32(_PACK_SHIFT)
                equal_before = packed_before & fx.Int32(_PACK_MASK)
                better_total = packed_total >> fx.Int32(_PACK_SHIFT)
                equal_total = packed_total & fx.Int32(_PACK_MASK)

                equal_seen = topk_state[_STREAM_EQUAL_SEEN]
                room = threshold_equal_needed - equal_seen
                room = (room < 0).select(fx.Int32(0), room)
                admit_equal = equal & (equal_before < room)
                keep = better | admit_equal
                admitted_equal_before = (equal_before < room).select(
                    equal_before,
                    room,
                )
                destination = (
                    topk_state[_STREAM_WRITE_CURSOR]
                    + better_before
                    + admitted_equal_before
                )
                if keep:
                    selected_values[destination] = value
                    selected_indices[destination] = raw_index
                gpu.barrier()
                if tid == 0:
                    admitted_equal_total = (equal_total < room).select(
                        equal_total,
                        room,
                    )
                    topk_state[_STREAM_WRITE_CURSOR] = (
                        topk_state[_STREAM_WRITE_CURSOR]
                        + better_total
                        + admitted_equal_total
                    )
                    topk_state[_STREAM_EQUAL_SEEN] = (
                        topk_state[_STREAM_EQUAL_SEEN] + equal_total
                    )
                gpu.barrier()

            for step in range_constexpr(output_steps):
                out_pos = fx.Int32(step * TOPK_BLOCK_THREADS) + tid
                if out_pos < fx.Int32(candidate_topk):
                    pool_values[out_pos] = selected_values[out_pos]
                    pool_indices[out_pos] = selected_indices[out_pos]
            gpu.barrier()
            if tid == 0:
                topk_state[_STREAM_RETAINED] = fx.Int32(candidate_topk)
                topk_state[_STREAM_PENDING] = 0
            gpu.barrier()

        def _commit_score_chunk(
            c_i32_arg,
            topk_state,
            topk_histogram,
            topk_scan,
            pool_values,
            pool_indices,
            selected_values,
            selected_indices,
        ):
            """Publish one score chunk and periodically reduce the bounded pool."""
            gpu.barrier()
            chunk_first = (chunk_start + c_i32_arg) * fx.Int32(block_k)
            live_first = (chunk_first > local_start).select(
                chunk_first,
                local_start,
            )
            chunk_last = chunk_first + fx.Int32(block_k)
            live_last = (chunk_last < local_end).select(chunk_last, local_end)
            live_count = live_last - live_first
            live_count = (live_count < 0).select(fx.Int32(0), live_count)
            if tid == 0:
                topk_state[_STREAM_PENDING] = topk_state[_STREAM_PENDING] + live_count
            gpu.barrier()

            batch_boundary = (
                (c_i32_arg + fx.Int32(1)) % fx.Int32(score_batch_chunks)
            ) == 0
            final_chunk = c_i32_arg == chunk_count - fx.Int32(1)
            if batch_boundary | final_chunk:
                pool_count = topk_state[_STREAM_RETAINED] + topk_state[_STREAM_PENDING]
                if pool_count > fx.Int32(candidate_topk):
                    _select_score_pool(
                        pool_count,
                        topk_state,
                        topk_histogram,
                        topk_scan,
                        pool_values,
                        pool_indices,
                        selected_values,
                        selected_indices,
                    )
                else:
                    if tid == 0:
                        topk_state[_STREAM_RETAINED] = pool_count
                        topk_state[_STREAM_PENDING] = 0
                    gpu.barrier()

        def _finish_score_candidates(
            topk_state,
            pool_values,
            pool_indices,
            candidate_values_row,
            candidate_indices_row,
            candidate_counts_ptr,
        ):
            retained = topk_state[_STREAM_RETAINED]
            for step in range_constexpr(output_steps):
                out_pos = fx.Int32(step * TOPK_BLOCK_THREADS) + tid
                if out_pos < fx.Int32(candidate_topk):
                    live = out_pos < retained
                    safe_pos = live.select(out_pos, fx.Int32(0))
                    candidate_values_row[out_pos] = live.select(
                        pool_values[safe_pos],
                        fx.Float32(float("-inf")),
                    )
                    candidate_indices_row[out_pos] = live.select(
                        pool_indices[safe_pos],
                        fx.Int32(-1),
                    )
            if tid == 0:
                candidate_counts_ptr[pid] = retained

        def _post_process_nt(accs, nt, c_i32_arg):
            """relu + weighted reduction + score callback."""
            zero = fx.Vector.filled(4, 0.0, fx.Float32)
            ni_warp = warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt)
            token_base = (chunk_start + c_i32_arg) * fx.Int32(
                block_k
            ) + ni_warp * fx.Int32(MFMA_N)

            thread_sum = ZERO_F
            for mi_idx in range_constexpr(m_tiles):
                relu_v = fx.Vector(accs[mi_idx]).maximumf(zero)
                w_v = fx.Vector(w_per_lane[mi_idx])
                for elem in [0, 1, 2, 3]:
                    thread_sum = fx.fma(relu_v[elem], w_v[elem], thread_sum)

            lane_i32 = fx.Int32(lane_id)

            def _bperm_xor_add(val, sh):
                peer_byte = (lane_i32 ^ fx.Int32(sh)) * fx.Int32(4)
                peer_i32 = fx.Int32(
                    rocdl.ds_bpermute(T.i32, peer_byte, val.bitcast(fx.Int32))
                )
                return val + peer_i32.bitcast(fx.Float32)

            thread_sum = _bperm_xor_add(thread_sum, 16)
            thread_sum = _bperm_xor_add(thread_sum, 32)
            # `weight_scale` already folded into `w_per_lane` (hoisted, once/wave).

            # Only [local_start, local_end) is written (one writer lane per token);
            # the rest stays at the caller's -inf pre-fill. output_token_base is
            # normally zero. The bounded score+top-k operator sets it to its
            # aligned tile start, allowing this same score kernel to target a
            # fixed [rows, tile_tokens] buffer instead of context-sized logits.
            # Sparse 1-writer scatter: guard the plain store instead of a V# OOB
            # sentinel. Row base is folded into out_base.
            is_writer = lane_div_16 < fx.Int32(1)
            out_token = token_base + lane_mod_16
            in_window = (out_token >= local_start) & (out_token < local_end)
            should_write = is_writer & in_window
            if const_expr(emit_candidates):
                chunk_first = (chunk_start + c_i32_arg) * fx.Int32(block_k)
                live_first = (chunk_first > local_start).select(
                    chunk_first,
                    local_start,
                )
                destination = (
                    topk_state[_STREAM_RETAINED]
                    + topk_state[_STREAM_PENDING]
                    + out_token
                    - live_first
                )

                def _write_candidate(
                    _values=pool_values,
                    _indices=pool_indices,
                    _dst=destination,
                    _value=thread_sum,
                    _index=out_token,
                ):
                    _values[_dst] = _value
                    _indices[_dst] = _index

                @flyc.jit
                def _guarded_candidate_write(
                    _pred=should_write,
                    _write=_write_candidate,
                ):
                    if _pred:
                        _write()

                _guarded_candidate_write()
            else:
                if should_write:
                    fx.ptr_store(
                        thread_sum,
                        fx.add_offset(out_base, out_token - output_token_base),
                    )

        def _compute_chunk(kv_list_in, kvs_packed_list_in, c_i32_arg, nt0_accs_in=None):
            assert (
                N_TILES_PER_WARP == 4
            ), "pipelined-nt structure currently hardcoded for NTPW=4"

            accs_nt0 = (
                _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 0)
                if nt0_accs_in is None
                else list(nt0_accs_in)
            )

            accs_nt1 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 1)
            _post_process_nt(accs_nt0, 0, c_i32_arg)

            accs_nt2 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 2)
            _post_process_nt(accs_nt1, 1, c_i32_arg)

            accs_nt3 = _issue_nt_mfmas(kv_list_in, kvs_packed_list_in, 3)
            _post_process_nt(accs_nt2, 2, c_i32_arg)

            _post_process_nt(accs_nt3, 3, c_i32_arg)

        # A bare ``if inactive: return`` is not an early exit under FlyDSL's
        # kernel rewriter. Keep the complete page-dependent pipeline in a
        # closure invoked by a block-uniform dynamic guard instead. This makes
        # padded schedule slots and empty windows perform no block-table or KV
        # access, even when every page-table entry is -1.
        def _process_chunks():
            # === Prologue ===
            N_KV = k_tiles * N_TILES_PER_WARP
            last_c_i32 = chunk_count - fx.Int32(1)

            phys_pre = _load_phys(c0_i32)
            kv_pre, kvs_pre = _prefetch_chunk(c0_i32, phys_pre)
            phys_next_pre = _load_phys(fx.Int32(1))

            nt0_accs_init = _issue_nt_mfmas(list(kv_pre), list(kvs_pre), 0)
            nt0_init_scalars = []
            for v in nt0_accs_init:
                vv = fx.Vector(v)
                for i in range(4):
                    nt0_init_scalars.append(vv[i])

            # === Main loop: chunk_count - 1 iterations ===
            N_KVS = k_tiles
            chunk_count_minus_1_i32 = chunk_count - fx.Int32(1)
            chunk_count_minus_1_idx = fx.Int64(chunk_count_minus_1_i32)
            init_args = (
                list(kv_pre) + list(kvs_pre) + list(phys_next_pre) + nt0_init_scalars
            )
            for c_idx, state in range(0, chunk_count_minus_1_idx, 1, init=init_args):
                kv_cur_list = [state[i] for i in range(N_KV)]
                kvs_cur_list = [state[N_KV + i] for i in range(N_KVS)]
                phys_next_list = [
                    state[N_KV + N_KVS + i] for i in range(N_TILES_PER_WARP)
                ]
                nt0_acc_base = N_KV + N_KVS + N_TILES_PER_WARP
                nt0_accs_cur = [
                    fx.Vector.from_elements(
                        [state[nt0_acc_base + mi * 4 + i] for i in range(4)],
                        dtype=fx.Float32,
                    )
                    for mi in range(m_tiles)
                ]
                c_idx_i32 = fx.Int32(c_idx)
                c_next_i32 = c_idx_i32 + fx.Int32(1)
                c_next_next_i32 = c_next_i32 + fx.Int32(1)

                _compute_chunk(
                    kv_cur_list,
                    kvs_cur_list,
                    c_idx_i32,
                    nt0_accs_in=nt0_accs_cur,
                )
                if const_expr(emit_candidates):
                    _commit_score_chunk(
                        c_idx_i32,
                        topk_state,
                        topk_histogram,
                        topk_scan,
                        pool_values,
                        pool_indices,
                        selected_values,
                        selected_indices,
                    )

                kv_next, kvs_next = _prefetch_chunk(c_next_i32, phys_next_list)

                phys_next_next_list = _load_phys(c_next_next_i32)

                nt0_accs_next = _issue_nt_mfmas(list(kv_next), list(kvs_next), 0)
                nt0_next_scalars = []
                for v in nt0_accs_next:
                    vv = fx.Vector(v)
                    for i in range(4):
                        nt0_next_scalars.append(vv[i])

                results = yield (
                    list(kv_next)
                    + list(kvs_next)
                    + list(phys_next_next_list)
                    + nt0_next_scalars
                )

            # === Epilogue: process last chunk (chunk_count - 1) ===
            kv_last_list = [results[i] for i in range(N_KV)]
            kvs_last_list = [results[N_KV + i] for i in range(N_KVS)]
            nt0_acc_base = N_KV + N_KVS + N_TILES_PER_WARP
            nt0_accs_last = [
                fx.Vector.from_elements(
                    [results[nt0_acc_base + mi * 4 + i] for i in range(4)],
                    dtype=fx.Float32,
                )
                for mi in range(m_tiles)
            ]
            _compute_chunk(
                kv_last_list,
                kvs_last_list,
                last_c_i32,
                nt0_accs_in=nt0_accs_last,
            )
            if const_expr(emit_candidates):
                _commit_score_chunk(
                    last_c_i32,
                    topk_state,
                    topk_histogram,
                    topk_scan,
                    pool_values,
                    pool_indices,
                    selected_values,
                    selected_indices,
                )
                _finish_score_candidates(
                    topk_state,
                    pool_values,
                    pool_indices,
                    candidate_values_row,
                    candidate_indices_row,
                    candidate_counts_ptr,
                )

        if chunk_count > fx.Int32(0):
            _process_chunks()

    return pa_mqa_logits_fp4_prefill_kernel, block_threads_k


# ============================================================================
# Cached compile + public host API
# ============================================================================


@lru_cache(maxsize=32)
def compile_pa_mqa_logits_fp4_prefill(
    *,
    block_k: int = 256,
    kv_block_size: int = 64,
    max_blocks_per_seq: int = 256,
    num_warps: int = DEFAULT_NUM_WARPS,
    heads: int = DEFAULT_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
):
    kfn, block_threads = build_pa_mqa_logits_fp4_prefill_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
    )

    @flyc.jit
    def launch_pa_mqa_logits_fp4_prefill(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info_,
        stride_out: fx.Int32,
        output_token_base: fx.Int32,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        gxi = fx.Int64(gx)
        kfn(
            out,
            out,
            out,
            out,
            q,
            qs,
            kv,
            kvs,
            bt,
            w,
            cta_info_,
            stride_out,
            output_token_base,
            weight_scale,
        ).launch(grid=(gxi,), block=(block_threads, 1, 1), stream=stream)

    return launch_pa_mqa_logits_fp4_prefill, block_threads


@cache
def compile_pa_mqa_logits_fp4_prefill_topk(
    *,
    topk: int,
    block_k: int = 256,
    kv_block_size: int = 64,
    max_blocks_per_seq: int = 256,
    num_warps: int = DEFAULT_NUM_WARPS,
    heads: int = DEFAULT_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    score_batch_chunks: int = DEFAULT_SCORE_BATCH_CHUNKS,
):
    """Compile the score callback that emits bounded per-CTA TopK planes."""

    kfn, block_threads = build_pa_mqa_logits_fp4_prefill_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
        candidate_topk=topk,
        score_batch_chunks=score_batch_chunks,
    )

    @flyc.jit
    def launch_pa_mqa_logits_fp4_prefill_topk(
        candidate_values,
        candidate_indices,
        candidate_counts,
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info_,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        gxi = fx.Int64(gx)
        kfn(
            candidate_values,
            candidate_values,
            candidate_indices,
            candidate_counts,
            q,
            qs,
            kv,
            kvs,
            bt,
            w,
            cta_info_,
            fx.Int32(0),
            fx.Int32(0),
            weight_scale,
        ).launch(
            grid=(gxi,),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_pa_mqa_logits_fp4_prefill_topk, block_threads


def flydsl_pa_mqa_logits_fp4_prefill(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    *,
    weight_scale: float = 1.0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int = 512,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    n_ctas: int | None = None,
    output_token_base: int = 0,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Ragged-prefill FP4 paged MQA logits (gfx950).

    ``output_token_base`` is an internal tiling hook: logical token ``j`` is
    stored at output column ``j - output_token_base``. Public full-logit calls
    leave it at zero.
    """
    total_tokens, heads, head_dim_packed = q_fp4.shape
    head_dim = head_dim_packed * 2
    max_blocks_per_seq = block_tables.shape[1]

    if (cta_info is None) != (n_ctas is None):
        raise ValueError("Pass both cta_info and n_ctas, or neither.")
    schedule_internal = cta_info is None
    if schedule_internal:
        _, cta_info, n_ctas = compute_prefill_schedule(
            row_to_batch,
            local_starts,
            local_ends,
            block_k,
            parallel_unit_num,
            max_seq_len,
        )

    if out is None:
        out = torch.full(
            (total_tokens, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    elif schedule_internal:
        out.fill_(float("-inf"))

    launcher, _ = compile_pa_mqa_logits_fp4_prefill(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
    )

    if stream is None:
        stream = torch.cuda.current_stream()

    launcher(
        out,
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        cta_info,
        out.stride(0),
        output_token_base,
        float(weight_scale),
        n_ctas,
        stream,
    )
    return out


def flydsl_pa_mqa_topk_fp4_prefill(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    max_seq_len: int,
    *,
    topk: int = 512,
    weight_scale: float = 1.0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int | None = None,
    score_batch_chunks: int = DEFAULT_SCORE_BATCH_CHUNKS,
    workspace: FP4PrefillTopKWorkspace | None = None,
    out: FP4PrefillTopKResult | None = None,
    stream: torch.cuda.Stream | None = None,
) -> FP4PrefillTopKResult:
    """True kernel-fused FP4 paged-MQA scoring and exact TopK for gfx950.

    The MFMA kernel never materializes ``[rows, max_seq_len]`` logits. Each CTA
    streams up to ``score_batch_chunks * block_k`` new scores through LDS,
    repeatedly retaining its exact ``topk`` under score-descending/index-
    ascending order. When the launch has one slot per row, a no-radix copier
    forwards that CTA's exact set using the schedule offsets and candidate
    count. Other geometries retain the exact split-candidate radix merge. A
    bounded finalizer returns long rows in total order and maps logical sequence
    indices through ``block_tables``. Rows no longer than ``topk`` preserve
    sequential logical-index order.

    ``workspace`` exposes the per-CTA values, raw indices, and counts for
    callers that need to exchange or inspect local candidates. Reuse both the
    workspace and output buffers to avoid allocations. HIP/CUDA graph capture
    is intentionally rejected: the row-plan scratch and first-call compiles
    have not been made capture-owned/preflighted.
    """

    if topk not in (512, 1024):
        raise ValueError("topk must be 512 or 1024")
    if block_k != 256 or num_warps != 4:
        raise ValueError("the fused TopK path requires block_k=256 and num_warps=4")
    if kv_block_size != 64:
        raise ValueError("the fused TopK path requires kv_block_size=64")
    if score_batch_chunks not in SUPPORTED_SCORE_BATCH_CHUNKS:
        raise ValueError(
            "score_batch_chunks must be one of "
            f"{SUPPORTED_SCORE_BATCH_CHUNKS}"
        )

    if q_fp4.ndim != 3:
        raise ValueError("q_fp4 must have shape [rows, 64, 64]")
    rows, heads, head_dim_packed = q_fp4.shape
    if rows <= 0:
        raise ValueError("the fused TopK path requires at least one query row")
    if parallel_unit_num is None:
        parallel_unit_num = max(512, rows)
    head_dim = head_dim_packed * 2
    if heads != 64 or head_dim != 128:
        raise ValueError("the fused TopK path requires production H=64 and D=128")
    if q_fp4.dtype != torch.uint8 or q_scale.dtype != torch.uint8:
        raise ValueError("q_fp4 and q_scale must use the packed uint8 FP4 ABI")
    if q_scale.shape != (rows, 1, 4, 16, 4):
        raise ValueError("q_scale must have shape [rows, 1, 4, 16, 4]")
    if weights.shape != (rows, 64) or weights.dtype != torch.bfloat16:
        raise ValueError("weights must be contiguous bfloat16 [rows, 64]")
    if kv_cache.dtype != torch.uint8 or tuple(kv_cache.shape[1:]) != (
        1,
        4,
        64,
        16,
    ):
        raise ValueError("kv_cache must be uint8 [blocks, 1, 4, 64, 16]")
    if kv_scale.dtype != torch.uint8 or tuple(kv_scale.shape[1:]) != (
        1,
        4,
        64,
    ):
        raise ValueError("kv_scale must be uint8 [blocks, 1, 4, 64]")
    device = q_fp4.device
    if device.type != "cuda":
        raise ValueError("q_fp4 must be a CUDA tensor")
    arch = str(torch.cuda.get_device_properties(device).gcnArchName).split(":")[0]
    if arch != "gfx950":
        raise ValueError(f"the fused TopK path requires gfx950, got {arch}")
    metadata = (row_to_batch, local_starts, local_ends)
    if any(t.dtype != torch.int32 or t.shape != (rows,) for t in metadata):
        raise ValueError("row_to_batch/local_starts/local_ends must be int32 [rows]")
    if block_tables.dtype != torch.int32 or block_tables.ndim != 2:
        raise ValueError("block_tables must be a 2D int32 tensor")
    if max_seq_len < 0:
        raise ValueError("max_seq_len must be non-negative")
    if max_seq_len > block_tables.shape[1] * kv_block_size:
        raise ValueError("max_seq_len exceeds block_tables capacity")
    inputs = (
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        *metadata,
    )
    if any(t.device != device for t in inputs):
        raise ValueError("all fused FP4 TopK inputs must be on one device")
    if any(not t.is_contiguous() for t in inputs):
        raise ValueError("all fused FP4 TopK inputs must be contiguous")

    if stream is None:
        stream = torch.cuda.current_stream(device)
    else:
        try:
            stream_device = torch.device(stream.device)
        except (AttributeError, TypeError) as exc:
            raise TypeError("stream must be a torch.cuda.Stream") from exc
        if stream_device != device:
            raise ValueError("stream must belong to the FP4 input device")

    # Resolve the launch stream before any allocation. Besides providing the
    # expected allocation-stream affinity for internally owned scratch, making
    # it current here lets capture detection cover an explicitly supplied
    # non-current stream.
    with torch.cuda.stream(stream):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "fused FP4 prefill TopK does not support HIP/CUDA graph capture"
            )
        if workspace is None:
            workspace = allocate_fp4_prefill_topk_workspace(
                rows,
                parallel_unit_num,
                topk,
                device,
            )
    expected_workspace = (
        ((parallel_unit_num, CTA_INFO_WIDTH), torch.int32, workspace.cta_info),
        ((rows + 1,), torch.int32, workspace.row_offsets),
        (
            (parallel_unit_num, topk),
            torch.float32,
            workspace.candidate_values,
        ),
        (
            (parallel_unit_num, topk),
            torch.int32,
            workspace.candidate_indices,
        ),
        ((parallel_unit_num,), torch.int32, workspace.candidate_counts),
        ((rows, topk), torch.float32, workspace.merge_values),
        ((rows, topk), torch.int32, workspace.merge_indices),
        ((rows, topk), torch.int32, workspace.merge_physical_indices),
    )
    for expected_shape, expected_dtype, tensor in expected_workspace:
        if (
            tuple(tensor.shape) != expected_shape
            or tensor.dtype != expected_dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "workspace tensor mismatch: expected "
                f"{expected_shape} {expected_dtype} contiguous on {device}"
            )

    with torch.cuda.stream(stream):
        if out is None:
            out = FP4PrefillTopKResult(
                values=torch.empty(rows, topk, dtype=torch.float32, device=device),
                raw_indices=torch.empty(rows, topk, dtype=torch.int32, device=device),
                physical_indices=torch.empty(
                    rows,
                    topk,
                    dtype=torch.int32,
                    device=device,
                ),
                counts=torch.empty(rows, dtype=torch.int32, device=device),
            )
    expected_outputs = (
        ((rows, topk), torch.float32, out.values),
        ((rows, topk), torch.int32, out.raw_indices),
        ((rows, topk), torch.int32, out.physical_indices),
        ((rows,), torch.int32, out.counts),
    )
    for expected_shape, expected_dtype, tensor in expected_outputs:
        if (
            tuple(tensor.shape) != expected_shape
            or tensor.dtype != expected_dtype
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                "output tensor mismatch: expected "
                f"{expected_shape} {expected_dtype} contiguous on {device}"
            )

    if max_seq_len == 0:
        with torch.cuda.stream(stream):
            workspace.row_offsets.zero_()
            workspace.candidate_counts.zero_()
            out.values.fill_(float("-inf"))
            out.raw_indices.fill_(-1)
            out.physical_indices.fill_(-1)
            out.counts.zero_()
        return out

    single_cta_per_row = parallel_unit_num == rows
    with torch.cuda.stream(stream):
        compute_prefill_schedule(
            row_to_batch,
            local_starts,
            local_ends,
            block_k,
            parallel_unit_num,
            max_seq_len,
            cta_info_out=workspace.cta_info,
            row_offsets_out=workspace.row_offsets,
            single_cta_per_row=single_cta_per_row,
        )
    max_blocks_per_seq = block_tables.shape[1]
    launcher, _ = compile_pa_mqa_logits_fp4_prefill_topk(
        topk=topk,
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        heads=heads,
        head_dim=head_dim,
        score_batch_chunks=score_batch_chunks,
    )
    launcher(
        workspace.candidate_values,
        workspace.candidate_indices,
        workspace.candidate_counts,
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        workspace.cta_info,
        float(weight_scale),
        parallel_unit_num,
        stream,
    )

    if single_cta_per_row:
        with torch.cuda.stream(stream):
            _copy_single_cta_topk_candidates(
                workspace.candidate_values,
                workspace.candidate_indices,
                workspace.candidate_counts,
                workspace.row_offsets,
                workspace.merge_values,
                workspace.merge_indices,
                out.counts,
            )
    else:
        from ...candidate_topk_merge import flydsl_candidate_topk_merge

        flydsl_candidate_topk_merge(
            workspace.candidate_values,
            workspace.candidate_indices,
            workspace.candidate_counts,
            workspace.row_offsets,
            row_to_batch,
            block_tables,
            workspace.merge_values,
            workspace.merge_indices,
            workspace.merge_physical_indices,
            out.counts,
            kv_block_size,
            stream=stream,
        )
    from ...mqa_topk_finalize import order_and_map_mqa_topk

    with torch.cuda.stream(stream):
        order_and_map_mqa_topk(
            workspace.merge_values,
            workspace.merge_indices,
            out.counts,
            local_starts,
            local_ends,
            row_to_batch,
            block_tables,
            out.values,
            out.raw_indices,
            out.physical_indices,
            max_seq_len,
            topk,
            kv_block_size,
        )
    return out


# Keep the logits-prefixed spelling as a compatibility alias for the original
# prototype branch.
flydsl_pa_mqa_logits_fp4_prefill_topk = flydsl_pa_mqa_topk_fp4_prefill


@triton.jit
def _varqlen_windows_kernel(
    cu_ptr,  # [B+1] int32, prefix-sum of per-batch qlen
    ctx_ptr,  # [B] int32, per-batch KV length
    row_to_batch_ptr,  # [total_q] int32 (out)
    local_starts_ptr,  # [total_q] int32 (out)
    local_ends_ptr,  # [total_q] int32 (out)
    total_q,
    B,
    BLOCK: tl.constexpr,
):
    """Fused build of ragged-row metadata for per-batch variable qlen (MTP)."""
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    rmask = r < total_q

    lo = tl.zeros([BLOCK], tl.int32)
    hi = tl.full([BLOCK], B, tl.int32)
    for _ in tl.static_range(32):
        mid = (lo + hi) // 2
        cu_mid = tl.load(
            cu_ptr + 1 + tl.minimum(mid, B - 1), mask=(mid < B), other=2147483647
        )
        go_right = cu_mid <= r
        lo = tl.where(go_right, mid + 1, lo)
        hi = tl.where(go_right, hi, mid)
    b = tl.minimum(lo, B - 1)

    cu_b = tl.load(cu_ptr + b, mask=rmask, other=0)
    cu_b1 = tl.load(cu_ptr + b + 1, mask=rmask, other=0)
    ctx_b = tl.load(ctx_ptr + b, mask=rmask, other=0)
    n = r - cu_b
    qlen = cu_b1 - cu_b
    le = tl.maximum(ctx_b - qlen + n + 1, 0)
    # Rows beyond the real total Σ (cu[B]) are FLAT tail-padding — force an empty
    # window so the mqa kernel / top_k skip them (used when `total_q` is the padded
    # count, e.g. the CUDAGraph decode path scores all padded rows in one shot).
    real_total = tl.load(cu_ptr + B)
    le = tl.where(r < real_total, le, tl.zeros([BLOCK], tl.int32))

    tl.store(row_to_batch_ptr + r, b, mask=rmask)
    tl.store(local_starts_ptr + r, tl.zeros([BLOCK], tl.int32), mask=rmask)
    tl.store(local_ends_ptr + r, le, mask=rmask)


def compute_varqlen_windows(cu_seq_q, context_lens, total_q, *, out=None):
    """Build ragged-row metadata for per-batch variable query length (MTP).

    Pass `out=(row_to_batch, local_starts, local_ends)` (fixed int32 buffers each
    >= total_q long) to write into stable addresses — the CUDAGraph decode path
    scores all padded rows, so top_k replays from these window pointers while
    `build()` refreshes their contents. Rows past the real total (cu[B]) get an
    empty window (local_ends == 0) so they are skipped.
    """
    dev = cu_seq_q.device
    cu = cu_seq_q.to(torch.int32).contiguous()
    ctx = context_lens.to(torch.int32).contiguous()
    B = ctx.shape[0]
    if out is None:
        row_to_batch = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_starts = torch.empty(total_q, dtype=torch.int32, device=dev)
        local_ends = torch.empty(total_q, dtype=torch.int32, device=dev)
    else:
        row_to_batch, local_starts, local_ends = out
    if total_q > 0:
        BLOCK = 256
        grid = (triton.cdiv(total_q, BLOCK),)
        _varqlen_windows_kernel[grid](
            cu,
            ctx,
            row_to_batch,
            local_starts,
            local_ends,
            total_q,
            B,
            BLOCK=BLOCK,
        )
    return row_to_batch, local_starts, local_ends


def flydsl_pa_mqa_logits_fp4_varqlen(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    max_seq_len: int,
    *,
    cu_seq_q: torch.Tensor | None = None,
    context_lens: torch.Tensor | None = None,
    windows: tuple | None = None,
    weight_scale: float = 1.0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int | None = None,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    n_ctas: int | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Variable-qlen (per-batch MTP) FP4 paged MQA logits (gfx950)."""
    total_q = q_fp4.shape[0]
    if windows is None:
        if cu_seq_q is None or context_lens is None:
            raise ValueError(
                "flydsl_pa_mqa_logits_fp4_varqlen: pass windows=(row_to_batch, "
                "local_starts, local_ends) built once via "
                "compute_varqlen_windows, or both cu_seq_q and context_lens "
                "to build them here."
            )
        windows = compute_varqlen_windows(cu_seq_q, context_lens, total_q)
    row_to_batch, local_starts, local_ends = windows
    if parallel_unit_num is None:
        chunks_per_seq = max(1, (max_seq_len + block_k - 1) // block_k)
        parallel_unit_num = total_q * chunks_per_seq
    return flydsl_pa_mqa_logits_fp4_prefill(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_to_batch,
        local_starts,
        local_ends,
        max_seq_len,
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
        num_warps=num_warps,
        parallel_unit_num=parallel_unit_num,
        out=out,
        cta_info=cta_info,
        n_ctas=n_ctas,
        stream=stream,
    )
