# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: PLR0124

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_seed_calibrate_repr = make_kernel_repr(
    "_pa_mqa_litetopk_seed_calibrate_kernel",
    ["SAMPLE_LEN", "BLOCK"],
)
_seed_histogram_repr = make_kernel_repr(
    "_pa_mqa_litetopk_seed_histogram_kernel",
    ["SAMPLE_LEN", "NUM_BUCKETS", "TOPK", "BLOCK"],
)
_seed_emit_repr = make_kernel_repr(
    "_pa_mqa_litetopk_seed_emit_kernel",
    ["SAMPLE_LEN", "NUM_BUCKETS", "TOPK", "BLOCK"],
)
_scan_schedule_repr = make_kernel_repr(
    "_pa_mqa_litetopk_scan_schedule_kernel",
    [
        "VALIDATE_ROWS",
        "RESET_STATE",
        "DUAL_SCHEDULE",
        "SAMPLE_LEN",
        "BLOCK_K",
        "BLOCK",
    ],
)
_refresh_threshold_repr = make_kernel_repr(
    "_pa_mqa_litetopk_refresh_threshold_kernel", ["NUM_BUCKETS", "TOPK"]
)


@triton.jit
def _finite_f32(value):
    return (value == value) & (value != float("inf")) & (value != -float("inf"))


@triton.jit
def _bucket_f32(value, origin, inv_delta, NUM_BUCKETS: tl.constexpr):
    affine = (-value - origin) * inv_delta
    affine = tl.where(_finite_f32(affine), affine, float(NUM_BUCKETS - 1))
    affine = tl.maximum(0.0, tl.minimum(affine, float(NUM_BUCKETS - 1)))
    return affine.to(tl.int32)


@triton.jit(repr=_seed_calibrate_repr, do_not_specialize=["max_seq_len"])
def _pa_mqa_litetopk_seed_calibrate_kernel(
    sample_logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    origin_ptr,
    inv_delta_ptr,
    status_ptr,
    sample_stride,
    max_seq_len,
    STATUS_NONFINITE: tl.constexpr,
    STATUS_INVALID_INDEX: tl.constexpr,
    SAMPLE_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    invalid_window = (row_start < 0) | (row_end < row_start) | (row_end > max_seq_len)
    row_len = tl.maximum(row_end - row_start, 0)
    sample_count = tl.minimum(row_len, SAMPLE_LEN)

    row_max = -float("inf")
    row_min = float("inf")
    finite_count = 0
    nonfinite_count = 0
    offsets = tl.arange(0, BLOCK)
    for base in tl.static_range(0, SAMPLE_LEN, BLOCK):
        columns = base + offsets
        genuine = columns < sample_count
        values = tl.load(
            sample_logits_ptr + row * sample_stride + columns,
            mask=genuine,
            other=0.0,
        ).to(tl.float32)
        finite = genuine & _finite_f32(values)
        row_max = tl.maximum(
            row_max,
            tl.max(tl.where(finite, values, -float("inf")), axis=0),
        )
        row_min = tl.minimum(
            row_min,
            tl.min(tl.where(finite, values, float("inf")), axis=0),
        )
        finite_count += tl.sum(finite.to(tl.int32), axis=0)
        nonfinite_count += tl.sum((genuine & ~finite).to(tl.int32), axis=0)

    has_finite = finite_count > 0
    safe_max = tl.where(has_finite, row_max, 0.0)
    safe_min = tl.where(has_finite, row_min, 0.0)
    origin = -safe_max
    magnitude = tl.maximum(tl.abs(safe_max), tl.abs(safe_min))
    span = tl.maximum(safe_max - safe_min, magnitude * (1.0 / 256.0))
    span = tl.maximum(span, 1.0e-6)
    inv_delta = 255.0 / span

    bad_affine = ~_finite_f32(origin) | ~_finite_f32(inv_delta)
    status = tl.load(status_ptr + row)
    status |= tl.where(invalid_window, STATUS_INVALID_INDEX, 0)
    status |= tl.where(
        (nonfinite_count > 0) | ((sample_count > 0) & ~has_finite) | bad_affine,
        STATUS_NONFINITE,
        0,
    )
    tl.store(origin_ptr + row, tl.where(bad_affine, 0.0, origin))
    tl.store(inv_delta_ptr + row, tl.where(bad_affine, 1.0, inv_delta))
    tl.store(status_ptr + row, status)


@triton.jit(repr=_seed_histogram_repr)
def _pa_mqa_litetopk_seed_histogram_kernel(
    sample_logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    origin_ptr,
    inv_delta_ptr,
    histogram_ptr,
    threshold_ptr,
    sample_stride,
    SAMPLE_LEN: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    row_len = tl.maximum(row_end - row_start, 0)
    sample_count = tl.minimum(row_len, SAMPLE_LEN)
    origin = tl.load(origin_ptr + row)
    inv_delta = tl.load(inv_delta_ptr + row)

    histogram = tl.zeros((NUM_BUCKETS,), dtype=tl.int32)
    offsets = tl.arange(0, BLOCK)
    for base in tl.static_range(0, SAMPLE_LEN, BLOCK):
        columns = base + offsets
        genuine = columns < sample_count
        values = tl.load(
            sample_logits_ptr + row * sample_stride + columns,
            mask=genuine,
            other=0.0,
        ).to(tl.float32)
        finite = genuine & _finite_f32(values)
        buckets = _bucket_f32(values, origin, inv_delta, NUM_BUCKETS)
        histogram += tl.histogram(buckets, NUM_BUCKETS, mask=finite)

    bucket_offsets = tl.arange(0, NUM_BUCKETS)
    tl.store(histogram_ptr + row * NUM_BUCKETS + bucket_offsets, histogram)
    required = tl.minimum(row_len, TOPK)
    inclusive = tl.cumsum(histogram, axis=0)
    first = tl.min(tl.where(inclusive >= required, bucket_offsets, NUM_BUCKETS), axis=0)
    threshold = tl.where(required > 0, tl.minimum(first, NUM_BUCKETS - 1), 0)
    tl.store(threshold_ptr + row, threshold)


@triton.jit(repr=_seed_emit_repr)
def _pa_mqa_litetopk_seed_emit_kernel(
    sample_logits_ptr,
    row_starts_ptr,
    row_ends_ptr,
    origin_ptr,
    inv_delta_ptr,
    threshold_ptr,
    candidate_values_ptr,
    candidate_indices_ptr,
    candidate_counts_ptr,
    select_starts_ptr,
    select_ends_ptr,
    status_ptr,
    sample_stride,
    candidate_stride,
    merge_cap,
    STATUS_CANDIDATE_OVERFLOW: tl.constexpr,
    STATUS_UNDERFILLED: tl.constexpr,
    SAMPLE_LEN: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    row_len = tl.maximum(row_end - row_start, 0)
    sample_count = tl.minimum(row_len, SAMPLE_LEN)
    origin = tl.load(origin_ptr + row)
    inv_delta = tl.load(inv_delta_ptr + row)
    threshold = tl.load(threshold_ptr + row)
    row_i64 = row.to(tl.int64)
    sample_row_base = row_i64 * sample_stride.to(tl.int64)
    candidate_row_base = row_i64 * candidate_stride.to(tl.int64)

    attempted = 0
    offsets = tl.arange(0, BLOCK)
    for base in tl.static_range(0, SAMPLE_LEN, BLOCK):
        columns = base + offsets
        genuine = columns < sample_count
        values = tl.load(
            sample_logits_ptr + sample_row_base + columns,
            mask=genuine,
            other=0.0,
        ).to(tl.float32)
        finite = genuine & _finite_f32(values)
        buckets = _bucket_f32(values, origin, inv_delta, NUM_BUCKETS)
        keep = finite & (buckets <= threshold)
        inclusive = tl.cumsum(keep.to(tl.int32), axis=0)
        positions = attempted + inclusive - 1
        in_capacity = positions < merge_cap
        store_mask = keep & in_capacity
        safe_positions = tl.where(store_mask, positions, 0)
        tl.store(
            candidate_values_ptr + candidate_row_base + safe_positions,
            values,
            mask=store_mask,
        )
        tl.store(
            candidate_indices_ptr + candidate_row_base + safe_positions,
            row_start + columns,
            mask=store_mask,
        )
        attempted += tl.sum(keep.to(tl.int32), axis=0)

    required = tl.minimum(row_len, TOPK)
    status = tl.load(status_ptr + row)
    status |= tl.where(attempted > merge_cap, STATUS_CANDIDATE_OVERFLOW, 0)
    status |= tl.where(attempted < required, STATUS_UNDERFILLED, 0)
    tl.store(candidate_counts_ptr + row, attempted)
    tl.store(select_starts_ptr + row, 0)
    tl.store(select_ends_ptr + row, tl.minimum(attempted, merge_cap))
    tl.store(status_ptr + row, status)


@triton.jit(
    repr=_scan_schedule_repr,
    do_not_specialize=[
        "rows",
        "segment_start",
        "segment_end",
        "max_seq_len",
        "block_table_rows",
    ],
)
def _pa_mqa_litetopk_scan_schedule_kernel(
    row_to_batch_ptr,
    row_starts_ptr,
    row_ends_ptr,
    cta_info_ptr,
    suffix_cta_info_ptr,
    page_errors_ptr,
    score_errors_ptr,
    status_ptr,
    sample_ends_ptr,
    status_ok_ptr,
    rows,
    segment_start,
    segment_end,
    max_seq_len,
    block_table_rows,
    VALIDATE_ROWS: tl.constexpr,
    RESET_STATE: tl.constexpr,
    RESET_STATUS_OK: tl.constexpr,
    DUAL_SCHEDULE: tl.constexpr,
    SAMPLE_LEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row_live = offsets < rows
    safe_row = tl.where(row_live, offsets, 0)
    row_start = tl.load(row_starts_ptr + safe_row, mask=row_live, other=0)
    row_end = tl.load(row_ends_ptr + safe_row, mask=row_live, other=0)
    batch = tl.load(row_to_batch_ptr + safe_row, mask=row_live, other=0)
    work_live = row_live
    if VALIDATE_ROWS:
        invalid = (
            (row_start < 0)
            | (row_end < row_start)
            | (row_end > max_seq_len)
            | (batch < 0)
            | (batch >= block_table_rows)
        )
        errors = invalid.to(tl.int32)
        tl.store(page_errors_ptr + safe_row, errors, mask=row_live)
        work_live &= errors == 0
    if RESET_STATE:
        tl.store(score_errors_ptr + safe_row, 0, mask=row_live)
        tl.store(status_ptr + safe_row, 0, mask=row_live)
    if RESET_STATUS_OK & (tl.program_id(0) == 0):
        tl.store(status_ok_ptr, 1)
    window_start = tl.where(work_live, tl.maximum(row_start, segment_start), 0)
    window_end = tl.where(work_live, tl.minimum(row_end, segment_end), 0)
    scan_start = window_start
    scan_end = (
        tl.minimum(window_start + SAMPLE_LEN, window_end)
        if DUAL_SCHEDULE
        else window_end
    )
    nonempty = work_live & (scan_end > scan_start)
    first_chunk = scan_start // BLOCK_K
    last_chunk = (scan_end + BLOCK_K - 1) // BLOCK_K
    chunk_count = tl.where(nonempty, tl.maximum(last_chunk - first_chunk, 1), 1)
    base = offsets * 6
    tl.store(cta_info_ptr + base + 0, safe_row, mask=row_live)
    tl.store(cta_info_ptr + base + 1, tl.where(work_live, batch, 0), mask=row_live)
    tl.store(cta_info_ptr + base + 2, tl.where(nonempty, first_chunk, 0), mask=row_live)
    tl.store(cta_info_ptr + base + 3, chunk_count, mask=row_live)
    tl.store(cta_info_ptr + base + 4, tl.where(nonempty, scan_start, 0), mask=row_live)
    tl.store(cta_info_ptr + base + 5, tl.where(nonempty, scan_end, 0), mask=row_live)
    if DUAL_SCHEDULE:
        tl.store(sample_ends_ptr + safe_row, scan_end, mask=row_live)
        suffix_start = scan_end
        suffix_end = window_end
        suffix_nonempty = work_live & (suffix_end > suffix_start)
        suffix_first_chunk = suffix_start // BLOCK_K
        suffix_last_chunk = (suffix_end + BLOCK_K - 1) // BLOCK_K
        suffix_chunk_count = tl.where(
            suffix_nonempty,
            tl.maximum(suffix_last_chunk - suffix_first_chunk, 1),
            1,
        )
        tl.store(suffix_cta_info_ptr + base + 0, safe_row, mask=row_live)
        tl.store(
            suffix_cta_info_ptr + base + 1,
            tl.where(work_live, batch, 0),
            mask=row_live,
        )
        tl.store(
            suffix_cta_info_ptr + base + 2,
            tl.where(suffix_nonempty, suffix_first_chunk, 0),
            mask=row_live,
        )
        tl.store(suffix_cta_info_ptr + base + 3, suffix_chunk_count, mask=row_live)
        tl.store(
            suffix_cta_info_ptr + base + 4,
            tl.where(suffix_nonempty, suffix_start, 0),
            mask=row_live,
        )
        tl.store(
            suffix_cta_info_ptr + base + 5,
            tl.where(suffix_nonempty, suffix_end, 0),
            mask=row_live,
        )


@triton.jit(repr=_refresh_threshold_repr)
def _pa_mqa_litetopk_refresh_threshold_kernel(
    histogram_ptr,
    threshold_ptr,
    NUM_BUCKETS: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    buckets = tl.arange(0, NUM_BUCKETS)
    histogram = tl.load(histogram_ptr + row * NUM_BUCKETS + buckets)
    inclusive = tl.cumsum(histogram, axis=0)
    first = tl.min(tl.where(inclusive >= TOPK, buckets, NUM_BUCKETS), axis=0)
    old_threshold = tl.load(threshold_ptr + row)
    has_topk = tl.sum(histogram, axis=0) >= TOPK
    new_threshold = tl.where(
        has_topk,
        tl.minimum(old_threshold, tl.minimum(first, NUM_BUCKETS - 1)),
        old_threshold,
    )
    tl.store(threshold_ptr + row, new_threshold)
