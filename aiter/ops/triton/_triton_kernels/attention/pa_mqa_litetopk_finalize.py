# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: PLR0124

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_preselect_repr = make_kernel_repr(
    "_pa_mqa_litetopk_preselect_kernel", ["NUM_BUCKETS", "TOPK"]
)
_finalize_repr = make_kernel_repr("_pa_mqa_litetopk_finalize_kernel", ["TOPK", "BLOCK"])
_validate_pages_repr = make_kernel_repr(
    "_pa_mqa_litetopk_validate_pages_kernel", ["PAGE_SIZE", "BLOCK"]
)


@triton.jit(
    repr=_validate_pages_repr,
    do_not_specialize=[
        "rows",
        "max_seq_len",
        "block_table_stride",
        "block_table_rows",
        "block_table_capacity",
        "physical_page_capacity",
    ],
)
def _pa_mqa_litetopk_validate_pages_kernel(
    row_to_batch_ptr,
    row_starts_ptr,
    row_ends_ptr,
    block_tables_ptr,
    page_errors_ptr,
    status_ptr,
    rows,
    max_seq_len,
    block_table_stride,
    block_table_rows,
    block_table_capacity,
    physical_page_capacity,
    STATUS_INVALID_INDEX: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= rows:
        return
    batch = tl.load(row_to_batch_ptr + row)
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    valid_window = (row_start >= 0) & (row_end >= row_start) & (row_end <= max_seq_len)
    batch_ok = (batch >= 0) & (batch < block_table_rows)
    safe_batch = tl.where(batch_ok, batch, 0)
    block_table_row_base = safe_batch.to(tl.int64) * block_table_stride.to(tl.int64)
    first_page = tl.maximum(row_start, 0) // PAGE_SIZE
    last_page = (tl.minimum(row_end, max_seq_len) + PAGE_SIZE - 1) // PAGE_SIZE
    errors = tl.where(valid_window & batch_ok, 0, 1)
    offsets = tl.arange(0, BLOCK)
    for base in range(0, block_table_capacity, BLOCK):
        page_offsets = base + offsets
        live = (
            valid_window
            & batch_ok
            & (page_offsets >= first_page)
            & (page_offsets < last_page)
            & (page_offsets < block_table_capacity)
        )
        pages = tl.load(
            block_tables_ptr + block_table_row_base + tl.where(live, page_offsets, 0),
            mask=live,
            other=0,
        )
        invalid = live & ((pages < 0) | (pages >= physical_page_capacity))
        errors += tl.sum(invalid.to(tl.int32), axis=0)
    status = tl.load(status_ptr + row)
    status |= tl.where(errors != 0, STATUS_INVALID_INDEX, 0)
    tl.store(page_errors_ptr + row, errors)
    tl.store(status_ptr + row, status)


@triton.jit(
    repr=_preselect_repr,
    do_not_specialize=["merge_cap"],
)
def _pa_mqa_litetopk_preselect_kernel(
    candidate_counts_ptr,
    page_errors_ptr,
    score_errors_ptr,
    histogram_ptr,
    row_starts_ptr,
    row_ends_ptr,
    threshold_ptr,
    select_ends_ptr,
    status_ptr,
    merge_cap,
    STATUS_CANDIDATE_OVERFLOW: tl.constexpr,
    STATUS_UNDERFILLED: tl.constexpr,
    STATUS_NONFINITE: tl.constexpr,
    STATUS_INVALID_INDEX: tl.constexpr,
    STATUS_BAD_CERTIFICATE: tl.constexpr,
    NUM_BUCKETS: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    buckets = tl.arange(0, NUM_BUCKETS)
    histogram = tl.load(histogram_ptr + row * NUM_BUCKETS + buckets)
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    row_len = tl.maximum(row_end - row_start, 0)
    required = tl.minimum(row_len, TOPK)
    attempted = tl.load(candidate_counts_ptr + row)
    threshold = tl.load(threshold_ptr + row)
    status = tl.load(status_ptr + row)
    status |= tl.where(attempted > merge_cap, STATUS_CANDIDATE_OVERFLOW, 0)
    status |= tl.where(attempted < required, STATUS_UNDERFILLED, 0)
    status |= tl.where(tl.load(score_errors_ptr + row) != 0, STATUS_NONFINITE, 0)
    status |= tl.where(tl.load(page_errors_ptr + row) != 0, STATUS_INVALID_INDEX, 0)
    threshold_count = tl.sum(tl.where(buckets <= threshold, histogram, 0), axis=0)
    bad_certificate = (
        (threshold < 0)
        | (threshold >= NUM_BUCKETS)
        | (threshold_count < required)
        | (attempted < threshold_count)
        | (attempted > row_len)
    )
    status |= tl.where(bad_certificate, STATUS_BAD_CERTIFICATE, 0)
    tl.store(status_ptr + row, status)
    tl.store(select_ends_ptr + row, tl.where(status == 0, attempted, 0))


@triton.jit(
    repr=_finalize_repr,
    do_not_specialize=[
        "candidate_stride",
        "block_table_stride",
        "block_table_rows",
        "block_table_capacity",
        "page_size",
        "physical_page_capacity",
    ],
)
def _pa_mqa_litetopk_finalize_kernel(
    candidate_indices_ptr,
    select_ends_ptr,
    selected_slots_ptr,
    selected_values_ptr,
    row_to_batch_ptr,
    row_starts_ptr,
    row_ends_ptr,
    block_tables_ptr,
    status_ptr,
    out_values_ptr,
    out_raw_indices_ptr,
    out_physical_indices_ptr,
    out_counts_ptr,
    status_ok_ptr,
    candidate_stride,
    block_table_stride,
    block_table_rows,
    block_table_capacity,
    page_size,
    physical_page_capacity,
    STATUS_NONFINITE: tl.constexpr,
    STATUS_INVALID_INDEX: tl.constexpr,
    STATUS_SELECTOR_FAILURE: tl.constexpr,
    AGGREGATE_STATUS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    lane_mask = lanes < TOPK
    row_start = tl.load(row_starts_ptr + row)
    row_end = tl.load(row_ends_ptr + row)
    required = tl.minimum(tl.maximum(row_end - row_start, 0), TOPK)
    needed = lane_mask & (lanes < required)
    select_end = tl.load(select_ends_ptr + row)
    row_i64 = row.to(tl.int64)
    candidate_row_base = row_i64 * candidate_stride.to(tl.int64)
    output_row_base = row_i64 * TOPK

    slots = tl.load(
        selected_slots_ptr + output_row_base + lanes,
        mask=lane_mask,
        other=-1,
    )
    slot_ok = needed & (slots >= 0) & (slots < select_end)
    safe_slots = tl.where(slot_ok, slots, 0)
    values = tl.load(
        selected_values_ptr + output_row_base + lanes,
        mask=lane_mask,
        other=-float("inf"),
    ).to(tl.float32)
    raw_indices = tl.load(
        candidate_indices_ptr + candidate_row_base + safe_slots,
        mask=slot_ok,
        other=-1,
    ).to(tl.int32)
    finite = (values == values) & (values != float("inf")) & (values != -float("inf"))
    raw_ok = needed & (raw_indices >= row_start) & (raw_indices < row_end)
    safe_raw = tl.where(raw_ok, raw_indices, 0)
    page_offsets = safe_raw // page_size
    page_ok = (
        needed
        & raw_ok
        & (raw_indices >= 0)
        & (page_offsets >= 0)
        & (page_offsets < block_table_capacity)
    )
    batch = tl.load(row_to_batch_ptr + row)
    batch_ok = (batch >= 0) & (batch < block_table_rows)
    safe_batch = tl.where(batch_ok, batch, 0)
    block_table_row_base = safe_batch.to(tl.int64) * block_table_stride.to(tl.int64)
    pages = tl.load(
        block_tables_ptr + block_table_row_base + page_offsets,
        mask=page_ok & batch_ok,
        other=-1,
    ).to(tl.int32)
    physical_i64 = pages.to(tl.int64) * page_size.to(tl.int64) + (
        safe_raw.to(tl.int64) % page_size.to(tl.int64)
    )
    physical_ok = (
        page_ok
        & batch_ok
        & (pages >= 0)
        & (pages < physical_page_capacity)
        & (physical_i64 >= 0)
        & (physical_i64 <= 0x7FFFFFFF)
    )
    physical = tl.where(physical_ok, physical_i64, 0).to(tl.int32)

    status = tl.load(status_ptr + row)
    status |= tl.where(
        tl.sum((needed & ~slot_ok).to(tl.int32), axis=0) > 0,
        STATUS_SELECTOR_FAILURE,
        0,
    )
    status |= tl.where(
        tl.sum((needed & ~finite).to(tl.int32), axis=0) > 0,
        STATUS_NONFINITE,
        0,
    )
    status |= tl.where(
        tl.sum((needed & ~physical_ok).to(tl.int32), axis=0) > 0,
        STATUS_INVALID_INDEX,
        0,
    )
    success = status == 0
    if AGGREGATE_STATUS:
        tl.atomic_min(status_ok_ptr, 0, mask=~success)
    publish = lane_mask & (lanes < required) & success
    tl.store(
        out_values_ptr + output_row_base + lanes,
        tl.where(publish, values, -float("inf")),
        mask=lane_mask,
    )
    tl.store(
        out_raw_indices_ptr + output_row_base + lanes,
        tl.where(publish, raw_indices, -1),
        mask=lane_mask,
    )
    tl.store(
        out_physical_indices_ptr + output_row_base + lanes,
        tl.where(publish, physical, -1),
        mask=lane_mask,
    )
    tl.store(status_ptr + row, status)
    tl.store(out_counts_ptr + row, tl.where(success, required, 0))
