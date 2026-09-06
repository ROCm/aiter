# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Final ordering and paged-slot mapping for FP4 MQA TopK candidates."""

import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.topk import argsort as triton_argsort


@triton.jit
def _order_and_map_topk_kernel(
    source_values,
    source_raw_indices,
    valid_counts,
    local_starts,
    local_ends,
    row_to_batch,
    block_tables,
    output_values,
    output_raw_indices,
    output_page_slots,
    rows,
    max_seq_len,
    block_table_stride,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, TOPK)
    count = tl.minimum(tl.maximum(tl.load(valid_counts + row), 0), TOPK)
    valid = offsets < count
    row_offset = row * TOPK
    values = tl.load(
        source_values + row_offset + offsets,
        mask=valid,
        other=-float("inf"),
    )
    raw_indices = tl.load(
        source_raw_indices + row_offset + offsets,
        mask=valid,
        other=-1,
    ).to(tl.int32)

    # Signed ordinal comparison preserves +0 > -0. Every NaN maps below every
    # numeric value; raw logical index is the exact secondary key.
    bits = values.to(tl.int32, bitcast=True)
    absolute_bits = bits & 0x7FFFFFFF
    is_nan = absolute_bits > 0x7F800000
    score_ordinal = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    score_ordinal = tl.where(is_nan, -2147483648, score_ordinal)
    raw_i64 = raw_indices.to(tl.int64)
    score_key = score_ordinal.to(tl.int64) * 4294967296 + (2147483647 - raw_i64)

    start = tl.maximum(
        0,
        tl.minimum(tl.load(local_starts + row), max_seq_len),
    )
    end = tl.maximum(
        start,
        tl.minimum(tl.load(local_ends + row), max_seq_len),
    )
    # Preserve the serving contract for a short row: return its complete
    # logical window sequentially. Long rows use score-desc/index-asc order.
    sequential_key = 2147483647 - raw_i64
    key = tl.where(end - start > TOPK, score_key, sequential_key)
    key = tl.where(valid, key, -9223372036854775808)
    positions = offsets.to(tl.int32)
    _, sorted_positions = triton_argsort(key, positions, 0, descending=True)

    selected_values = tl.load(
        source_values + row_offset + sorted_positions,
        mask=valid,
        other=-float("inf"),
    )
    selected_raw = tl.load(
        source_raw_indices + row_offset + sorted_positions,
        mask=valid,
        other=-1,
    ).to(tl.int32)
    safe_raw = tl.maximum(selected_raw, 0)
    batch = tl.load(row_to_batch + row)
    logical_page = safe_raw // PAGE_SIZE
    page_offset = safe_raw % PAGE_SIZE
    physical_page = tl.load(
        block_tables + batch * block_table_stride + logical_page,
        mask=valid,
        other=-1,
    ).to(tl.int32)
    page_slot = physical_page * PAGE_SIZE + page_offset

    tl.store(
        output_values + row_offset + offsets,
        tl.where(valid, selected_values, -float("inf")),
    )
    tl.store(
        output_raw_indices + row_offset + offsets,
        tl.where(valid, selected_raw, -1),
    )
    tl.store(
        output_page_slots + row_offset + offsets,
        tl.where(valid, page_slot, -1),
    )


def order_and_map_mqa_topk(
    source_values: torch.Tensor,
    source_raw_indices: torch.Tensor,
    valid_counts: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
    row_to_batch: torch.Tensor,
    block_tables: torch.Tensor,
    output_values: torch.Tensor,
    output_raw_indices: torch.Tensor,
    output_page_slots: torch.Tensor,
    max_seq_len: int,
    topk: int,
    page_size: int,
) -> None:
    """Order an exact winning set and map logical indices to physical slots."""
    rows = source_values.shape[0]
    _order_and_map_topk_kernel[(rows,)](
        source_values,
        source_raw_indices,
        valid_counts,
        local_starts,
        local_ends,
        row_to_batch,
        block_tables,
        output_values,
        output_raw_indices,
        output_page_slots,
        rows,
        max_seq_len,
        block_tables.stride(0),
        TOPK=topk,
        PAGE_SIZE=page_size,
        num_warps=8,
    )


__all__ = ["order_and_map_mqa_topk"]
