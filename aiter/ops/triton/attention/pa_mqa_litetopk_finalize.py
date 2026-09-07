# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.pa_mqa_litetopk_finalize import (
    _pa_mqa_litetopk_finalize_kernel,
    _pa_mqa_litetopk_preselect_kernel,
    _pa_mqa_litetopk_validate_pages_kernel,
)


def validate_pa_mqa_litetopk_pages(
    row_to_batch: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    block_tables: torch.Tensor,
    page_errors: torch.Tensor,
    status: torch.Tensor,
    *,
    max_seq_len: int,
    physical_page_capacity: int,
    page_size: int,
    status_invalid_index: int,
    stream: torch.cuda.Stream,
) -> None:
    rows = row_to_batch.shape[0]
    block = 256
    device = row_to_batch.device
    if torch.device(stream.device) != device:
        raise ValueError("stream and page metadata must be on the same device")
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _pa_mqa_litetopk_validate_pages_kernel[(rows,)](
            row_to_batch,
            row_starts,
            row_ends,
            block_tables,
            page_errors,
            status,
            rows,
            max_seq_len,
            block_tables.stride(0),
            block_tables.shape[0],
            block_tables.shape[1],
            physical_page_capacity,
            STATUS_INVALID_INDEX=status_invalid_index,
            PAGE_SIZE=page_size,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )


def prepare_pa_mqa_litetopk_select(
    candidate_counts: torch.Tensor,
    page_errors: torch.Tensor,
    score_errors: torch.Tensor,
    histogram: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    threshold: torch.Tensor,
    select_ends: torch.Tensor,
    status: torch.Tensor,
    *,
    topk: int,
    merge_cap: int,
    num_buckets: int,
    status_candidate_overflow: int,
    status_underfilled: int,
    status_nonfinite: int,
    status_invalid_index: int,
    status_bad_certificate: int,
    stream: torch.cuda.Stream,
) -> None:
    rows = candidate_counts.shape[0]
    device = candidate_counts.device
    if torch.device(stream.device) != device:
        raise ValueError("stream and selection metadata must be on the same device")
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _pa_mqa_litetopk_preselect_kernel[(rows,)](
            candidate_counts,
            page_errors,
            score_errors,
            histogram,
            row_starts,
            row_ends,
            threshold,
            select_ends,
            status,
            merge_cap,
            STATUS_CANDIDATE_OVERFLOW=status_candidate_overflow,
            STATUS_UNDERFILLED=status_underfilled,
            STATUS_NONFINITE=status_nonfinite,
            STATUS_INVALID_INDEX=status_invalid_index,
            STATUS_BAD_CERTIFICATE=status_bad_certificate,
            NUM_BUCKETS=num_buckets,
            TOPK=topk,
            num_warps=4,
            num_stages=1,
        )


def finalize_pa_mqa_litetopk(
    candidate_indices: torch.Tensor,
    select_ends: torch.Tensor,
    selected_slots: torch.Tensor,
    selected_values: torch.Tensor,
    row_to_batch: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    block_tables: torch.Tensor,
    status: torch.Tensor,
    out_values: torch.Tensor,
    out_raw_indices: torch.Tensor,
    out_physical_indices: torch.Tensor,
    out_counts: torch.Tensor,
    *,
    topk: int,
    page_size: int,
    physical_page_capacity: int,
    status_nonfinite: int,
    status_invalid_index: int,
    status_selector_failure: int,
    stream: torch.cuda.Stream,
    status_ok: torch.Tensor | None = None,
) -> None:
    rows = candidate_indices.shape[0]
    block = triton.next_power_of_2(topk)
    device = candidate_indices.device
    if torch.device(stream.device) != device:
        raise ValueError("stream and finalizer tensors must be on the same device")
    aggregate_status = status_ok is not None
    if status_ok is None:
        status_ok = status
    elif (
        status_ok.dtype != torch.int32
        or status_ok.numel() != 1
        or status_ok.device != device
        or not status_ok.is_contiguous()
    ):
        raise ValueError("status_ok must be a contiguous one-element int32 tensor")
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _pa_mqa_litetopk_finalize_kernel[(rows,)](
            candidate_indices,
            select_ends,
            selected_slots,
            selected_values,
            row_to_batch,
            row_starts,
            row_ends,
            block_tables,
            status,
            out_values,
            out_raw_indices,
            out_physical_indices,
            out_counts,
            status_ok,
            candidate_indices.stride(0),
            block_tables.stride(0),
            block_tables.shape[0],
            block_tables.shape[1],
            page_size,
            physical_page_capacity,
            STATUS_NONFINITE=status_nonfinite,
            STATUS_INVALID_INDEX=status_invalid_index,
            STATUS_SELECTOR_FAILURE=status_selector_failure,
            AGGREGATE_STATUS=aggregate_status,
            TOPK=topk,
            BLOCK=block,
            num_warps=4 if topk <= 512 else 8,
            num_stages=1,
        )
