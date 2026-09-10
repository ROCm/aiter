# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from itertools import pairwise

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.pa_mqa_litetopk_seed import (
    _pa_mqa_litetopk_refresh_threshold_kernel,
    _pa_mqa_litetopk_scan_schedule_kernel,
    _pa_mqa_litetopk_seed_calibrate_kernel,
    _pa_mqa_litetopk_seed_emit_kernel,
    _pa_mqa_litetopk_seed_histogram_kernel,
)


def prepare_pa_mqa_litetopk_seed(
    sample_logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    *,
    origin: torch.Tensor,
    inv_delta: torch.Tensor,
    threshold: torch.Tensor,
    histogram: torch.Tensor,
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    select_starts: torch.Tensor,
    select_ends: torch.Tensor,
    status: torch.Tensor,
    max_seq_len: int,
    topk: int,
    sample_len: int,
    num_buckets: int,
    merge_cap: int,
    status_candidate_overflow: int,
    status_underfilled: int,
    status_nonfinite: int,
    status_invalid_index: int,
    emit_candidates: bool = True,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Calibrate the LiteTopK gate and emit a row's bounded seed candidates."""
    rows = sample_logits.shape[0]
    if sample_logits.dtype != torch.float32 or sample_logits.ndim != 2:
        raise ValueError("sample_logits must be a 2D float32 tensor")
    if sample_logits.shape[1] != sample_len:
        raise ValueError(
            f"sample_logits width must equal sample_len={sample_len}, "
            f"got {sample_logits.shape[1]}"
        )
    if sample_len < topk:
        raise ValueError("sample_len must be at least topk")
    if num_buckets != 256:
        raise ValueError("the initial LiteTopK seed implementation requires 256 bins")
    if merge_cap < topk or candidate_values.shape != (rows, merge_cap):
        raise ValueError("candidate_values shape does not match rows and merge_cap")
    expected_i32_rows = (
        row_starts,
        row_ends,
        threshold,
        candidate_counts,
        select_starts,
        select_ends,
        status,
    )
    if any(t.dtype != torch.int32 or t.shape != (rows,) for t in expected_i32_rows):
        raise ValueError("row metadata and status tensors must be int32 [rows]")
    if origin.dtype != torch.float32 or origin.shape != (rows,):
        raise ValueError("origin must be float32 [rows]")
    if inv_delta.dtype != torch.float32 or inv_delta.shape != (rows,):
        raise ValueError("inv_delta must be float32 [rows]")
    if histogram.dtype != torch.int32 or histogram.shape != (rows, num_buckets):
        raise ValueError("histogram must be int32 [rows, num_buckets]")
    if candidate_indices.dtype != torch.int32 or candidate_indices.shape != (
        rows,
        merge_cap,
    ):
        raise ValueError("candidate_indices shape does not match rows and merge_cap")
    tensors = (
        sample_logits,
        *expected_i32_rows,
        origin,
        inv_delta,
        histogram,
        candidate_values,
        candidate_indices,
    )
    device = sample_logits.device
    if device.type != "cuda" or any(t.device != device for t in tensors):
        raise ValueError("all seed tensors must be on the same GPU device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all seed tensors must be contiguous")
    if rows == 0:
        return
    if stream is None:
        stream = torch.cuda.current_stream(device)
    if torch.device(stream.device) != device:
        raise ValueError("stream and seed tensors must be on the same device")

    block = 256
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _pa_mqa_litetopk_seed_calibrate_kernel[(rows,)](
            sample_logits,
            row_starts,
            row_ends,
            origin,
            inv_delta,
            status,
            sample_logits.stride(0),
            max_seq_len,
            STATUS_NONFINITE=status_nonfinite,
            STATUS_INVALID_INDEX=status_invalid_index,
            SAMPLE_LEN=sample_len,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
        _pa_mqa_litetopk_seed_histogram_kernel[(rows,)](
            sample_logits,
            row_starts,
            row_ends,
            origin,
            inv_delta,
            histogram,
            threshold,
            sample_logits.stride(0),
            SAMPLE_LEN=sample_len,
            NUM_BUCKETS=num_buckets,
            TOPK=topk,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )
        if emit_candidates:
            _pa_mqa_litetopk_seed_emit_kernel[(rows,)](
                sample_logits,
                row_starts,
                row_ends,
                origin,
                inv_delta,
                threshold,
                candidate_values,
                candidate_indices,
                candidate_counts,
                select_starts,
                select_ends,
                status,
                sample_logits.stride(0),
                candidate_values.stride(0),
                merge_cap,
                STATUS_CANDIDATE_OVERFLOW=status_candidate_overflow,
                STATUS_UNDERFILLED=status_underfilled,
                SAMPLE_LEN=sample_len,
                NUM_BUCKETS=num_buckets,
                TOPK=topk,
                BLOCK=block,
                num_warps=4,
                num_stages=1,
            )


def build_pa_mqa_litetopk_scan_schedule(
    row_to_batch: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    cta_info: torch.Tensor,
    *,
    segment_start: int,
    segment_end: int,
    block_k: int,
    stream: torch.cuda.Stream,
    page_errors: torch.Tensor | None = None,
    max_seq_len: int | None = None,
    block_table_rows: int | None = None,
    suffix_cta_info: torch.Tensor | None = None,
    sample_len: int = 0,
    score_errors: torch.Tensor | None = None,
    status: torch.Tensor | None = None,
    sample_ends: torch.Tensor | None = None,
    status_ok: torch.Tensor | None = None,
) -> None:
    rows = row_to_batch.shape[0]
    block = 256
    device = row_to_batch.device
    if torch.device(stream.device) != device:
        raise ValueError("stream and scan metadata must be on the same device")
    validate_rows = page_errors is not None
    if validate_rows:
        if page_errors.dtype != torch.int32 or page_errors.shape != (rows,):
            raise ValueError("page_errors must be int32 [rows]")
        if page_errors.device != device or not page_errors.is_contiguous():
            raise ValueError("page_errors must be contiguous and on the scan device")
        if max_seq_len is None or block_table_rows is None:
            raise ValueError(
                "max_seq_len and block_table_rows are required with page_errors"
            )
    else:
        page_errors = cta_info
        max_seq_len = segment_end
        block_table_rows = 1
    dual_schedule = suffix_cta_info is not None
    if dual_schedule:
        if (
            suffix_cta_info.dtype != torch.int32
            or suffix_cta_info.shape != (rows, 6)
            or suffix_cta_info.device != device
            or not suffix_cta_info.is_contiguous()
        ):
            raise ValueError("suffix_cta_info must be contiguous int32 [rows, 6]")
        if sample_len <= 0:
            raise ValueError("sample_len must be positive with suffix_cta_info")
        if (
            sample_ends is None
            or sample_ends.dtype != torch.int32
            or sample_ends.shape != (rows,)
            or sample_ends.device != device
            or not sample_ends.is_contiguous()
        ):
            raise ValueError("sample_ends must be contiguous int32 [rows]")
    else:
        suffix_cta_info = cta_info
        sample_ends = cta_info
    reset_state = score_errors is not None or status is not None
    if reset_state:
        if score_errors is None or status is None:
            raise ValueError("score_errors and status must be provided together")
        if not validate_rows:
            raise ValueError("page_errors is required when resetting scan state")
        for name, tensor in (("score_errors", score_errors), ("status", status)):
            if (
                tensor.dtype != torch.int32
                or tensor.shape != (rows,)
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous int32 [rows]")
    else:
        score_errors = cta_info
        status = cta_info
    reset_status_ok = status_ok is not None
    if status_ok is None:
        status_ok = cta_info
    elif (
        status_ok.dtype != torch.int32
        or status_ok.numel() != 1
        or status_ok.device != device
        or not status_ok.is_contiguous()
    ):
        raise ValueError("status_ok must be a contiguous one-element int32 tensor")
    written_outputs = [cta_info]
    if dual_schedule:
        written_outputs.extend((suffix_cta_info, sample_ends))
    if validate_rows:
        written_outputs.append(page_errors)
    if reset_state:
        written_outputs.extend((score_errors, status))
    if reset_status_ok:
        written_outputs.append(status_ok)
    byte_ranges = []
    for tensor in written_outputs:
        start = tensor.data_ptr()
        byte_ranges.append((start, start + tensor.numel() * tensor.element_size()))
    byte_ranges.sort()
    if any(
        left_start < right_end and right_start < left_end
        for (left_start, left_end), (right_start, right_end) in pairwise(byte_ranges)
    ):
        raise ValueError("scan schedule outputs must not overlap")
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _pa_mqa_litetopk_scan_schedule_kernel[(triton.cdiv(rows, block),)](
            row_to_batch,
            row_starts,
            row_ends,
            cta_info,
            suffix_cta_info,
            page_errors,
            score_errors,
            status,
            sample_ends,
            status_ok,
            rows,
            segment_start,
            segment_end,
            max_seq_len,
            block_table_rows,
            VALIDATE_ROWS=validate_rows,
            RESET_STATE=reset_state,
            RESET_STATUS_OK=reset_status_ok,
            DUAL_SCHEDULE=dual_schedule,
            SAMPLE_LEN=sample_len,
            BLOCK_K=block_k,
            BLOCK=block,
            num_warps=4,
            num_stages=1,
        )


def refresh_pa_mqa_litetopk_threshold(
    histogram: torch.Tensor,
    threshold: torch.Tensor,
    *,
    topk: int,
    num_buckets: int,
    stream: torch.cuda.Stream,
) -> None:
    rows = histogram.shape[0]
    device = histogram.device
    if torch.cuda.device_count() and torch.device(stream.device) != device:
        raise ValueError("stream and threshold tensors must be on the same device")
    with torch.cuda.device(device), torch.cuda.stream(stream):
        _pa_mqa_litetopk_refresh_threshold_kernel[(rows,)](
            histogram,
            threshold,
            NUM_BUCKETS=num_buckets,
            TOPK=topk,
            num_warps=4,
            num_stages=1,
        )
