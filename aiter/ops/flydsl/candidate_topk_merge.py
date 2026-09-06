# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public interface for exact split-candidate TopK merge."""

from functools import lru_cache

import torch

from .kernels.candidate_topk_merge import build_candidate_topk_merge_module
from .kernels.kernels_common import get_warp_size
from .kernels.tensor_shim import _run_compiled


@lru_cache(maxsize=32)
def _get_launcher(topk: int, page_size: int):
    return build_candidate_topk_merge_module(topk, page_size)


@lru_cache(maxsize=128)
def _validate(
    candidate_values_shape,
    candidate_values_stride,
    candidate_values_dtype,
    candidate_indices_shape,
    candidate_indices_stride,
    candidate_indices_dtype,
    candidate_counts_shape,
    candidate_counts_dtype,
    row_offsets_shape,
    row_offsets_dtype,
    row_to_batch_shape,
    row_to_batch_dtype,
    block_table_shape,
    block_table_dtype,
    out_values_shape,
    out_values_dtype,
    out_raw_shape,
    out_raw_dtype,
    out_physical_shape,
    out_physical_dtype,
    out_counts_shape,
    out_counts_dtype,
    page_size,
    wave_size,
):
    if len(candidate_values_shape) != 2 or candidate_values_dtype != torch.float32:
        raise ValueError("candidate_values must be float32 [ctas, topk]")
    ctas, topk = candidate_values_shape
    if candidate_values_stride != (topk, 1):
        raise ValueError("candidate_values must be contiguous")
    if (
        candidate_indices_shape != candidate_values_shape
        or candidate_indices_stride != (topk, 1)
        or candidate_indices_dtype != torch.int32
    ):
        raise ValueError(
            "candidate_indices must be contiguous int32 with candidate_values shape"
        )
    if candidate_counts_shape != (ctas,) or candidate_counts_dtype != torch.int32:
        raise ValueError("candidate_counts must be int32 [ctas]")
    if len(row_to_batch_shape) != 1 or row_to_batch_dtype != torch.int32:
        raise ValueError("row_to_batch must be int32 [rows]")
    rows = row_to_batch_shape[0]
    if row_offsets_shape != (rows + 1,) or row_offsets_dtype != torch.int32:
        raise ValueError("row_offsets must be int32 [rows + 1]")
    if len(block_table_shape) != 2 or block_table_dtype != torch.int32:
        raise ValueError("block_table must be a 2D int32 tensor")
    expected_output = (rows, topk)
    if out_values_shape != expected_output or out_values_dtype != torch.float32:
        raise ValueError("out_values must be float32 [rows, topk]")
    if out_raw_shape != expected_output or out_raw_dtype != torch.int32:
        raise ValueError("out_raw_indices must be int32 [rows, topk]")
    if out_physical_shape != expected_output or out_physical_dtype != torch.int32:
        raise ValueError("out_physical_indices must be int32 [rows, topk]")
    if out_counts_shape != (rows,) or out_counts_dtype != torch.int32:
        raise ValueError("out_counts must be int32 [rows]")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if wave_size != 64:
        raise ValueError("candidate TopK merge requires a wave64 GPU")


def flydsl_candidate_topk_merge(
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    candidate_counts: torch.Tensor,
    row_offsets: torch.Tensor,
    row_to_batch: torch.Tensor,
    block_table: torch.Tensor,
    out_values: torch.Tensor,
    out_raw_indices: torch.Tensor,
    out_physical_indices: torch.Tensor,
    out_counts: torch.Tensor,
    page_size: int,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Merge CTA candidate planes and map logical indices through a page table.

    The winning set is exact under ``score descending, raw index ascending``.
    NaNs sort below every numeric value and ``+0`` sorts above ``-0``. Candidate
    planes are grouped by row through ``row_offsets``; only the first
    ``candidate_counts[cta]`` entries of each plane are read.

    Output slots are compact but not score-sorted. Unused slots are written as
    ``(-inf, -1, -1)``. The op allocates no device memory and can be captured
    after its shape-specialized first-call compilation.
    """

    tensors = (
        candidate_values,
        candidate_indices,
        candidate_counts,
        row_offsets,
        row_to_batch,
        block_table,
        out_values,
        out_raw_indices,
        out_physical_indices,
        out_counts,
    )
    device = candidate_values.device
    if device.type != "cuda":
        raise ValueError("candidate tensors must be CUDA tensors")
    if any(t.device != device for t in tensors):
        raise ValueError("all candidate merge tensors must be on one device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all candidate merge tensors must be contiguous")

    _validate(
        tuple(candidate_values.shape),
        tuple(candidate_values.stride()),
        candidate_values.dtype,
        tuple(candidate_indices.shape),
        tuple(candidate_indices.stride()),
        candidate_indices.dtype,
        tuple(candidate_counts.shape),
        candidate_counts.dtype,
        tuple(row_offsets.shape),
        row_offsets.dtype,
        tuple(row_to_batch.shape),
        row_to_batch.dtype,
        tuple(block_table.shape),
        block_table.dtype,
        tuple(out_values.shape),
        out_values.dtype,
        tuple(out_raw_indices.shape),
        out_raw_indices.dtype,
        tuple(out_physical_indices.shape),
        out_physical_indices.dtype,
        tuple(out_counts.shape),
        out_counts.dtype,
        page_size,
        get_warp_size(torch.cuda.get_device_properties(device).gcnArchName),
    )

    topk = candidate_values.shape[1]
    launcher = _get_launcher(topk, page_size)
    if stream is None:
        stream = torch.cuda.current_stream(device)
    elif stream.device != device:
        raise ValueError("stream must belong to the candidate tensor device")
    _run_compiled(
        launcher,
        *tensors,
        row_to_batch.shape[0],
        stream,
    )
