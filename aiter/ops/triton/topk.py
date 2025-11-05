# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from FlagGems' topk:
# https://github.com/FlagOpen/FlagGems/blob/master/src/flag_gems/ops/topk.py

#  Top-K on GPU:  1-stage (tiny rows) + 2-stage (large rows) Triton kernels,
from __future__ import annotations
from typing import Tuple
import math
import torch
import triton
import triton.language as tl

# import triton.language.core as core
# from triton.language.standard import _log2, zeros_like
from aiter.ops.triton._triton_kernels.topk import (
    _topk_kernel,
    topk_stage1_kernel,
    topk_stage2_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _pick_block(m: int, k: int) -> int:
    blk = max(128, k)
    while blk < m and blk < 1024:
        blk <<= 1
    return blk


def one_stage_topk(
    x: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, M = x.shape
    BLOCK = _pick_block(M, k)
    if M > BLOCK or BLOCK > 1024:
        raise ValueError("row length too large for this kernel (<=1024)")

    out_v = torch.empty((B, k), device=x.device, dtype=x.dtype)
    out_i = torch.empty((B, k), device=x.device, dtype=torch.int64)

    _topk_kernel[(B,)](
        x.contiguous(),
        out_v,
        out_i,
        x.stride(0),
        out_v.stride(0),
        out_i.stride(0),
        M=M,
        K=k,
        BLOCK=BLOCK,
        FILL_VALUE=torch.finfo(torch.float32).min,
        num_warps=4,
        num_stages=2,
    )
    return out_v, out_i


def two_stage_topk(x, k, dim=-1, largest=True):
    descending = True
    if not largest:
        descending = False

    topk_elem_cnt = x.shape[dim]
    batch_size = math.prod(x.shape) // topk_elem_cnt

    if topk_elem_cnt < 1024:
        chunk_size = 256
    else:
        chunk_size = 1024

    if chunk_size < k:
        chunk_size = triton.next_power_of_2(k)

    chunk_num = triton.cdiv(topk_elem_cnt, chunk_size)

    stage1_out = torch.empty(batch_size * chunk_num * k, device=x.device, dtype=x.dtype)
    stage1_out_idx = torch.empty(
        batch_size * chunk_num * k, device=x.device, dtype=torch.int64
    )

    out_shape = x.shape[:-1] + (k,)
    stage2_out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    stage2_out_idx = torch.empty(out_shape, device=x.device, dtype=torch.int64)

    topk_stage1_kernel[
        batch_size,
        chunk_num,
    ](
        stage1_out,  # pointer to the output
        stage1_out_idx,  # pointer to the output
        x,  # pointer to the input
        k,
        topk_elem_cnt,
        chunk_size,
        descending,
        (
            torch.finfo(torch.float32).min
            if descending
            else torch.finfo(torch.float32).max
        ),
    )
    stage2_elem_cnt = chunk_num * k
    BLOCK_SIZE = triton.next_power_of_2(stage2_elem_cnt)

    (
        topk_stage2_kernel[batch_size,](
            stage2_out,
            stage2_out_idx,
            stage1_out,
            stage1_out_idx,
            k,
            stage2_elem_cnt,
            BLOCK_SIZE,
            descending,
            (
                torch.finfo(torch.float32).min
                if descending
                else torch.finfo(torch.float32).max
            ),
            torch.iinfo(torch.int32).min,
        )
        if descending
        else tl.constexpr(torch.iinfo(torch.int32).max)
    )

    return (stage2_out, stage2_out_idx)


# For dispatcher
MAX_TINY_ROW = 1024

"""
Triton Top-K operator
=========================================

Selects the "k" largest elements (and their indices) along the "last"
dimension of a 2-D input tensor.  A fast path and a hierarchical path are
chosen automatically based on the row length "M".

Algorithm selection
-------------------
- 1-stage kernel - used when M <= 1024 ("tiny" rows).
  Each row is processed by one Triton launch.
- 2-stage kernel - used when M > 1024 ("large" rows).
  The row is first tiled, each tile computes a local Top-K, and the partial
  results are merged in a second stage.

Interface & constraints
-----------------------
1. Only the last dimension can be reduced.
2. Input must be a 2-D tensor of shape (B, M).
3. Exactly k largest elements are returned.
4. Returned values are **sorted in descending order.

Returns
-------
(values, indices) - both tensors have shape (B, k) and reside on the
same device as the input.

"""


def topk(
    x: torch.Tensor,
    k: int,
    *,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    tiny_row_thresh: int = MAX_TINY_ROW,
):
    """
    Selects k largest elements along last dimension using 1-stage or 2-stage algorithm.

    Args:
        x (torch.Tensor): Input tensor with shape (B, M). Must be 2D.
        k (int): Number of top elements to select.
        dim (int): Dimension to reduce. Must be -1 (last dimension).
        largest (bool): Select largest elements. Must be True.
        sorted (bool): Return sorted results. Must be True.
        tiny_row_thresh (int): Threshold for choosing 1-stage vs 2-stage algorithm.

    Returns:
        tuple: (values, indices) both with shape (B, k), sorted in descending order.
    """
    _LOGGER.info(f"TOPK: x={tuple(x.shape)}, k={k}, largest={largest}, sorted={sorted}")
    if dim < 0:
        dim += x.ndim
    if dim != x.ndim - 1:
        raise ValueError("only last-dim Top-K is implemented")
    if x.ndim != 2:
        raise ValueError("input tensor must be 2-D (batch, M)")
    if not largest:
        raise ValueError("only largest=True supported")
    if not sorted:
        raise ValueError("sorted=False not supported")

    if not x.is_contiguous():
        x = x.contiguous()

    row_len = x.shape[-1]
    if row_len <= tiny_row_thresh:
        # if (row_len <= tiny_row_thresh) and (k <= 8):
        return one_stage_topk(x.view(-1, row_len), k)
    else:
        return two_stage_topk(x, k, dim=dim, largest=True)
