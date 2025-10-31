# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import functools
import json
import os
import torch
import triton
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _softmax_kernel_online(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    SIZE_PER_THREAD: gl.constexpr,
    THREADS_PER_WARP: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
):
    row_start = gl.program_id(0)
    row_idx = row_start

    gl.static_assert(
        SIZE_PER_THREAD <= triton.cdiv(BLOCK_SIZE, gl.num_warps() * THREADS_PER_WARP)
    )
    blocked_cols: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[SIZE_PER_THREAD],
        threads_per_warp=[THREADS_PER_WARP],
        warps_per_cta=[gl.num_warps()],
        order=[0],
    )
    col_offsets_range = gl.arange(0, BLOCK_SIZE, layout=blocked_cols)

    # loop 1: find the max and sum of each row
    m = -float("inf")
    row_sum = 0.0
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # prologue
    start_col_offsets = col_offsets_range
    start_mask = start_col_offsets < n_cols
    start_row_block = gl.amd.cdna4.buffer_load(
        ptr=row_start_ptr,
        offsets=start_col_offsets,
        mask=start_mask,
        other=-float("inf"),
        cache=".cg",
    )

    # iterate through blocks of columns
    col_offsets = start_col_offsets
    mask = start_mask
    row_block = start_row_block
    for b in tl.range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        # get column offsets from row starting pointer
        col_offsets = b + col_offsets_range

        # create mask to ensure in-bounds offsets
        mask = col_offsets < n_cols

        # load next block of columns of row from global memory
        next_row_block = gl.amd.cdna4.buffer_load(
            ptr=row_start_ptr,
            offsets=col_offsets,
            mask=mask,
            other=-float("inf"),
            cache=".cg",
        )

        # find the max within the block
        m_p = gl.max(row_block)

        # find new max among all blocks
        m_p = gl.maximum(m, m_p)

        # correct previous row sum
        row_sum = row_sum * gl.exp(m - m_p)

        # add new exponential to row sum
        row_sum += gl.sum(gl.exp(row_block - m_p))

        # save the new max and update block
        m = m_p
        row_block = next_row_block

    # epilogue
    m_p = gl.max(row_block)
    m_p = gl.maximum(m, m_p)
    row_sum = row_sum * gl.exp(m - m_p)
    row_sum += gl.sum(gl.exp(row_block - m_p))
    m = m_p

    # loop 2: divide each row by respective norms
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    col_offsets = start_col_offsets
    mask = start_mask
    row_block = start_row_block
    for b in tl.range(BLOCK_SIZE, n_cols, BLOCK_SIZE):
        next_col_offsets = b + gl.arange(0, BLOCK_SIZE, layout=blocked_cols)
        next_mask = col_offsets < n_cols
        next_row_block = gl.amd.cdna4.buffer_load(
            ptr=row_start_ptr,
            offsets=next_col_offsets,
            mask=next_mask,
            other=-float("inf"),
            cache=".cg",
        )

        # subtract, exponentiate and divide by sum
        softmax_output = gl.exp(row_block - m) / row_sum
        softmax_output = softmax_output.to(output_ptr.type.element_ty)

        # store in output array
        gl.amd.cdna4.buffer_store(
            stored_value=softmax_output,
            ptr=output_row_start_ptr,
            offsets=col_offsets,
            mask=mask,
            cache=".cg",
        )

        col_offsets = next_col_offsets
        mask = next_mask
        row_block = next_row_block
    softmax_output = gl.exp(row_block - m) / row_sum
    softmax_output = softmax_output.to(output_ptr.type.element_ty)
    gl.amd.cdna4.buffer_store(
        stored_value=softmax_output,
        ptr=output_row_start_ptr,
        offsets=col_offsets,
        mask=mask,
        cache=".cg",
    )


def softmax(x):
    """
    Computes the row-wise softmax of a 2D input tensor.

    Key parameters:
        x (torch.Tensor): A 2D input tensor.

    Returns:
        torch.Tensor: A tensor of the same shape as 'x', where softmax has been
        applied along the last dimension (row-wise).

    Note:
        - The input tensor 'x' must reside on the GPU.
    """
    _LOGGER.info(f"SOFTMAX: x={tuple(x.shape)}")
    n_rows, n_cols = x.shape

    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(n_cols))
    y = torch.empty_like(x)

    waves_per_eu = 2
    num_warps = 8

    buffer_size = 16
    threads_per_warp = 64

    # each thread should only be loading as many elements
    # as can fit in the load buffer (determined by element size)
    cols_per_thread = triton.cdiv(BLOCK_SIZE, num_warps * threads_per_warp)
    size_per_thread = min(cols_per_thread, buffer_size // x.element_size())

    num_programs = n_rows

    grid = lambda meta: (num_programs,)  # noqa: E731
    _softmax_kernel_online[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        size_per_thread,
        threads_per_warp,
        BLOCK_SIZE,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
    )

    return y
