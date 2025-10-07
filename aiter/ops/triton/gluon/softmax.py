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
    BLOCK_SIZE: gl.constexpr,
):
    row_start = gl.program_id(0)
    row_idx = row_start
    
    cols_per_thread: gl.constexpr = triton.cdiv(
        BLOCK_SIZE, gl.num_warps() * 64
    )
    blocked_cols: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[cols_per_thread],
        threads_per_warp=[64],
        warps_per_cta=[gl.num_warps()],
        order=[0],
    )
    
    # loop 1: find the max and sum of each row
    m = -float("inf")
    row_sum = 0.0
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # iterate through blocks of columns
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        # get column offsets from row starting pointer
        col_offsets = b + gl.arange(0, BLOCK_SIZE, layout=blocked_cols)
        
        # create mask to ensure in-bounds offsets
        mask = col_offsets < n_cols
        
        # load block of columns of row from global memory
        row_block = gl.amd.cdna4.buffer_load(
            ptr=row_start_ptr, offsets=col_offsets, mask=mask, other=-float("inf"), cache=".cg"
        )
        
        # find the max within the block
        m_p = gl.max(row_block, axis=0)
        
        # find new max among all blocks
        m_p = gl.maximum(m, m_p)
        
        # correct previous row sum
        row_sum = row_sum * gl.exp(m - m_p)
        
        # add new exponential to row sum
        row_sum += gl.sum(
            gl.exp(row_block - m_p),
            axis=0
        )
        
        # save the new max
        m = m_p
    
    
    # loop 2: divide each row by respective norms
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    for b in tl.range(0, n_cols, BLOCK_SIZE):
        col_offsets = b + gl.arange(0, BLOCK_SIZE, layout=blocked_cols)
        mask = col_offsets < n_cols
        row_block = gl.amd.cdna4.buffer_load(
            ptr=row_start_ptr, offsets=col_offsets, mask=mask, other=-float("inf"), cache=".cg"
        )
        
        # subtract, exponentiate and divide by sum
        softmax_output = gl.exp(row_block - m) / row_sum
        
        # store in output array
        output_ptrs = output_row_start_ptr + col_offsets
        gl.store(output_ptrs, softmax_output, mask=mask)


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
    num_stages = 2

    num_programs = n_rows

    grid = lambda meta: (num_programs,)  # noqa: E731
    _softmax_kernel_online[grid](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE,
        # waves_per_eu=waves_per_eu,
        num_warps=num_warps,
        # num_stages=num_stages,
    )

    return y

