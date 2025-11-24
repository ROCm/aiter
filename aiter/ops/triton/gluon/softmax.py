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
from triton.experimental.gluon.language.amd.cdna4 import async_copy as acp


@gluon.jit
def _issue_loads(
    copy_idx,
    cols_smem,
    row_start_ptr,
    n_cols,
    layout: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    col_offsets = copy_idx * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = col_offsets < n_cols

    acp.buffer_load_to_shared(
        cols_smem.index(copy_idx % NUM_STAGES),
        row_start_ptr,
        col_offsets,
        mask,
        other=-float("inf"),
        cache_modifier=".cg",
    )
    # acp.global_load_to_shared(
    #     cols_smem.index(copy_idx % NUM_STAGES),
    #     row_start_ptr + col_offsets,
    #     mask=mask,
    #     other=-float("inf"),
    #     cache_modifier=".cg",
    # )
    acp.commit_group()
    return copy_idx + 1


@gluon.jit
def _perform_loop1(
    m, row_sum, read_idx, cols_smem, layout: gl.constexpr, NUM_STAGES: gl.constexpr
):
    row_block = cols_smem.index(read_idx % NUM_STAGES).load(layout)

    # find the max within the block
    m_p = gl.max(row_block, axis=0)

    # find new max among all blocks
    m_p = gl.maximum(m, m_p)

    # correct previous row sum
    row_sum = row_sum * gl.exp(m - m_p)

    # add new exponential to row sum
    row_sum += gl.sum(gl.exp(row_block - m_p), axis=0)

    # save the new max and update block
    m = m_p

    return m, row_sum, read_idx + 1


@gluon.jit
def _perform_loop2(
    m,
    row_sum,
    read_idx,
    cols_smem,
    output_row_start_ptr,
    n_cols,
    output_dtype,
    layout: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    col_offsets = read_idx * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = col_offsets < n_cols
    row_block = cols_smem.index(read_idx % NUM_STAGES).load(layout)

    # subtract, exponentiate and divide by sum
    softmax_output = gl.exp(row_block - m) / row_sum
    softmax_output = softmax_output.to(output_dtype)

    # store in output array
    gl.amd.cdna4.buffer_store(
        stored_value=softmax_output,
        ptr=output_row_start_ptr,
        offsets=col_offsets,
        mask=mask,
        cache=".cg",
    )

    return read_idx + 1


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
    NUM_STAGES: gl.constexpr,
):
    row_start = gl.program_id(0)
    row_idx = row_start

    blocked_cols: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[SIZE_PER_THREAD],
        threads_per_warp=[THREADS_PER_WARP],
        warps_per_cta=[gl.num_warps()],
        order=[0],
    )
    shared_cols: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0]
    )
    cols_smem = gl.allocate_shared_memory(
        input_ptr.type.element_ty, [NUM_STAGES, BLOCK_SIZE], layout=shared_cols
    )
    copy_idx = 0
    read_idx = 0

    # loop 1: find the max and sum of each row
    m = -float("inf")
    row_sum = 0.0
    row_start_ptr = input_ptr + row_idx * input_row_stride

    # prefill the pipeline
    for _ in gl.static_range(NUM_STAGES - 1):
        copy_idx = _issue_loads(
            copy_idx,
            cols_smem,
            row_start_ptr,
            n_cols,
            blocked_cols,
            BLOCK_SIZE,
            NUM_STAGES,
        )

    # steady state
    for _ in range(gl.cdiv(n_cols, BLOCK_SIZE) - (NUM_STAGES - 1)):
        # issue the overlapping copy
        copy_idx = _issue_loads(
            copy_idx,
            cols_smem,
            row_start_ptr,
            n_cols,
            blocked_cols,
            BLOCK_SIZE,
            NUM_STAGES,
        )

        # wait for a copy to finish before doing any computation
        acp.wait_group(NUM_STAGES - 1)
        m, row_sum, read_idx = _perform_loop1(
            m, row_sum, read_idx, cols_smem, blocked_cols, NUM_STAGES
        )

    # finish the pipeline
    for i in gl.static_range(NUM_STAGES - 1):
        acp.wait_group(NUM_STAGES - 2 - i)
        m, row_sum, read_idx = _perform_loop1(
            m, row_sum, read_idx, cols_smem, blocked_cols, NUM_STAGES
        )

    # loop 2: divide each row by respective norms, and then store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    copy_idx = 0
    read_idx = 0

    # prefill the pipeline
    for _ in gl.static_range(NUM_STAGES - 1):
        copy_idx = _issue_loads(
            copy_idx,
            cols_smem,
            row_start_ptr,
            n_cols,
            blocked_cols,
            BLOCK_SIZE,
            NUM_STAGES,
        )

    # steady state
    for _ in range(gl.cdiv(n_cols, BLOCK_SIZE) - (NUM_STAGES - 1)):
        # issue the overlapping copy
        copy_idx = _issue_loads(
            copy_idx,
            cols_smem,
            row_start_ptr,
            n_cols,
            blocked_cols,
            BLOCK_SIZE,
            NUM_STAGES,
        )

        # wait for a copy to finish before doing any computation
        acp.wait_group(NUM_STAGES - 1)
        read_idx = _perform_loop2(
            m,
            row_sum,
            read_idx,
            cols_smem,
            output_row_start_ptr,
            n_cols,
            output_ptr.type.element_ty,
            blocked_cols,
            BLOCK_SIZE,
            NUM_STAGES,
        )

    # finish the pipeline
    for i in gl.static_range(NUM_STAGES - 1):
        acp.wait_group(NUM_STAGES - 2 - i)
        read_idx = _perform_loop2(
            m,
            row_sum,
            read_idx,
            cols_smem,
            output_row_start_ptr,
            n_cols,
            output_ptr.type.element_ty,
            blocked_cols,
            BLOCK_SIZE,
            NUM_STAGES,
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
    num_stages = 2

    buffer_size_128 = 16
    buffer_size_32 = 4
    threads_per_warp = 64

    # Each thread should only be loading as many elements as can fit in the load buffer (determined by element size)
    # We also need to make sure that either size_per_thread * (x.element_size() * 8) == 128 or 32,
    # for async_copy load functions
    cols_per_thread = triton.cdiv(BLOCK_SIZE, num_warps * threads_per_warp)
    size_per_thread = min(cols_per_thread, buffer_size_128 // x.element_size())
    if (
        size_per_thread * x.element_size() != buffer_size_128
        and size_per_thread * x.element_size() != buffer_size_32
    ):
        size_per_thread = buffer_size_32 // x.element_size()

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
        num_stages,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
    )

    return y
