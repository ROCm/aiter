# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

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
    USE_ASYNC_COPY: gl.constexpr = True,
):
    col_offsets = copy_idx * BLOCK_SIZE + gl.arange(0, BLOCK_SIZE, layout=layout)
    mask = col_offsets < n_cols

    if USE_ASYNC_COPY:
        # acp.buffer_load_to_shared(
        #     cols_smem.index(copy_idx % NUM_STAGES),
        #     row_start_ptr,
        #     col_offsets,
        #     mask,
        #     other=-float("inf"),
        #     cache_modifier=".cg",
        # )
        acp.global_load_to_shared(
            cols_smem.index(copy_idx % NUM_STAGES),
            row_start_ptr + col_offsets,
            mask=mask,
            other=-float("inf"),
            cache_modifier=".cg",
        )
        acp.commit_group()
    else:
        cols_smem.index(copy_idx % NUM_STAGES).store(
            gl.amd.cdna4.buffer_load(
                ptr=row_start_ptr,
                offsets=col_offsets,
                mask=mask,
                other=-float("inf"),
                cache=".cg",
            )
        )
    return copy_idx + 1


@gluon.jit
def _perform_loop1(
    m, row_sum, read_idx, cols_smem, layout: gl.constexpr, NUM_STAGES: gl.constexpr
):
    row_block = cols_smem.index(read_idx % NUM_STAGES).load(layout)
    # row_block = acp.load_shared_relaxed(cols_smem.index(read_idx % NUM_STAGES), layout)

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
    # row_block = acp.load_shared_relaxed(cols_smem.index(read_idx % NUM_STAGES), layout)

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
    USE_ASYNC_COPY: gl.constexpr,
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
            USE_ASYNC_COPY,
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
            USE_ASYNC_COPY,
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
            USE_ASYNC_COPY,
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
            USE_ASYNC_COPY,
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
