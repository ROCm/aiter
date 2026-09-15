# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 Gluon kernel for BF16 QSA paged-MQA scoring."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan

_MAX_PROPAGATE_NAN_ALL = gl.constexpr(PropagateNan.ALL)


@gluon.jit
def _relu_f32(x):
    return gl.maximum(x, 0.0, propagate_nan=_MAX_PROPAGATE_NAN_ALL)


@gluon.jit
def _gluon_qsa_paged_mqa_logits_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    logits_ptr,
    stride_q_token,
    stride_q_head: gl.constexpr,
    stride_q_dim: gl.constexpr,
    stride_cache_page,
    stride_cache_token: gl.constexpr,
    stride_cache_dim: gl.constexpr,
    stride_table_request,
    stride_table_page: gl.constexpr,
    stride_logits_token,
    num_tokens,
    num_columns,
    num_cache_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: gl.constexpr,
    PAGE_TABLE_WIDTH: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    COMPRESS_RATIO: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
):
    """One program scores one query token against one compressed-key tile.

    BLOCK_H and BLOCK_D pad the logical MQA-head and head-dimension axes to MFMA
    shapes. Invalid rows/columns are zero-filled before the BF16 MFMA.
    """
    NUM_WARPS: gl.constexpr = gl.num_warps()
    WARP_SIZE: gl.constexpr = 64

    gl.static_assert(BLOCK_H >= NUM_HEADS, "BLOCK_H must cover NUM_HEADS")
    gl.static_assert(BLOCK_D >= HEAD_DIM, "BLOCK_D must cover HEAD_DIM")
    gl.static_assert(BLOCK_H % 16 == 0, "BLOCK_H must be a multiple of 16")
    gl.static_assert(BLOCK_N % 16 == 0, "BLOCK_N must be a multiple of 16")
    gl.static_assert(BLOCK_D % 16 == 0, "BLOCK_D must be a multiple of 16")

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    q_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=8
    )
    k_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=8
    )

    # Q is contiguous in D. K is represented as [D, N], so D is also the
    # contiguous axis for each independently gathered paged-cache row.
    d_threads: gl.constexpr = BLOCK_D // 16
    q_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[WARP_SIZE // d_threads, d_threads],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    k_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[d_threads, WARP_SIZE // d_threads],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    token = gl.program_id(0)
    column_block = gl.program_id(1)

    request = gl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = gl.minimum(gl.maximum(request, 0), num_requests - 1)
    query_position = gl.load(query_positions_ptr + token)
    context_len = gl.load(context_lens_ptr + safe_request, mask=request_valid, other=0)
    visible_groups = gl.maximum(
        0,
        gl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            context_len // COMPRESS_RATIO,
        ),
    )
    if column_block == 0:
        gl.store(visible_groups_ptr + token, visible_groups)

    q_heads = gl.arange(0, BLOCK_H, layout=gl.SliceLayout(1, q_layout))
    q_dims = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(0, q_layout))
    q_offsets = q_heads[:, None] * stride_q_head + q_dims[None, :] * stride_q_dim
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr + token * stride_q_token,
        offsets=q_offsets.to(gl.int32),
        mask=(q_heads[:, None] < NUM_HEADS) & (q_dims[None, :] < HEAD_DIM),
        other=0.0,
        cache=".cg",
    )
    q_dot = gl.convert_layout(q, q_dot_layout)

    k_dims = gl.arange(0, BLOCK_D, layout=gl.SliceLayout(1, k_layout))
    tile_columns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, k_layout))
    columns = column_block * BLOCK_N + tile_columns
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    column_valid = (
        (token < num_tokens)
        & (columns < num_columns)
        & (columns < visible_groups)
        & request_valid
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = gl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = gl.load(
        page_table_ptr
        + safe_request * stride_table_request
        + safe_logical_page * stride_table_page,
        mask=column_valid,
        other=-1,
    )
    page_valid = (physical_page >= 0) & (physical_page < num_cache_pages)
    column_valid = column_valid & page_valid
    safe_physical_page = gl.where(column_valid, physical_page, 0).to(gl.int32)

    cache_offsets = (
        k_dims[:, None] * stride_cache_dim
        + safe_physical_page[None, :] * stride_cache_page
        + page_offset[None, :] * stride_cache_token
    )
    keys = gl.amd.cdna4.buffer_load(
        ptr=k_cache_ptr,
        offsets=cache_offsets.to(gl.int32),
        mask=(k_dims[:, None] < HEAD_DIM) & column_valid[None, :],
        other=0.0,
        cache=".cg",
    )
    k_dot = gl.convert_layout(keys, k_dot_layout)

    scores = gl.amd.cdna4.mfma(
        q_dot,
        k_dot,
        gl.zeros([BLOCK_H, BLOCK_N], gl.float32, layout=mfma_layout),
    )
    scores = gl.sum(_relu_f32(scores), axis=0) / score_divisor

    score_columns = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout))
    score_valid = gl.convert_layout(column_valid, gl.SliceLayout(0, mfma_layout))
    gl.amd.cdna4.buffer_store(
        gl.where(score_valid, scores, -float("inf")),
        ptr=logits_ptr + token * stride_logits_token,
        offsets=(column_block * BLOCK_N + score_columns).to(gl.int32),
        mask=(column_block * BLOCK_N + score_columns) < num_columns,
    )
