# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    output_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim,
    stride_indices_token,
    stride_indices_column,
    stride_table_request,
    stride_table_page,
    stride_output_token,
    stride_output_head,
    stride_output_dim,
    num_tokens,
    num_cache_pages,
    num_requests,
    softmax_scale,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Apply GQA over arbitrary logical tokens in separate paged BF16 K/V."""
    token = tl.program_id(0)
    kv_head = tl.program_id(1)
    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_q_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + token * stride_q_token
        + (first_q_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < GROUP_SIZE) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    running_max = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    for start in tl.range(0, TOPK, BLOCK_N):
        columns = start + column_offsets
        logical_token = tl.load(
            logical_indices_ptr
            + token * stride_indices_token
            + columns * stride_indices_column,
            mask=columns < TOPK,
            other=-1,
        )
        safe_logical_token = tl.maximum(logical_token, 0)
        logical_page = safe_logical_token // PAGE_SIZE
        page_offset = safe_logical_token % PAGE_SIZE
        valid = (
            (token < num_tokens)
            & request_valid
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
        safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)

        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[None, :] * stride_k_page
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None] * stride_k_dim,
            mask=(dim_offsets[:, None] < HEAD_DIM) & valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_physical_page[:, None] * stride_v_page
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )

        scores = tl.where(valid[None, :], tl.dot(query, keys), -1.0e20)
        next_max = tl.maximum(running_max, tl.max(scores, axis=1))
        alpha = tl.math.exp2(running_max - next_max)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_max[:, None]),
            0.0,
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = next_max

    output = tl.where(
        running_sum[:, None] > 0,
        accumulator / tl.maximum(running_sum[:, None], 1.0e-20),
        0.0,
    )
    tl.store(
        output_ptr
        + token * stride_output_token
        + (first_q_head + head_offsets[:, None]) * stride_output_head
        + dim_offsets[None, :] * stride_output_dim,
        output,
        mask=(token < num_tokens)
        & (head_offsets[:, None] < GROUP_SIZE)
        & (dim_offsets[None, :] < HEAD_DIM),
    )
