# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _qsa_paged_mqa_logits_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    logits_ptr,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_cache_page,
    stride_cache_token,
    stride_cache_dim,
    stride_table_request,
    stride_table_page,
    stride_logits_token,
    num_tokens,
    num_columns,
    num_cache_pages,
    num_requests,
    score_divisor,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    """Score BF16 compressed QSA keys directly from a paged cache."""
    token = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)

    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + token)
    context_len = tl.load(
        context_lens_ptr + safe_request,
        mask=request_valid,
        other=0,
    )
    visible_groups = tl.maximum(
        0,
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            context_len // COMPRESS_RATIO,
        ),
    )
    if tl.program_id(1) == 0:
        tl.store(visible_groups_ptr + token, visible_groups)

    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    valid = (
        (token < num_tokens)
        & (columns < num_columns)
        & (columns < visible_groups)
        & request_valid
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    safe_logical_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    physical_page = tl.load(
        page_table_ptr
        + safe_request * stride_table_request
        + safe_logical_page * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid &= (physical_page >= 0) & (physical_page < num_cache_pages)
    safe_physical_page = tl.maximum(physical_page, 0).to(tl.int64)

    score = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for head in tl.static_range(0, NUM_HEADS):
        query = tl.load(
            q_ptr
            + token * stride_q_token
            + head * stride_q_head
            + dims * stride_q_dim,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_page
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        score += tl.maximum(tl.sum(keys * query[None, :], axis=1), 0.0)

    score /= score_divisor
    tl.store(
        logits_ptr + token * stride_logits_token + columns,
        tl.where(valid, score, -float("inf")),
        mask=(token < num_tokens) & (columns < num_columns),
    )
