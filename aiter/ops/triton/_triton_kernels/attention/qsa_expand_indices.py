# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _qsa_expand_block_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    context_lens_ptr,
    token_to_request_ptr,
    output_ptr,
    stride_blocks_token,
    stride_blocks_column,
    stride_output_token,
    stride_output_column,
    num_tokens,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    """Expand selected compressed groups and append the incomplete causal tail."""
    token_id = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    request = tl.load(token_to_request_ptr + token_id)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_position = tl.load(query_positions_ptr + token_id)
    context_len = tl.load(
        context_lens_ptr + safe_request,
        mask=request_valid,
        other=0,
    )
    complete_groups = tl.minimum(
        tl.maximum(
            0,
            tl.minimum(
                (query_position + 1) // COMPRESS_RATIO,
                context_len // COMPRESS_RATIO,
            ),
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_groups * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    selected_group = tl.load(
        block_indices_ptr
        + token_id * stride_blocks_token
        + tl.minimum(block_rank, BLOCK_TOPK - 1) * stride_blocks_column,
        mask=(token_id < num_tokens) & is_expanded,
        other=-1,
    )
    expanded_token = selected_group * COMPRESS_RATIO + offset

    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    logical_token = tl.where(
        is_expanded,
        expanded_token,
        tail_start + tail_offset,
    )
    valid = (
        (token_id < num_tokens)
        & (columns < OUTPUT_WIDTH)
        & request_valid
        & (is_expanded | is_tail)
        & (logical_token >= 0)
        & (logical_token < context_len)
    )
    tl.store(
        output_ptr
        + token_id * stride_output_token
        + columns * stride_output_column,
        tl.where(valid, logical_token, -1),
        mask=(token_id < num_tokens) & (columns < OUTPUT_WIDTH),
    )
