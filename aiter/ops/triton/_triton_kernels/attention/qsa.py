# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA triton kernels for index scoring and sparse paged GQA."""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_qsa_paged_mqa_logits_repr = make_kernel_repr(
    "_qsa_paged_mqa_logits",
    ["NUM_HEADS", "HEAD_DIM", "COMPRESS_RATIO", "BLOCK_N", "BLOCK_D"],
)

_qsa_paged_mqa_logits_mfma_repr = make_kernel_repr(
    "_qsa_paged_mqa_logits_mfma",
    ["NUM_HEADS", "HEAD_DIM", "COMPRESS_RATIO", "BMT", "BN"],
)

_expand_selected_groups_repr = make_kernel_repr(
    "_expand_selected_groups",
    ["RATIO", "BLOCK"],
)

_qsa_sparse_paged_gqa_repr = make_kernel_repr(
    "_qsa_sparse_paged_gqa",
    [
        "TOPK",
        "NUM_KV_HEADS",
        "GROUP_SIZE",
        "HEAD_DIM",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_D",
        "KV_SPLITS",
    ],
)


@triton.jit(repr=_qsa_paged_mqa_logits_repr)
def _qsa_paged_mqa_logits_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    row_starts_ptr,
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
        if row_starts_ptr is not None:
            tl.store(row_starts_ptr + token, 0)
    q0 = tl.load(
        q_ptr + token * stride_q_token + dims * stride_q_dim,
        mask=dims < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + token * stride_q_token + stride_q_head + dims * stride_q_dim,
        mask=(dims < HEAD_DIM) & (NUM_HEADS > 1),
        other=0.0,
    ).to(tl.float32)
    q2 = tl.load(
        q_ptr + token * stride_q_token + 2 * stride_q_head + dims * stride_q_dim,
        mask=(dims < HEAD_DIM) & (NUM_HEADS > 2),
        other=0.0,
    ).to(tl.float32)
    q3 = tl.load(
        q_ptr + token * stride_q_token + 3 * stride_q_head + dims * stride_q_dim,
        mask=(dims < HEAD_DIM) & (NUM_HEADS > 3),
        other=0.0,
    ).to(tl.float32)
    tl.static_assert(NUM_HEADS <= 4)
    num_tiles = tl.cdiv(tl.minimum(visible_groups, num_columns), BLOCK_N)
    for tile in range(tl.program_id(1), num_tiles, tl.num_programs(1)):
        columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
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
        keys = tl.load(
            k_cache_ptr
            + safe_physical_page[:, None] * stride_cache_page
            + page_offset[:, None] * stride_cache_token
            + dims[None, :] * stride_cache_dim,
            mask=valid[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        score = tl.maximum(tl.sum(keys * q0[None, :], axis=1), 0.0)
        if NUM_HEADS > 1:
            score += tl.maximum(tl.sum(keys * q1[None, :], axis=1), 0.0)
        if NUM_HEADS > 2:
            score += tl.maximum(tl.sum(keys * q2[None, :], axis=1), 0.0)
        if NUM_HEADS > 3:
            score += tl.maximum(tl.sum(keys * q3[None, :], axis=1), 0.0)
        score /= score_divisor
        tl.store(
            logits_ptr + token * stride_logits_token + columns,
            tl.where(valid, score, -float("inf")),
            mask=(token < num_tokens) & (columns < num_columns),
        )


@triton.jit(repr=_qsa_paged_mqa_logits_mfma_repr)
def _qsa_paged_mqa_logits_mfma_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    token_to_request_ptr,
    query_positions_ptr,
    context_lens_ptr,
    visible_groups_ptr,
    row_starts_ptr,
    logits_ptr,
    stride_q_token,
    stride_q_head,
    stride_cache_page,
    stride_cache_token,
    stride_table_request,
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
    BMT: tl.constexpr,
    BN: tl.constexpr,
) -> None:
    """`_qsa_paged_mqa_logits_kernel` on matrix cores."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BMT + tl.arange(0, BMT)
    rvalid = rows < num_tokens
    request = tl.load(token_to_request_ptr + rows, mask=rvalid, other=-1)
    request_ok = rvalid & (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    position = tl.load(query_positions_ptr + rows, mask=rvalid, other=-1)
    context = tl.load(context_lens_ptr + safe_request, mask=request_ok, other=0)
    visible = tl.maximum(
        0,
        tl.minimum((position + 1) // COMPRESS_RATIO, context // COMPRESS_RATIO),
    )
    visible = tl.where(request_ok, visible, 0)
    if pid_n == 0:
        tl.store(visible_groups_ptr + rows, visible, mask=rvalid)
        if row_starts_ptr is not None:
            tl.store(row_starts_ptr + rows, 0, mask=rvalid)
    col0 = pid_n * BN
    if col0 >= tl.max(visible, axis=0):
        return

    dims = tl.arange(0, HEAD_DIM)
    q_rows = q_ptr + rows[:, None].to(tl.int64) * stride_q_token + dims[None, :]
    columns = col0 + tl.arange(0, BN)
    logical_page = columns // PAGE_SIZE
    page_offset = columns % PAGE_SIZE
    page_ok = logical_page < PAGE_TABLE_WIDTH
    safe_page = tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1)
    active = visible > 0
    req_lo = tl.min(tl.where(active, safe_request, num_requests), axis=0)
    req_hi = tl.max(tl.where(active, safe_request, -1), axis=0)
    score = tl.zeros((BMT, BN), dtype=tl.float32)
    for req in range(req_lo, req_hi + 1):
        physical = tl.load(
            page_table_ptr + req.to(tl.int64) * stride_table_request + safe_page,
            mask=page_ok,
            other=-1,
        )
        key_ok = page_ok & (physical >= 0) & (physical < num_cache_pages)
        keys_t = tl.load(
            k_cache_ptr
            + tl.maximum(physical, 0).to(tl.int64)[None, :] * stride_cache_page
            + page_offset[None, :] * stride_cache_token
            + dims[:, None],
            mask=key_ok[None, :],
            other=0.0,
        )
        s = tl.zeros((BMT, BN), dtype=tl.float32)
        for head in tl.static_range(NUM_HEADS):
            qh = tl.load(q_rows + head * stride_q_head, mask=rvalid[:, None], other=0.0)
            s += tl.maximum(tl.dot(qh, keys_t), 0.0)
        score = tl.where((safe_request == req)[:, None], s, score)
    score = score / score_divisor
    tl.store(
        logits_ptr
        + rows[:, None].to(tl.int64) * stride_logits_token
        + columns[None, :],
        tl.where(columns[None, :] < visible[:, None], score, -float("inf")),
        mask=rvalid[:, None] & (columns < num_columns)[None, :],
    )


@triton.jit(repr=_expand_selected_groups_repr)
def _expand_selected_groups(
    Groups,
    Context,
    Positions,
    Requests,
    Out,
    num_requests,
    n_groups,
    out_cols,
    group_s0,
    group_s1,
    out_s0,
    out_s1,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(Requests + row)
    valid_req = (req >= 0) & (req < num_requests)
    context = tl.load(Context + req, valid_req, 0)
    pos = tl.load(Positions + row)
    length = tl.where(valid_req & (pos >= 0) & (pos < context), pos + 1, 0)
    full_groups = length // RATIO
    tail_start = full_groups * RATIO
    history_end = tl.minimum(full_groups, n_groups) * RATIO
    is_history = cols < history_end
    group = tl.load(
        Groups + row * group_s0 + (cols // RATIO) * group_s1,
        (cols < out_cols) & is_history,
        -1,
    )
    history = tl.where(
        (group >= 0) & (group < full_groups), group * RATIO + cols % RATIO, -1
    )
    tail_offset = cols - history_end
    tail = tl.where(
        (tail_offset >= 0) & (tail_offset < length - tail_start),
        tail_start + tail_offset,
        -1,
    )
    tl.store(
        Out + row * out_s0 + cols * out_s1,
        tl.where(is_history, history, tail).to(tl.int32),
        cols < out_cols,
    )


@triton.jit(repr=_qsa_sparse_paged_gqa_repr)
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    output_ptr,
    partial_max_ptr,
    partial_sum_ptr,
    widths_ptr,
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
    KV_SPLITS: tl.constexpr = 1,
    SUB_GROUP: tl.constexpr = 0,
) -> None:
    """Apply GQA over arbitrary logical tokens in separate paged BF16 K/V."""
    token = tl.program_id(0)
    part = tl.program_id(2)
    ACTIVE_SUB: tl.constexpr = SUB_GROUP if SUB_GROUP > 0 else GROUP_SIZE
    NUM_HEAD_GROUPS: tl.constexpr = (GROUP_SIZE + ACTIVE_SUB - 1) // ACTIVE_SUB
    TAIL: tl.constexpr = GROUP_SIZE % ACTIVE_SUB
    kv_head = tl.program_id(1) // NUM_HEAD_GROUPS
    head_group = tl.program_id(1) % NUM_HEAD_GROUPS
    if TAIL != 0:
        valid_sub = tl.where(head_group < NUM_HEAD_GROUPS - 1, ACTIVE_SUB, TAIL)
    else:
        valid_sub = ACTIVE_SUB
    if KV_SPLITS > 1:  # noqa: SIM102
        if tl.program_id(1) == 0 and part == 0:
            tl.store(widths_ptr + token, TOPK)
    request = tl.load(token_to_request_ptr + token)
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_q_head = kv_head * GROUP_SIZE + head_group * ACTIVE_SUB
    query = tl.load(
        q_ptr
        + token * stride_q_token
        + (first_q_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim,
        mask=(head_offsets[:, None] < valid_sub) & (dim_offsets[None, :] < HEAD_DIM),
        other=0.0,
    )
    query = (query * softmax_scale * 1.4426950408889634).to(query.dtype)

    running_max = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    column_offsets = tl.arange(0, BLOCK_N)

    partition_size = tl.cdiv(TOPK, KV_SPLITS * BLOCK_N) * BLOCK_N
    if part * partition_size >= TOPK:
        return
    for start in tl.range(
        part * partition_size, tl.minimum((part + 1) * partition_size, TOPK), BLOCK_N
    ):
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
            cache_modifier=".cg",
        )
        values = tl.load(
            v_cache_ptr
            + safe_physical_page[:, None] * stride_v_page
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim,
            mask=valid[:, None] & (dim_offsets[None, :] < HEAD_DIM),
            other=0.0,
            cache_modifier=".cg",
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

    if KV_SPLITS > 1:
        partial_offset = (
            token * NUM_KV_HEADS * GROUP_SIZE + first_q_head + head_offsets
        ) * KV_SPLITS + part
        tl.store(
            partial_max_ptr + partial_offset, running_max, head_offsets < valid_sub
        )
        tl.store(
            partial_sum_ptr + partial_offset, running_sum, head_offsets < valid_sub
        )
        output_ptr += part * HEAD_DIM
        output = accumulator
    else:
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
        & (head_offsets[:, None] < valid_sub)
        & (dim_offsets[None, :] < HEAD_DIM),
    )
