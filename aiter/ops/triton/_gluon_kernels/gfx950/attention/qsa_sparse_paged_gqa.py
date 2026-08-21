# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon gfx950 sparse grouped-query attention over paged BF16 K/V caches.

Logical token indices are translated through the request's page table before
the separate K and V gathers. One program computes one query-token/KV-head
group and maintains FP32 online-softmax state across the sparse selection.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _cache_load(ptr, offsets, USE_BUFFER_LOAD: gl.constexpr, mask=None, other=None):
    """Load a gathered cache tile, retaining a 64-bit fallback for large caches."""
    if USE_BUFFER_LOAD:
        return gl.amd.cdna4.buffer_load(
            ptr=ptr,
            offsets=offsets.to(gl.int32),
            mask=mask,
            other=other,
            cache=".cg",
        )
    return gl.load(
        ptr + offsets.to(gl.int64),
        mask=mask,
        other=other,
        cache_modifier=".cg",
    )


@gluon.jit
def _sparse_tile(
    q_dot,
    k_cache_ptr,
    v_cache_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token,
    request,
    start,
    kv_head,
    running_max,
    running_sum,
    accumulator,
    head_mask,
    qk_scale,
    k_smem,
    v_smem,
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
    num_tokens,
    num_cache_pages,
    num_requests,
    qk_layout: gl.constexpr,
    pv_layout: gl.constexpr,
    k_layout: gl.constexpr,
    v_layout: gl.constexpr,
    p_layout: gl.constexpr,
    gather_layout: gl.constexpr,
    column_layout: gl.constexpr,
    TOPK: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    PAGE_TABLE_WIDTH: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    """Gather one sparse K/V tile and update FP32 online-softmax state."""
    columns = start + gl.arange(0, BLOCK_N, layout=column_layout)
    in_selection = columns < TOPK
    logical_token = gl.load(
        logical_indices_ptr
        + token * stride_indices_token
        + columns * stride_indices_column,
        mask=in_selection,
        other=-1,
    )

    request_valid = (request >= 0) & (request < num_requests)
    safe_request = gl.minimum(gl.maximum(request, 0), num_requests - 1)
    safe_logical_token = gl.maximum(logical_token, 0)
    logical_page = safe_logical_token // PAGE_SIZE
    page_offset = safe_logical_token % PAGE_SIZE
    valid = (
        (token < num_tokens)
        & request_valid
        & in_selection
        & (logical_token >= 0)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = gl.load(
        block_table_ptr
        + safe_request * stride_table_request
        + gl.minimum(logical_page, PAGE_TABLE_WIDTH - 1) * stride_table_page,
        mask=valid,
        other=-1,
    )
    valid = valid & (physical_page >= 0) & (physical_page < num_cache_pages)
    safe_physical_page = gl.maximum(physical_page, 0)

    page_g = gl.convert_layout(
        safe_physical_page, gl.SliceLayout(1, gather_layout)
    )
    offset_g = gl.convert_layout(page_offset, gl.SliceLayout(1, gather_layout))
    valid_g = gl.convert_layout(valid, gl.SliceLayout(1, gather_layout))
    dim_offsets = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, gather_layout))

    k_offsets = (
        page_g[:, None] * stride_k_page
        + offset_g[:, None] * stride_k_token
        + kv_head * stride_k_head
        + dim_offsets[None, :] * stride_k_dim
    )
    v_offsets = (
        page_g[:, None] * stride_v_page
        + offset_g[:, None] * stride_v_token
        + kv_head * stride_v_head
        + dim_offsets[None, :] * stride_v_dim
    )
    keys = _cache_load(
        k_cache_ptr,
        k_offsets,
        USE_BUFFER_LOAD,
        mask=valid_g[:, None],
        other=0.0,
    )
    values = _cache_load(
        v_cache_ptr,
        v_offsets,
        USE_BUFFER_LOAD,
        mask=valid_g[:, None],
        other=0.0,
    )
    k_smem.store(keys)

    keys_dot = k_smem.permute([1, 0]).load(k_layout)
    scores = gl.amd.cdna4.mfma(
        q_dot,
        keys_dot,
        gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=qk_layout),
    )
    valid_qk = gl.convert_layout(valid, gl.SliceLayout(0, qk_layout))[None, :]
    head_mask_qk = gl.convert_layout(head_mask, gl.SliceLayout(1, qk_layout))[
        :, None
    ]
    scores = gl.where(valid_qk & head_mask_qk, scores, float("-inf"))

    tile_max = gl.max(scores, axis=1)
    next_max = gl.maximum(running_max, tile_max)
    safe_next_max = gl.where(next_max > float("-inf"), next_max, 0.0)
    safe_next_scaled = safe_next_max * qk_scale
    probabilities = gl.exp2(scores * qk_scale - safe_next_scaled[:, None])
    alpha = gl.where(
        running_max > float("-inf"),
        gl.exp2(running_max * qk_scale - safe_next_scaled),
        0.0,
    )
    next_sum = running_sum * alpha + gl.sum(probabilities, axis=1)

    v_smem.store(values)
    values_dot = v_smem.load(v_layout)
    probabilities_dot = gl.convert_layout(
        probabilities.to(gl.bfloat16), p_layout
    )
    alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, pv_layout))
    tile_accumulator = gl.amd.cdna4.mfma(
        probabilities_dot,
        values_dot,
        gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, layout=pv_layout),
    )
    accumulator = accumulator * alpha_pv[:, None] + tile_accumulator
    return next_max, next_sum, accumulator


_qsa_sparse_paged_gqa_repr = make_kernel_repr(
    "_qsa_sparse_paged_gqa",
    ["TOPK", "PAGE_SIZE", "NUM_KV_HEADS", "GROUP_SIZE", "HEAD_DIM", "BLOCK_N"],
)


@gluon.jit(repr=_qsa_sparse_paged_gqa_repr)
def _qsa_sparse_paged_gqa_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    logical_indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    output_ptr,
    stride_q_token: gl.constexpr,
    stride_q_head: gl.constexpr,
    stride_q_dim: gl.constexpr,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_k_dim: gl.constexpr,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_v_dim: gl.constexpr,
    stride_indices_token,
    stride_indices_column: gl.constexpr,
    stride_table_request,
    stride_table_page: gl.constexpr,
    stride_output_token: gl.constexpr,
    stride_output_head: gl.constexpr,
    stride_output_dim: gl.constexpr,
    num_tokens,
    num_cache_pages,
    num_requests,
    selection_width,
    softmax_scale: gl.constexpr,
    TOPK: gl.constexpr,
    PAGE_SIZE: gl.constexpr,
    PAGE_TABLE_WIDTH: gl.constexpr,
    NUM_KV_HEADS: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    HEAD_DIM: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_D: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    """One program = one (query token, KV head) grouped-attention result."""
    gl.static_assert(BLOCK_M >= GROUP_SIZE, "BLOCK_M must cover GROUP_SIZE")
    gl.static_assert(BLOCK_M % 16 == 0, "BLOCK_M must be a multiple of 16")
    gl.static_assert(BLOCK_D == HEAD_DIM, "BLOCK_D must equal HEAD_DIM")
    gl.static_assert(BLOCK_N == 16, "gfx950 sparse GQA currently requires BLOCK_N=16")
    gl.static_assert(HEAD_DIM % 16 == 0, "HEAD_DIM must be a multiple of 16")

    token = gl.program_id(0)
    kv_head = gl.program_id(1)
    request = gl.load(token_to_request_ptr + token)

    num_warps: gl.constexpr = gl.num_warps()
    qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_layout, 8)
    k_layout: gl.constexpr = gl.DotOperandLayout(1, qk_layout, 8)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_layout, 8)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_layout, 8)

    gather_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    column_layout: gl.constexpr = gl.SliceLayout(1, gather_layout)
    q_load_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[HEAD_DIM, 8]], [BLOCK_N, HEAD_DIM], [1, 0]
    )

    first_q_head = kv_head * GROUP_SIZE
    head_offsets = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, q_load_layout))
    dim_offsets = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, q_load_layout))
    head_mask = head_offsets < GROUP_SIZE
    q_offsets = (
        token * stride_q_token
        + (first_q_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :] * stride_q_dim
    )
    query = gl.amd.cdna4.buffer_load(
        ptr=q_ptr,
        offsets=q_offsets.to(gl.int32),
        mask=head_mask[:, None],
        other=0.0,
        cache=".cg",
    )
    q_dot = gl.convert_layout(query, q_layout)

    running_max = gl.full(
        [BLOCK_M], float("-inf"), gl.float32, layout=gl.SliceLayout(1, qk_layout)
    )
    running_sum = gl.zeros(
        [BLOCK_M], gl.float32, layout=gl.SliceLayout(1, qk_layout)
    )
    accumulator = gl.zeros([BLOCK_M, HEAD_DIM], gl.float32, layout=pv_layout)
    kv_smem = gl.allocate_shared_memory(
        gl.bfloat16, [BLOCK_N, HEAD_DIM], kv_shared_layout
    )

    qk_scale: gl.constexpr = softmax_scale * 1.4426950408889634
    for start in range(0, selection_width, BLOCK_N):
        running_max, running_sum, accumulator = _sparse_tile(
            q_dot,
            k_cache_ptr,
            v_cache_ptr,
            logical_indices_ptr,
            block_table_ptr,
            token,
            request,
            start,
            kv_head,
            running_max,
            running_sum,
            accumulator,
            head_mask,
            qk_scale,
            kv_smem,
            kv_smem,
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
            num_tokens,
            num_cache_pages,
            num_requests,
            qk_layout,
            pv_layout,
            k_layout,
            v_layout,
            p_layout,
            gather_layout,
            column_layout,
            TOPK,
            PAGE_SIZE,
            PAGE_TABLE_WIDTH,
            GROUP_SIZE,
            HEAD_DIM,
            BLOCK_M,
            BLOCK_N,
            USE_BUFFER_LOAD,
        )

    sum_pv = gl.convert_layout(running_sum, gl.SliceLayout(1, pv_layout))
    output = gl.where(
        sum_pv[:, None] > 0.0,
        accumulator / gl.maximum(sum_pv[:, None], 1.0e-20),
        0.0,
    )
    output_heads = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, pv_layout))
    output_dims = gl.arange(0, HEAD_DIM, layout=gl.SliceLayout(0, pv_layout))
    output_head_mask = output_heads < GROUP_SIZE
    output_offsets = (
        token * stride_output_token
        + (first_q_head + output_heads[:, None]) * stride_output_head
        + output_dims[None, :] * stride_output_dim
    )
    gl.amd.cdna4.buffer_store(
        output.to(output_ptr.dtype.element_ty),
        ptr=output_ptr,
        offsets=output_offsets.to(gl.int32),
        mask=output_head_mask[:, None],
    )
