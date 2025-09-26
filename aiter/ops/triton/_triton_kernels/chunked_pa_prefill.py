# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# The kernel in this file is adapted from the VLLM project:
# https://github.com/ROCm/vllm/blob/aiter_integration_final/vllm/attention/ops/chunked_prefill_paged_decode.py

# Authors:
#  - Burkhard Ringlein
#  - Jan van Lunteren
#  - Thomas Parnell


import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def _kernel_paged_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.constexpr,  # int
    query_stride_0: tl.constexpr,  # int
    query_stride_1: tl.constexpr,  # int, should be equal to head_size
    output_stride_0: tl.constexpr,  # int
    output_stride_1: tl.constexpr,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    x: tl.constexpr,  # int
    stride_k_cache_0: tl.constexpr,  # int
    stride_k_cache_1: tl.constexpr,  # int
    stride_k_cache_2: tl.constexpr,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_k_cache_4: tl.constexpr,  # int
    stride_v_cache_0: tl.constexpr,  # int
    stride_v_cache_1: tl.constexpr,  # int
    stride_v_cache_2: tl.constexpr,  # int
    stride_v_cache_3: tl.constexpr,  # int
    filter_by_query_len: tl.constexpr,  # bool
    query_start_len_ptr,  # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)
    kv_head_idx = query_head_idx // num_queries_per_kv

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0 + query_head_idx * query_stride_1
    )

    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # Q : (HEAD_SIZE,)
    Q = tl.load(
        query_ptr + query_offset + tl.arange(0, HEAD_SIZE_PADDED),
        mask=dim_mask,
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = tl.full([1], float("-inf"), dtype=tl.float32)
    L = tl.full([1], 1.0, dtype=tl.float32)
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_head_idx)

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles
    for j in range(0, num_blocks):

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)

        v_offset = (
            physical_block_idx * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[:, None] * stride_v_cache_2
            + offs_n[None, :] * stride_v_cache_3
        )

        k_offset = (
            physical_block_idx * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + offs_n[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None], other=0.0)

        if K_load.dtype.is_fp8():
            K = (K_load.to(tl.float32) * tl.load(k_scale)).to(Q.dtype)
        else:
            K = K_load

        # V : (HEAD_SIZE, BLOCK_SIZE)
        V_load = tl.load(value_cache_ptr + v_offset, mask=dim_mask[:, None], other=0.0)

        if V_load.dtype.is_fp8():
            V = (V_load.to(tl.float32) * tl.load(v_scale)).to(Q.dtype)
        else:
            V = V_load

        tmp = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        boundary = tl.full([BLOCK_SIZE], seq_len, dtype=tl.int32)
        mask_new = tmp < boundary
        # S : (BLOCK_SIZE,)
        S = tl.where(mask_new, 0.0, float("-inf")).to(tl.float32)
        S += scale * tl.sum(K * Q[:, None], axis=0)

        if SLIDING_WINDOW > 0:
            S = tl.where((seq_len - 1 - tmp) < SLIDING_WINDOW, S, -10000)

        if USE_ALIBI_SLOPES:
            S += alibi_slope * (tmp - seq_len + 1)

        # compute running maximum
        # m_j : (1,)
        m_j = tl.maximum(M, tl.max(S, axis=0))

        # P : (BLOCK_SIZE,)
        P = tl.exp(S - m_j)

        # l_j : (1,)
        l_j = tl.sum(P, axis=0)

        # alpha : (1, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_SIZE,)
        acc = acc * alpha

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_SIZE,)
        acc += tl.sum(V * P[None, :], axis=1)

    # epilogue
    acc = acc / L

    output_offset = (
        cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
    )

    tl.store(
        output_ptr + output_offset + tl.arange(0, HEAD_SIZE_PADDED), acc, mask=dim_mask
    )
