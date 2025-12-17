# Copyright(C) [2025] Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

######################################## Imports ######################################## 
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
from typing import Optional
import sys
import os
import math

# Add aiter to path for imports
sys.path.insert(0, '/home/upandey/kernelgen/openevolve/TriVolve/aiter')

from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.types import get_fp8_dtypes

e5m2_type, e4m3_type = get_fp8_dtypes()
float8_info = torch.finfo(e4m3_type)

dtype_mapping = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
}
######################################## Imports ######################################## 


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.exp(Sdiv)
    p2 = tl.exp(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.autotune(
    configs=[
        # Small block sizes for decode scenarios
        triton.Config({'BLOCK_M': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=1),
        
        # Medium block sizes for balanced workloads
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 64}, num_warps=16, num_stages=1),
        
        # Large block sizes for high-throughput scenarios
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 128}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 256}, num_warps=16, num_stages=1),
        
        # Very large block sizes for maximum throughput
        triton.Config({'BLOCK_M': 512}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_M': 1024}, num_warps=16, num_stages=1),
        
        # Memory bandwidth optimized configs
        triton.Config({'BLOCK_M': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=16, num_stages=2),
        
        # Extreme throughput configs for very large problems
        triton.Config({'BLOCK_M': 512}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 1024}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 2048}, num_warps=16, num_stages=1),
        
        # Pipeline optimization configs
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256}, num_warps=16, num_stages=3),
    ],
    key=['BLOCK_SIZE', 'HEAD_SIZE', 'ALL_DECODE', 'num_query_heads'],
)
@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    out_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ALL_DECODE: tl.constexpr = False,
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    tl.assume(kv_head_idx >= 0)
    tl.assume(q_block_global_idx >= 0)
    tl.assume(block_table_stride > 0)
    tl.assume(query_stride_0 > 0)
    tl.assume(query_stride_1 > 0)
    tl.assume(output_stride_0 > 0)
    tl.assume(output_stride_1 > 0)
    tl.assume(stride_k_cache_0 > 0)
    tl.assume(stride_k_cache_1 > 0)
    tl.assume(stride_k_cache_2 > 0)
    tl.assume(stride_v_cache_0 > 0)
    tl.assume(stride_v_cache_1 > 0)
    tl.assume(stride_v_cache_2 > 0)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.load(
            sink_ptr + query_offset_1,
            mask=query_mask_1,
            other=float("-inf"),
        ).to(dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles (blocks) that need to be processed to
    # cover the longest sequence prefix (due to causal masking, blocks beyond
    # this prefix can be skipped)
    num_blocks = cdiv_fn(max_seq_prefix_len, BLOCK_SIZE)
    if SLIDING_WINDOW > 0:
        num_blocks_start = (
            max_seq_prefix_len - SLIDING_WINDOW - BLOCK_Q - 1
        ) // BLOCK_SIZE
        num_blocks_start = max(0, num_blocks_start)
    else:
        num_blocks_start = 0
    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    
    # Pre-load FP8 scale factors outside the loop to eliminate redundant memory accesses
    k_scale_val = tl.load(k_scale) if USE_FP8 else 1.0
    v_scale_val = tl.load(v_scale) if USE_FP8 else 1.0
    
    # Precompute base offsets for better memory coalescing
    offs_n = tl.arange(0, BLOCK_SIZE)
    base_v_offset = kv_head_idx * stride_v_cache_2 + offs_d[None, :] * stride_v_cache_3
    base_k_offset = kv_head_idx * stride_k_cache_2 + offs_d[:, None] * stride_k_cache_3
    
    # iterate through tiles with optimized memory access
    for j in range(num_blocks_start, num_blocks):
        # Prefetch next block index for better pipelining
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)
        
        # Compute offsets using precomputed bases for better coalescing
        block_base_v = physical_block_idx * stride_v_cache_0
        block_base_k = physical_block_idx * stride_k_cache_0
        
        v_offset = block_base_v + base_v_offset + offs_n[:, None] * stride_v_cache_1
        k_offset = block_base_k + base_k_offset + offs_n[None, :] * stride_k_cache_1

        # Pre-load FP8 scale factors outside loop to reduce memory traffic
        k_scale_val = tl.load(k_scale) if USE_FP8 else 1.0
        v_scale_val = tl.load(v_scale) if USE_FP8 else 1.0

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * k_scale_val).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * v_scale_val).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + offs_n

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, BLOCK_SIZE)
        S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))
        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, BLOCK_SIZE)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    # epilogue
    one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L
    if USE_FP8:
        acc = acc * tl.load(out_scale)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.autotune(
    configs=[
        # Small block sizes optimized for segmented processing
        triton.Config({'BLOCK_M': 16}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 16}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 32}, num_warps=8, num_stages=1),
        
        # Medium block sizes with aggressive warp scaling
        triton.Config({'BLOCK_M': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 64}, num_warps=16, num_stages=1),
        
        # Large block sizes for high throughput segments
        triton.Config({'BLOCK_M': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 128}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_M': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_M': 256}, num_warps=16, num_stages=1),
        
        # Very large block sizes for extreme throughput
        triton.Config({'BLOCK_M': 512}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_M': 1024}, num_warps=16, num_stages=1),
        
        # Memory-bound configs with deeper pipelining
        triton.Config({'BLOCK_M': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256}, num_warps=16, num_stages=2),
    ],
    key=['BLOCK_SIZE', 'HEAD_SIZE', 'NUM_SEGMENTS_PER_SEQ'],
)
@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    k_scale,  # float32
    v_scale,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    tl.assume(kv_head_idx >= 0)
    tl.assume(q_block_global_idx >= 0)
    tl.assume(segm_idx >= 0)

    tl.assume(block_table_stride > 0)
    tl.assume(query_stride_0 > 0)
    tl.assume(query_stride_1 > 0)
    tl.assume(stride_k_cache_0 > 0)
    tl.assume(stride_k_cache_1 > 0)
    tl.assume(stride_k_cache_2 > 0)
    tl.assume(stride_v_cache_0 > 0)
    tl.assume(stride_v_cache_1 > 0)
    tl.assume(stride_v_cache_2 > 0)

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

    if segm_idx * blocks_per_segment * BLOCK_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv

    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            M = tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    num_blocks = cdiv_fn(seq_len, BLOCK_SIZE)

    # iterate through tiles within current segment
    for j in range(
        segm_idx * blocks_per_segment,
        min((segm_idx + 1) * blocks_per_segment, num_blocks),
    ):
        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + j)

        offs_n = tl.arange(0, BLOCK_SIZE)

        v_offset = (
            physical_block_idx * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + offs_n[:, None] * stride_v_cache_1
        )

        k_offset = (
            physical_block_idx * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + offs_n[None, :] * stride_k_cache_1
        )

        # K : (HEAD_SIZE, BLOCK_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )

        if K_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                K = K_load
            else:
                K = (K_load.to(tl.float32) * k_scale_val).to(Q.dtype)
        else:
            K = K_load

        # V : (BLOCK_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :],
            other=0.0,
            cache_modifier=KV_cache_modifier,
        )

        if V_load.dtype.is_fp8():
            if Q.dtype.is_fp8():
                V = V_load
            else:
                V = (V_load.to(tl.float32) * v_scale_val).to(Q.dtype)
        else:
            V = V_load

        seq_offset = j * BLOCK_SIZE + offs_n

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, BLOCK_SIZE)
        S = tl.zeros(shape=(BLOCK_M, BLOCK_SIZE), dtype=tl.float32)

        S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            S += alibi_slope[:, None] * (seq_offset - context_len)

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            S += qq_bias

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))
        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, BLOCK_SIZE,)
        P = tl.exp(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.exp(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc += tl.dot(P.to(V.dtype), V)

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['NUM_SEGMENTS_PER_SEQ', 'HEAD_SIZE'],
)
@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    blocks_per_segment = cdiv_fn(seq_len, num_segments * BLOCK_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, blocks_per_segment * BLOCK_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = v.shape[1]
    assert (
        q.element_size() >= 2 or block_size >= 32
    ), "Block size must be at least 32 for fp8"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    SLIDING_WINDOW = 1 + window_size[0]
    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = 16
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    if BLOCK_Q == 0:
        BLOCK_M = triton.next_power_of_2(num_queries_per_kv)
        BLOCK_Q = BLOCK_M // num_queries_per_kv
    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs

    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
    target_num_prgms = get_num_sms() * 4
    num_2d_prgms = total_num_q_blocks * num_kv_heads
    ALL_DECODE = max_seqlen_q == 1

    # call 2d if sliding window is used
    if SLIDING_WINDOW > 0 or num_2d_prgms >= target_num_prgms or max_seqlen_k <= 1024:
        if ALL_DECODE == False:
            num_stages_2d = 4
            num_warps = 4
        else:
            num_stages_2d = 3
            num_warps = 2
        # make the block_m bigger if we already have enough parallelism
        if num_2d_prgms >= 2 * target_num_prgms:
            if num_2d_prgms <= 4 * target_num_prgms:
                BLOCK_M = 64
                num_stages_2d = 2 if SLIDING_WINDOW > 0 else 4
            elif num_2d_prgms <= 8 * target_num_prgms:
                BLOCK_M = 64
                num_stages_2d = 1 if SLIDING_WINDOW > 0 else 2
            else:
                BLOCK_M = 64
                num_stages_2d = 1
            BLOCK_Q = BLOCK_M // num_queries_per_kv
            total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

        if max_seqlen_q >= 512 and block_size == 64:
            BLOCK_M = 128
            num_stages_2d = 1
            num_warps = 4
            BLOCK_Q = BLOCK_M // num_queries_per_kv
            total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs
        kernel_unified_attention_2d[
            (
                num_kv_heads,
                total_num_q_blocks,
            )
        ](
            output_ptr=out,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            out_scale=1 / output_scale if output_scale is not None else 1.0,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=SLIDING_WINDOW,
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            USE_FP8=output_scale is not None,
            ALL_DECODE=ALL_DECODE,
            waves_per_eu=2,
        )
    else:
        NUM_SEGMENTS = math.ceil(target_num_prgms / num_2d_prgms)
        NUM_SEGMENTS = triton.next_power_of_2(NUM_SEGMENTS)
        NUM_SEGMENTS = min(NUM_SEGMENTS, 256)
        MIN_SEGMENTS = 16 if block_size <= 16 else 8
        NUM_SEGMENTS = max(NUM_SEGMENTS, MIN_SEGMENTS)
        segm_output = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            triton.next_power_of_2(head_size),
            dtype=torch.float32,
            device=q.device,
        )
        segm_max = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )
        segm_expsum = torch.empty(
            q.shape[0],
            num_query_heads,
            NUM_SEGMENTS,
            dtype=torch.float32,
            device=q.device,
        )

        kernel_unified_attention_3d[(total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)](
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            query_ptr=q,
            key_cache_ptr=k,
            value_cache_ptr=v,
            sink_ptr=sinks,
            block_tables_ptr=block_table,
            seq_lens_ptr=seqused_k,
            alibi_slopes_ptr=alibi_slopes,
            qq_bias_ptr=qq_bias,
            scale=softmax_scale,
            k_scale=k_descale,
            v_scale=v_descale,
            softcap=softcap,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            block_table_stride=block_table.stride(0),
            query_stride_0=q.stride(0),
            query_stride_1=q.stride(1),
            qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            USE_QQ_BIAS=use_qq_bias,
            USE_SOFTCAP=(softcap > 0),
            USE_SINKS=(sinks is not None),
            SLIDING_WINDOW=(1 + window_size[0]),
            stride_k_cache_0=k.stride(0),
            stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2),
            stride_k_cache_3=k.stride(3),
            stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1),
            stride_v_cache_2=v.stride(2),
            stride_v_cache_3=v.stride(3),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            num_seqs=num_seqs,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
            ALL_DECODE=ALL_DECODE,
            waves_per_eu=2,
        )

        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=segm_output,
            segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            BLOCK_SIZE=block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,
            USE_FP8=output_scale is not None,
            num_stages=1,
        )

##################################################################################################################################################  

import numpy as np
import random
from tb_eval.perf.ROCm.performance_utils_pytest import PytestBenchmarker, do_bench_config, save_all_benchmark_results
from typing import Dict
from aiter.ops.triton.utils.types import e4m3_dtype

result_gold = {}

######################################## HELPERS for Eval ######################################## 

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across multiple libraries and configure PyTorch for deterministic behavior.

    Args:
        seed (int): The seed value to set. Default is 42.
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch on CPU
    torch.manual_seed(seed)
    # Set seed for PyTorch on all GPUs (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list,
    kv_lens: list,
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    sinks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation for paged attention."""
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len, device=q.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        if sinks is not None:
            s_aux = sinks[:, None, None].repeat_interleave(attn.shape[-2], dim=-2)
            attn = torch.cat((attn, s_aux), dim=-1)
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        if sinks is not None:
            attn = attn[..., :-1]
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


def calculate_attention_tflops(params: Dict, ms: float) -> float:
    """Calculate TFLOPS for unified attention."""
    total_q_tokens = sum(params['query_lens'])
    total_kv_tokens = sum(params['kv_lens'])
    num_query_heads = params['num_query_heads']
    head_size = params['head_size']
    
    # Approximate FLOPs: Q@K^T + softmax + attn@V
    # For each query token attending to kv tokens
    flops = 2 * total_q_tokens * total_kv_tokens * num_query_heads * head_size
    tflops = flops / (ms / 1000) / 1e12
    return tflops


def calculate_attention_gbps(params: Dict, ms: float) -> float:
    """Calculate GB/s for unified attention."""
    total_q_tokens = sum(params['query_lens'])
    total_kv_tokens = sum(params['kv_lens'])
    num_query_heads = params['num_query_heads']
    num_kv_heads = params['num_kv_heads']
    head_size = params['head_size']
    dtype_str = params['dtype_str']
    dtype = dtype_mapping[dtype_str]
    
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    
    # Read Q, K, V, Write output
    total_bytes = (
        total_q_tokens * num_query_heads * head_size * bytes_per_element +  # Q
        total_kv_tokens * num_kv_heads * head_size * bytes_per_element +    # K
        total_kv_tokens * num_kv_heads * head_size * bytes_per_element +    # V
        total_q_tokens * num_query_heads * head_size * bytes_per_element    # Out
    )
    
    gbps = total_bytes / (ms / 1000) / 1e9
    return gbps

######################################## HELPERS for Eval ######################################## 


# Simplified test cases for faster execution
@pytest.mark.parametrize(
    'seq_lens,num_heads,head_size,block_size,dtype_str',
    [
        ([(1, 128), (5, 64)], (4, 4), 128, 16, 'bfloat16'),
        ([(1, 256)], (8, 2), 128, 64, 'float16'),
        ([(2, 128), (3, 256)], (4, 2), 128, 16, 'bfloat16'),
    ]
)
def test_unified_attention(seq_lens, num_heads, head_size, block_size, dtype_str, request):
    """Test correctness of unified attention kernel."""
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (-1, -1)  # No sliding window for basic tests
    scale = head_size**-0.5
    num_blocks = 512  # Small number for testing
    
    # Create inputs
    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device='cuda'
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda'
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device='cuda'
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device='cuda')
    
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device='cuda',
    )
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device='cuda')
    output = torch.empty_like(query)
    
    # Run kernel
    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
    )
    
    result_gold['_CALL_SUCCESS_'] = torch.tensor([[1.0]])
    
    # Save result
    test_case_name = request.node.name
    sanitized_key_name = test_case_name.replace("::", "_").replace("[", "_").replace("]", "").replace("-", "_")
    result_gold[sanitized_key_name] = output.clone().detach().cpu()
    
    # Reference implementation
    ref_output = ref_paged_attn(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        sliding_window=None,
        soft_cap=None,
        sinks=sinks,
    )
    
    # Check correctness
    torch.set_printoptions(profile='full')
    assert_close(output, ref_output, atol=1.5e-2, rtol=1e-2, check_dtype=False)


OP_NAME_FOR_BENCHMARK = "unified_attention_perf"


@pytest.mark.parametrize(
    'seq_lens,num_heads,head_size,block_size,dtype_str',
    [
        ([(1, 256)], (8, 2), 128, 64, 'bfloat16'),
        ([(2, 512), (3, 384)], (4, 2), 128, 16, 'bfloat16'),
    ]
)
def test_performance(seq_lens, num_heads, head_size, block_size, dtype_str, request):
    """Benchmark performance of unified attention kernel."""
    set_seed()
    
    dtype = dtype_mapping[dtype_str]
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    window_size = (-1, -1)
    scale = head_size**-0.5
    num_blocks = 1024
    
    # Create inputs
    query = torch.randn(
        sum(query_lens), num_query_heads, head_size, dtype=dtype, device='cuda'
    )
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device='cuda'
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device='cuda'
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device='cuda')
    
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32,
        device='cuda',
    )
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device='cuda')
    output = torch.empty_like(query)
    
    # Create op_lambda for benchmarking
    op_lambda = lambda: unified_attention(
        q=query, k=key_cache, v=value_cache, out=output,
        cu_seqlens_q=cu_query_lens, seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len, max_seqlen_k=max_kv_len,
        softmax_scale=scale, causal=True, window_size=window_size,
        block_table=block_tables, softcap=0,
        q_descale=None, k_descale=None, v_descale=None, sinks=sinks,
    )
    
    # Benchmarking
    bench_config = do_bench_config(warm_up=25, repetition=100)
    benchmarker = PytestBenchmarker(
        op_callable=op_lambda,
        op_name=OP_NAME_FOR_BENCHMARK,
        config=bench_config
    )
    
    current_params = {
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "block_size": block_size,
        "dtype_str": dtype_str,
    }
    
    benchmarker.run_benchmark(
        current_params_dict=current_params,
        gbps_calculator=calculate_attention_gbps,
        tflops_calculator=calculate_attention_tflops
    )


######################################## HELPERS for Eval ########################################     

def test_save_results():  
    """  
    Called after whole test run finished, right before returning the exit status to the system.  
    """
    print('Inside session finish...')
    if "_CALL_SUCCESS_" not in result_gold:
        result_gold['_CALL_SUCCESS_'] = torch.tensor([[0.0]])
    OUTPUT_FILENAME = __file__.replace('.','_') + '.pt'
    print(f"\nSaving all y_triton results to {OUTPUT_FILENAME}...")  
    # Ensure the directory for the output file exists if it's in a subdirectory  
    output_dir = os.path.dirname(OUTPUT_FILENAME)  
    if output_dir and not os.path.exists(output_dir):  
        os.makedirs(output_dir, exist_ok=True)  
    torch.save(result_gold, OUTPUT_FILENAME)       
    print(f"Successfully saved {len(result_gold)} y_triton tensors to {OUTPUT_FILENAME}.")  


def test_save_performance_results():
    """
    Called after the test_performance function finishes.
    This is a separate hook to ensure performance results are saved.
    """
    print('\nPytest session finishing... Saving benchmark results...')

    output_directory = os.path.join(os.path.dirname(__file__), "perf")  # Save in a "perf" subdirectory next to the test file
    os.makedirs(output_directory, exist_ok=True)
    
    save_all_benchmark_results(output_directory)
    print(f"All benchmark results attempted to save to: {output_directory}")


######################################## HELPERS for Eval ########################################