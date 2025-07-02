#pragma once

/*
 * Copyright (C) 2024-2025, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pa_kernels.cuh"

#if defined(__HIPCC__) && \
    (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
  #define __HIP__GFX9__
#endif

#if defined(__HIP__GFX9__)

///////////////////////////////////////
// grid (num_seqs, num_partitions,num_kv_heads)
// block (256)
template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE,
          int HEAD_SIZE,
          int NUM_THREADS,
          bool ALIBI_ENABLED,
          int GQA_RATIO,
          int MTP,
          typename AttentionVariant>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,      // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache, // [num_blocks, block_size, num_kv_heads,
                                         // head_size]
    const cache_t* __restrict__ v_cache, // [num_blocks, block_size, num_kv_heads,
                                         // head_size]
    const float scale,
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_page_indices,   // [max_num_blocks]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const float* __restrict__ alibi_slopes,    // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int kv_seq_stride,
    float* __restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads,
                                    // max_num_partitions]
    scalar_t* __restrict__ out,     // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    float logits_soft_cap,
    float logits_soft_cap_rcp,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const AttentionVariant* variant)
{
    const int seq_idx       = blockIdx.x;
    const int partition_idx = blockIdx.y;

    constexpr int T_PAR_SIZE = 256; // token partition size set to 256

    int context_len;
    if constexpr(BLOCK_SIZE > 1)
    {
        context_len = (kv_indptr[seq_idx + 1] - kv_indptr[seq_idx] - 1) * BLOCK_SIZE +
                      kv_last_page_lens[seq_idx];
    }
    else
    {
        context_len = kv_indptr[seq_idx + 1] - kv_indptr[seq_idx];
    }

    const int partition_start_token_idx = partition_idx * T_PAR_SIZE; // partition_size;
    // exit if partition is out of context for seq
    if(partition_start_token_idx >= context_len)
    {
        return;
    }
    const int64_t query_loc = static_cast<int64_t>(seq_idx * MTP);
    const int* block_table_seq = kv_page_indices + kv_indptr[seq_idx];
    _paged_attention_kernel<scalar_t, cache_t, KV_DTYPE, BLOCK_SIZE, HEAD_SIZE, NUM_THREADS, ALIBI_ENABLED, GQA_RATIO, MTP, AttentionVariant>(block_table_seq, query_loc, context_len, partition_start_token_idx, q, k_cache, v_cache, scale, alibi_slopes, q_stride, kv_block_stride, kv_head_stride, kv_seq_stride, exp_sums, max_logits, out, logits_soft_cap, logits_soft_cap_rcp, k_scale_ptr, v_scale_ptr, variant);
}

// Grid: (num_heads, num_seqs, mtp).
template <typename scalar_t,
          typename OUTT,
          int HEAD_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE,
          int NPAR_LOOPS,
          bool ENABLE_LAST_PAGE_LENS>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                    // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,        // [num_seqs, num_heads,
                                               // max_num_partitions]
    const float* __restrict__ max_logits,      // [num_seqs, num_heads,
                                               // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,      // [num_seqs, num_heads,
                                               // max_num_partitions, head_size]
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const int block_size,
    const int max_num_partitions,
    const float* __restrict__ fp8_out_scale_ptr)
{
    const int num_heads = gridDim.x;
    const auto MTP = gridDim.z;
    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    int context_len;
    if constexpr(ENABLE_LAST_PAGE_LENS)
    {
        context_len = (kv_indptr[seq_idx + 1] - kv_indptr[seq_idx] - 1) * block_size +
                      kv_last_page_lens[seq_idx];
    }
    else
    {
        context_len = kv_indptr[seq_idx + 1] - kv_indptr[seq_idx];
    }
    const int64_t query_loc = static_cast<int64_t>(seq_idx * MTP);
    _paged_attention_ll4mi_reduce_kernel<scalar_t, OUTT, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE, NPAR_LOOPS>(query_loc, context_len, out, exp_sums, max_logits, tmp_out, max_num_partitions, fp8_out_scale_ptr);
}

#else // !defined(__HIP__MI3XX_MI250__) TODO: Add NAVI support


template <typename scalar_t,
          typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE,
          int HEAD_SIZE,
          int NUM_THREADS,
          bool ALIBI_ENABLED,
          int GQA_RATIO,
          int MTP,
          typename AttentionVariant>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,      // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache, // [num_blocks, block_size, num_kv_heads,
                                         // head_size]
    const cache_t* __restrict__ v_cache, // [num_blocks, block_size, num_kv_heads,
                                         // head_size]
    const float scale,
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_page_indices,   // [max_num_blocks]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const float* __restrict__ alibi_slopes,    // [num_heads]
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const int kv_seq_stride,
    float* __restrict__ exp_sums,   // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits, // [num_seqs, num_heads,
                                    // max_num_partitions]
    scalar_t* __restrict__ out,     // [num_seqs, num_heads, max_num_partitions,
                                    // head_size]
    float logits_soft_cap,
    float logits_soft_cap_rcp,
    const float* k_scale_ptr,
    const float* v_scale_ptr,
    const AttentionVariant* variant)
{
    UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t,
          typename OUTT,
          int HEAD_SIZE,
          int NUM_THREADS,
          int PARTITION_SIZE,
          int NPAR_LOOPS,
          bool ENABLE_LAST_PAGE_LENS>
__global__ __launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                    // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,        // [num_seqs, num_heads,
                                               // max_num_partitions]
    const float* __restrict__ max_logits,      // [num_seqs, num_heads,
                                               // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,      // [num_seqs, num_heads,
                                               // max_num_partitions, head_size]
    const int* __restrict__ kv_indptr,         // [num_seqs + 1]
    const int* __restrict__ kv_last_page_lens, // [num_seqs]
    const int block_size,
    const int max_num_partitions,
    const float* __restrict__ fp8_out_scale_ptr)
{
    UNREACHABLE_CODE
}

#endif // defined(__HIP__MI3XX_MI250__) TODO: Add NAVI support