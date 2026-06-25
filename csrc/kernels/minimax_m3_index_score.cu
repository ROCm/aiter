// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "minimax_m3_index_score.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include "aiter_dispatch.h"
#include "aiter_hip_common.h"
#include "aiter_stream.h"

namespace aiter {
namespace minimax_m3_index_score_ops {

constexpr int kHeadDim = 128;
constexpr int kWarpSize = 32;
constexpr int kGroupLanes = 16;
constexpr int kGroupsPerWarp = kWarpSize / kGroupLanes;
constexpr int kNumWarps = 8;
constexpr int kNumTokenGroups = kNumWarps * kGroupsPerWarp;
constexpr int kHeadsPerCta = 2;
constexpr int kElemsPerLane = kHeadDim / kGroupLanes;

__device__ __forceinline__ float groupReduceSum(float val)
{
#pragma unroll
    for(int mask = kGroupLanes / 2; mask > 0; mask >>= 1)
    {
        val += __shfl_xor(val, mask, kGroupLanes);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ void load8Vec(const scalar_t* __restrict__ src,
                                         float (&dst)[kElemsPerLane])
{
    static_assert(kElemsPerLane == 8);
    union Vec8
    {
        uint4 raw;
        scalar_t vals[kElemsPerLane];
    };
    Vec8 tmp;
    tmp.raw = *reinterpret_cast<const uint4*>(src);
#pragma unroll
    for(int i = 0; i < kElemsPerLane; ++i)
    {
        dst[i] = static_cast<float>(tmp.vals[i]);
    }
}

template <typename scalar_t>
__global__ void decodeIndexScoreKernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ index_kv_cache,
    float* __restrict__ score,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    int64_t total_q,
    int64_t num_idx_heads,
    int64_t init_blocks,
    int64_t local_blocks,
    float sm_scale_log2e,
    int64_t max_query_len,
    int64_t block_size,
    int64_t num_kv_chunks,
    int64_t stride_q_n,
    int64_t stride_q_h,
    int64_t stride_q_d,
    int64_t stride_k_blk,
    int64_t stride_k_pos,
    int64_t stride_k_d,
    int64_t stride_s_h,
    int64_t stride_s_n,
    int64_t stride_s_k,
    int64_t stride_bt_b)
{
    const int warp_id = threadIdx.x / kWarpSize;
    const int warp_lane = threadIdx.x & (kWarpSize - 1);
    const int group_in_warp = warp_lane / kGroupLanes;
    const int lane = warp_lane & (kGroupLanes - 1);
    const int token_group = warp_id * kGroupsPerWarp + group_in_warp;
    const int64_t pid_tc = blockIdx.x;
    const int64_t pid_h_base = blockIdx.y * kHeadsPerCta;
    const int64_t pid_t = pid_tc % total_q;
    const int64_t pid_c = pid_tc / total_q;
    const int64_t pid_b = pid_t / max_query_len;
    const int64_t tok = pid_t - pid_b * max_query_len;

    const int64_t seq_len = static_cast<int64_t>(seq_lens[pid_b]);
    const int64_t causal_len = seq_len - max_query_len + tok + 1;
    if(causal_len <= 0)
    {
        return;
    }

    const int64_t num_blocks = (causal_len + block_size - 1) / block_size;
    const int64_t chunk_size_blocks = (num_blocks + num_kv_chunks - 1) / num_kv_chunks;
    const int64_t chunk_start = pid_c * chunk_size_blocks;
    const int64_t chunk_limit = chunk_start + chunk_size_blocks;
    const int64_t chunk_end = chunk_limit < num_blocks ? chunk_limit : num_blocks;
    if(chunk_start >= chunk_end)
    {
        return;
    }

    const int64_t local_start = (num_blocks > local_blocks) ? (num_blocks - local_blocks) : 0;
    const int dim0 = lane * kElemsPerLane;
    const bool valid_head[kHeadsPerCta] = {pid_h_base < num_idx_heads,
                                           (pid_h_base + 1) < num_idx_heads};

    __shared__ float q_shared[kHeadsPerCta][kHeadDim];
    if(threadIdx.x < kGroupLanes)
    {
        for(int h = 0; h < kHeadsPerCta; ++h)
        {
            const int64_t pid_h = pid_h_base + h;
            if(valid_head[h])
            {
                float q_load[kElemsPerLane];
                load8Vec(q + pid_t * stride_q_n + pid_h * stride_q_h + dim0 * stride_q_d,
                         q_load);
#pragma unroll
                for(int i = 0; i < kElemsPerLane; ++i)
                {
                    q_shared[h][dim0 + i] = q_load[i];
                }
            }
        }
    }
    __syncthreads();

    float q_vals[kHeadsPerCta][kElemsPerLane];
#pragma unroll
    for(int h = 0; h < kHeadsPerCta; ++h)
    {
#pragma unroll
        for(int i = 0; i < kElemsPerLane; ++i)
        {
            q_vals[h][i] = valid_head[h] ? q_shared[h][dim0 + i] : 0.0f;
        }
    }

    const int32_t* __restrict__ bt_row = block_table + pid_b * stride_bt_b;
    __shared__ float wave_scores[kHeadsPerCta][kNumTokenGroups];
    for(int64_t blk = chunk_start; blk < chunk_end; ++blk)
    {
        const bool is_init = blk < init_blocks;
        const bool is_local = blk >= local_start && blk < num_blocks;
        if(is_init || is_local)
        {
            if(threadIdx.x < kHeadsPerCta)
            {
                const int64_t pid_h = pid_h_base + threadIdx.x;
                if(valid_head[threadIdx.x])
                {
                    score[pid_h * stride_s_h + pid_t * stride_s_n + blk * stride_s_k] =
                        is_local ? 1.0e29f : 1.0e30f;
                }
            }
            continue;
        }

        float wave_score[kHeadsPerCta];
#pragma unroll
        for(int h = 0; h < kHeadsPerCta; ++h)
        {
            wave_score[h] = -INFINITY;
        }

        const int64_t page = static_cast<int64_t>(bt_row[blk]);
        const scalar_t* __restrict__ k_blk =
            index_kv_cache + page * stride_k_blk + dim0 * stride_k_d;

        for(int64_t off = token_group; off < block_size; off += kNumTokenGroups)
        {
            const int64_t pos = blk * block_size + off;
            if(pos >= causal_len)
            {
                break;
            }
            float k_vals[kElemsPerLane];
            load8Vec(k_blk + off * stride_k_pos, k_vals);
#pragma unroll
            for(int h = 0; h < kHeadsPerCta; ++h)
            {
                float dot = 0.0f;
#pragma unroll
                for(int i = 0; i < kElemsPerLane; ++i)
                {
                    dot += q_vals[h][i] * k_vals[i];
                }
                dot = groupReduceSum(dot) * sm_scale_log2e;
                if(lane == 0 && valid_head[h])
                {
                    wave_score[h] = fmaxf(wave_score[h], dot);
                }
            }
        }

        if(lane == 0)
        {
#pragma unroll
            for(int h = 0; h < kHeadsPerCta; ++h)
            {
                wave_scores[h][token_group] = wave_score[h];
            }
        }
        __syncthreads();

        if(threadIdx.x < kHeadsPerCta)
        {
            const int h = threadIdx.x;
            const int64_t pid_h = pid_h_base + h;
            if(valid_head[h])
            {
                float block_score = wave_scores[h][0];
#pragma unroll
                for(int i = 1; i < kNumTokenGroups; ++i)
                {
                    block_score = fmaxf(block_score, wave_scores[h][i]);
                }
                score[pid_h * stride_s_h + pid_t * stride_s_n + blk * stride_s_k] =
                    block_score;
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t>
void launchDecodeIndexScore(const aiter_tensor_t& idx_q,
                            const aiter_tensor_t& index_kv_cache,
                            aiter_tensor_t& score,
                            const aiter_tensor_t& block_table,
                            const aiter_tensor_t& seq_lens,
                            int64_t total_q,
                            int64_t head_dim,
                            int64_t init_blocks,
                            int64_t local_blocks,
                            double sm_scale,
                            int64_t max_query_len,
                            int64_t block_size,
                            int64_t num_kv_chunks,
                            hipStream_t stream)
{
    AITER_CHECK(head_dim == kHeadDim, "MiniMax-M3 HIP index score expects head_dim=128");
    AITER_CHECK(block_size == 128, "MiniMax-M3 HIP index score expects block_size=128");
    AITER_CHECK(idx_q.stride(2) == 1, "idx_q last dimension must be contiguous");
    AITER_CHECK(index_kv_cache.stride(2) == 1,
                "index_kv_cache last dimension must be contiguous");
    AITER_CHECK(score.dtype() == AITER_DTYPE_fp32, "score must be float32");
    AITER_CHECK(block_table.dtype() == AITER_DTYPE_i32, "block_table must be int32");
    AITER_CHECK(seq_lens.dtype() == AITER_DTYPE_i32, "seq_lens must be int32");

    const int64_t num_idx_heads = idx_q.size(1);
    const dim3 grid(static_cast<unsigned int>(total_q * num_kv_chunks),
                    static_cast<unsigned int>((num_idx_heads + kHeadsPerCta - 1) /
                                              kHeadsPerCta));
    const dim3 block(kWarpSize * kNumWarps);
    const float sm_scale_log2e = static_cast<float>(sm_scale) * 1.4426950409f;

    hipLaunchKernelGGL((decodeIndexScoreKernel<scalar_t>),
                       grid,
                       block,
                       0,
                       stream,
                       reinterpret_cast<const scalar_t*>(idx_q.data_ptr()),
                       reinterpret_cast<const scalar_t*>(index_kv_cache.data_ptr()),
                       reinterpret_cast<float*>(score.data_ptr()),
                       reinterpret_cast<const int32_t*>(block_table.data_ptr()),
                       reinterpret_cast<const int32_t*>(seq_lens.data_ptr()),
                       total_q,
                       num_idx_heads,
                       init_blocks,
                       local_blocks,
                       sm_scale_log2e,
                       max_query_len,
                       block_size,
                       num_kv_chunks,
                       idx_q.stride(0),
                       idx_q.stride(1),
                       idx_q.stride(2),
                       index_kv_cache.stride(0),
                       index_kv_cache.stride(1),
                       index_kv_cache.stride(2),
                       score.stride(0),
                       score.stride(1),
                       score.stride(2),
                       block_table.stride(0));
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace minimax_m3_index_score_ops

void minimax_m3_decode_index_score(
    const aiter_tensor_t& idx_q,
    const aiter_tensor_t& index_kv_cache,
    aiter_tensor_t& score,
    const aiter_tensor_t& block_table,
    const aiter_tensor_t& seq_lens,
    int64_t total_q,
    int64_t head_dim,
    int64_t init_blocks,
    int64_t local_blocks,
    double sm_scale,
    int64_t max_query_len,
    int64_t block_size,
    int64_t num_kv_chunks)
{
    const hipStream_t stream = aiter::getCurrentHIPStream();
    AITER_DISPATCH_FLOATING16_TYPES_rmTorch(idx_q.dtype(), "minimax_m3_decode_index_score", [&] {
        minimax_m3_index_score_ops::launchDecodeIndexScore<scalar_t>(idx_q,
                                                                     index_kv_cache,
                                                                     score,
                                                                     block_table,
                                                                     seq_lens,
                                                                     total_q,
                                                                     head_dim,
                                                                     init_blocks,
                                                                     local_blocks,
                                                                     sm_scale,
                                                                     max_query_len,
                                                                     block_size,
                                                                     num_kv_chunks,
                                                                     stream);
    });
}

} // namespace aiter
