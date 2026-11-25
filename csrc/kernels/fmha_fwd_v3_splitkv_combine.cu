// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_bfloat16.h>

#include <type_traits>

#include "aiter_hip_common.h"
#include "mha_fwd.h"

namespace aiter {
namespace {

constexpr int kValueDim     = 128;
constexpr int kCombineBlock = 256;
constexpr int kWaveSize     = 64;
constexpr int kRowsPerBlock = kCombineBlock / kWaveSize;

template <int NumSplits>
__global__ __launch_bounds__(kCombineBlock) void fmha_fwd_v3_splitkv_combine_kernel(
    const hip_bfloat16* __restrict__ partial_out,
    const float* __restrict__ partial_lse,
    hip_bfloat16* __restrict__ out,
    float* __restrict__ lse_out,
    const int32_t* __restrict__ cu_seqlens_q,
    int seqlen_q,
    int num_heads,
    int64_t out_token_stride,
    int64_t out_head_stride)
{
    const int lane      = threadIdx.x & (kWaveSize - 1);
    const int wave      = threadIdx.x / kWaveSize;
    const int row       = blockIdx.x * kRowsPerBlock + wave;
    const int row_count = seqlen_q * num_heads;
    if(row >= row_count)
        return;

    const int token = row / num_heads;
    if(token < cu_seqlens_q[0] || token >= cu_seqlens_q[1])
        return;

    const int head              = row - token * num_heads;
    const int64_t partial_plane = static_cast<int64_t>(row_count) * kValueDim;
    const int64_t lse_plane     = static_cast<int64_t>(num_heads) * seqlen_q;
    const int64_t lse_index     = static_cast<int64_t>(head) * seqlen_q + token;

    float weights[NumSplits];
    int all_empty = 0;
    if(lane == 0)
    {
        float lse_max = partial_lse[lse_index];
#pragma unroll
        for(int split = 1; split < NumSplits; ++split)
            lse_max = fmaxf(lse_max, partial_lse[split * lse_plane + lse_index]);

        all_empty = lse_max == -INFINITY;
        if(all_empty)
        {
            if(lse_out != nullptr)
                lse_out[lse_index] = INFINITY;
        }
        else
        {
            float denominator = 0.0f;
#pragma unroll
            for(int split = 0; split < NumSplits; ++split)
            {
                weights[split] =
                    expf(partial_lse[split * lse_plane + lse_index] - lse_max);
                denominator += weights[split];
            }
            denominator         = fmaxf(denominator, 1.0e-20f);
            const float inverse = 1.0f / denominator;
#pragma unroll
            for(int split = 0; split < NumSplits; ++split)
                weights[split] *= inverse;

            if(lse_out != nullptr)
                lse_out[lse_index] = lse_max + logf(denominator);
        }
    }

    all_empty = __shfl(all_empty, 0);

    const int64_t partial_index = static_cast<int64_t>(row) * kValueDim;
    const int64_t out_index     = static_cast<int64_t>(token) * out_token_stride +
                              static_cast<int64_t>(head) * out_head_stride;
    if(all_empty)
    {
        for(int value = lane; value < kValueDim; value += kWaveSize)
            out[out_index + value] = hip_bfloat16(0.0f);
        return;
    }

#pragma unroll
    for(int split = 0; split < NumSplits; ++split)
        weights[split] = __shfl(weights[split], 0);

    for(int value = lane; value < kValueDim; value += kWaveSize)
    {
        float result = 0.0f;
#pragma unroll
        for(int split = 0; split < NumSplits; ++split)
        {
            result +=
                weights[split] *
                static_cast<float>(partial_out[split * partial_plane + partial_index + value]);
        }
        out[out_index + value] = hip_bfloat16(result);
    }
}

} // namespace

void launch_fmha_fwd_v3_splitkv_combine(const void* partial_out,
                                        const void* partial_lse,
                                        void* out,
                                        void* lse,
                                        const int32_t* cu_seqlens_q,
                                        int seqlen_q,
                                        int num_heads,
                                        int num_splits,
                                        int64_t out_token_stride,
                                        int64_t out_head_stride,
                                        hipStream_t stream)
{
    AITER_CHECK(num_splits >= kFmhaHd192SplitKvMinSplits &&
                    num_splits <= kFmhaHd192SplitKvMaxSplits,
                "unsupported fmha_fwd_v3_splitkv combine split count");

    const int row_count = seqlen_q * num_heads;
    const dim3 grid((row_count + kRowsPerBlock - 1) / kRowsPerBlock);
    const auto launch = [&](auto splits) {
        hipLaunchKernelGGL((fmha_fwd_v3_splitkv_combine_kernel<decltype(splits)::value>),
                           grid,
                           dim3(kCombineBlock),
                           0,
                           stream,
                           static_cast<const hip_bfloat16*>(partial_out),
                           static_cast<const float*>(partial_lse),
                           static_cast<hip_bfloat16*>(out),
                           static_cast<float*>(lse),
                           cu_seqlens_q,
                           seqlen_q,
                           num_heads,
                           out_token_stride,
                           out_head_stride);
    };
    switch(num_splits)
    {
    case 2: launch(std::integral_constant<int, 2>{}); break;
    case 3: launch(std::integral_constant<int, 3>{}); break;
    case 4: launch(std::integral_constant<int, 4>{}); break;
    case 5: launch(std::integral_constant<int, 5>{}); break;
    case 6: launch(std::integral_constant<int, 6>{}); break;
    case 7: launch(std::integral_constant<int, 7>{}); break;
    case 8: launch(std::integral_constant<int, 8>{}); break;
    }
    HIP_CALL(hipGetLastError());
}

} // namespace aiter
