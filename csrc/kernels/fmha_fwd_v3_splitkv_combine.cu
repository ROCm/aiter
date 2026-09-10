// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_bfloat16.h>

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

    const int token             = row / num_heads;
    const int head              = row - token * num_heads;
    const int64_t partial_plane = static_cast<int64_t>(row_count) * kValueDim;
    const int64_t lse_plane     = static_cast<int64_t>(num_heads) * seqlen_q;
    const int64_t lse_index     = static_cast<int64_t>(head) * seqlen_q + token;

    float weights[NumSplits];
    if(lane == 0)
    {
        float lse_max = partial_lse[lse_index];
#pragma unroll
        for(int split = 1; split < NumSplits; ++split)
            lse_max = fmaxf(lse_max, partial_lse[split * lse_plane + lse_index]);

        float denominator = 0.0f;
#pragma unroll
        for(int split = 0; split < NumSplits; ++split)
        {
            weights[split] = expf(partial_lse[split * lse_plane + lse_index] - lse_max);
            denominator += weights[split];
        }
        denominator = fmaxf(denominator, 1.0e-20f);
        const float inverse = 1.0f / denominator;
#pragma unroll
        for(int split = 0; split < NumSplits; ++split)
            weights[split] *= inverse;

        if(lse_out != nullptr)
            lse_out[lse_index] = lse_max + logf(denominator);
    }

#pragma unroll
    for(int split = 0; split < NumSplits; ++split)
        weights[split] = __shfl(weights[split], 0);

    const int64_t partial_index = static_cast<int64_t>(row) * kValueDim;
    const int64_t out_index = static_cast<int64_t>(token) * out_token_stride +
                              static_cast<int64_t>(head) * out_head_stride;
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

template <int NumSplits>
void launch_combine(const hip_bfloat16* partial_out,
                    const float* partial_lse,
                    hip_bfloat16* out,
                    float* lse,
                    int seqlen_q,
                    int num_heads,
                    int64_t out_token_stride,
                    int64_t out_head_stride,
                    dim3 grid,
                    hipStream_t stream)
{
    hipLaunchKernelGGL((fmha_fwd_v3_splitkv_combine_kernel<NumSplits>),
                       grid,
                       dim3(kCombineBlock),
                       0,
                       stream,
                       partial_out,
                       partial_lse,
                       out,
                       lse,
                       seqlen_q,
                       num_heads,
                       out_token_stride,
                       out_head_stride);
}

} // namespace

void launch_fmha_fwd_v3_splitkv_combine(const void* partial_out,
                                        const void* partial_lse,
                                        void* out,
                                        void* lse,
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
    const auto* partial_out_ptr = static_cast<const hip_bfloat16*>(partial_out);
    const auto* partial_lse_ptr = static_cast<const float*>(partial_lse);
    auto* out_ptr               = static_cast<hip_bfloat16*>(out);
    auto* lse_ptr               = static_cast<float*>(lse);

    using combine_launcher                        = decltype(&launch_combine<2>);
    static constexpr combine_launcher launchers[] = {
        nullptr,
        nullptr,
        &launch_combine<2>,
        &launch_combine<3>,
        &launch_combine<4>,
        &launch_combine<5>,
        &launch_combine<6>,
        &launch_combine<7>,
        &launch_combine<8>,
    };
    launchers[num_splits](partial_out_ptr,
                          partial_lse_ptr,
                          out_ptr,
                          lse_ptr,
                          seqlen_q,
                          num_heads,
                          out_token_stride,
                          out_head_stride,
                          grid,
                          stream);
    HIP_CALL(hipGetLastError());
}

} // namespace aiter
