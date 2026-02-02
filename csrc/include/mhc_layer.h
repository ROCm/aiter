#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_cooperative_groups.h>
#include "aiter_hip_common.h"

namespace cooperative_groups {
template <typename T>
struct plus {
    __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T, typename Op>
__device__ __forceinline__ T reduce(::cooperative_groups::thread_block_tile<32> tile, T val, Op op)
{
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset);
        val = op(val, other);
    }
    return val;
}
}  // namespace cooperative_groups

namespace mhc {

struct MHCConfig {
    int sinkhorn_iters;
    int nC;
    float eps;
    bool use_pdl;
};

struct RMSNormParams {
    int n;
    float eps;
};

template<int BLOCK_SIZE>
__global__ void float_to_bf16_kernel(__hip_bfloat16* __restrict__ out,
                                     const float* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (__hip_bfloat16)inp[idx];
    }
}

template<int BLOCK_SIZE>
__global__ void bf16_to_float_kernel(float* __restrict__ out,
                                     const __hip_bfloat16* __restrict__ inp,
                                     int size) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = (float)inp[idx];
    }
}

inline void float_to_bf16(__hip_bfloat16* out, const float* inp, int size,
                          hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float_to_bf16_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

inline void bf16_to_float(float* out, const __hip_bfloat16* inp, int size,
                          hipStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bf16_to_float_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

__device__ __forceinline__ __hip_bfloat162 mhc_floats2bfloat162(float x, float y) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
    __hip_bfloat162 out;
    out.x = __float2bfloat16(x);
    out.y = __float2bfloat16(y);
    return out;
#else
    return mhc_floats2bfloat162(x, y);
#endif
}

} // namespace mhc

namespace aiter {
void mhc_layer_fwd(torch::Tensor &out,
                   torch::Tensor &x_expanded,
                   torch::Tensor &rmsnorm_weight,
                   torch::Tensor &phi_pre,
                   torch::Tensor &phi_post,
                   torch::Tensor &phi_res,
                   torch::Tensor &b_pre,
                   torch::Tensor &b_post,
                   torch::Tensor &b_res,
                   double alpha_pre,
                   double alpha_post,
                   double alpha_res,
                   int64_t sinkhorn_iters,
                   double eps,
                   bool use_pdl);

void mhc_layer_fwd_debug(torch::Tensor &out,
                         torch::Tensor &x_expanded,
                         torch::Tensor &rmsnorm_weight,
                         torch::Tensor &phi_pre,
                         torch::Tensor &phi_post,
                         torch::Tensor &phi_res,
                         torch::Tensor &b_pre,
                         torch::Tensor &b_post,
                         torch::Tensor &b_res,
                         double alpha_pre,
                         double alpha_post,
                         double alpha_res,
                         int64_t sinkhorn_iters,
                         double eps,
                         torch::Tensor &H_proj_raw,
                         torch::Tensor &H_pre,
                         torch::Tensor &H_post,
                         torch::Tensor &M,
                         torch::Tensor &x_agg_bf16,
                         torch::Tensor &layer_out_bf16,
                         torch::Tensor &rms_values,
                         bool use_pdl);

} // namespace aiter
