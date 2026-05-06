// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "dispatch_utils.h"
#include "hip_reduce.h"
#include "py_itfs_common.h"
#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <torch/all.h>

namespace aiter {

__inline__ __device__ void warpReduceMax_softplus(float& val_o, int& idx)
{
    using kvp = hipcub::KeyValuePair<int, float>;
    kvp thread_kvp;
    thread_kvp.key   = idx;
    thread_kvp.value = val_o;
    auto arg_max     = [](kvp a, kvp b) { return a.value > b.value ? a : b; };
    const kvp result_kvp =
        wave_reduce<kvp, decltype(arg_max), WARP_SIZE, false>(thread_kvp, arg_max);
    val_o = __builtin_bit_cast(
        float,
        __builtin_amdgcn_readlane(__builtin_bit_cast(int, result_kvp.value), WARP_SIZE - 1));
    idx = __builtin_bit_cast(
        int, __builtin_amdgcn_readlane(result_kvp.key, WARP_SIZE - 1));
}

template <typename DTYPE_I, typename f32vec, bool need_renorm>
__global__ void topk_softplus_kernel(
    const DTYPE_I* __restrict__ gating_output,    // [num_tokens, num_experts]
    const DTYPE_I* __restrict__ correction_bias,  // [num_experts] or nullptr
    float* __restrict__ topk_weights,             // [num_tokens, topk]
    int* __restrict__ topk_ids,                   // [num_tokens, topk]
    const size_t stride_tk,
    const int num_experts,
    const int topk,
    const int num_tokens,
    const float routed_scaling_factor)
{
    extern __shared__ char shared_mem[];
    const int token_idx = blockIdx.x;

    float* scores = reinterpret_cast<float*>(shared_mem);

    using cktype_i                = typename t2opus<DTYPE_I>::type;
    f32vec* scores_vec            = reinterpret_cast<f32vec*>(scores);
    static constexpr int vec_size = opus::vector_traits<f32vec>::size();
    using vec_i                   = opus::vector_t<cktype_i, vec_size>;
    const int num_experts_vec     = num_experts / vec_size;

    // Step 1: compute sqrt(softplus(x)) and optionally add bias for topk selection
    auto const* input_ptr = gating_output + token_idx * num_experts;
    for(int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
    {
        vec_i tmp = reinterpret_cast<vec_i const*>(input_ptr)[e];
        f32vec gating;
#pragma unroll
        for(size_t i = 0; i < vec_size; i++)
        {
            float x  = static_cast<float>(tmp[i]);
            // sqrt(softplus(x)) = sqrt(log1p(exp(x)))
            // For numerical stability: when x > 20, softplus(x) ≈ x
            float sp = x > 20.0f ? x : log1pf(expf(x));
            gating[i] = sqrtf(sp);
            if(correction_bias != nullptr)
            {
                int idx        = e * vec_size + i;
                float bias_val = static_cast<float>(
                    reinterpret_cast<cktype_i const*>(correction_bias)[idx]);
                gating[i] += bias_val;
            }
        }
        scores_vec[e] = gating;
    }
    // Handle remainder if num_experts is not divisible by vec_size
    for(int e = num_experts_vec * vec_size + threadIdx.x; e < num_experts; e += blockDim.x)
    {
        float x  = static_cast<float>(input_ptr[e]);
        float sp = x > 20.0f ? x : log1pf(expf(x));
        scores[e] = sqrtf(sp);
        if(correction_bias != nullptr)
        {
            scores[e] += static_cast<float>(
                reinterpret_cast<cktype_i const*>(correction_bias)[e]);
        }
    }
    __syncthreads();

    // Step 2: find topk
    float sum = 0.0f;
    int topk_indice;
    float topk_value;
    for(int k = 0; k < topk; ++k)
    {
        float max_val = -INFINITY;
        int max_idx   = k;

        for(int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
        {
            f32vec tmp = scores_vec[e];
#pragma unroll
            for(size_t i = 0; i < vec_size; i++)
            {
                if(tmp[i] > max_val)
                {
                    max_val = tmp[i];
                    max_idx = e * vec_size + i;
                }
            }
        }

        warpReduceMax_softplus(max_val, max_idx);

        {
            // Subtract bias to get original score as the routing weight
            if(correction_bias != nullptr)
            {
                max_val -= static_cast<float>(
                    reinterpret_cast<cktype_i const*>(correction_bias)[max_idx]);
            }
            scores[max_idx] = -INFINITY;
            topk_indice     = threadIdx.x == k ? max_idx : topk_indice;
            topk_value      = threadIdx.x == k ? max_val : topk_value;
            if(need_renorm)
            {
                sum += max_val;
            }
        }
    }

    // Step 3: apply renorm and route_scale
    if(need_renorm)
    {
        sum = routed_scaling_factor / sum;
    }
    else
    {
        sum = routed_scaling_factor;
    }

    for(int k = threadIdx.x; k < topk; k += blockDim.x)
    {
        topk_weights[token_idx * stride_tk + k] = topk_value * sum;
        topk_ids[token_idx * stride_tk + k]     = topk_indice;
    }
}

#define LAUNCH_TOPK_SOFTPLUS_KERNEL(VEC_F, need_renorm_val)                                     \
    VLLM_DISPATCH_FLOATING_TYPES(gating_output.scalar_type(), "topk_softplus_kernel", [&] {      \
        hipLaunchKernelGGL(                                                                      \
            (aiter::topk_softplus_kernel<scalar_t, VEC_F, need_renorm_val>),                     \
            dim3(grid),                                                                          \
            dim3(block),                                                                         \
            shared_mem_size,                                                                     \
            stream,                                                                              \
            gating_output.data_ptr<scalar_t>(),                                                  \
            has_bias ? correction_bias.data_ptr<scalar_t>() : nullptr,                           \
            topk_weights.data_ptr<float>(),                                                      \
            topk_indices.data_ptr<int>(),                                                            \
            stride_tk,                                                                           \
            num_experts,                                                                         \
            topk,                                                                                \
            num_tokens,                                                                          \
            routed_scaling_factor);                                                              \
    });

void topk_softplus(torch::Tensor& topk_weights,    // [num_tokens, topk]
                   torch::Tensor& topk_indices,     // [num_tokens, topk]
                   torch::Tensor& gating_output,    // [num_tokens, num_experts]
                   torch::Tensor& correction_bias,  // [num_experts]
                   bool need_renorm,
                   float routed_scaling_factor)
{
    int num_tokens  = gating_output.size(0);
    int num_experts = gating_output.size(1);
    int topk        = topk_indices.size(1);
    size_t stride_tk = topk_indices.stride(0);
    bool has_bias   = correction_bias.numel() > 0;

    dim3 grid(num_tokens);
    dim3 block(WARP_SIZE);
    size_t shared_mem_size = num_experts * sizeof(float);

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    switch(num_experts % 4)
    {
    case 0: {
        using vec4_type = opus::vector_t<float, 4>;
        if(need_renorm)
        {
            LAUNCH_TOPK_SOFTPLUS_KERNEL(vec4_type, true)
        }
        else
        {
            LAUNCH_TOPK_SOFTPLUS_KERNEL(vec4_type, false)
        }
        break;
    }
    case 2: {
        using vec2_type = opus::vector_t<float, 2>;
        if(need_renorm)
        {
            LAUNCH_TOPK_SOFTPLUS_KERNEL(vec2_type, true)
        }
        else
        {
            LAUNCH_TOPK_SOFTPLUS_KERNEL(vec2_type, false)
        }
        break;
    }
    default: {
        using vec1_type = opus::vector_t<float, 1>;
        if(need_renorm)
        {
            LAUNCH_TOPK_SOFTPLUS_KERNEL(vec1_type, true)
        }
        else
        {
            LAUNCH_TOPK_SOFTPLUS_KERNEL(vec1_type, false)
        }
        break;
    }
    }
}

} // namespace aiter
