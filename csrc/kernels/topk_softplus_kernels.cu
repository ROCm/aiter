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

// ---- Register-only kernel for moderate expert counts (256 / 384) ----
// Sorts each thread's partition in registers via an optimal sorting network,
// then extracts global top-K through a warp-level k-way merge (iterative
// argmax).  Shared memory and __syncthreads are completely eliminated.

#define _CAS_DESC(v, o, id, i, j)                                    \
    do                                                               \
    {                                                                \
        if((v)[i] < (v)[j])                                          \
        {                                                            \
            float _tv = (v)[i]; (v)[i] = (v)[j]; (v)[j] = _tv;      \
            float _to = (o)[i]; (o)[i] = (o)[j]; (o)[j] = _to;      \
            int _ti   = (id)[i]; (id)[i] = (id)[j]; (id)[j] = _ti;  \
        }                                                            \
    } while(0)

template <int N>
__device__ __forceinline__ void sort_network_desc(float* vals, float* orig, int* idxs)
{
    if constexpr(N == 4)
    {
        _CAS_DESC(vals, orig, idxs, 0, 1);
        _CAS_DESC(vals, orig, idxs, 2, 3);
        _CAS_DESC(vals, orig, idxs, 0, 2);
        _CAS_DESC(vals, orig, idxs, 1, 3);
        _CAS_DESC(vals, orig, idxs, 1, 2);
    }
    else if constexpr(N == 6)
    {
        _CAS_DESC(vals, orig, idxs, 0, 1);
        _CAS_DESC(vals, orig, idxs, 2, 3);
        _CAS_DESC(vals, orig, idxs, 4, 5);
        _CAS_DESC(vals, orig, idxs, 0, 2);
        _CAS_DESC(vals, orig, idxs, 1, 4);
        _CAS_DESC(vals, orig, idxs, 3, 5);
        _CAS_DESC(vals, orig, idxs, 0, 1);
        _CAS_DESC(vals, orig, idxs, 2, 3);
        _CAS_DESC(vals, orig, idxs, 4, 5);
        _CAS_DESC(vals, orig, idxs, 1, 2);
        _CAS_DESC(vals, orig, idxs, 3, 4);
        _CAS_DESC(vals, orig, idxs, 2, 3);
    }
    else
    {
#pragma unroll
        for(int i = 0; i < N - 1; i++)
        {
#pragma unroll
            for(int j = 0; j < N - 1 - i; j++)
            {
                _CAS_DESC(vals, orig, idxs, j, j + 1);
            }
        }
    }
}

#undef _CAS_DESC

template <typename DTYPE_I, int NUM_EXPERTS, int TOPK, bool need_renorm>
__global__ void topk_softplus_kernel_opt(
    const DTYPE_I* __restrict__ gating_output,    // [num_tokens, NUM_EXPERTS]
    const DTYPE_I* __restrict__ correction_bias,  // [NUM_EXPERTS] or nullptr
    float* __restrict__ topk_weights,             // [num_tokens, TOPK]
    int* __restrict__ topk_ids,                   // [num_tokens, TOPK]
    const size_t stride_tk,
    const int num_tokens,
    const float routed_scaling_factor)
{
    using cktype_i               = typename t2opus<DTYPE_I>::type;
    static constexpr int EPT     = NUM_EXPERTS / WARP_SIZE;
    static_assert(NUM_EXPERTS % WARP_SIZE == 0);

    const int token_idx = blockIdx.x;

    auto const* input_ptr = reinterpret_cast<cktype_i const*>(
        gating_output + token_idx * NUM_EXPERTS);

    float vals[EPT];
    float orig[EPT];
    int   idxs[EPT];

    // Step 1: load + softplus + bias — entirely in registers (strided access)
    // orig[] caches the unbiased score so the merge phase never re-reads
    // correction_bias from global memory (eliminates TOPK fp16→f32 conversions
    // on the critical path).  Sorted alongside vals[]/idxs[] so that all three
    // arrays share the same cursor index — the compiler emits one set of
    // compare instructions for the dynamic register-array selects.
#pragma unroll
    for(int i = 0; i < EPT; i++)
    {
        int   e     = threadIdx.x + i * static_cast<int>(WARP_SIZE);
        float x     = static_cast<float>(input_ptr[e]);
        float sp    = x > 20.0f ? x : log1pf(expf(x));
        float score = sqrtf(sp);
        orig[i]     = score;
        vals[i]     = score;
        idxs[i]     = e;
        if(correction_bias != nullptr)
        {
            vals[i] += static_cast<float>(
                reinterpret_cast<cktype_i const*>(correction_bias)[e]);
        }
    }

    // Step 2: sort thread-local partition descending (optimal sorting network)
    sort_network_desc<EPT>(vals, orig, idxs);

    // Step 3: warp-level k-way merge — iterative argmax across sorted lists
    // The winning lane is derived from the expert index layout
    // (e = threadIdx.x + i * WARP_SIZE  ⇒  lane = e & (WARP_SIZE-1)),
    // so we broadcast the pre-cached unbiased score via readlane instead of
    // loading + converting correction_bias[my_idx] each round.
    int   cursor = 0;
    float sum    = 0.0f;
    int   topk_indice;
    float topk_value;

#pragma unroll
    for(int k = 0; k < TOPK; ++k)
    {
        float my_val = (cursor < EPT) ? vals[cursor] : -INFINITY;
        int   my_idx = (cursor < EPT) ? idxs[cursor] : 0;

        warpReduceMax_softplus(my_val, my_idx);

        bool  i_won   = (cursor < EPT && idxs[cursor] == my_idx);
        float my_orig = i_won ? orig[cursor] : 0.0f;
        if(i_won)
            cursor++;

        int   win_lane = my_idx & (static_cast<int>(WARP_SIZE) - 1);
        float weight   = __builtin_bit_cast(
            float,
            __builtin_amdgcn_readlane(__builtin_bit_cast(int, my_orig), win_lane));

        topk_indice = (threadIdx.x == k) ? my_idx : topk_indice;
        topk_value  = (threadIdx.x == k) ? weight : topk_value;
        if constexpr(need_renorm)
        {
            sum += weight;
        }
    }

    // Step 4: apply renorm / route_scale and write
    if constexpr(need_renorm)
    {
        sum = routed_scaling_factor / fmaxf(sum, 1e-20f);
    }
    else
    {
        sum = routed_scaling_factor;
    }

    if(threadIdx.x < TOPK)
    {
        topk_weights[token_idx * stride_tk + threadIdx.x] = topk_value * sum;
        topk_ids[token_idx * stride_tk + threadIdx.x]     = topk_indice;
    }
}

// ---- Generic fallback kernel (shared-memory based) ----
// Uses shared memory to stage all expert scores, then runs topk sequential
// passes: each pass scans the full shared-memory array to find the next
// global maximum via warp reduce, marks the winner as -INF, and repeats.
// Works for any (num_experts, topk) combination but performs topk full scans
// of shared memory, which becomes the bottleneck for large topk values.
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
        sum = routed_scaling_factor / fmaxf(sum, 1e-20f);
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

#define LAUNCH_TOPK_SOFTPLUS_KERNEL_OPT(NUM_EXP, TK, need_renorm_val)                            \
    VLLM_DISPATCH_FLOATING_TYPES(gating_output.scalar_type(), "topk_softplus_kernel_opt", [&] {   \
        hipLaunchKernelGGL(                                                                       \
            (aiter::topk_softplus_kernel_opt<scalar_t, NUM_EXP, TK, need_renorm_val>),            \
            dim3(grid),                                                                           \
            dim3(block),                                                                          \
            0,                                                                                    \
            stream,                                                                               \
            gating_output.data_ptr<scalar_t>(),                                                   \
            has_bias ? correction_bias.data_ptr<scalar_t>() : nullptr,                            \
            topk_weights.data_ptr<float>(),                                                       \
            topk_indices.data_ptr<int>(),                                                         \
            stride_tk,                                                                            \
            num_tokens,                                                                           \
            routed_scaling_factor);                                                               \
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
    dim3 block(get_warp_size_func());

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    if(topk == 6 && (num_experts == 256 || num_experts == 384))
    {
        if(num_experts == 256)
        {
            if(need_renorm)
            {
                LAUNCH_TOPK_SOFTPLUS_KERNEL_OPT(256, 6, true)
            }
            else
            {
                LAUNCH_TOPK_SOFTPLUS_KERNEL_OPT(256, 6, false)
            }
        }
        else
        {
            if(need_renorm)
            {
                LAUNCH_TOPK_SOFTPLUS_KERNEL_OPT(384, 6, true)
            }
            else
            {
                LAUNCH_TOPK_SOFTPLUS_KERNEL_OPT(384, 6, false)
            }
        }
        return;
    }

    size_t shared_mem_size = num_experts * sizeof(float);

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
