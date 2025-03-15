// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
/*
 * @Script: topk_softmax_kernels_group.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2025-03-01 12:16:14
 * @Last Modified By: valarLip
 * @Last Modified At: 2025-03-04 22:27:28
 * @Description: This is description.
 */

#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "dispatch_utils.h"

#define WARP_SIZE 64
namespace aiter
{
    template <typename T, typename F>
    __device__ constexpr T wave_reduce(T local, F reduce_f)
    {
      constexpr int reduce_stage = 6; // 1<<6=64
      T v_local = local;
#pragma unroll
      for (int i_stage = 0; i_stage < reduce_stage; i_stage++)
      {
        int src_lane = __lane_id() ^ (1 << i_stage);
        int32_t v_remote_tmp =
            __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, v_local));
        T v_remote = __builtin_bit_cast(T, v_remote_tmp);
        v_local = reduce_f(v_local, v_remote);
      }
      return v_local;
    }

    __inline__ __device__ void warpReduceMax(float &val, int &idx)
    {
        static_assert(64 == WARP_SIZE, "WARP_SIZE == 64");
#pragma unroll
        for (int i = 0; i < 6; i++)
        {
            int offset = 1 << i;
            float tmp_val = __shfl_down(val, offset);
            int tmp_idx = __shfl_down(idx, offset);
            if (tmp_val > val)
            {
                val = tmp_val;
                idx = tmp_idx;
            }
        }
    }

    __device__ void blockReduceMax(float &val, int &idx)
    {
        __shared__ float shared_vals[32];
        __shared__ int shared_idxs[32];

        int lane = threadIdx.x % WARP_SIZE;
        int wid = threadIdx.x / WARP_SIZE;

        warpReduceMax(val, idx);

        if (lane == 0)
        {
            shared_vals[wid] = val;
            shared_idxs[wid] = idx;
        }
        __syncthreads();

        if (wid == 0)
        {
            val = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_vals[lane] : -INFINITY;
            idx = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_idxs[lane] : -1;

            warpReduceMax(val, idx);
        }
        __syncthreads();
    }

    template <typename DTYPE_I, typename fvec, int NUM_GRP, bool need_renorm, bool isBiased, bool isSoftmax>
    __global__ void grouped_topk_kernel(
        const DTYPE_I *__restrict__ gating_output,   // [num_tokens, hidden_size]
        const DTYPE_I *__restrict__ correction_bias, // [num_expert]
        float *__restrict__ topk_weights,            // [num_tokens, topk]
        int *__restrict__ topk_ids,                  // [num_tokens, topk]
        const size_t stride_tk,
        const int num_experts,
        const int topk,
        const int topk_group,
        const int num_tokens,
        const float routed_scaling_factor)
    {
        static_assert(NUM_GRP <= WARP_SIZE, "NUM_GRP must be <= WARP_SIZE");
        // 256 E, 8->4 group, 32 e/group
        const int experts_per_group = num_experts / NUM_GRP;
        extern __shared__ char shared_mem[];
        const int token_idx = blockIdx.x;

        char *ptr = (char *)(((size_t)shared_mem + 255) & ~255);
        float *scores = reinterpret_cast<float *>(ptr);
        ptr += num_experts * sizeof(float);

        float *group_scores = reinterpret_cast<float *>(ptr);
        ptr += NUM_GRP * sizeof(float);

        bool *group_mask = reinterpret_cast<bool *>(ptr);
        ptr += NUM_GRP * sizeof(bool);

        int *topk_indices = reinterpret_cast<int *>(ptr);
        ptr += topk * sizeof(int);

        float *topk_values = reinterpret_cast<float *>(ptr);
        // ptr += topk * sizeof(float);

        // int *topk_indices_f = reinterpret_cast<int *>(ptr);
        // ptr += topk * sizeof(int);

        // float *topk_values_f = reinterpret_cast<float *>(ptr);

        fvec *scores_vec = reinterpret_cast<fvec *>(scores);
        constexpr uint32_t vec_size = sizeof(fvec) / sizeof(float);

        if constexpr (!isSoftmax)
        {
            for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            {
                float gating = static_cast<float>(gating_output[token_idx * num_experts + e]);
                gating = 1.0f / (1.0f + expf(-gating));
                if constexpr (isBiased)
                {
                    gating += correction_bias[e];
                }
                scores[e] = gating;
            }
            __syncthreads();
        }
        else
        {
            __shared__ float sdata;
            float max_val = -INFINITY;
            for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            {

                float gating = gating_output[token_idx * num_experts + e];
                scores[e] = gating;
                if (gating > max_val)
                {
                    max_val = gating;
                }
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 6; i++)
            {
                int offset = 1 << i;
                float tmp_val = __shfl_down(max_val, offset);
                if (tmp_val > max_val)
                {
                    max_val = tmp_val;
                }
            }
            if (threadIdx.x == 0)
            {
                sdata = max_val;
            }
            __syncthreads();
            max_val = sdata;
            float thread_sum = 0.0;
            for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            {
                scores[e] = expf(scores[e] - max_val);
                thread_sum += scores[e];
            }
            __syncthreads();
            thread_sum = wave_reduce(thread_sum, [](float a, float b) { return a + b; });
            for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            {
                scores[e] /= thread_sum;
            }
            __syncthreads();
        }

        if constexpr (isBiased)
        {
#pragma unroll
            for (int g = threadIdx.x; g < NUM_GRP; g += blockDim.x)
            {
                float max1 = -INFINITY, max2 = -INFINITY;
                const int start = g * experts_per_group;
                const int end = start + experts_per_group;

                for (int e = start; e < end; ++e)
                {
                    if (scores[e] > max1)
                    {
                        max2 = max1;
                        max1 = scores[e];
                    }
                    else if (scores[e] > max2)
                    {
                        max2 = scores[e];
                    }
                }
                group_scores[g] = max1 + max2;
                group_mask[g] = false;
            }
            __syncthreads();
        }
        else
        {
#pragma unroll
            for (int g = threadIdx.x; g < NUM_GRP; g += blockDim.x)
            {
                float max1 = -INFINITY;
                const int start = g * experts_per_group;
                const int end = start + experts_per_group;
                for (int e = start; e < end; ++e)
                {
                    if (scores[e] > max1)
                    {
                        max1 = scores[e];
                    }
                }
                group_scores[g] = max1;
                group_mask[g] = false;
            }
            __syncthreads();
        }

        for (int k = 0; k < topk_group; k++)
        {
            float max_val = -INFINITY;
            int max_idx = -1;
#pragma unroll
            for (int g = threadIdx.x; g < NUM_GRP; g += blockDim.x)
            {
                if (group_scores[g] > max_val)
                {
                    max_val = group_scores[g];
                    max_idx = g;
                }
            }
            warpReduceMax(max_val, max_idx);
            if (threadIdx.x == 0 && max_idx != -1)
            {
                group_mask[max_idx] = true;
                group_scores[max_idx] = -INFINITY;
            }
            __syncthreads();
        }

        // lip: TODO we can do vec here if experts_per_group%vec_size==0
        for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        {
            int group_idx = e / experts_per_group;
            if (!group_mask[group_idx])
            {
                scores[e] = -INFINITY;
            }
        }
        __syncthreads();

        float sum = 0.0f;
        for (int k = 0; k < topk; ++k)
        {
            float max_val = -INFINITY;
            int max_idx = -1;

            for (int e = threadIdx.x; e < num_experts / vec_size; e += blockDim.x)
            {
                union
                {
                    fvec vec;
                    float f[vec_size];
                } tmp;
                tmp.vec = scores_vec[e];
#pragma unroll
                for (size_t i = 0; i < vec_size; i++)
                {
                    if (tmp.f[i] > max_val)
                    {
                        max_val = tmp.f[i];
                        max_idx = e * vec_size + i;
                    }
                }
            }

            warpReduceMax(max_val, max_idx);
            // blockReduceMax(max_val, max_idx);

            if (threadIdx.x == 0)
            {
                if (max_idx == -1)
                {
                    max_idx = k;
                    max_val = scores[max_idx];
                }
                if constexpr (isBiased)
                {
                    max_val -= correction_bias[max_idx];
                }
                scores[max_idx] = -INFINITY;
                topk_indices[k] = max_idx;
                topk_values[k] = max_val;
                if (need_renorm)
                {
                    sum += max_val;
                }
            }
            __syncthreads();
        }

        if (need_renorm)
        {
            if (threadIdx.x == 0)
            {
                scores[0] = routed_scaling_factor / sum; // reuse lds
            }
            __syncthreads();
            sum = scores[0];
        }
        else
        {
            sum = routed_scaling_factor;
        }

        for (int k = threadIdx.x; k < topk; k += blockDim.x)
        {
            topk_weights[token_idx * stride_tk + k] = topk_values[k] * sum;
            topk_ids[token_idx * stride_tk + k] = topk_indices[k];
        }
    }
} // namespace aiter end

#define LAUNCH_KERNEL()      \
    switch (num_experts % 4) \
    {                        \
    case 0:                  \
        LAUNCHER2(float4)    \
        break;               \
    case 2:                  \
        LAUNCHER2(float2)    \
        break;               \
    default:                 \
        LAUNCHER2(float)     \
        break;               \
    }
#define LAUNCHER2(VEC_F)                                                        \
    switch (num_expert_group)                                                   \
    {                                                                           \
    case 8:                                                                     \
        LAUNCHER3(VEC_F, 8)                                                     \
        break;                                                                  \
    case 4:                                                                     \
        LAUNCHER3(VEC_F, 4)                                                     \
        break;                                                                  \
    case 2:                                                                     \
        LAUNCHER3(VEC_F, 2)                                                     \
        break;                                                                  \
    case 1:                                                                     \
        LAUNCHER3(VEC_F, 1)                                                     \
        break;                                                                  \
    default:                                                                    \
        TORCH_CHECK(false, "Unsupported num_expert_group: ", num_expert_group); \
        break;                                                                  \
    }
#define LAUNCHER3(VEC_F, NUM_GRP)        \
    switch (need_renorm)                 \
    {                                    \
    case true:                           \
        LAUNCHER4(VEC_F, NUM_GRP, true)  \
        break;                           \
    default:                             \
        LAUNCHER4(VEC_F, NUM_GRP, false) \
    }

#define LAUNCHER4(VEC_F, NUM_GRP, need_renorm)                                        \
    if constexpr (isBiased)                                                           \
    {                                                                                 \
        LAUNCHER_biased_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, true, false) \
    }                                                                                 \
    else                                                                              \
    {                                                                                 \
        if (isSoftmax)                                                                \
        {                                                                             \
            LAUNCHER_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, false, true)    \
        }                                                                             \
        else                                                                          \
        {                                                                             \
            LAUNCHER_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, false, false)   \
        }                                                                             \
    }

#define LAUNCHER_biased_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax)    \
    VLLM_DISPATCH_FLOATING_TYPES(                                                                \
        gating_output.scalar_type(), "biased_grouped_topk_kernel", [&]                           \
        { aiter::grouped_topk_kernel<scalar_t, VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax> \
              <<<grid, block, shared_mem_size, stream>>>(                                        \
                  gating_output.data_ptr<scalar_t>(),                                            \
                  correction_bias.data_ptr<scalar_t>(),                                          \
                  topk_weights.data_ptr<float>(),                                                \
                  topk_ids.data_ptr<int>(),                                                      \
                  stride_tk,                                                                     \
                  num_experts,                                                                   \
                  topk,                                                                          \
                  topk_grp, num_tokens, routed_scaling_factor); });

#define LAUNCHER_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax)           \
    VLLM_DISPATCH_FLOATING_TYPES(                                                                \
        gating_output.scalar_type(), "grouped_topk_kernel", [&]                                  \
        { aiter::grouped_topk_kernel<scalar_t, VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax> \
              <<<grid, block, shared_mem_size, stream>>>(                                        \
                  gating_output.data_ptr<scalar_t>(),                                            \
                  nullptr,                                                                       \
                  topk_weights.data_ptr<float>(),                                                \
                  topk_ids.data_ptr<int>(),                                                      \
                  stride_tk,                                                                     \
                  num_experts,                                                                   \
                  topk,                                                                          \
                  topk_grp, num_tokens, routed_scaling_factor); });

void biased_grouped_topk(
    torch::Tensor &gating_output,   // [num_tokens, num_experts]
    torch::Tensor &correction_bias, // [num_expert]
    torch::Tensor &topk_weights,    // [num_tokens, topk]
    torch::Tensor &topk_ids,        // [num_tokens, topk]
    int num_expert_group,
    int topk_grp,
    bool need_renorm,
    const float routed_scaling_factor=1.)
{
    const bool isBiased = true;
    bool isSoftmax = false;
    int num_tokens = gating_output.size(0);
    int num_experts = gating_output.size(1);
    int topk = topk_ids.size(1);
    size_t stride_tk = topk_ids.stride(0);
    TORCH_CHECK(stride_tk == topk_weights.stride(0), "topk_ids.stride(0) == topk_weights.stride(0)");
    TORCH_CHECK(gating_output.dtype() == correction_bias.dtype(), "gating_output.dtype() == correction_bias.dtype()");

    dim3 grid(num_tokens);
    dim3 block(64);
    size_t shared_mem_size = (num_experts * sizeof(float) +
                              num_expert_group * sizeof(float) +
                              num_expert_group * sizeof(bool) +
                              topk * sizeof(int) +
                              topk * sizeof(float) + 255) &
                             ~255;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    LAUNCH_KERNEL()
}

void grouped_topk(
    torch::Tensor &gating_output, // [num_tokens, num_experts]
    torch::Tensor &topk_weights,  // [num_tokens, topk]
    torch::Tensor &topk_ids,      // [num_tokens, topk]
    int num_expert_group,
    int topk_grp,
    bool need_renorm,
    std::string scoring_func = "softmax",
    const float routed_scaling_factor = 1.)
{
    TORCH_CHECK((scoring_func == "softmax") || (scoring_func == "sigmoid"), "grouped_topk scoring_func only suppot softmax or sigmoid");
    const bool isBiased = false;
    bool isSoftmax = scoring_func == "softmax" ? true : false;
    int num_tokens = gating_output.size(0);
    int num_experts = gating_output.size(1);
    int topk = topk_ids.size(1);
    size_t stride_tk = topk_ids.stride(0);
    auto correction_bias = topk_ids;
    TORCH_CHECK(stride_tk == topk_weights.stride(0), "topk_ids.stride(0) == topk_weights.stride(0)");

    dim3 grid(num_tokens);
    dim3 block(64);
    size_t shared_mem_size = (num_experts * sizeof(float) +
                              num_expert_group * sizeof(float) +
                              num_expert_group * sizeof(bool) +
                              topk * sizeof(int) +
                              topk * sizeof(float) + 255) &
                             ~255;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(gating_output));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    LAUNCH_KERNEL()
}

#undef LAUNCHER4
#undef LAUNCHER3
#undef LAUNCHER2
#undef LAUNCH_KERNEL