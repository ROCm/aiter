// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/*
 * @Script: topk_softmax_kernels_group.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2025-03-01 12:16:14
 * @Last Modified By: valarLip
 * @Last Modified At: 2025-05-02 15:52:13
 * @Description: This is description.
 */

#include <hip/hip_runtime.h>
#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "dispatch_utils.h"
#include "py_itfs_common.h"
#include "warp_sort.h"
#include "hip_reduce.h"
#include <hipcub/util_type.hpp>
#include <hipcub/hipcub.hpp>

#define WARP_SIZE 64
namespace aiter
{
    template <typename T, typename F, int wave_size_ = 64>
    __device__ constexpr T wave_reduce(T local, F reduce_f, ck_tile::number<wave_size_> = {})
    {
        constexpr int reduce_stage = [](){
            if constexpr(wave_size_ == 2) return 1;
            else if constexpr(wave_size_ == 4) return 2;
            else if constexpr(wave_size_ == 8) return 3;
            else if constexpr(wave_size_ == 16) return 4;
            else if constexpr(wave_size_ == 32) return 5;
            else if constexpr(wave_size_ == 64) return 6;
            else return 0;
        }();
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

    // make sure local_max is local_value, local_max_2 is -INF
    template <typename T, int wave_size_ = 64>
    __device__ constexpr void wave_reduce_max2(T& local_max, T& local_max_2, ck_tile::number<wave_size_> = {})
    {
        constexpr int reduce_stage = [](){
            if constexpr(wave_size_ == 2) return 1;
            else if constexpr(wave_size_ == 4) return 2;
            else if constexpr(wave_size_ == 8) return 3;
            else if constexpr(wave_size_ == 16) return 4;
            else if constexpr(wave_size_ == 32) return 5;
            else if constexpr(wave_size_ == 64) return 6;
            else return 0;
        }();
        // T v_local = local_max;
#pragma unroll
        for (int i_stage = 0; i_stage < reduce_stage; i_stage++)
        {
            int src_lane = __lane_id() ^ (1 << i_stage);
            int32_t remote_max_ =
                __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, local_max));
            T remote_max = __builtin_bit_cast(T, remote_max_);
            if ( remote_max > local_max)
            {
                local_max_2 = local_max;
                local_max = remote_max;
            }
            else if (remote_max > local_max_2)
            {
                local_max_2 = remote_max;
            }
        }
    }

    template <typename T, typename I, int wave_size_ = 64>
    __device__ constexpr void wave_reduce_argmax2(T& local_max, I& idx, T& local_max_2, I& idx_2, ck_tile::number<wave_size_> = {})
    {
        constexpr int reduce_stage = [](){
            if constexpr(wave_size_ == 2) return 1;
            else if constexpr(wave_size_ == 4) return 2;
            else if constexpr(wave_size_ == 8) return 3;
            else if constexpr(wave_size_ == 16) return 4;
            else if constexpr(wave_size_ == 32) return 5;
            else if constexpr(wave_size_ == 64) return 6;
            else return 0;
        }();
        // T v_local = local_max;
#pragma unroll
        for (int i_stage = 0; i_stage < reduce_stage; i_stage++)
        {
            int src_lane = __lane_id() ^ (1 << i_stage);
            int32_t remote_max_ =
                __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, local_max));
            T remote_max = __builtin_bit_cast(T, remote_max_);
            if ( remote_max > local_max)
            {
                idx_2 = idx;
                local_max_2 = local_max;
                idx = src_lane;
                local_max = remote_max;
            }
            else if (remote_max > local_max_2)
            {
                local_max_2 = remote_max;
                idx_2 = src_lane;
            }
        }
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

    template <typename DTYPE_I, typename f32vec, int NUM_GRP, bool need_renorm, bool isBiased, bool isSoftmax>
    __global__ void grouped_topk_kernel(
        DTYPE_I *__restrict__ gating_output,         // [num_tokens, hidden_size]
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

        int *topk_indices = reinterpret_cast<int *>(ptr);
        ptr += topk * sizeof(int);

        float *topk_values = reinterpret_cast<float *>(ptr);
        // ptr += topk * sizeof(float);

        // int *topk_indices_f = reinterpret_cast<int *>(ptr);
        // ptr += topk * sizeof(int);

        // float *topk_values_f = reinterpret_cast<float *>(ptr);

        f32vec *scores_vec = reinterpret_cast<f32vec *>(scores);
        using cktype_i = typename t2ck<DTYPE_I>::type;
        static constexpr int vec_size = ck_tile::vector_traits<f32vec>::vector_size;
        using vec_i = ck_tile::ext_vector_t<cktype_i, vec_size>;
        const int num_experts_vec = num_experts / vec_size;

        if constexpr (!isSoftmax)
        {
            auto const *input_ptr = gating_output + token_idx * num_experts;
            for (int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
            {
                vec_i tmp = reinterpret_cast<vec_i const *>(input_ptr)[e];
                vec_i tmp2;
                if constexpr (isBiased)
                    tmp2 = reinterpret_cast<vec_i const *>(correction_bias)[e];
                f32vec gating;
#pragma unroll
                for (size_t i = 0; i < vec_size; i++)
                {
                    gating[i] = ck_tile::type_convert<float>(tmp[i]);
                    gating[i] = 1.0f / (1.0f + expf(-gating[i]));
                    if constexpr (isBiased)
                    {
                        gating[i] += ck_tile::type_convert<float>(tmp2[i]);
                    }
                }
                scores_vec[e] = gating;
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
            thread_sum = wave_reduce(thread_sum, [](float a, float b)
                                     { return a + b; });
            for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            {
                scores[e] /= thread_sum;
            }
            __syncthreads();
        }

        if constexpr (isBiased)
        {
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
            }
            __syncthreads();
        }

        for (int k = 0; k < topk_group; k++)
        {
            float max_val = -INFINITY;
            int max_idx = NUM_GRP;
#pragma unroll
            for (int g = 0; g < NUM_GRP; g++)
            {
                if (group_scores[g] > max_val)
                {
                    max_val = group_scores[g];
                    max_idx = g;
                }
            }
            group_scores[max_idx] = -INFINITY;
        }

        for (int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
        {
            int group_idx = e * vec_size / experts_per_group;
            if (group_scores[group_idx] != -INFINITY)
            {
                scores_vec[e] = -INFINITY;
            }
        }
        __syncthreads();

        using kvp = hipcub::KeyValuePair<int, float>;
        using BlockReduce = hipcub::BlockReduce<kvp, WARP_SIZE>;
        __shared__ typename BlockReduce::TempStorage tmpStorage;
        kvp thread_kvp;
        hipcub::ArgMax arg_max;

        float sum = 0.0f;
        for (int k = 0; k < topk; ++k)
        {
            float max_val = scores[k];
            int max_idx = k;

            for (int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
            {
                f32vec tmp = scores_vec[e];
#pragma unroll
                for (size_t i = 0; i < vec_size; i++)
                {
                    if (tmp[i] > max_val)
                    {
                        max_val = tmp[i];
                        max_idx = e * vec_size + i;
                    }
                }
            }
            thread_kvp.key = max_idx;
            thread_kvp.value = max_val;
            const kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
            // warpReduceMax(max_val, max_idx);
            // blockReduceMax(max_val, max_idx);

            if (threadIdx.x == 0)
            {
                max_val = result_kvp.value;
                max_idx = result_kvp.key;
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

    template <typename DTYPE_I, typename f32vec, int NUM_GRP, bool need_renorm, bool isBiased, bool isSoftmax>
    __global__ void grouped_topk_opt_sort_kernel(
        DTYPE_I *__restrict__ gating_output,         // [num_tokens, hidden_size]
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

        int *group_map_idx = reinterpret_cast<int *>(ptr);
        ptr += NUM_GRP * sizeof(int);

        int *final_topk_idx = reinterpret_cast<int *>(ptr);
        int *topk_indices = final_topk_idx; // reuse
        ptr += topk * sizeof(int);

        float *topk_values = reinterpret_cast<float *>(ptr);
        ptr += topk * sizeof(float);

        float *bias = reinterpret_cast<float *>(ptr);
        ptr += num_experts * sizeof(float);

        // float * sorting_smem = reinterpret_cast<float *>(ptr);

        // int *topk_indices_f = reinterpret_cast<int *>(ptr);
        // ptr += topk * sizeof(int);

        // float *topk_values_f = reinterpret_cast<float *>(ptr);

        f32vec *scores_vec = reinterpret_cast<f32vec *>(scores);
        using cktype_i = typename t2ck<DTYPE_I>::type;
        static constexpr int vec_size = ck_tile::vector_traits<f32vec>::vector_size;
        using vec_i = ck_tile::ext_vector_t<cktype_i, vec_size>;
        const int num_experts_vec = num_experts / vec_size;

        if constexpr (!isSoftmax)
        {
            auto const *input_ptr = gating_output + token_idx * num_experts;
            for (int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
            {
                vec_i tmp = reinterpret_cast<vec_i const *>(input_ptr)[e];
                vec_i tmp2;
                f32vec tmp2_f32;
                if constexpr (isBiased) {
                    tmp2 = reinterpret_cast<vec_i const *>(correction_bias)[e];
                }
                f32vec gating;
#pragma unroll
                for (size_t i = 0; i < vec_size; i++)
                {
                    gating[i] = ck_tile::type_convert<float>(tmp[i]);
                    gating[i] = 1.0f / (1.0f + expf(-gating[i]));
                    if constexpr (isBiased)
                    {
                        tmp2_f32[i] = ck_tile::type_convert<float>(tmp2[i]);
                        gating[i] += tmp2_f32[i];
                    }
                }
                scores_vec[e] = gating;
                if constexpr (isBiased)  {
                    reinterpret_cast<f32vec*>(bias)[e] = tmp2_f32;
                }
            }
            // __syncthreads();
        }
        else
        {
            float max_val = -INFINITY;
            float scores_[4];
            // for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            for(int i_ = 0; i_ < 4; i_++)
            {
                int e = threadIdx.x + i_ * blockDim.x;

                float gating = gating_output[token_idx * num_experts + e];
                // scores[e] = gating;
                scores_[i_] = gating;
                if (gating > max_val)
                {
                    max_val = gating;
                }
            }
#if 0
            __shared__ float sdata;
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
#else
            max_val = wave_reduce(max_val, [](auto a, auto b)
                                     { return a > b ? a : b; });
#endif
            float thread_sum = 0.0;
            //for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            for(int i_ = 0; i_ < 4; i_++)
            {
                scores_[i_] = expf(scores_[i_] - max_val);
                thread_sum += scores_[i_];
            }
            __syncthreads();
            thread_sum = wave_reduce(thread_sum, [](float a, float b)
                                     { return a + b; });
            // for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
            for(int i_ = 0; i_ < 4; i_++)
            {
                int e = threadIdx.x + i_ * blockDim.x;

                scores[e] = scores_[i_] / thread_sum;
            }
            __syncthreads();
        }

        if constexpr (isBiased)
        {
#if 0
            for (int g = threadIdx.x; g < NUM_GRP; g += blockDim.x)
            {
                float max1 = -INFINITY, max2 = -INFINITY;
                const int start = g * experts_per_group;
                const int end = start + experts_per_group;

                for (int e = start; e < end; ++e)
                {
                    auto s_tmp = scores[e];
                    max2 = s_tmp > max2 ? s_tmp : max2;
                    max2 = s_tmp > max1 ? max1 : max2;
                    max1 = s_tmp > max1 ? s_tmp : max1;
                }
                group_scores[g] = max1 + max2;
            }
#else
            constexpr int lane_group_size = 8;  // experts_per_group / 4 per thread to iter
            constexpr int lane_steps = [&](){
                if constexpr (lane_group_size == 8) return 3;
                if constexpr (lane_group_size == 4) return 2;
                if constexpr (lane_group_size == 2) return 1;
                else return 0;
            }();
            const int lane_id = threadIdx.x % lane_group_size;
            for (int g = threadIdx.x / lane_group_size; g < NUM_GRP; g += blockDim.x / lane_group_size)
            {
                float max1 = -INFINITY, max2 = -INFINITY;
                const int start = g * experts_per_group;
                const int end = start + experts_per_group;

                for (int e = start; e < end; e += lane_group_size)
                {
                    auto s_tmp = scores[e + lane_id];
                    // max2 = s_tmp > max2 ? s_tmp : max2;
                    // max2 = s_tmp > max1 ? max1 : max2;
                    // max1 = s_tmp > max1 ? s_tmp : max1;
                    max2 = dev_max_(s_tmp, max2);
                    max2 = s_tmp > max1 ? max1 : max2;
                    max1 = dev_max_(s_tmp, max1);
                }

                {
                    constexpr int row_mask    = 0xf;
                    constexpr int bank_mask   = 0xf;
                    constexpr bool bound_ctrl = true;   // ! out-of-bound is zero !

                    constexpr auto get_dpp_i = [&](auto i_step){
                        if constexpr (i_step.value == 0) return 0xb1; // quad_perm:[1,0,3,2]
                        if constexpr (i_step.value == 1) return 0x4e; // quad_perm:[2,3,0,1]
                        if constexpr (i_step.value == 2) return 0x141; // row_half_mirror
                        else return 0xffff;  // return a value to let compile crash
                    };
                    ck_tile::static_for<0, lane_steps, 1>{}([&](auto i_step){
                        constexpr int dpp_i = get_dpp_i(i_step);
                        float remote_max_1 = __builtin_bit_cast(float,
                                                        __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, max1),
                                                                    dpp_i,
                                                                    row_mask,
                                                                    bank_mask,
                                                                    bound_ctrl));
                        float remote_max_2 = __builtin_bit_cast(float,
                                                        __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, max2),
                                                                    dpp_i,
                                                                    row_mask,
                                                                    bank_mask,
                                                                    bound_ctrl));

                        // max2 = remote_max_1 > max2 ? remote_max_1 : max2;
                        // max2 = remote_max_1 > max1 ? max1 : max2;
                        // max1 = remote_max_1 > max1 ? remote_max_1 : max1;
                        max2 = dev_max_(remote_max_1, max2);
                        max2 = remote_max_1 > max1 ? max1 : max2;
                        max1 = dev_max_(remote_max_1, max1);

                        // max2 = max2 > remote_max_2 ? max2 : remote_max_2;
                        max2 = dev_max_(max2, remote_max_2);
                    });
                }
                if(lane_id == 0)
                group_scores[g] = max1 + max2;
            }
#endif
            __syncthreads();
        }
        else
        {
#if 1
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
            }
            __syncthreads();
#else
            for(int i_ = 0; i_ < 8; i_++) {
                float max_ = -INFINITY;
                if(threadIdx.x < experts_per_group) {
                    max_ = scores[i_ * experts_per_group + threadIdx.x];
                }
                max_ = wave_reduce(max_, [](auto a, auto b)
                                     { return a > b ? a : b; }, ck_tile::number<32>{});
                group_scores[i_] = max_;
            }
            __syncthreads();
#endif
        }
#if 0
        for (int k = 0; k < topk_group; k++)
        {
            float max_val = -INFINITY;
            int max_idx = NUM_GRP;
#pragma unroll
            for (int g = 0; g < NUM_GRP; g++)
            {
                auto gs_tmp = group_scores[g];
                max_idx = gs_tmp > max_val ? g : max_idx;
                max_val = gs_tmp > max_val ? gs_tmp : max_val;
            }
            group_scores[max_idx] = -INFINITY;
        }
#else
        if constexpr (NUM_GRP == 8 || NUM_GRP == 4 || NUM_GRP == 2) {
            float gs_tmp = -INFINITY;
            if(threadIdx.x < NUM_GRP)
                gs_tmp = group_scores[threadIdx.x];
#if 0
            warp_merge_sort_to_smem(sorting_smem, gs_tmp, ck_tile::number<NUM_GRP>{});
            auto pivot = sorting_smem[topk_group - 1];
#else
            auto sort_res = warp_merge_sort_to_reg(gs_tmp, ck_tile::number<NUM_GRP>{});
            // auto pivot = sort_res[3];
            auto pivot = __shfl(sort_res[topk_group - 1], 0);
#endif

#if 1
            if(gs_tmp >= pivot ) {
                group_scores[threadIdx.x] = -INFINITY;
            }

            int local_cnt = gs_tmp >= pivot ? 1 : 0;
            warp_cumsum(local_cnt, ck_tile::number<NUM_GRP>{});
            if(gs_tmp >= pivot) {
                group_map_idx[local_cnt - 1] = threadIdx.x;
            }
#else
            int local_cnt = gs_tmp >= pivot ? 1 : 0;
            warp_cumsum(local_cnt, ck_tile::number<NUM_GRP>{});
            if(gs_tmp >= pivot) {
                group_map_idx[local_cnt - 1] = threadIdx.x;
            }
            if(threadIdx.x < (NUM_GRP - topk_group)) {
                group_map_idx[topk_group + threadIdx.x] = -1;
            }
#endif
            //__syncthreads();
        } else {
#pragma unroll
            for (int k = 0; k < topk_group; k++)
            {
                float max_val = -INFINITY;
                int max_idx = NUM_GRP;
    #pragma unroll
                for (int g = 0; g < NUM_GRP; g++)
                {
                    auto gs_tmp = group_scores[g];
                    max_idx = gs_tmp > max_val ? g : max_idx;
                    max_val = gs_tmp > max_val ? gs_tmp : max_val;
                }
                group_scores[max_idx] = -INFINITY;
            }
        }
#endif

#if 1
        for (int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
        {
            int group_idx = e * vec_size / experts_per_group;
            if (group_scores[group_idx] != -INFINITY)
            {
                scores_vec[e] = -INFINITY;
            }
        }
#else
        for (int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
        {
            constexpr int experts_per_group___ = 32;
            int group_idx = e * vec_size / experts_per_group___;
            if (group_map_idx[group_idx] == -1)
            {
                // auto remapped_idx = group_map_idx[group_idx] * experts_per_group___ / vec_size;
                // TODO: this is not correct, need remap the real group-idx
                scores_vec[e] = -INFINITY;
            }
        }
#endif
        __syncthreads();

        float sum = 0.0f;
#if 0
        using kvp = hipcub::KeyValuePair<int, float>;
        using BlockReduce = hipcub::BlockReduce<kvp, WARP_SIZE>;
        __shared__ typename BlockReduce::TempStorage tmpStorage;
        kvp thread_kvp;
        hipcub::ArgMax arg_max;
        for (int k = 0; k < topk; ++k)
        {
            float max_val = scores[k];
            int max_idx = k;

            for (int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
            {
                f32vec tmp = scores_vec[e];
#pragma unroll
                for (size_t i = 0; i < vec_size; i++)
                {
                    max_idx = tmp[i] > max_val ? (e * vec_size + i) : max_idx;
                    max_val = tmp[i] > max_val ? tmp[i] : max_val;
                    // if (tmp[i] > max_val)
                    // {
                    //     max_val = tmp[i];
                    //     max_idx = e * vec_size + i;
                    // }
                }
            }
            thread_kvp.key = max_idx;
            thread_kvp.value = max_val;
            const kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
            // warpReduceMax(max_val, max_idx);
            // blockReduceMax(max_val, max_idx);

            if (threadIdx.x == 0)
            {
                max_val = result_kvp.value;
                max_idx = result_kvp.key;
                if constexpr (isBiased)
                {
                    max_val -= bias[max_idx];
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
#else

        if constexpr (NUM_GRP == 8 ) {
            constexpr int experts_per_group___ = 32;
            constexpr int final_score_vec = 2;

            using final_score_vec_t = ck_tile::ext_vector_t<float, final_score_vec>;
            final_score_vec_t s;

            for(int i = 0; i < final_score_vec; i++) {
                int expert_group_id = (threadIdx.x + i * 64) / experts_per_group___; //
                int expert_id_inside_group = threadIdx.x % experts_per_group___;
                int remapped_group_id = group_map_idx[expert_group_id];
                // if(remapped_group_id != -1)
                s[i] = scores[remapped_group_id * experts_per_group___ + expert_id_inside_group];
            }

            using vec8_t = ck_tile::ext_vector_t<float, 8>;
            vec8_t local_res[final_score_vec];
            for(int i = 0; i < final_score_vec; i++) {
                auto res_16 = warp_merge_sort_to_reg(s[i], ck_tile::number<16>{});
                auto res_32 = warp_merge_sort_combine2(vec8_t{res_16[0], res_16[1], res_16[2], res_16[3], res_16[4], res_16[5], res_16[6], res_16[7]}, ck_tile::number<16>{}, ck_tile::number<16>{});
                auto res_64 = warp_merge_sort_combine2(vec8_t{res_32[0], res_32[1], res_32[2], res_32[3], res_32[4], res_32[5], res_32[6], res_32[7]}, ck_tile::number<16>{}, ck_tile::number<32>{});

                local_res[i] = vec8_t{res_64[0], res_64[1], res_64[2], res_64[3], res_64[4], res_64[5], res_64[6], res_64[7]};
            }

            auto local_res_16 = warp_merge_sort_combine2(local_res[0], local_res[1], ck_tile::number<16>{});
            // vec8_t local_res_8 = vec8_t{local_res_16[0], local_res_16[1], local_res_16[2], local_res_16[3], local_res_16[4], local_res_16[5], local_res_16[6], local_res_16[7]};
            // float pivot = local_res_16[7];
            float pivot = __shfl(local_res_16[7], 0);

            int offset = 0;
            for(int i = 0; i < final_score_vec; i++) {
                int local_cnt = s[i] >= pivot ? 1 : 0;
                warp_cumsum(local_cnt, ck_tile::number<64>{});
                if(s[i] >= pivot) {
                    int expert_group_id = (threadIdx.x + i * 64) / experts_per_group___;
                    int expert_id_inside_group = threadIdx.x % experts_per_group___;
                    int remapped_group_id = group_map_idx[expert_group_id];
                    final_topk_idx[offset + local_cnt - 1] = remapped_group_id * experts_per_group___ + expert_id_inside_group;
                    topk_values[offset + local_cnt - 1] = s[i];
                }
                offset = __shfl(local_cnt, 63); 
            }

            // int topk_exp_id = -1;
            // float topk_value = -INFINITY;
            // if(threadIdx.x < topk) {
            //     topk_exp_id = final_topk_idx[threadIdx.x];
            //     topk_value = scores[topk_exp_id];
            //     // if constexpr (isBiased) {
            //     //     topk_value -= bias[topk_exp_id];
            //     // }

            // }

            // auto [topk_v, topk_e] = warp_arg_merge_sort_to_reg(topk_value, topk_exp_id, ck_tile::number<8>{});

            if constexpr (isBiased) {
                if (threadIdx.x < topk) {
                    topk_values[threadIdx.x] -= bias[final_topk_idx[threadIdx.x]];
                }
            }
            if (need_renorm)
            {
                if (threadIdx.x < topk) {
                    sum = multithread_reduce(topk_values[threadIdx.x], [&](auto x_, auto y_){ return x_ + y_;}, 8);
                    // sum = wave_reduce(topk_values[threadIdx.x], [&](auto x_, auto y_){ return x_ + y_;}, ck_tile::number<8>{});
                    topk_values[threadIdx.x] = topk_values[threadIdx.x] * routed_scaling_factor / sum;
                }
            }
            if (threadIdx.x < topk) {
                topk_weights[token_idx * stride_tk + threadIdx.x] = topk_values[threadIdx.x];
                topk_ids[token_idx * stride_tk + threadIdx.x] = final_topk_idx[threadIdx.x];
            }

            // if(threadIdx.x == 0) {
            //     using vec_int8_t = ck_tile::ext_vector_t<int, 8>;
            //     *reinterpret_cast<vec8_t*>(&topk_weights[token_idx * stride_tk]) = *reinterpret_cast<vec8_t*>(topk_values); 
            //     *reinterpret_cast<vec_int8_t*>(&topk_ids[token_idx * stride_tk]) = *reinterpret_cast<vec_int8_t*>(final_topk_idx);
            //     // topk_indices[threadIdx.x ] = topk_e[0];
            //     // topk_values[threadIdx.x ] = topk_v[0];
            // }
        }
#endif

    }
} // namespace aiter end

#define LAUNCH_KERNEL()                                    \
    switch (num_experts % 4)                               \
    {                                                      \
    case 0:                                                \
        using vec4_type = ck_tile::ext_vector_t<float, 4>; \
        LAUNCHER2(vec4_type)                               \
        break;                                             \
    case 2:                                                \
        using vec2_type = ck_tile::ext_vector_t<float, 2>; \
        LAUNCHER2(vec2_type)                               \
        break;                                             \
    default:                                               \
        using vec1_type = ck_tile::ext_vector_t<float, 1>; \
        LAUNCHER2(vec1_type)                               \
        break;                                             \
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

#define LAUNCHER4(VEC_F, NUM_GRP, need_renorm)                                          \
    if constexpr (isBiased)                                                             \
    {                                                                                   \
        if(use_opt_sort) {                                                              \
            LAUNCHER_biased_grouped_topk_opt_sort_kernel(VEC_F, NUM_GRP, need_renorm, true, false) \
        }                                                                                   \
        else {                                                                              \
            LAUNCHER_biased_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, true, false)   \
        }                                                                                   \
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

#define LAUNCHER_biased_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax)                                                                            \
    VLLM_DISPATCH_FLOATING_TYPES(                                                                                                                                        \
        gating_output.scalar_type(), "biased_grouped_topk_kernel", [&]                                                                                                   \
        { hipLaunchKernelGGL((aiter::grouped_topk_kernel<scalar_t, VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax>), dim3(grid), dim3(block), shared_mem_size, stream, \
                             gating_output.data_ptr<scalar_t>(),                                                                                                         \
                             correction_bias.data_ptr<scalar_t>(),                                                                                                       \
                             topk_weights.data_ptr<float>(),                                                                                                             \
                             topk_ids.data_ptr<int>(),                                                                                                                   \
                             stride_tk,                                                                                                                                  \
                             num_experts,                                                                                                                                \
                             topk,                                                                                                                                       \
                             topk_grp, num_tokens, routed_scaling_factor); });

#define LAUNCHER_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax)                                                                                   \
    VLLM_DISPATCH_FLOATING_TYPES(                                                                                                                                        \
        gating_output.scalar_type(), "grouped_topk_kernel", [&]                                                                                                          \
        { hipLaunchKernelGGL((aiter::grouped_topk_kernel<scalar_t, VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax>), dim3(grid), dim3(block), shared_mem_size, stream, \
                             gating_output.data_ptr<scalar_t>(),                                                                                                         \
                             nullptr,                                                                                                                                    \
                             topk_weights.data_ptr<float>(),                                                                                                             \
                             topk_ids.data_ptr<int>(),                                                                                                                   \
                             stride_tk,                                                                                                                                  \
                             num_experts,                                                                                                                                \
                             topk,                                                                                                                                       \
                             topk_grp, num_tokens, routed_scaling_factor); });

#define LAUNCHER_biased_grouped_topk_opt_sort_kernel(VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax)                                                                   \
    VLLM_DISPATCH_FLOATING_TYPES(                                                                                                                                        \
        gating_output.scalar_type(), "biased_grouped_topk_opt_sort_kernel", [&]                                                                                          \
        { hipLaunchKernelGGL((aiter::grouped_topk_opt_sort_kernel<scalar_t, VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax>), dim3(grid), dim3(block), shared_mem_size, stream, \
                             gating_output.data_ptr<scalar_t>(),                                                                                                         \
                             correction_bias.data_ptr<scalar_t>(),                                                                                                       \
                             topk_weights.data_ptr<float>(),                                                                                                             \
                             topk_ids.data_ptr<int>(),                                                                                                                   \
                             stride_tk,                                                                                                                                  \
                             num_experts,                                                                                                                                \
                             topk,                                                                                                                                       \
                             topk_grp, num_tokens, routed_scaling_factor); });

void biased_grouped_topk(
    torch::Tensor &gating_output,   // [num_tokens, num_experts]
    torch::Tensor &correction_bias, // [num_expert]
    torch::Tensor &topk_weights,    // [num_tokens, topk]
    torch::Tensor &topk_ids,        // [num_tokens, topk]
    int num_expert_group,
    int topk_grp,
    bool need_renorm,
    const float routed_scaling_factor = 1.)
{
    const bool isBiased = true;
    bool isSoftmax = false;
    int num_tokens = gating_output.size(0);
    int num_experts = gating_output.size(1);
    int topk = topk_ids.size(1);
    size_t stride_tk = topk_ids.stride(0);
    TORCH_CHECK(stride_tk == topk_weights.stride(0), "topk_ids.stride(0) == topk_weights.stride(0)");
    TORCH_CHECK(gating_output.dtype() == correction_bias.dtype(), "gating_output.dtype() == correction_bias.dtype()");

    // TODO: expand usage in the future
    bool use_opt_sort = (topk == 8) && (num_expert_group == 8) && (num_experts == 256) && (topk_grp == 4) && (isBiased == true);

    dim3 grid(num_tokens);
    dim3 block(64);
    size_t shared_mem_size = (num_experts * sizeof(float) +
                              (num_expert_group + 1) * sizeof(float) +
                              topk * sizeof(int) +
                              topk * sizeof(float)
                              + num_experts * sizeof(float) /*bias*/
                              + 255) &
                             ~255;
                            //   + 64 / num_expert_group * sizeof(float) /* for sorting */

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    LAUNCH_KERNEL()
}

void grouped_topk(
    torch::Tensor &gating_output, // [num_tokens, num_experts]
    torch::Tensor &topk_weights,  // [num_tokens, topk]
    torch::Tensor &topk_ids,      // [num_tokens, topk]
    int num_expert_group,
    int topk_grp,
    bool need_renorm,
    bool is_softmax = true,
    const float routed_scaling_factor = 1.)
{
    const bool isBiased = false;
    bool isSoftmax = is_softmax;
    int num_tokens = gating_output.size(0);
    int num_experts = gating_output.size(1);
    int topk = topk_ids.size(1);
    size_t stride_tk = topk_ids.stride(0);
    auto correction_bias = topk_ids;
    TORCH_CHECK(stride_tk == topk_weights.stride(0), "topk_ids.stride(0) == topk_weights.stride(0)");

    // TODO: expand usage in the future
    bool use_opt_sort = false;

    dim3 grid(num_tokens);
    dim3 block(64);
    size_t shared_mem_size = (num_experts * sizeof(float) +
                              (num_expert_group + 1) * sizeof(float) +
                              topk * sizeof(int) +
                              topk * sizeof(float) + 255) &
                             ~255;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    LAUNCH_KERNEL()
}

#undef LAUNCHER4
#undef LAUNCHER3
#undef LAUNCHER2
#undef LAUNCH_KERNEL
