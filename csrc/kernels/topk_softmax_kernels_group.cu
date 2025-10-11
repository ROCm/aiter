// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
/*
 * @Script: topk_softmax_kernels_group.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2025-03-01 12:16:14
 * @Last Modified By: valarLip
 * @Last Modified At: 2025-09-15 15:08:25
 * @Description: This is description.
 */

#include "dispatch_utils.h"
#include "hip_reduce.h"
#include "py_itfs_common.h"
#include "warp_sort.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <hipcub/util_type.hpp>
#include <torch/all.h>

#ifndef AITER_TOPK_SOFTMAX_GROUP_PERMUTE_SCORE
#define AITER_TOPK_SOFTMAX_GROUP_PERMUTE_SCORE 0
#endif

// rocm 6.4.1 has compiler problem that can't generate proper dependency for ds_permute
#ifndef AITER_TOPK_SOFTMAX_GROUP_PERMUTE_SCORE_USE_INLINE_ASM
#define AITER_TOPK_SOFTMAX_GROUP_PERMUTE_SCORE_USE_INLINE_ASM 1
#endif

#ifndef C_LOG2E
#define C_LOG2E 1.44269504088896340736 // log2(e)
#endif

#define WARP_SIZE 64
namespace aiter {
namespace impl {
// use this type for argsort
template <typename KType, typename VType>
struct kvpair_t
{
    KType key;
    VType value;
};

using topk_score_t = kvpair_t<int, float>;
} // namespace impl

template <typename T, typename F, int wave_size_ = 64>
__device__ constexpr T wave_reduce(T local, F reduce_f, ck_tile::number<wave_size_> = {})
{
    constexpr int reduce_stage = []() {
        if constexpr(wave_size_ == 2)
            return 1;
        else if constexpr(wave_size_ == 4)
            return 2;
        else if constexpr(wave_size_ == 8)
            return 3;
        else if constexpr(wave_size_ == 16)
            return 4;
        else if constexpr(wave_size_ == 32)
            return 5;
        else if constexpr(wave_size_ == 64)
            return 6;
        else
            return 0;
    }();
    T v_local = local;
#pragma unroll
    for(int i_stage = 0; i_stage < reduce_stage; i_stage++)
    {
        int src_lane = __lane_id() ^ (1 << i_stage);
        int32_t v_remote_tmp =
            __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, v_local));
        T v_remote = __builtin_bit_cast(T, v_remote_tmp);
        v_local    = reduce_f(v_local, v_remote);
    }
    return v_local;
}

// every thread hold one value, with pivot (topk value)
// need to find the first topk value that is >= pivot
// e.g.
// value   : [7, 5, 3, 5, 7]
// lane id : [0, 1, 2, 3, 4]
//
// topk = 3 => pivot = 5
// => (out)
// value(out) : [7, 7, 5]
// index(out) : [0, 4, 1]
//
// we need cumsum result
//
// cumsum  : [1, 3, 4, 2, 2] => this is the return value per lane
// lane id : [0, 1, 2, 3, 4]
//
// here cumsum is the index into LDS, with condition (v >= pivot && cumsum <= topk)
// mask    : [1, 1, 0, 0, 1] (of above condition per lane)
// position: [0, 2, x, x, 1] => this is cumsum-1
//
template <typename T, int lanegroup_size_ = 64>
__device__ constexpr int
cumsum_topk_with_pivot(const T& v, const T& pivot, ck_tile::number<lanegroup_size_> = {})
{
    // lanegroup_size_ must be power of 2!

    // fisrt count larger than pivot
    int cnt_ = v > pivot ? 1 : 0;
    warp_cumsum(cnt_, ck_tile::number<lanegroup_size_>{});
    int total_ = __builtin_amdgcn_readlane(cnt_, lanegroup_size_ - 1);

    // 2nd count equal to pivot
    int cnt_2_ = v == pivot ? 1 : 0;
    warp_cumsum(cnt_2_, ck_tile::number<lanegroup_size_>{});
    return v == pivot ? (total_ + cnt_2_) : cnt_;
}

// make sure local_max is local_value, local_max_2 is -INF
template <typename T, int wave_size_ = 64>
__device__ constexpr void
wave_reduce_max2(T& local_max, T& local_max_2, ck_tile::number<wave_size_> = {})
{
    constexpr int reduce_stage = []() {
        if constexpr(wave_size_ == 2)
            return 1;
        else if constexpr(wave_size_ == 4)
            return 2;
        else if constexpr(wave_size_ == 8)
            return 3;
        else if constexpr(wave_size_ == 16)
            return 4;
        else if constexpr(wave_size_ == 32)
            return 5;
        else if constexpr(wave_size_ == 64)
            return 6;
        else
            return 0;
    }();
    // T v_local = local_max;
#pragma unroll
    for(int i_stage = 0; i_stage < reduce_stage; i_stage++)
    {
        int src_lane = __lane_id() ^ (1 << i_stage);
        int32_t remote_max_ =
            __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, local_max));
        T remote_max = __builtin_bit_cast(T, remote_max_);
        if(remote_max > local_max)
        {
            local_max_2 = local_max;
            local_max   = remote_max;
        }
        else if(remote_max > local_max_2)
        {
            local_max_2 = remote_max;
        }
    }
}

template <typename T, typename I, int wave_size_ = 64>
__device__ constexpr void wave_reduce_argmax2(
    T& local_max, I& idx, T& local_max_2, I& idx_2, ck_tile::number<wave_size_> = {})
{
    constexpr int reduce_stage = []() {
        if constexpr(wave_size_ == 2)
            return 1;
        else if constexpr(wave_size_ == 4)
            return 2;
        else if constexpr(wave_size_ == 8)
            return 3;
        else if constexpr(wave_size_ == 16)
            return 4;
        else if constexpr(wave_size_ == 32)
            return 5;
        else if constexpr(wave_size_ == 64)
            return 6;
        else
            return 0;
    }();
    // T v_local = local_max;
#pragma unroll
    for(int i_stage = 0; i_stage < reduce_stage; i_stage++)
    {
        int src_lane = __lane_id() ^ (1 << i_stage);
        int32_t remote_max_ =
            __builtin_amdgcn_ds_bpermute(src_lane << 2, __builtin_bit_cast(int32_t, local_max));
        T remote_max = __builtin_bit_cast(T, remote_max_);
        if(remote_max > local_max)
        {
            idx_2       = idx;
            local_max_2 = local_max;
            idx         = src_lane;
            local_max   = remote_max;
        }
        else if(remote_max > local_max_2)
        {
            local_max_2 = remote_max;
            idx_2       = src_lane;
        }
    }
}

__inline__ __device__ void warpReduceMax(float& val_o, int& idx)
{
    static_assert(64 == WARP_SIZE, "WARP_SIZE == 64");
    constexpr int lane_steps  = 6;
    constexpr int row_mask    = 0xf;
    constexpr int bank_mask   = 0xf;
    constexpr bool bound_ctrl = true;
    float val                 = val_o;

    constexpr auto get_dpp_i = [&](auto i_step) {
        if constexpr(i_step.value == 0)
            return 0xb1; // quad_perm:[1,0,3,2]
        if constexpr(i_step.value == 1)
            return 0x4e; // quad_perm:[2,3,0,1]
        if constexpr(i_step.value == 2)
            return 0x114; // row_shr:4
        if constexpr(i_step.value == 3)
            return 0x118; // row_shr:8
        if constexpr(i_step.value == 4)
            return 0x142; // row_bcast:15
        if constexpr(i_step.value == 5)
            return 0x143; // row_bcast:31
        else
            return 0xffff; // return a value to let compile crash
    };
    ck_tile::static_for<0, lane_steps, 1>{}([&](auto i_step) {
        constexpr int dpp_i = get_dpp_i(i_step);

        float remote_val = __builtin_bit_cast(
            float,
            __builtin_amdgcn_mov_dpp(
                __builtin_bit_cast(int, val), dpp_i, row_mask, bank_mask, bound_ctrl));
        int remote_idx = __builtin_bit_cast(
            int,
            __builtin_amdgcn_mov_dpp(
                __builtin_bit_cast(int, idx), dpp_i, row_mask, bank_mask, bound_ctrl));

        idx = val > remote_val ? idx : remote_idx;
        val = val > remote_val ? val : remote_val;
    });
    val_o = __builtin_bit_cast(
        float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, val), WARP_SIZE - 1));
    idx = __builtin_amdgcn_readlane(idx, WARP_SIZE - 1);

    // val = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, val), 63));
    // if (val==val_o)
    // {
    //     unsigned long long active_mask = __builtin_amdgcn_read_exec();
    //     int first_lane;
    //     asm volatile("s_ff1_i32_b64 %0, %1\n" : "=s"(first_lane) : "s"(active_mask));
    //     if(threadIdx.x == first_lane)
    //     {
    //         int tmp = __builtin_amdgcn_readlane(idx, first_lane);
    //         asm volatile("v_writelane_b32 %0, %1, 63\n" : "=v"(idx) : "s"(tmp));
    //     }
    // }
    // val_o = val;
    // idx = __builtin_amdgcn_readlane(idx, 63);

    // #pragma unroll
    //         for(int i = 0; i < 6; i++)
    //         {
    //             int offset = 1 << i;
    //             float tmp_val = __shfl_down(val, offset);
    //             int tmp_idx = __shfl_down(idx, offset);
    //             if (tmp_val > val)
    //             {
    //                 val = tmp_val;
    //                 idx = tmp_idx;
    //             }
    //         }
}

__device__ void blockReduceMax(float& val, int& idx)
{
    __shared__ float shared_vals[32];
    __shared__ int shared_idxs[32];

    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    warpReduceMax(val, idx);

    if(lane == 0)
    {
        shared_vals[wid] = val;
        shared_idxs[wid] = idx;
    }
    __syncthreads();

    if(wid == 0)
    {
        val = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_vals[lane] : -INFINITY;
        idx = (lane < (blockDim.x + WARP_SIZE - 1) / WARP_SIZE) ? shared_idxs[lane] : -1;

        warpReduceMax(val, idx);
    }
    __syncthreads();
}

template <typename DTYPE_I,
          typename f32vec,
          int NUM_GRP,
          bool need_renorm,
          bool isBiased,
          bool isSoftmax>
__global__ void
grouped_topk_kernel(DTYPE_I* __restrict__ gating_output,         // [num_tokens, hidden_size]
                    const DTYPE_I* __restrict__ correction_bias, // [num_expert]
                    float* __restrict__ topk_weights,            // [num_tokens, topk]
                    int* __restrict__ topk_ids,                  // [num_tokens, topk]
                    const size_t stride_tk,
                    const int num_experts,
                    const int topk,
                    const int topk_group,
                    const int num_tokens,
                    const float routed_scaling_factor)
{
    static_assert(NUM_GRP <= WARP_SIZE, "NUM_GRP must be <= WARP_SIZE");
    static constexpr int THREAD_PER_GRP = (WARP_SIZE + NUM_GRP - 1) / NUM_GRP;
    // 256 E, 8->4 group, 32 e/group
    const int experts_per_group = num_experts / NUM_GRP;
    extern __shared__ char shared_mem[];
    const int token_idx = blockIdx.x;

    char* ptr     = shared_mem;
    float* scores = reinterpret_cast<float*>(ptr);
    ptr += num_experts * sizeof(float);

    float* group_scores = reinterpret_cast<float*>(ptr);
    ptr += NUM_GRP * sizeof(float);

    // float* bias = reinterpret_cast<float*>(ptr);
    // ptr += num_experts * sizeof(float);

    // int* topk_indices   = reinterpret_cast<int*>(ptr);
    // ptr += topk * sizeof(int);

    // float* topk_values = reinterpret_cast<float*>(ptr);
    // ptr += topk * sizeof(float);

    // int *topk_indices_f = reinterpret_cast<int *>(ptr);
    // ptr += topk * sizeof(int);

    // float *topk_values_f = reinterpret_cast<float *>(ptr);

    f32vec* scores_vec            = reinterpret_cast<f32vec*>(scores);
    using cktype_i                = typename t2ck<DTYPE_I>::type;
    static constexpr int vec_size = ck_tile::vector_traits<f32vec>::vector_size;
    using vec_i                   = ck_tile::ext_vector_t<cktype_i, vec_size>;
    const int num_experts_vec     = num_experts / vec_size;

    if constexpr(!isSoftmax)
    {
        auto const* input_ptr = gating_output + token_idx * num_experts;
        for(int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
        {
            vec_i tmp = reinterpret_cast<vec_i const*>(input_ptr)[e];
            vec_i tmp2;
            f32vec tmp2_f32;
            if constexpr(isBiased)
                tmp2 = reinterpret_cast<vec_i const*>(correction_bias)[e];
            f32vec gating;
#pragma unroll
            for(size_t i = 0; i < vec_size; i++)
            {
                gating[i] = ck_tile::type_convert<float>(tmp[i]);
                gating[i] = 1.0f / (1.0f + expf(-gating[i]));
                if constexpr(isBiased)
                {
                    tmp2_f32[i] = ck_tile::type_convert<float>(tmp2[i]);
                    gating[i] += tmp2_f32[i];
                }
            }
            scores_vec[e] = gating;
            // if constexpr(isBiased)
            // {
            //     reinterpret_cast<f32vec*>(bias)[e] = tmp2_f32;
            // }
        }
        __syncthreads();
    }
    else
    {
        __shared__ float sdata;
        float max_val = -INFINITY;
        for(int e = threadIdx.x; e < num_experts; e += blockDim.x)
        {

            float gating = gating_output[token_idx * num_experts + e];
            scores[e]    = gating;
            if(gating > max_val)
            {
                max_val = gating;
            }
        }
        __syncthreads();
#pragma unroll
        for(int i = 0; i < 6; i++)
        {
            int offset    = 1 << i;
            float tmp_val = __shfl_down(max_val, offset);
            if(tmp_val > max_val)
            {
                max_val = tmp_val;
            }
        }
        if(threadIdx.x == 0)
        {
            sdata = max_val;
        }
        __syncthreads();
        max_val          = sdata;
        float thread_sum = 0.0;
        for(int e = threadIdx.x; e < num_experts; e += blockDim.x)
        {
            scores[e] = expf(scores[e] - max_val);
            thread_sum += scores[e];
        }
        __syncthreads();
        thread_sum = wave_reduce(thread_sum, [](float a, float b) { return a + b; });
        for(int e = threadIdx.x; e < num_experts; e += blockDim.x)
        {
            scores[e] /= thread_sum;
        }
        __syncthreads();
    }

    if constexpr(isBiased)
    {
        constexpr int lane_steps = [&]() {
            if constexpr(THREAD_PER_GRP == 8)
                return 3;
            if constexpr(THREAD_PER_GRP == 4)
                return 2;
            if constexpr(THREAD_PER_GRP == 2)
                return 1;
            else
                return 0;
        }();
        const int lane_id = threadIdx.x % THREAD_PER_GRP;
        for(int g = threadIdx.x / THREAD_PER_GRP; g < NUM_GRP; g += blockDim.x / THREAD_PER_GRP)
        {
            float max1 = -INFINITY, max2 = -INFINITY;
            const int start = g * experts_per_group;
            const int end   = experts_per_group / vec_size;
            f32vec* sc      = reinterpret_cast<f32vec*>(scores + start);

            for(int e = lane_id; e < end; e += THREAD_PER_GRP)
            {
                auto s_vec = sc[e];
                for(int j = 0; j < vec_size; j++)
                {
                    auto s_tmp = s_vec[j];
                    max2       = dev_max_(s_tmp, max2);
                    max2       = s_tmp > max1 ? max1 : max2;
                    max1       = dev_max_(s_tmp, max1);
                }
            }

            {
                constexpr int row_mask    = 0xf;
                constexpr int bank_mask   = 0xf;
                constexpr bool bound_ctrl = true; // ! out-of-bound is zero !

                constexpr auto get_dpp_i = [&](auto i_step) {
                    if constexpr(i_step.value == 0)
                        return 0xb1; // quad_perm:[1,0,3,2]
                    if constexpr(i_step.value == 1)
                        return 0x4e; // quad_perm:[2,3,0,1]
                    if constexpr(i_step.value == 2)
                        return 0x141; // row_half_mirror
                    else
                        return 0xffff; // return a value to let compile crash
                };
                ck_tile::static_for<0, lane_steps, 1>{}([&](auto i_step) {
                    constexpr int dpp_i = get_dpp_i(i_step);
                    float remote_max_1  = __builtin_bit_cast(
                        float,
                        __builtin_amdgcn_mov_dpp(
                            __builtin_bit_cast(int, max1), dpp_i, row_mask, bank_mask, bound_ctrl));
                    float remote_max_2 = __builtin_bit_cast(
                        float,
                        __builtin_amdgcn_mov_dpp(
                            __builtin_bit_cast(int, max2), dpp_i, row_mask, bank_mask, bound_ctrl));

                    max2 = dev_max_(remote_max_1, max2);
                    max2 = remote_max_1 > max1 ? max1 : max2;
                    max1 = dev_max_(remote_max_1, max1);
                    max2 = dev_max_(max2, remote_max_2);
                });
            }
            if(lane_id == 0)
                group_scores[g] = max1 + max2;
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int g = threadIdx.x; g < NUM_GRP; g += blockDim.x)
        {
            float max1      = -INFINITY;
            const int start = g * experts_per_group;
            const int end   = start + experts_per_group;
            for(int e = start; e < end; ++e)
            {
                max1 = scores[e] > max1 ? scores[e] : max1;
            }
            group_scores[g] = max1;
        }
        __syncthreads();
    }

    for(int k = 0; k < topk_group; k++)
    {
        float max_val = -INFINITY;
        int max_idx   = NUM_GRP;
#pragma unroll
        for(int g = 0; g < NUM_GRP; g++)
        {
            auto gs_tmp = group_scores[g];
            max_idx     = gs_tmp > max_val ? g : max_idx;
            max_val     = gs_tmp > max_val ? gs_tmp : max_val;
        }
        group_scores[max_idx] = -INFINITY;
    }

    for(int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
    {
        int group_idx = e * vec_size / experts_per_group;
        if(group_scores[group_idx] != -INFINITY)
        {
            scores_vec[e] = -INFINITY;
        }
    }
    __syncthreads();

    // using kvp = hipcub::KeyValuePair<int, float>;
    // using BlockReduce = hipcub::BlockReduce<kvp, WARP_SIZE>;
    // __shared__ typename BlockReduce::TempStorage tmpStorage;
    // kvp thread_kvp;
    // hipcub::ArgMax arg_max;

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
        // thread_kvp.key = max_idx;
        // thread_kvp.value = max_val;
        // const kvp result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);

        warpReduceMax(max_val, max_idx);
        // blockReduceMax(max_val, max_idx);

        // if (threadIdx.x == 0)
        {
            // max_val = result_kvp.value;
            // max_idx = result_kvp.key;
            if constexpr(isBiased)
            {
                max_val -= correction_bias[max_idx];
                // max_val -= bias[max_idx];
            }
            scores[max_idx] = -INFINITY;
            // topk_indices[k] = max_idx;
            // topk_values[k] = max_val;
            topk_indice = threadIdx.x == k ? max_idx : topk_indice;
            topk_value  = threadIdx.x == k ? max_val : topk_value;
            if(need_renorm)
            {
                sum += max_val;
            }
        }
        // __syncthreads();
    }

    if(need_renorm)
    {
        sum = routed_scaling_factor / sum;
        ;
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

template <typename DTYPE_I,
          typename f32vec,
          int NUM_GRP,
          bool need_renorm,
          bool isBiased,
          bool isSoftmax>
__global__ void
grouped_topk_opt_sort_kernel(DTYPE_I* __restrict__ gating_output, // [num_tokens, hidden_size]
                             const DTYPE_I* __restrict__ correction_bias, // [num_expert]
                             float* __restrict__ topk_weights,            // [num_tokens, topk]
                             int* __restrict__ topk_ids,                  // [num_tokens, topk]
                             const size_t stride_tk,
                             const int num_experts,
                             const int topk,
                             const int topk_group,
                             const int num_tokens,
                             const float routed_scaling_factor)
{
    static_assert(NUM_GRP <= WARP_SIZE, "NUM_GRP must be <= WARP_SIZE");
    // number of lanes responsible for a expert group
    static constexpr int THREAD_PER_GRP = (WARP_SIZE + NUM_GRP - 1) / NUM_GRP;
    // 256 E, 8->4 group, 32 e/group
    const int experts_per_group = num_experts / NUM_GRP;
    extern __shared__ char shared_mem[];
    const int token_idx = blockIdx.x;

    char* ptr     = shared_mem; // (char *)(((size_t)shared_mem + 255) & ~255);
    float* scores = reinterpret_cast<float*>(ptr);
    ptr += num_experts * sizeof(float);

    float* group_scores = reinterpret_cast<float*>(ptr);
    ptr += NUM_GRP * sizeof(float);

    int* group_map_idx = reinterpret_cast<int*>(ptr);
    ptr += NUM_GRP * sizeof(int);

    int* final_topk_idx = reinterpret_cast<int*>(ptr);
    // int *topk_indices = final_topk_idx; // reuse
    ptr += topk * sizeof(int);

    float* topk_values = reinterpret_cast<float*>(ptr);
    ptr += topk * sizeof(float);

    float* bias = reinterpret_cast<float*>(ptr);
    ptr += num_experts * sizeof(float);

    // used for arg sort
    int* sorted_k = reinterpret_cast<int*>(ptr);
    ptr += max(topk, topk_group) * sizeof(int);

    float* sorted_v = reinterpret_cast<float*>(ptr);

    // float * sorting_smem = reinterpret_cast<float *>(ptr);

    // int *topk_indices_f = reinterpret_cast<int *>(ptr);
    // ptr += topk * sizeof(int);

    // float *topk_values_f = reinterpret_cast<float *>(ptr);

    f32vec* scores_vec            = reinterpret_cast<f32vec*>(scores);
    using cktype_i                = typename t2ck<DTYPE_I>::type;
    static constexpr int vec_size = ck_tile::vector_traits<f32vec>::vector_size;
    using vec_i                   = ck_tile::ext_vector_t<cktype_i, vec_size>;
    const int num_experts_vec     = num_experts / vec_size;

    f32vec gating;
    if constexpr(!isSoftmax)
    {
        auto const* input_ptr = gating_output + token_idx * num_experts;
        // for(int e = threadIdx.x; e < num_experts_vec; e += blockDim.x)
        int e = threadIdx.x;
        {
            vec_i tmp = reinterpret_cast<vec_i const*>(input_ptr)[e];
            vec_i tmp2;
            f32vec tmp2_f32;
            if constexpr(isBiased)
            {
                tmp2 = reinterpret_cast<vec_i const*>(correction_bias)[e];
            }
            
#pragma unroll
            for(size_t i = 0; i < vec_size; i++)
            {
                gating[i] = ck_tile::type_convert<float>(tmp[i]);
                // gating[i] = __builtin_amdgcn_rcpf(1.0f + expf(-gating[i]));
                gating[i] = __builtin_amdgcn_rcpf(1.0f + exp2f(-C_LOG2E * gating[i]));
                if constexpr(isBiased)
                {
                    tmp2_f32[i] = ck_tile::type_convert<float>(tmp2[i]);
                    gating[i] += tmp2_f32[i];
                }
                gating[i] = ::isnan(gating[i]) ? -INFINITY : gating[i];
            }
            scores_vec[e] = gating;
        }
        //__syncthreads();
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
            if(gating > max_val)
            {
                max_val = gating;
            }
        }

        max_val = wave_reduce(max_val, [](auto a, auto b) { return a > b ? a : b; });

        float thread_sum = 0.0;
        // for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        for(int i_ = 0; i_ < 4; i_++)
        {
            scores_[i_] = expf(scores_[i_] - max_val);
            thread_sum += scores_[i_];
        }
        __syncthreads();
        thread_sum = wave_reduce(thread_sum, [](float a, float b) { return a + b; });
        // for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
        for(int i_ = 0; i_ < 4; i_++)
        {
            int e = threadIdx.x + i_ * blockDim.x;

            scores[e] = scores_[i_] / thread_sum;
        }
        __syncthreads();
    }

    float group_score_;

    if constexpr(isBiased)
    {
        constexpr int lane_steps = [&]() {
            if constexpr(THREAD_PER_GRP == 8)
                return 3;
            if constexpr(THREAD_PER_GRP == 4)
                return 2;
            if constexpr(THREAD_PER_GRP == 2)
                return 1;
            else
                return 0;
        }();
        const int lane_id = threadIdx.x % THREAD_PER_GRP;
        int g = threadIdx.x / THREAD_PER_GRP;
        {
            float max1, max2;
            const int start = g * experts_per_group;
            const int end   = experts_per_group / vec_size;
            //f32vec* sc      = reinterpret_cast<f32vec*>(scores + start);

            {
                int e = lane_id;
                //auto s_vec = sc[e];
                auto s_vec = gating;
                max1 = s_vec[0];
                max2 = -INFINITY;
#pragma unroll
                for(int j = 1; j < vec_size; j++)
                {
                    auto s_tmp = s_vec[j];
                    max2       = dev_med3_(s_tmp, max1, max2);
                    max1       = dev_max_(s_tmp, max1);
                }
            }

            {
                constexpr int row_mask    = 0xf;
                constexpr int bank_mask   = 0xf;
                constexpr bool bound_ctrl = true; // ! out-of-bound is zero !

                constexpr auto get_dpp_i = [&](auto i_step) {
                    if constexpr(i_step.value == 0)
                        return 0xb1; // quad_perm:[1,0,3,2]
                    if constexpr(i_step.value == 1)
                        return 0x4e; // quad_perm:[2,3,0,1]
                    if constexpr(i_step.value == 2)
                        return 0x141; // row_half_mirror
                    else
                        return 0xffff; // return a value to let compile crash
                };
                ck_tile::static_for<0, lane_steps, 1>{}([&](auto i_step) {
                    constexpr int dpp_i = get_dpp_i(i_step);
                    float remote_max_1  = __builtin_bit_cast(
                        float,
                        __builtin_amdgcn_mov_dpp(
                            __builtin_bit_cast(int, max1), dpp_i, row_mask, bank_mask, bound_ctrl));
                    float remote_max_2 = __builtin_bit_cast(
                        float,
                        __builtin_amdgcn_mov_dpp(
                            __builtin_bit_cast(int, max2), dpp_i, row_mask, bank_mask, bound_ctrl));

                    max2 = dev_max_(remote_max_2, max2);
                    max2 = dev_med3_(remote_max_1, max1, max2);
                    max1 = dev_max_(remote_max_1, max1);
                });
            }
            // not all lanes store the correct result!
            group_score_ = max1 + max2;
        }
        // scores_vec[threadIdx.x] = gating;
    }
    else
    {
#if 1
#pragma unroll
        for(int g = threadIdx.x; g < NUM_GRP; g += blockDim.x)
        {
            float max1      = -INFINITY;
            const int start = g * experts_per_group;
            const int end   = start + experts_per_group;
            for(int e = start; e < end; ++e)
            {
                if(scores[e] > max1)
                {
                    max1 = scores[e];
                }
            }
            group_scores[g] = max1;
        }
        __syncthreads();
#else
        for(int i_ = 0; i_ < 8; i_++)
        {
            float max_ = -INFINITY;
            if(threadIdx.x < experts_per_group)
            {
                max_ = scores[i_ * experts_per_group + threadIdx.x];
            }
            max_ = wave_reduce(
                max_, [](auto a, auto b) { return a > b ? a : b; }, ck_tile::number<32>{});
            group_scores[i_] = max_;
        }
        __syncthreads();
#endif
    }

    if constexpr(NUM_GRP == 8 || NUM_GRP == 4 || NUM_GRP == 2)
    {
        float gs_tmp_remote = __shfl(group_score_, threadIdx.x * THREAD_PER_GRP);
        float gs_tmp        =  gs_tmp_remote;

        auto sort_res = warp_bitonic_merge_sort_to_reg(gs_tmp, ck_tile::number<NUM_GRP>{});
        auto pivot    = __shfl(sort_res, 3);

        int local_cnt = cumsum_topk_with_pivot(gs_tmp, pivot, ck_tile::number<NUM_GRP>{});

        if(gs_tmp >= pivot && local_cnt <= topk_group && threadIdx.x < NUM_GRP)
        {
            group_map_idx[local_cnt - 1] = threadIdx.x;
        }
        __syncthreads();
    }
    else
    {
#pragma unroll
        for(int k = 0; k < topk_group; k++)
        {
            float max_val = -INFINITY;
            int max_idx   = NUM_GRP;
#pragma unroll
            for(int g = 0; g < NUM_GRP; g++)
            {
                auto gs_tmp = group_scores[g];
                max_idx     = gs_tmp > max_val ? g : max_idx;
                max_val     = gs_tmp > max_val ? gs_tmp : max_val;
            }
            group_scores[max_idx] = -INFINITY;
        }
    }

    float sum = 0.0f;

    if constexpr(NUM_GRP == 8)
    {
        constexpr int experts_per_group___ = 32;
        constexpr int final_score_vec      = 2;

        using final_score_vec_t = ck_tile::ext_vector_t<float, final_score_vec>;
        using final_expid_vec_t = ck_tile::ext_vector_t<int, final_score_vec>;
        final_score_vec_t s;
        final_expid_vec_t e;
        final_expid_vec_t remapped_group_ids;

        for(int i = 0; i < final_score_vec; i++)
        {
            int expert_group_id        = (threadIdx.x + i * 64) / experts_per_group___; //
            int expert_id_inside_group = threadIdx.x % experts_per_group___;
            remapped_group_ids[i]      = group_map_idx[expert_group_id];
            e[i] = remapped_group_ids[i] * experts_per_group___ + expert_id_inside_group;
            s[i] = scores[remapped_group_ids[i] * experts_per_group___ + expert_id_inside_group];
        }

#if 1
        float o_sorted_32_0 = warp_bitonic_merge_sort_build_with_early_stop(s[0], threadIdx.x, ck_tile::number<64>{}, ck_tile::number<32>{}, ck_tile::number<1>{});
        float o_sorted_32_1 = warp_bitonic_merge_sort_build_with_early_stop(s[1], threadIdx.x, ck_tile::number<64>{}, ck_tile::number<32>{}, ck_tile::number<1>{});

        // descending
        // 0..>..15, 16..<..31, 32..>..47, 48..<..63
        // 0..7=>0..7, 24..31=> 8..15, 32..39=>16..23, 56..63=>24..31
        // 0           16              48              32
        // now we rearrange the data, pick up 8 out of 16 number, which could just fit in wave64
        // lane0~31 hold o_sorted_32_0 (8 out of 16), lane 32~63 hold o_sorted_32_1 (8 out of 16)
        int half_lane_id = threadIdx.x % 32;
        int m8_gid = half_lane_id / 8;
        int twi_0_ = m8_gid ^ (m8_gid / 2); // 0,1,2,3 ^ 0,0,1,1 -> 0,1,3,2
        int twi_1_ = twi_0_ * 16;

        float o_t_0 = __shfl(o_sorted_32_0, half_lane_id ^ twi_1_);
        float o_t_1 = __shfl(o_sorted_32_1, half_lane_id ^ twi_1_);
        float o_t  = threadIdx.x < 32 ? o_t_0 : o_t_1;
        float o_r = warp_swap_(o_t, threadIdx.x, ck_tile::number<16>{});
        // 0..>..15, 16..<..31, 32..>..47, 48..<..63
        float o_y = warp_bitonic_merge_sort_combine(o_t, o_r, threadIdx.x, (threadIdx.x / 16) & 1, ck_tile::number<16>{}, ck_tile::number<1>{});
        float o_q = __shfl(o_y, threadIdx.x ^ twi_1_);
        float o_w = warp_swap_(o_q, threadIdx.x, ck_tile::number<16>{});
        // 0..>..15, 16..<..31
        float o_o = warp_bitonic_merge_sort_combine(o_q, o_w, threadIdx.x, (threadIdx.x / 16) & 1, ck_tile::number<16>{}, ck_tile::number<1>{});
        float o_p = __shfl(o_o, threadIdx.x ^ twi_1_);
        float o_z = warp_swap_(o_p, threadIdx.x, ck_tile::number<16>{});
        // 0..>..15,
        float o_n = warp_bitonic_merge_sort_combine(o_p, o_z, threadIdx.x, 0, ck_tile::number<16>{}, ck_tile::number<1>{});
        float pivot = __shfl(o_n, 7);
#else
        auto bitonic_get = [&](float v_, auto is_descending_){
            
            float o_x = warp_bitonic_merge_sort_build_with_early_stop(v_, threadIdx.x, ck_tile::number<64>{}, ck_tile::number<32>{}, ck_tile::number<1>{});
            // descending
            // 0..>..15, 16..<..31, 32..>..47, 48..<..63
            // 0..7=>0..7, 24..31=> 8..15, 32..39=>16..23, 56..63=>24..31
            // 0           16              48              32
            // 
            int m8_gid = threadIdx.x / 8;
            int twi_0_ = m8_gid ^ (m8_gid / 2); // 0,1,2,3 ^ 0,0,1,1 -> 0,1,3,2
            int twi_1_ = twi_0_ * 16;
            float o_t = __shfl(o_x, threadIdx.x ^ twi_1_);
            float o_r = warp_swap_(o_t, threadIdx.x, ck_tile::number<16>{});
            

            // 0..>..15, 16..<..31,
            float o_y = warp_bitonic_merge_sort_combine(o_t, o_r, threadIdx.x, (threadIdx.x / 16) & 1, ck_tile::number<16>{}, ck_tile::number<1>{});
            float o_q = __shfl(o_y, threadIdx.x ^ twi_1_);

            
            float o_w = warp_swap_(o_q, threadIdx.x, ck_tile::number<16>{});
            float o_o = warp_bitonic_merge_sort_combine(o_q, o_w, threadIdx.x, 0, ck_tile::number<16>{}, is_descending_);
            // printf("[%2d] v:%f, o_x:%f, o_t:%f, o_r:%f o_y:%f, o_q:%f, o_w:%f, o_o:%f\n", threadIdx.x, v_, o_x, o_t, o_r, o_y, o_q, o_w, o_o);
            return o_o;
        };

        // o_0: 0..>.16, o_1: 0..<..15
        float o_0 = bitonic_get(s[0], ck_tile::number<1>{});
        float o_1 = bitonic_get(s[1], ck_tile::number<0>{});
        float o_m = ((threadIdx.x / 8) == 0) ? o_0 : o_1;
        float o_t = warp_swap_(o_m, threadIdx.x, ck_tile::number<16>{});
        float o_sorted = warp_bitonic_merge_sort_combine(o_m, o_t, threadIdx.x, 0, ck_tile::number<16>{}, ck_tile::number<1>{});
        float pivot = __shfl(o_sorted, 7);
#endif
        int offset = 0;
        final_expid_vec_t cumsum_cnt{0};
        // 1st
        for(int i = 0; i < final_score_vec; i++)
        {
            int cnt_ = s[i] > pivot ? 1 : 0;
            warp_cumsum(cnt_, ck_tile::number<64>{});
            cnt_ += offset;

            offset        = __builtin_amdgcn_readlane(cnt_, 63);
            cumsum_cnt[i] = cnt_;
        }
        // 2nd
        for(int i = 0; i < final_score_vec; i++)
        {
            int cnt_ = s[i] == pivot ? 1 : 0;
            warp_cumsum(cnt_, ck_tile::number<64>{});
            cnt_ += offset;

            offset        = __builtin_amdgcn_readlane(cnt_, 63);
            cumsum_cnt[i] = s[i] == pivot ? cnt_ : cumsum_cnt[i];
        }

#if AITER_TOPK_SOFTMAX_GROUP_PERMUTE_SCORE
        float topk_vs[final_score_vec];
        int topk_is[final_score_vec];
#if AITER_TOPK_SOFTMAX_GROUP_PERMUTE_SCORE_USE_INLINE_ASM
        {
            int remote_lane_id_0 = (s[0] >= pivot && cumsum_cnt[0] <= topk) ?  ((cumsum_cnt[0] - 1) << 2) : (topk << 2);
            int remote_lane_id_1 = (s[1] >= pivot && cumsum_cnt[1] <= topk) ?  ((cumsum_cnt[1] - 1) << 2) : (topk << 2);
            asm volatile(
                "ds_permute_b32 %[v_topk_v_0], %[v_lane_id_0], %[v_src_v_0]\n"
                "ds_permute_b32 %[v_topk_v_1], %[v_lane_id_1], %[v_src_v_1]\n"
                "ds_permute_b32 %[v_topk_i_0], %[v_lane_id_0], %[v_src_i_0]\n"
                "ds_permute_b32 %[v_topk_i_1], %[v_lane_id_1], %[v_src_i_1]\n"
                "s_waitcnt lgkmcnt(0)\n"
                :
                [v_topk_v_0]"+v"(topk_vs[0]),
                [v_topk_i_0]"+v"(topk_is[0]),
                [v_topk_v_1]"+v"(topk_vs[1]),
                [v_topk_i_1]"+v"(topk_is[1]),
                [v_lane_id_0]"+v"(remote_lane_id_0),
                [v_lane_id_1]"+v"(remote_lane_id_1),
                [v_src_v_0]"+v"(s[0]),
                [v_src_v_1]"+v"(s[1]),
                [v_src_i_0]"+v"(e[0]),
                [v_src_i_1]"+v"(e[1])
                :
                :
            );
        }
#else
        ck_tile::static_for<0, final_score_vec, 1>{}([&](auto i_){
            constexpr int i = i_.value;

            int remote_lane_id = (s[i] >= pivot && cumsum_cnt[i] <= topk) ?  ((cumsum_cnt[i] - 1) << 2) : (topk << 2);
            int s_ = __builtin_bit_cast(int, s[i]);
            int e_ = __builtin_bit_cast(int, e[i]);
            topk_vs[i] = __builtin_bit_cast(float,
                    __builtin_amdgcn_ds_permute(remote_lane_id, s_));
            topk_is[i] = __builtin_bit_cast(int,
                    __builtin_amdgcn_ds_permute(remote_lane_id, e_));
            // printf("[%2d:%d] remote:%d valid:%d topk_vs:%f, topk_is:%d (s:%f, e:%d)\n", threadIdx.x, i, remote_lane_id >> 2,  (s[i] >= pivot && cumsum_cnt[i] <= topk)? 1 : 0, 
            //   topk_vs[i],  topk_is[i]  , s[i], e[i]);
        });
#endif
        float topk_v = topk_vs[0] + topk_vs[1];
        int topk_i = topk_is[0] + topk_is[1];

        if(threadIdx.x < topk)
        {
            if constexpr(isBiased)
            {
                topk_v -= ck_tile::type_convert<float>(correction_bias[topk_i]);
            }
            if(need_renorm)
            {
                sum    = multithread_reduce(topk_v, [&](auto x_, auto y_) { return x_ + y_; }, 8);
                topk_v = topk_v *  routed_scaling_factor * __builtin_amdgcn_rcpf(sum);
            }
            topk_weights[token_idx * stride_tk + threadIdx.x] = topk_v;
            topk_ids[token_idx * stride_tk + threadIdx.x]     = topk_i;
        }
#else
        for(int i = 0; i < final_score_vec; i++)
        {
            if(s[i] >= pivot && cumsum_cnt[i] <= topk)
            {
                int expert_id_inside_group = threadIdx.x % experts_per_group___;
                final_topk_idx[cumsum_cnt[i] - 1] = e[i] ;
                topk_values[cumsum_cnt[i] - 1] = s[i];
            }
        }

        __syncthreads();

        if(threadIdx.x < topk)
        {
            float topk_v = topk_values[threadIdx.x];
            int topk_i   = final_topk_idx[threadIdx.x];
            if constexpr(isBiased)
            {
                topk_v -= ck_tile::type_convert<float>(correction_bias[topk_i]);
            }
            if(need_renorm)
            {
                sum    = multithread_reduce(topk_v, [&](auto x_, auto y_) { return x_ + y_; }, 8);
                topk_v = topk_v *  routed_scaling_factor * __builtin_amdgcn_rcpf(sum);
            }
            topk_weights[token_idx * stride_tk + threadIdx.x] = topk_v;
            topk_ids[token_idx * stride_tk + threadIdx.x]     = topk_i;
        }
#endif
    }
}
} // namespace aiter

#define LAUNCH_KERNEL()                                    \
    switch(num_experts % 4)                                \
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
#define LAUNCHER2(VEC_F)                                                                    \
    switch(num_expert_group)                                                                \
    {                                                                                       \
    case 8: LAUNCHER3(VEC_F, 8) break;                                                      \
    case 4: LAUNCHER3(VEC_F, 4) break;                                                      \
    case 2: LAUNCHER3(VEC_F, 2) break;                                                      \
    case 1: LAUNCHER3(VEC_F, 1) break;                                                      \
    default: TORCH_CHECK(false, "Unsupported num_expert_group: ", num_expert_group); break; \
    }
#define LAUNCHER3(VEC_F, NUM_GRP)                     \
    switch(need_renorm)                               \
    {                                                 \
    case true: LAUNCHER4(VEC_F, NUM_GRP, true) break; \
    default: LAUNCHER4(VEC_F, NUM_GRP, false)         \
    }

#define LAUNCHER4(VEC_F, NUM_GRP, need_renorm)                                                     \
    if constexpr(isBiased)                                                                         \
    {                                                                                              \
        if(use_opt_sort)                                                                           \
        {                                                                                          \
            LAUNCHER_biased_grouped_topk_opt_sort_kernel(VEC_F, NUM_GRP, need_renorm, true, false) \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            LAUNCHER_biased_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, true, false)          \
        }                                                                                          \
    }                                                                                              \
    else                                                                                           \
    {                                                                                              \
        if(isSoftmax)                                                                              \
        {                                                                                          \
            LAUNCHER_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, false, true)                 \
        }                                                                                          \
        else                                                                                       \
        {                                                                                          \
            LAUNCHER_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, false, false)                \
        }                                                                                          \
    }

#define LAUNCHER_biased_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax)      \
    VLLM_DISPATCH_FLOATING_TYPES(gating_output.scalar_type(), "biased_grouped_topk_kernel", [&] {  \
        hipLaunchKernelGGL(                                                                        \
            (aiter::                                                                               \
                 grouped_topk_kernel<scalar_t, VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax>), \
            dim3(grid),                                                                            \
            dim3(block),                                                                           \
            shared_mem_size,                                                                       \
            stream,                                                                                \
            gating_output.data_ptr<scalar_t>(),                                                    \
            correction_bias.data_ptr<scalar_t>(),                                                  \
            topk_weights.data_ptr<float>(),                                                        \
            topk_ids.data_ptr<int>(),                                                              \
            stride_tk,                                                                             \
            num_experts,                                                                           \
            topk,                                                                                  \
            topk_grp,                                                                              \
            num_tokens,                                                                            \
            routed_scaling_factor);                                                                \
    });

#define LAUNCHER_grouped_topk_kernel(VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax)             \
    VLLM_DISPATCH_FLOATING_TYPES(gating_output.scalar_type(), "grouped_topk_kernel", [&] {         \
        hipLaunchKernelGGL(                                                                        \
            (aiter::                                                                               \
                 grouped_topk_kernel<scalar_t, VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax>), \
            dim3(grid),                                                                            \
            dim3(block),                                                                           \
            shared_mem_size,                                                                       \
            stream,                                                                                \
            gating_output.data_ptr<scalar_t>(),                                                    \
            nullptr,                                                                               \
            topk_weights.data_ptr<float>(),                                                        \
            topk_ids.data_ptr<int>(),                                                              \
            stride_tk,                                                                             \
            num_experts,                                                                           \
            topk,                                                                                  \
            topk_grp,                                                                              \
            num_tokens,                                                                            \
            routed_scaling_factor);                                                                \
    });

#define LAUNCHER_biased_grouped_topk_opt_sort_kernel(                             \
    VEC_F, NUM_GRP, need_renorm, isBiased, isSoftmax)                             \
    VLLM_DISPATCH_FLOATING_TYPES(                                                 \
        gating_output.scalar_type(), "biased_grouped_topk_opt_sort_kernel", [&] { \
            hipLaunchKernelGGL((aiter::grouped_topk_opt_sort_kernel<scalar_t,     \
                                                                    VEC_F,        \
                                                                    NUM_GRP,      \
                                                                    need_renorm,  \
                                                                    isBiased,     \
                                                                    isSoftmax>),  \
                               dim3(grid),                                        \
                               dim3(block),                                       \
                               shared_mem_size,                                   \
                               stream,                                            \
                               gating_output.data_ptr<scalar_t>(),                \
                               correction_bias.data_ptr<scalar_t>(),              \
                               topk_weights.data_ptr<float>(),                    \
                               topk_ids.data_ptr<int>(),                          \
                               stride_tk,                                         \
                               num_experts,                                       \
                               topk,                                              \
                               topk_grp,                                          \
                               num_tokens,                                        \
                               routed_scaling_factor);                            \
        });

void biased_grouped_topk(torch::Tensor& gating_output,   // [num_tokens, num_experts]
                         torch::Tensor& correction_bias, // [num_expert]
                         torch::Tensor& topk_weights,    // [num_tokens, topk]
                         torch::Tensor& topk_ids,        // [num_tokens, topk]
                         int num_expert_group,
                         int topk_grp,
                         bool need_renorm,
                         const float routed_scaling_factor = 1.)
{
    const bool isBiased = true;
    bool isSoftmax      = false;
    int num_tokens      = gating_output.size(0);
    int num_experts     = gating_output.size(1);
    int topk            = topk_ids.size(1);
    size_t stride_tk    = topk_ids.stride(0);
    TORCH_CHECK(stride_tk == topk_weights.stride(0),
                "topk_ids.stride(0) == topk_weights.stride(0)");
    TORCH_CHECK(gating_output.dtype() == correction_bias.dtype(),
                "gating_output.dtype() == correction_bias.dtype()");

    // TODO: expand usage in the future
    // bool use_opt_sort = false;
    bool use_opt_sort = (topk == 8) && (num_expert_group == 8) && (num_experts == 256) &&
                        (topk_grp == 4) && (isBiased == true);

    dim3 grid(num_tokens);
    dim3 block(64);
    size_t shared_mem_size = (num_experts * sizeof(float) + num_expert_group * sizeof(float));
    shared_mem_size += !use_opt_sort
                           ? 0
                           : (num_expert_group * sizeof(int) /*group_map_idx*/
                              + topk * sizeof(int)           /*idx+weight*/
                              + topk * sizeof(float)         /*idx+weight*/
                              //   + num_experts * sizeof(float)                         /*bias*/
                              + (topk > topk_grp ? topk : topk_grp) * sizeof(int)   /* sort_k*/
                              + (topk > topk_grp ? topk : topk_grp) * sizeof(float) /* sort_v*/
                              //    + 64 / num_expert_group * sizeof(float) /* for sorting */
                             );

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    LAUNCH_KERNEL()
}

void grouped_topk(torch::Tensor& gating_output, // [num_tokens, num_experts]
                  torch::Tensor& topk_weights,  // [num_tokens, topk]
                  torch::Tensor& topk_ids,      // [num_tokens, topk]
                  int num_expert_group,
                  int topk_grp,
                  bool need_renorm,
                  bool is_softmax                   = true,
                  const float routed_scaling_factor = 1.)
{
    const bool isBiased  = false;
    bool isSoftmax       = is_softmax;
    int num_tokens       = gating_output.size(0);
    int num_experts      = gating_output.size(1);
    int topk             = topk_ids.size(1);
    size_t stride_tk     = topk_ids.stride(0);
    auto correction_bias = topk_ids;
    TORCH_CHECK(stride_tk == topk_weights.stride(0),
                "topk_ids.stride(0) == topk_weights.stride(0)");

    // TODO: expand usage in the future
    bool use_opt_sort = false;

    dim3 grid(num_tokens);
    dim3 block(64);
    size_t shared_mem_size = (num_experts * sizeof(float) + (num_expert_group + 1) * sizeof(float) +
                              topk * sizeof(int) + topk * sizeof(float) + 255) &
                             ~255;

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));
    const hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();

    LAUNCH_KERNEL()
}

#undef LAUNCHER4
#undef LAUNCHER3
#undef LAUNCHER2
#undef LAUNCH_KERNEL
