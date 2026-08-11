// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "../../opus_moe_backward_common.cuh"
#include "opus_moe_backward_traits_gfx950.cuh"

#include "aiter_hip_common.h"
#include "opus/opus.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>

namespace opus_moe_backward::gfx950
{

template<typename Traits, int TopK>
__device__ __forceinline__ void
bias_dscore_process_tile_gfx950(BiasBwdKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    constexpr int routes_per_block = T::B_M;
    constexpr int group_size = T::DSCORE_GROUP_SIZE;
    static_assert(routes_per_block * group_size == T::BLOCK_SIZE);
    static_assert(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor ||
                  (TopK >= 1 && TopK <= T::MAX_TOPK));

    const int tid = static_cast<int>(threadIdx.x);
    const int route_tile = static_cast<int>(blockIdx.x);
    const int expert = kargs.route.sorted_expert_ids[route_tile];
    if(expert < 0 || expert >= kargs.route.num_experts)
        return;

    extern __shared__ hip_bfloat16 shared_b2[];
    for(int col = tid; col < kargs.model_dim; col += T::BLOCK_SIZE)
        shared_b2[col] =
            kargs.b2[static_cast<int64_t>(expert) * kargs.stride_b2_e + col];
    __syncthreads();

    const int local_route = tid / group_size;
    const int lane = tid % group_size;
    const int sorted_row = route_tile * kargs.route.sort_block_m + local_route;
    const int valid_rows = kargs.route.num_valid_ids[0];
    bool valid = sorted_row < valid_rows;
    const int32_t encoded =
        valid ? kargs.route.sorted_token_ids[sorted_row] : -1;
    const auto decoded = decode_sorted_route<T::ROUTE_LAYOUT>(
        kargs.route, encoded, valid);
    const int token = decoded.token;
    valid = decoded.valid;

    float partial = 0.0f;
    if(valid)
    {
        const int64_t do_base =
            static_cast<int64_t>(token) * kargs.stride_do_t;
        for(int col = lane; col < kargs.model_dim; col += group_size)
            partial += static_cast<float>(kargs.d_out[do_base + col]) *
                       static_cast<float>(shared_b2[col]);
    }
#pragma unroll
    for(int offset = group_size / 2; offset > 0; offset /= 2)
        partial += __shfl_down(partial, offset, group_size);

    if(valid && lane == 0)
        kargs.d_scores[route_value_offset<T::ROUTE_LAYOUT>(
            kargs.route, decoded, kargs.stride_ds_t)] += partial;
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits, int TopK>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void bias_dscore_kernel_gfx950(BiasBwdKargs kargs)
{
    bias_dscore_process_tile_gfx950<Traits, TopK>(kargs);
}

template<typename Traits, int TopK, bool Db1>
__device__ __forceinline__ void
bias_db_process_tile_gfx950(BiasBwdKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    const int expert = static_cast<int>(blockIdx.x);
    constexpr int group_size = T::DB_ROUTE_GROUP_SIZE;
    const int tid = static_cast<int>(threadIdx.x);
    const int route_lane = tid % group_size;
    const int col = static_cast<int>(blockIdx.y) * T::B_N + tid / group_size;
    const int output_dim = Db1 ? 2 * kargs.inter_dim : kargs.model_dim;
    if(expert >= kargs.route.num_experts || col >= output_dim)
        return;

    const int row_begin = kargs.route.expert_offsets[expert];
    const int row_end = kargs.route.expert_offsets[expert + 1];
    float accum = 0.0f;
    for(int row = row_begin + route_lane; row < row_end; row += group_size)
    {
        const int32_t encoded = kargs.route.sorted_token_ids[row];
        const auto decoded = decode_sorted_route<T::ROUTE_LAYOUT>(
            kargs.route, encoded, true);
        if(decoded.valid)
        {
            if constexpr(Db1)
            {
                accum += static_cast<float>(kargs.d_z[
                    static_cast<int64_t>(row) * kargs.stride_dz_r + col]);
            }
            else
            {
                const float score = kargs.scores[
                    route_value_offset<T::ROUTE_LAYOUT>(
                        kargs.route, decoded, kargs.stride_score_t)];
                accum += score * static_cast<float>(kargs.d_out[
                                      static_cast<int64_t>(decoded.token) *
                                              kargs.stride_do_t +
                                          col]);
            }
        }
    }
#pragma unroll
    for(int offset = group_size / 2; offset > 0; offset /= 2)
        accum += __shfl_down(accum, offset, group_size);

    if(route_lane == 0)
    {
        if constexpr(Db1)
            kargs.d_b1[static_cast<int64_t>(expert) * kargs.stride_db1_e + col] =
                hip_bfloat16(accum);
        else
            kargs.d_b2[static_cast<int64_t>(expert) * kargs.stride_db2_e + col] =
                hip_bfloat16(accum);
    }
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits, int TopK, bool Db1>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void bias_db_kernel_gfx950(BiasBwdKargs kargs)
{
    bias_db_process_tile_gfx950<Traits, TopK, Db1>(kargs);
}

template<typename Traits, int TopK>
inline void bias_bwd_launch_topk_gfx950(const BiasBwdKargs& kargs,
                                        hipStream_t stream)
{
    using T = opus::remove_cvref_t<Traits>;
    const dim3 block(T::BLOCK_SIZE);
    if(kargs.compute_dscore)
    {
        const dim3 grid(
            static_cast<unsigned int>(kargs.route.sorted_block_capacity));
        const std::size_t shared_bytes =
            static_cast<std::size_t>(kargs.model_dim) * sizeof(hip_bfloat16);
        hipLaunchKernelGGL((bias_dscore_kernel_gfx950<T, TopK>),
                           grid,
                           block,
                           shared_bytes,
                           stream,
                           kargs);
    }
    if(kargs.compute_db1)
    {
        const dim3 grid(
            static_cast<unsigned int>(kargs.route.num_experts),
            static_cast<unsigned int>(
                (2 * kargs.inter_dim + T::B_N - 1) / T::B_N));
        hipLaunchKernelGGL((bias_db_kernel_gfx950<T, TopK, true>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
    }
    if(kargs.compute_db2)
    {
        const dim3 grid(
            static_cast<unsigned int>(kargs.route.num_experts),
            static_cast<unsigned int>(
                (kargs.model_dim + T::B_N - 1) / T::B_N));
        hipLaunchKernelGGL((bias_db_kernel_gfx950<T, TopK, false>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
    }
}

template<typename Traits>
inline void bias_bwd_launch_gfx950(const BiasBwdKargs& kargs,
                                   hipStream_t stream)
{
    using T = opus::remove_cvref_t<Traits>;
    if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
    {
        bias_bwd_launch_topk_gfx950<Traits, 0>(kargs, stream);
        return;
    }
    AITER_CHECK(kargs.route.topk == 1 || kargs.route.topk == 2 ||
                    kargs.route.topk == 4 || kargs.route.topk == 8,
                "bias_bwd: fixed routing supports topk in {1,2,4,8}");
    switch(kargs.route.topk)
    {
    case 1: bias_bwd_launch_topk_gfx950<Traits, 1>(kargs, stream); break;
    case 2: bias_bwd_launch_topk_gfx950<Traits, 2>(kargs, stream); break;
    case 4: bias_bwd_launch_topk_gfx950<Traits, 4>(kargs, stream); break;
    case 8: bias_bwd_launch_topk_gfx950<Traits, 8>(kargs, stream); break;
    default: break;
    }
}

} // namespace opus_moe_backward::gfx950
