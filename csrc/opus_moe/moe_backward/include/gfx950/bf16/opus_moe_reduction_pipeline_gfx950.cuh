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

// ---- Route-to-token dX reduction ----------------------------------------
namespace opus_moe_backward::gfx950
{

#ifdef __HIP_DEVICE_COMPILE__

inline __device__ uint32_t route_reduce_cvt_pk_bf16_f32(float lo, float hi)
{
    uint32_t packed;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2"
                 : "=v"(packed)
                 : "v"(lo), "v"(hi));
    return packed;
}

inline __device__ uint64_t
route_reduce_pack_bf16x4(float v0, float v1, float v2, float v3)
{
    const uint32_t packed01 = route_reduce_cvt_pk_bf16_f32(v0, v1);
    const uint32_t packed23 = route_reduce_cvt_pk_bf16_f32(v2, v3);
    return static_cast<uint64_t>(packed01) |
           (static_cast<uint64_t>(packed23) << 32);
}

using route_reduce_f32x2 = float __attribute__((ext_vector_type(2)));
using route_reduce_u32x4 = uint32_t __attribute__((ext_vector_type(4)));

inline __device__ route_reduce_f32x2
route_reduce_unpack_bf16x2(uint32_t packed)
{
    return route_reduce_f32x2{
        __builtin_bit_cast(float, packed << 16),
        __builtin_bit_cast(float, packed & 0xffff0000u)};
}

#endif

template<typename Traits, int TopK>
__device__ __forceinline__ void
route_reduce_process_tile_gfx950(RouteReduceKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int VEC = T::VEC;
    constexpr int threads_per_row = BN / VEC;
    static_assert(BM * threads_per_row == T::BLOCK_SIZE);
    static_assert(TopK >= 1 && TopK <= T::MAX_TOPK);

    const int tid = static_cast<int>(threadIdx.x);
    const int local_m = tid / threads_per_row;
    const int lane_n = tid % threads_per_row;
    const int token = static_cast<int>(blockIdx.x) * BM + local_m;
    const int col = static_cast<int>(blockIdx.y) * BN + lane_n * VEC;

    if(token >= kargs.route.token_num || col + VEC > kargs.model_dim)
        return;

    static_assert(VEC == 8);
    route_reduce_f32x2 accum[VEC / 2];
    opus::gmem<opus::bf16_t> route_gmem(kargs.d_x_route);

    const auto load_route = [&](int route) {
        const int64_t element_offset =
            static_cast<int64_t>(route) * kargs.stride_dx_route_r + col;
        if constexpr(T::CACHECTL_ROUTE == 0)
        {
            return *reinterpret_cast<const route_reduce_u32x4*>(
                kargs.d_x_route + element_offset);
        }
        else
        {
            const auto values = route_gmem.template load<VEC>(
                element_offset,
                0,
                opus::number<T::CACHECTL_ROUTE>{});
            return __builtin_bit_cast(route_reduce_u32x4, values);
        }
    };

    const int logical_base = token * TopK;
    constexpr int route_broadcast_width =
        threads_per_row < 64 ? threads_per_row : 64;
    static_assert(!T::BROADCAST_ROUTE_ID ||
                  (route_broadcast_width > 0 &&
                   64 % route_broadcast_width == 0));
    const int route_lane = lane_n % route_broadcast_width;
    int distributed_route = 0;
    if constexpr(T::DISTRIBUTE_ROUTE_IDS)
        distributed_route = route_lane < TopK
                                ? kargs.route.reverse_sorted[logical_base +
                                                              route_lane]
                                : 0;
    const auto route_for_slot = [&](int slot) {
        if constexpr(T::DISTRIBUTE_ROUTE_IDS)
            return __shfl(distributed_route, slot, route_broadcast_width);
        int route = T::READ_SORTED_ROUTES
                        ? ((!T::BROADCAST_ROUTE_ID || route_lane == 0)
                               ? kargs.route.reverse_sorted[logical_base + slot]
                               : 0)
                        : logical_base + slot;
        if constexpr(T::BROADCAST_ROUTE_ID)
            route = __shfl(route, 0, route_broadcast_width);
        return route;
    };

    const int first_route = route_for_slot(0);
    const auto first = load_route(first_route);
#pragma unroll
    for(int pair = 0; pair < VEC / 2; ++pair)
        accum[pair] = route_reduce_unpack_bf16x2(first[pair]);

#pragma unroll
    for(int slot = 1; slot < TopK; ++slot)
    {
        const int route = route_for_slot(slot);
        const auto values = load_route(route);
#pragma unroll
        for(int pair = 0; pair < VEC / 2; ++pair)
            accum[pair] += route_reduce_unpack_bf16x2(values[pair]);
    }

    using u64x2 = uint64_t __attribute__((ext_vector_type(2)));
    *reinterpret_cast<u64x2*>(
        kargs.d_x + static_cast<int64_t>(token) * kargs.stride_dx_t + col) =
        u64x2{route_reduce_pack_bf16x4(
                  accum[0][0], accum[0][1], accum[1][0], accum[1][1]),
              route_reduce_pack_bf16x4(
                  accum[2][0], accum[2][1], accum[3][0], accum[3][1])};
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits, int TopK>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void route_reduce_kernel_gfx950(RouteReduceKargs kargs)
{
    route_reduce_process_tile_gfx950<Traits, TopK>(kargs);
}

template<typename Traits>
__device__ __forceinline__ void
route_reduce_varlen_process_tile_gfx950(RouteReduceKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    constexpr int VEC = T::VEC;
    constexpr int threads_per_row = BN / VEC;
    static_assert(BM * threads_per_row == T::BLOCK_SIZE);
    static_assert(VEC == 8);

    const int tid = static_cast<int>(threadIdx.x);
    const int local_m = tid / threads_per_row;
    const int lane_n = tid % threads_per_row;
    const int token = static_cast<int>(blockIdx.x) * BM + local_m;
    const int col = static_cast<int>(blockIdx.y) * BN + lane_n * VEC;
    if(token >= kargs.route.token_num || col + VEC > kargs.model_dim)
        return;

    route_reduce_f32x2 accum[VEC / 2];
#pragma unroll
    for(int pair = 0; pair < VEC / 2; ++pair)
        accum[pair] = route_reduce_f32x2{0.0f, 0.0f};

    const int route_begin = kargs.route.token_route_offsets[token];
    const int route_end = kargs.route.token_route_offsets[token + 1];
    for(int route = route_begin; route < route_end; ++route)
    {
        const auto values = *reinterpret_cast<const route_reduce_u32x4*>(
            kargs.d_x_route +
            static_cast<int64_t>(route) * kargs.stride_dx_route_r + col);
#pragma unroll
        for(int pair = 0; pair < VEC / 2; ++pair)
            accum[pair] += route_reduce_unpack_bf16x2(values[pair]);
    }

    using u64x2 = uint64_t __attribute__((ext_vector_type(2)));
    *reinterpret_cast<u64x2*>(
        kargs.d_x + static_cast<int64_t>(token) * kargs.stride_dx_t + col) =
        u64x2{route_reduce_pack_bf16x4(
                  accum[0][0], accum[0][1], accum[1][0], accum[1][1]),
              route_reduce_pack_bf16x4(
                  accum[2][0], accum[2][1], accum[3][0], accum[3][1])};
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void route_reduce_varlen_kernel_gfx950(RouteReduceKargs kargs)
{
    route_reduce_varlen_process_tile_gfx950<Traits>(kargs);
}

template<typename Traits>
inline void route_reduce_launch_gfx950(const RouteReduceKargs& kargs,
                                       hipStream_t stream)
{
    using T = opus::remove_cvref_t<Traits>;
    AITER_CHECK(kargs.model_dim % T::B_N == 0,
                "route_reduce: D must be divisible by ",
                T::B_N);
    const dim3 grid(
        static_cast<unsigned int>(
            (kargs.route.token_num + T::B_M - 1) / T::B_M),
        static_cast<unsigned int>(kargs.model_dim / T::B_N));
    const dim3 block(T::BLOCK_SIZE);
    if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
    {
        AITER_CHECK(kargs.route.token_route_offsets != nullptr,
                    "route_reduce varlen: token_route_offsets are required");
        hipLaunchKernelGGL((route_reduce_varlen_kernel_gfx950<T>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        return;
    }
    if constexpr(T::READ_SORTED_ROUTES)
        AITER_CHECK(kargs.route.reverse_sorted != nullptr,
                    "route_reduce sorted input: reverse_sorted is required");
    AITER_CHECK(kargs.route.topk == 1 || kargs.route.topk == 2 ||
                    kargs.route.topk == 4 || kargs.route.topk == 8,
                "route_reduce: first instance supports topk in {1,2,4,8}");
    switch(kargs.route.topk)
    {
    case 1:
        hipLaunchKernelGGL((route_reduce_kernel_gfx950<T, 1>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        break;
    case 2:
        hipLaunchKernelGGL((route_reduce_kernel_gfx950<T, 2>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        break;
    case 4:
        hipLaunchKernelGGL((route_reduce_kernel_gfx950<T, 4>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        break;
    case 8:
        hipLaunchKernelGGL((route_reduce_kernel_gfx950<T, 8>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        break;
    default: break;
    }
}

} // namespace opus_moe_backward::gfx950

// ---- Selected-softmax router Jacobian/scatter -----------------------------
namespace opus_moe_backward::gfx950
{

template<typename Traits, int TopK>
__device__ __forceinline__ void
router_bwd_process_tile_gfx950(RouterBwdKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    static_assert(BM * BN == T::BLOCK_SIZE);
    static_assert(TopK >= 1 && TopK <= T::MAX_TOPK);

    const int tid = static_cast<int>(threadIdx.x);
    const int local_token = tid / BN;
    const int local_expert = tid % BN;
    const int token = static_cast<int>(blockIdx.x) * BM + local_token;
    const int expert = static_cast<int>(blockIdx.y) * BN + local_expert;

    if(token >= kargs.token_num || expert >= kargs.num_experts)
        return;

    const int64_t ds_base =
        static_cast<int64_t>(token) * kargs.stride_ds_t;
    const int64_t score_base =
        static_cast<int64_t>(token) * kargs.stride_score_t;
    const int64_t id_base =
        static_cast<int64_t>(token) * kargs.stride_topk_id_t;

    float score_dot_dscore = 0.0f;
#pragma unroll
    for(int slot = 0; slot < TopK; ++slot)
        score_dot_dscore +=
            kargs.scores[score_base + slot] * kargs.d_scores[ds_base + slot];

    float d_logit = 0.0f;
#pragma unroll
    for(int slot = 0; slot < TopK; ++slot)
    {
        if(kargs.topk_ids[id_base + slot] == expert)
        {
            const float score = kargs.scores[score_base + slot];
            d_logit +=
                score * (kargs.d_scores[ds_base + slot] - score_dot_dscore);
        }
    }
    kargs.d_logits[static_cast<int64_t>(token) * kargs.stride_dl_t + expert] =
        d_logit;
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits, int TopK>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void router_bwd_kernel_gfx950(RouterBwdKargs kargs)
{
    router_bwd_process_tile_gfx950<Traits, TopK>(kargs);
}

template<typename Traits>
__device__ __forceinline__ void
router_bwd_varlen_process_tile_gfx950(RouterBwdKargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)
    using T = opus::remove_cvref_t<Traits>;
    constexpr int BM = T::B_M;
    constexpr int BN = T::B_N;
    static_assert(BM * BN == T::BLOCK_SIZE);

    const int tid = static_cast<int>(threadIdx.x);
    const int token =
        static_cast<int>(blockIdx.x) * BM + tid / BN;
    const int expert =
        static_cast<int>(blockIdx.y) * BN + tid % BN;
    if(token >= kargs.token_num || expert >= kargs.num_experts)
        return;

    const int route_begin = kargs.token_route_offsets[token];
    const int route_end = kargs.token_route_offsets[token + 1];
    float score_dot_dscore = 0.0f;
    for(int route = route_begin; route < route_end; ++route)
        score_dot_dscore +=
            kargs.scores[route] * kargs.d_scores[route];

    float d_logit = 0.0f;
    for(int route = route_begin; route < route_end; ++route)
    {
        if(kargs.topk_ids[route] == expert)
        {
            const float score = kargs.scores[route];
            d_logit +=
                score * (kargs.d_scores[route] - score_dot_dscore);
        }
    }
    kargs.d_logits[static_cast<int64_t>(token) * kargs.stride_dl_t + expert] =
        d_logit;
#else
    (void)kargs;
#endif
#else
    (void)kargs;
#endif
}

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU)
void router_bwd_varlen_kernel_gfx950(RouterBwdKargs kargs)
{
    router_bwd_varlen_process_tile_gfx950<Traits>(kargs);
}

template<typename Traits>
inline void router_bwd_launch_gfx950(const RouterBwdKargs& kargs,
                                     hipStream_t stream)
{
    using T = opus::remove_cvref_t<Traits>;
    const dim3 grid(
        static_cast<unsigned int>((kargs.token_num + T::B_M - 1) / T::B_M),
        static_cast<unsigned int>((kargs.num_experts + T::B_N - 1) / T::B_N));
    const dim3 block(T::BLOCK_SIZE);
    if constexpr(T::ROUTE_LAYOUT == RouteLayout::CompactRouteMajor)
    {
        AITER_CHECK(kargs.token_route_offsets != nullptr,
                    "router_bwd varlen: token_route_offsets are required");
        hipLaunchKernelGGL((router_bwd_varlen_kernel_gfx950<T>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        return;
    }
    AITER_CHECK(kargs.topk == 1 || kargs.topk == 2 ||
                    kargs.topk == 4 || kargs.topk == 8,
                "router_bwd: selected-softmax supports topk in {1,2,4,8}");
    switch(kargs.topk)
    {
    case 1:
        hipLaunchKernelGGL((router_bwd_kernel_gfx950<T, 1>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        break;
    case 2:
        hipLaunchKernelGGL((router_bwd_kernel_gfx950<T, 2>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        break;
    case 4:
        hipLaunchKernelGGL((router_bwd_kernel_gfx950<T, 4>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        break;
    case 8:
        hipLaunchKernelGGL((router_bwd_kernel_gfx950<T, 8>),
                           grid,
                           block,
                           0,
                           stream,
                           kargs);
        break;
    default: break;
    }
}

} // namespace opus_moe_backward::gfx950
