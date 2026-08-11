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

    const int64_t route_base =
        static_cast<int64_t>(token * TopK) * kargs.stride_dx_route_r + col;
    const auto first = *reinterpret_cast<const route_reduce_u32x4*>(
        kargs.d_x_route + route_base);
#pragma unroll
    for(int pair = 0; pair < VEC / 2; ++pair)
        accum[pair] = route_reduce_unpack_bf16x2(first[pair]);

#pragma unroll
    for(int slot = 1; slot < TopK; ++slot)
    {
        const auto values = *reinterpret_cast<const route_reduce_u32x4*>(
            kargs.d_x_route + route_base +
            static_cast<int64_t>(slot) * kargs.stride_dx_route_r);
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
