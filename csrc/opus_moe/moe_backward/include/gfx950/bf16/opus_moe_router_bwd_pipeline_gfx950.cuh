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
