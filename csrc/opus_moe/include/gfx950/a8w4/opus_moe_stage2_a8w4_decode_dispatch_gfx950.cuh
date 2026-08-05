// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_traits_stage2_a8w4_decode_gfx950.cuh"
#include "opus_moe_stage2_a8w4_manifest.h"

#include "aiter_hip_common.h"

template<typename Traits>
__global__ void
opus_moe_stage2_a8w4_decode_kernel_gfx950(opus_moe_stage2_a8w4_kargs);

template<typename Traits>
inline void opus_moe_stage2_a8w4_decode_launch_gfx950(
    const opus_moe_stage2_a8w4_kargs& kargs,
    hipStream_t stream)
{
    int route_blocks =
        (kargs.sorted_blocks * Traits::SORT_BLOCK_M + Traits::B_M - 1) / Traits::B_M;
    opus_moe_stage2_a8w4_kargs launch_kargs = kargs;
    if constexpr(Traits::DECODE_PACE_ROUTE_BLOCKS_TO_POW2)
    {
        int paced_route_blocks = 1;
        while(paced_route_blocks < route_blocks)
            paced_route_blocks <<= 1;
        route_blocks = paced_route_blocks;
    }
    launch_kargs.sorted_blocks = route_blocks;
    AITER_CHECK(kargs.model_dim % Traits::B_N == 0,
                "Opus A8W4 stage2 requires model_dim to be a multiple of block_n=",
                Traits::B_N,
                ", got ",
                kargs.model_dim);
    const dim3 grid(kargs.model_dim / Traits::B_N, route_blocks);
    const dim3 block(Traits::BLOCK_SIZE);
    opus_moe_stage2_a8w4_decode_kernel_gfx950<Traits>
        <<<grid, block, 0, stream>>>(launch_kargs);
}

inline void opus_moe_stage2_a8w4_decode_dispatch_gfx950(
    int kid,
    int effective_inter_dim,
    const opus_moe_stage2_a8w4_kargs& kargs,
    hipStream_t stream)
{
    switch(kid)
    {
    GENERATE_OPUS_MOE_STAGE2_A8W4_DECODE_DISPATCH_CASES
    default: break;
    }
    AITER_CHECK(false,
                "unreachable A8W4 kernel dispatch for kid=",
                kid,
                " effective_inter_dim=",
                effective_inter_dim);
}
