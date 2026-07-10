// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_pipeline_gfx950.cuh"
#include "opus_moe_stage1_a8w4_manifest.h"

#include <hip/hip_runtime.h>

namespace opus_moe
{
namespace stage1_a8w4
{

constexpr bool kid_is_valid(int kid)
{
    return opus_moe::stage1_a8w4_kid_is_valid(kid);
}

constexpr const char* kid_name(int kid)
{
    return opus_moe::stage1_a8w4_kid_name(kid);
}

constexpr int kid_sort_block_m(int kid)
{
    return opus_moe::stage1_a8w4_kid_sort_block_m(kid);
}

template<typename Traits>
inline void launch(const OpusMoeStage1A8W4Kargs& kargs, hipStream_t stream)
{
    // sorted route metadata includes per-expert padding; token_num * topk is
    // not enough to cover all valid sorted route tiles.
    int route_tiles =
        (kargs.sorted_blocks * Traits::SORT_BLOCK_M + Traits::B_M - 1) /
        Traits::B_M;
    constexpr int route_subtiles = Traits::SORT_BLOCK_M / Traits::B_M;
    if constexpr(Traits::CAP_ROUTE_TILES_TO_ROUTED_BLOCKS)
    {
        const int routed_blocks = kargs.token_num * kargs.topk * route_subtiles;
        route_tiles =
            route_tiles < routed_blocks ? route_tiles : routed_blocks;
    }
    dim3 grid(Traits::STAGE1_COL_TILES,
              route_tiles,
              1);
    dim3 block(Traits::BLOCK_SIZE);
    opus_moe_stage1_a8w4_kernel_gfx950<Traits><<<grid, block, 0, stream>>>(kargs);
}

inline void dispatch(const OpusMoeStage1A8W4Kargs& kargs, hipStream_t stream)
{
    switch(kargs.kernel_id)
    {
    GENERATE_OPUS_MOE_STAGE1_A8W4_DISPATCH_CASES
    default: AITER_CHECK(false, "unreachable A8W4 stage1 kernel dispatch");
    }
}

} // namespace stage1_a8w4
} // namespace opus_moe
