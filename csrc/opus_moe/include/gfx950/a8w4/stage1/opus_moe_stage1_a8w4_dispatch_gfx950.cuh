// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_traits_gfx950.cuh"
#include "opus_moe_stage1_a8w4_manifest.h"

#include "aiter_hip_common.h"

namespace opus_moe
{
namespace stage1_a8w4
{

namespace pipeline_group_split
{
template<typename Traits>
__global__ void
opus_moe_stage1_a8w4_kernel_group_split_gfx950(OpusMoeStage1A8W4Kargs);
}

namespace pipeline_pair_kwave
{
template<typename Traits>
__global__ void
opus_moe_stage1_a8w4_kernel_pair_kwave_gfx950(OpusMoeStage1A8W4Kargs);
}

template<typename Traits>
inline void launch_gfx950(int effective_inter_dim,
                          int sorted_blocks,
                          const OpusMoeStage1A8W4Kargs& kargs,
                          hipStream_t stream)
{
    const dim3 grid(effective_inter_dim / Traits::OUTPUT_COLS_PER_TILE, sorted_blocks);
    const dim3 block(Traits::BLOCK_SIZE);
    if constexpr(Traits::GATE_UP_GROUP_SPLIT)
    {
        pipeline_group_split::opus_moe_stage1_a8w4_kernel_group_split_gfx950<Traits>
            <<<grid, block, 0, stream>>>(kargs);
    }
    else
    {
        pipeline_pair_kwave::opus_moe_stage1_a8w4_kernel_pair_kwave_gfx950<Traits>
            <<<grid, block, 0, stream>>>(kargs);
    }
}

inline void dispatch(int kid,
                     int effective_inter_dim,
                     int sorted_blocks,
                     const OpusMoeStage1A8W4Kargs& kargs,
                     hipStream_t stream)
{
    switch(kid)
    {
    GENERATE_OPUS_MOE_STAGE1_A8W4_DISPATCH_CASES
    default: break;
    }
    AITER_CHECK(false,
                "unreachable A8W4 stage1 kernel dispatch for kid=",
                kid);
}

} // namespace stage1_a8w4
} // namespace opus_moe
