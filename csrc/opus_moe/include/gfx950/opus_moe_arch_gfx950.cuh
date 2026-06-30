// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950-specific Opus MoE dispatch implementations.
#pragma once

#include "../opus_moe_arch.cuh"
#include "../opus_moe_common.cuh"
#include "opus_moe_stage2_route_output_reduce_gfx950.cuh"
#include "a16w16/opus_moe_pipeline_stage2_gemmstyle_gfx950.cuh"
#include "a8w4/opus_moe_pipeline_stage2_a8w4_decode_main_gfx950.cuh"
#include "opus_moe_stage2_manifest.h"

#include "aiter_hip_common.h"

#include <algorithm>
#include <cstddef>
#include <hip/hip_runtime.h>

using OpusMoeStage2Bf16Kernel = void (*)(const opus_moe_stage2_bf16_kargs&,
                                         int,
                                         hipStream_t);

template<typename Traits>
inline void opus_moe_stage2_gemmstyle_launch_gfx950(const opus_moe_stage2_bf16_kargs& kargs,
                                                    int sorted_blocks,
                                                    hipStream_t stream)
{
    AITER_CHECK(kargs.block_m % Traits::B_M == 0,
                "opus_moe stage2 gemmstyle kernel requires block_m to be a multiple of ",
                Traits::B_M,
                ", got ",
                kargs.block_m);
    AITER_CHECK(kargs.model_dim % Traits::B_N == 0,
                "opus_moe stage2 gemmstyle kernel requires model_dim to be a multiple of ",
                Traits::B_N,
                ", got ",
                kargs.model_dim);
    AITER_CHECK(kargs.inter_dim % Traits::B_K == 0,
                "opus_moe stage2 gemmstyle kernel requires inter_dim to be a multiple of ",
                Traits::B_K,
                ", got ",
                kargs.inter_dim);
    const int metadata_tiles = sorted_blocks * (kargs.block_m / Traits::B_M);
    const int route_tiles =
        (kargs.token_num * kargs.topk + Traits::B_M - 1) / Traits::B_M;
    const int m_tiles = std::min(metadata_tiles, route_tiles);
    const int n_tiles = (kargs.model_dim + Traits::B_N - 1) / Traits::B_N;
    dim3 grid(n_tiles, m_tiles, 1);
    dim3 block(Traits::BLOCK_SIZE);
    opus_moe_stage2_gemmstyle_kernel_gfx950<Traits><<<grid, block, 0, stream>>>(kargs);
}

template<typename Traits>
inline void opus_moe_stage2_a8w4_decode_launch_gfx950(
    const opus_moe_stage2_a8w4_kargs& kargs,
    hipStream_t stream)
{
    int route_blocks =
        (kargs.sorted_blocks * Traits::SORT_BLOCK_M + Traits::B_M - 1) /
        Traits::B_M;
    opus_moe_stage2_a8w4_kargs launch_kargs = kargs;
    launch_kargs.sorted_blocks = route_blocks;
    if constexpr(Traits::DECODE_PACE_ROUTE_BLOCKS_TO_POW2)
    {
        int paced_route_blocks = 1;
        while(paced_route_blocks < route_blocks)
            paced_route_blocks <<= 1;
        route_blocks = paced_route_blocks;
        launch_kargs.sorted_blocks = route_blocks;
    }
    AITER_CHECK(kargs.model_dim % Traits::B_N == 0,
                "Opus A8W4 stage2 requires model_dim to be a multiple of block_n=",
                Traits::B_N,
                ", got ",
                kargs.model_dim);
    const int n_tiles = kargs.model_dim / Traits::B_N;
    dim3 grid(n_tiles, route_blocks, 1);
    dim3 block(Traits::BLOCK_SIZE);
    opus_moe_stage2_a8w4_decode_kernel_gfx950<Traits><<<grid, block, 0, stream>>>(
        launch_kargs);
}

namespace opus_moe_gfx950_detail
{
struct OpusMoeStage2TuneEntry
{
    int kid;
    OpusMoeStage2Bf16Kernel func;
};

struct TuneEntryLess
{
    template<typename Entry>
    constexpr bool operator()(const Entry& a, const Entry& b) const noexcept
    {
        return a.kid < b.kid;
    }
};
} // namespace opus_moe_gfx950_detail

inline OpusMoeStage2Bf16Kernel opus_moe_stage2_bf16_tune_dispatch_gfx950(int id)
{
    using namespace opus_moe_gfx950_detail;
    static constexpr OpusMoeStage2TuneEntry kTune[] = {
        GENERATE_OPUS_MOE_STAGE2_BF16_TUNE_LOOKUP
    };
    constexpr size_t kSize = OPUS_MOE_STAGE2_BF16_TUNE_LOOKUP_SIZE;
    static_assert(kSize == sizeof(kTune) / sizeof(kTune[0]));
    const OpusMoeStage2TuneEntry needle{id, nullptr};
    const auto it = std::lower_bound(kTune, kTune + kSize, needle, TuneEntryLess{});
    AITER_CHECK(it != kTune + kSize && it->kid == id,
                "Kernel id ",
                id,
                " (",
                opus_moe::stage2_bf16_kid_name(id),
                ") not found in gfx950 Opus MoE stage2 BF16 tune table");
    return it->func;
}
