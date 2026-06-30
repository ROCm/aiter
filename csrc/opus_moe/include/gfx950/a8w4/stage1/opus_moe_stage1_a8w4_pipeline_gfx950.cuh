// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_stage1_a8w4_epilogue_gfx950.cuh"
#include "opus_moe_stage1_a8w4_mainloop_gfx950.cuh"

#include <hip/hip_runtime.h>

namespace opus_moe
{
namespace stage1_a8w4
{

// ============================================================================
// Pipeline and kernel: compose the stage1 phases and expose the launch symbol.
// ============================================================================

template<typename Traits>
struct OpusMoeStage1A8W4Pipeline
{
#ifdef __HIP_DEVICE_COMPILE__
    static __device__ bool process_tile(
        const OpusMoeStage1A8W4Kargs& kargs,
        const Tile<Traits>& tile,
        uint8_t* __restrict__ smem_scratch,
        int* __restrict__ smem_token,
        int* __restrict__ smem_slot,
        uint8_t* __restrict__ smem_route_valid)
    {
        using D_ACC = opus::fp32_t;
        using D_MFMA_A = opus::fp8_t;
        using D_MFMA_B = opus::fp4_t;

        if(tile.route_base >= tile.valid_rows || tile.expert_id < 0 ||
           tile.expert_id >= Traits::EXPERTS)
            return false;

        const bool has_route = load_route_metadata_to_smem<Traits>(
            kargs, tile, smem_token, smem_slot, smem_route_valid);
        if(!has_route)
            return false;

        auto mma = opus::make_mfma<D_MFMA_A, D_MFMA_B, D_ACC>(
            opus::number<Traits::MMA_M>{},
            opus::number<Traits::MMA_N>{},
            opus::number<Traits::MMA_K>{});

        const int lane_id = tile.tid & (opus::get_warp_size() - 1);
        const int wave_id = tile.tid / opus::get_warp_size();
        auto u_ga = make_layout_ga<Traits>(lane_id, 0);
        auto u_gb = make_layout_gb<Traits>(lane_id, 0);

        typename decltype(mma)::vtype_c
            rc[Traits::M_MFMA_PER_WAVE]
              [Traits::ACC_SCALE_GROUPS_PER_TILE]{};

        if constexpr(Traits::GATE_UP_GROUP_SPLIT)
        {
            mainloop_gate_up_group_split<Traits>(
                mma,
                u_ga,
                u_gb,
                kargs,
                tile,
                wave_id,
                lane_id,
                smem_token,
                smem_slot,
                smem_route_valid,
                smem_scratch,
                rc);
            quant_epilogue_gate_up_group_split<Traits>(
                kargs,
                tile,
                u_ga,
                rc,
                wave_id,
                smem_token,
                smem_slot,
                smem_route_valid,
                reinterpret_cast<float*>(smem_scratch));
        }
        else
        {
            mainloop<Traits>(
                mma,
                u_ga,
                u_gb,
                kargs,
                tile,
                wave_id,
                lane_id,
                smem_token,
                smem_slot,
                smem_route_valid,
                smem_scratch,
                rc);
            quant_epilogue<Traits>(
                kargs,
                tile,
                u_ga,
                rc,
                wave_id,
                smem_token,
                smem_slot,
                smem_route_valid,
                reinterpret_cast<float*>(smem_scratch));
        }
        return true;
    }
#endif

    static __device__ void run(const OpusMoeStage1A8W4Kargs& kargs)
    {
#ifdef __HIP_DEVICE_COMPILE__
        __shared__ __align__(Traits::BYTES_PER_VEC) uint8_t
            smem_scratch[Traits::SHARED_SCRATCH_BYTES];
        __shared__ int smem_token[Traits::B_M];
        __shared__ int smem_slot[Traits::B_M];
        __shared__ uint8_t smem_route_valid[Traits::B_M];

        const auto tile = make_tile<Traits>(kargs);
        process_tile(
            kargs,
            tile,
            smem_scratch,
            smem_token,
            smem_slot,
            smem_route_valid);

#else
        (void)kargs;
#endif
    }
};

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS_PER_CU) void
opus_moe_stage1_a8w4_kernel_gfx950(OpusMoeStage1A8W4Kargs kargs)
{
    OpusMoeStage1A8W4Pipeline<Traits>::run(kargs);
}

} // namespace stage1_a8w4
} // namespace opus_moe
