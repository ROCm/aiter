// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_moe_pipeline_stage2_a8w4_decode_policy_gfx950.cuh"

#include "opus/opus.hpp"

#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx950__)

template<typename T,
         typename IssueAPayload,
         typename WaitAPayload,
         typename LoadBScale,
         typename LoadAScale,
         typename ComputeTile>
inline __device__ void opus_moe_stage2_a8w4_decode_run_generic_pipeline_gfx950(
    int col_base,
    IssueAPayload& issue_a_payload,
    WaitAPayload& wait_a_payload,
    LoadBScale& load_b_scale,
    LoadAScale& load_a_scale,
    ComputeTile& compute_tile)
{
    using namespace opus;

    static_assert(T::K_TILES > 0);
    using Schedule = OpusMoeStage2A8W4DecodeSchedule<T>;
    using MainloopSchedule = OpusMoeStage2A8W4DecodeMainloopSchedule;

    static_for<T::K_TILES>([&](auto kt) {
        issue_a_payload(kt, kt.value * T::K_STEP_PACKED);
    });
    wait_a_payload(number<0>{});

    static_for<(T::K_TILES + 1) / 2>([&](auto pair) {
        constexpr int k_tile0 = pair.value * 2;
        constexpr int k_tile1 = k_tile0 + 1;
        constexpr int scale_word_base =
            pair.value * T::SCALE_WORDS_PER_GROUP_PACK;

        int b_scale[T::HALF_N_MFMA_PER_WAVE];
        int a_scale[T::M_MFMA_PER_WAVE];
        load_b_scale(scale_word_base, b_scale);
        load_a_scale(scale_word_base, a_scale);

        const int b_tile_base0 =
            opus_moe_stage2_a8w4_b_payload_tile_base_byte_offset<T>(
                col_base, k_tile0 * T::K_STEP_PACKED);
        compute_tile(
            number<0>{},
            number<Schedule::Mainloop == MainloopSchedule::SplitALoadByNWave>{},
            number<k_tile0>{},
            b_tile_base0,
            b_scale,
            a_scale);

        if constexpr(k_tile1 < T::K_TILES)
        {
            const int b_tile_base1 =
                opus_moe_stage2_a8w4_b_payload_tile_base_byte_offset<T>(
                    col_base, k_tile1 * T::K_STEP_PACKED);
            compute_tile(
                number<1>{},
                number<Schedule::Mainloop == MainloopSchedule::SplitALoadByNWave>{},
                number<k_tile1>{},
                b_tile_base1,
                b_scale,
                a_scale);
        }
    });
}

#endif // __gfx950__
#endif // __HIP_DEVICE_COMPILE__
