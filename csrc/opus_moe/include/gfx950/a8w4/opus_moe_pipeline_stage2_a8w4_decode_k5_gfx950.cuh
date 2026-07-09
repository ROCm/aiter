// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// K5 loop schedule for non-production bring-up; instantiated from shared load/compute helpers.
#pragma once

#include "opus_moe_pipeline_stage2_a8w4_decode_policy_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#include "opus/opus.hpp"

template<typename T,
         typename LayoutA,
         typename LayoutASmem,
         typename SmemA,
         typename GmemA,
         typename GmemAScale,
         typename GmemWScale,
         typename ComputeKTile>
inline __device__ void opus_moe_stage2_a8w4_decode_run_k5_schedule_gfx950(
    int col_base,
    const LayoutA& u_ga,
    const LayoutASmem& u_sa,
    SmemA& s_a,
    GmemA& g_a,
    const int (&a_base)[T::M_MFMA_PER_WAVE],
    int wave_id_n,
    GmemAScale& g_a_scale,
    const int* __restrict__ a_scale_base_word,
    GmemWScale& g_w_scale,
    int b_scale_base_word,
    ComputeKTile& compute_k_tile_both_n_halves)
{
    using namespace opus;

    static_assert(T::K_TILES == 5);

    static_for<T::K_TILES>([&](auto kt) {
        opus_moe_stage2_a8w4_decode_issue_a<T>(
            u_ga, u_sa, s_a, g_a, a_base, wave_id_n, kt, kt.value * T::K_STEP_PACKED);
    });
    opus_moe_stage2_a8w4_decode_wait_a(wave_id_n, opus::number<0>{});

    static_for<(T::K_TILES + 1) / 2>([&](auto pair) {
        constexpr int k_tile0 = pair.value * 2;
        constexpr int k_tile1 = k_tile0 + 1;
        constexpr int scale_word_base = pair.value * T::SCALE_WORDS_PER_GROUP_PACK;
        const int b_tile_base0 =
            col_base * T::B_PAYLOAD_ROW_STRIDE_BYTES +
            k_tile0 * T::K_STEP_PACKED * T::B_PAYLOAD_K_STRIDE_BYTES;

        opus::array<int, T::HALF_N_MFMA_PER_WAVE> b_scale;
        opus::array<int, T::M_MFMA_PER_WAVE> a_scale;
        opus_moe_stage2_a8w4_decode_load_b_scale<T>(
            g_w_scale, b_scale_base_word, scale_word_base, b_scale);
        opus_moe_stage2_a8w4_decode_load_a_scale<T>(
            g_a_scale, a_scale_base_word, scale_word_base, a_scale);

        compute_k_tile_both_n_halves(
            opus::number<0>{},
            opus::number<k_tile0>{},
            b_tile_base0,
            b_scale,
            a_scale);

        if constexpr(k_tile1 < T::K_TILES)
        {
            const int b_tile_base1 =
                col_base * T::B_PAYLOAD_ROW_STRIDE_BYTES +
                k_tile1 * T::K_STEP_PACKED * T::B_PAYLOAD_K_STRIDE_BYTES;
            compute_k_tile_both_n_halves(
                opus::number<1>{},
                opus::number<k_tile1>{},
                b_tile_base1,
                b_scale,
                a_scale);
        }
    });
}

#endif // __HIP_DEVICE_COMPILE__ && __gfx950__
