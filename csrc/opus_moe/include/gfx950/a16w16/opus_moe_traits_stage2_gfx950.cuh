// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus/opus.hpp"

struct OpusMoeStage2Bf16GemmStyle256x256x64TokenSlotRouteOutNoOobNFast
{
    static constexpr int BLOCK_SIZE = 512;
    static constexpr int B_M = 256;
    static constexpr int B_N = 256;
    static constexpr int B_K = 64;

    static constexpr int T_M = 2;
    static constexpr int T_N = 4;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    using D_A   = opus::bf16_t;
    using D_B   = opus::bf16_t;
    using D_ACC = opus::fp32_t;

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;

    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);

    static constexpr int VEC_A = 16 / sizeof(D_A);
    static constexpr int VEC_B = 16 / sizeof(D_B);
    static constexpr int VEC_C = 4;

    static constexpr int smem_linear_wave = opus::get_warp_size() * 16 / sizeof(D_A);
    static_assert(smem_linear_wave % B_K == 0);
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;
    static constexpr int smem_padding = 2 * 16 / sizeof(D_A);

    static constexpr int CACHECTL_A = 0;
    static constexpr int CACHECTL_B = 2;

    static constexpr int NUM_XCD = 8;
    // Narrow row window keeps A-route reuse local while spreading CTAs across XCDs.
    static constexpr int SWIZZLE_W = 1;
    static constexpr int SWIZZLE_C = 64;

    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts =
        (E_M * E_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
};
