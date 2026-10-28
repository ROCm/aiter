// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

template<int BlockM = 256, int FixedK = 0, int GroupM = 4, bool XcdOrder = true,
    int CacheA = 0, int CacheB = 0, int WaveN = 4>
struct opus_gemm_bpreshuffle_wave_interleave_traits {
    // Thread block and wave geometry.
    static constexpr int WARP_SIZE = 64;
    static constexpr int T_M = 2;
    static constexpr int T_N = WaveN;
    static constexpr int T_K = 1;
    static constexpr int BLOCK_SIZE = T_M * T_N * WARP_SIZE;

    // MFMA geometry and matrix/scale copy widths.
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = W_M * W_N / WARP_SIZE;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_OUTPUT = 8;
    static constexpr int SCALE_WORD_BYTES = 4;

    // A B_K tile is consumed as two K_STAGE tiles.
    static constexpr int B_M = BlockM;
    static constexpr int B_N = 256;
    static constexpr int B_K = 256;
    static constexpr int K_STAGE = W_K;

    // Each wave computes MFMA fragments from both M and N halves.
    static constexpr int M_HALVES = 2;
    static constexpr int N_HALVES = 2;
    static constexpr int HALF_M = B_M / M_HALVES;
    static constexpr int HALF_N = B_N / N_HALVES;
    static constexpr int E_M = HALF_M / (T_M * W_M);
    static constexpr int E_N = HALF_N / (T_N * W_N);

    // K specialization, tile ordering, and global-memory cache policies.
    static constexpr int FIXED_K = FixedK;
    static constexpr int GROUP_M = GroupM;
    static constexpr bool XCD_ORDER = XcdOrder;
    static constexpr int CACHE_A = CacheA;
    static constexpr int CACHE_B = CacheB;

    // Two LDS stages, each containing two A slots and two B slots.
    static constexpr int LDS_STAGES = 2;
    static constexpr int A_STEPS = (HALF_M + BLOCK_SIZE / (K_STAGE / VEC_A) - 1) / (BLOCK_SIZE / (K_STAGE / VEC_A));
    static constexpr int B_STEPS = HALF_N / (BLOCK_SIZE / (K_STAGE / VEC_B));
    static constexpr int A_SLOT = A_STEPS * (BLOCK_SIZE / (K_STAGE / VEC_A)) * K_STAGE;
    static constexpr int B_SLOT = HALF_N * K_STAGE;
    static constexpr int MATRIX_BYTES = LDS_STAGES * (M_HALVES * A_SLOT + N_HALVES * B_SLOT);

    // K=0 A scales stay in registers; later groups occupy LDS.
    static constexpr int SCALE_GROUPS = 128;
    static constexpr bool PACKED_SCALE = E_M == SCALE_WORD_BYTES;
    static constexpr int SFA_BYTES = (SCALE_GROUPS - 1) * B_M;
    static constexpr int SFB_BYTES = SCALE_GROUPS * N_HALVES;
    static constexpr int LDS_BYTES = MATRIX_BYTES + SFA_BYTES + SFB_BYTES;

    // Reuse matrix LDS for padded BF16 rows during output writeback.
    static constexpr int C_PADDING = 8;
    static constexpr int C_STRIDE = B_N + C_PADDING;
    static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);

    // Supported geometry and LDS capacity.
    static_assert(BlockM == 128 || BlockM == 192 || BlockM == 256);
    static_assert(WaveN == 2 || WaveN == 4);
    static_assert(FixedK == 0 || (FixedK >= 256 && FixedK <= 16384 && FixedK % 256 == 0));
    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8);
    static_assert(!XcdOrder || GroupM > 0);
    static_assert(CacheA == 0 || CacheA == 1 || CacheA == 17);
    static_assert(CacheB == 0 || CacheB == 1 || CacheB == 17);
    static_assert(LDS_BYTES <= 163840);
    static_assert(B_M * C_STRIDE * 2 <= LDS_BYTES);
};
