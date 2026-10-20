// SPDX-License-Identifier: Apache-2.0
#pragma once

// Two wave groups alternate operand staging and scaled MFMA.
template<int BlockM = 256, int FixedK = 0, int GroupM = 4, bool XcdOrder = true,
    int CacheA = 0, int CacheB = 0, int WaveN = 4>
struct opus_gemm_bpreshuffle_wave_interleave_traits {
    static constexpr int BLOCK_SIZE = 2 * WaveN * 64;
    static constexpr int WARP_SIZE = 64;

    static constexpr int B_M = BlockM;
    static constexpr int B_N = 256;
    static constexpr int B_K = 256;
    static constexpr int K_STAGE = 128;

    static constexpr int T_M = 2;
    static constexpr int T_N = WaveN;
    static constexpr int T_K = 1;

    static constexpr int HALF_M = B_M / 2;
    static constexpr int HALF_N = B_N / 2;
    static constexpr int E_M = HALF_M / (T_M * 16);
    static constexpr int E_N = HALF_N / (T_N * 16);

    static constexpr int FIXED_K = FixedK;
    static constexpr int GROUP_M = GroupM;
    static constexpr bool XCD_ORDER = XcdOrder;
    static constexpr int CACHE_A = CacheA;
    static constexpr int CACHE_B = CacheB;

    static constexpr int A_STEPS = (HALF_M + BLOCK_SIZE / 8 - 1) / (BLOCK_SIZE / 8);
    static constexpr int B_STEPS = HALF_N / (BLOCK_SIZE / 8);
    static constexpr int A_SLOT = A_STEPS * (BLOCK_SIZE / 8) * K_STAGE;
    static constexpr int B_SLOT = HALF_N * K_STAGE;
    static constexpr int MATRIX_BYTES = 4 * (A_SLOT + B_SLOT);

    static constexpr int SCALE_GROUPS = 128;
    static constexpr bool PACKED_SCALE = E_M == 4;
    static constexpr int SFA_BYTES = (SCALE_GROUPS - 1) * B_M;
    static constexpr int SFB_BYTES = SCALE_GROUPS * 2;
    static constexpr int LDS_BYTES = MATRIX_BYTES + SFA_BYTES + SFB_BYTES;

    static constexpr int C_STRIDE = B_N + 8;
    static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * 8);

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
