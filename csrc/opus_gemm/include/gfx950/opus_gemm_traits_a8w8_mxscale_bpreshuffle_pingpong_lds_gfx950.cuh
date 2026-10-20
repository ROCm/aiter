// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opus/opus.hpp>

// Double-buffered LDS panel with matrix/scale overlap and staged output.
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
    int WaveM = 4, int GroupM = 0, int CacheA = 0, int CacheB = 0,
    bool FragmentArrays = false, bool PinA = false, bool PinB = false, bool PinC = false,
    bool XcdOrder = false, bool BoundedOutput = false, int PublishM = 0, int GroupN = 8>
struct opus_gemm_bpreshuffle_pingpong_lds_traits {
    static constexpr int BLOCK_SIZE = WaveM * 2 * 64;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int MIN_WGS_PER_CU = 1;

    static constexpr int B_M = BlockM;
    static constexpr int B_N = BlockN;
    static constexpr int B_K = 128;

    static constexpr int T_M = WaveM;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;

    static constexpr int HALF_B_M = B_M;
    static constexpr int HALF_B_N = B_N;

    static_assert(NUM_WAVES == T_M * T_N * T_K);
    static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
    static_assert((B_M == 64 || B_M == 96 || B_M == 128 || B_M == 192 || B_M == 256) &&
        (B_N == 64 || B_N == 128 || B_N == 256));
    static_assert((WaveM == 2 || WaveM == 4) && T_N == 2);

    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = 1;

    static_assert(E_M >= 2 && B_N >= NUM_WAVES * 16);

    static constexpr int FIXED_K = FixedK;
    static constexpr int GROUPED_M = GroupM;
    static constexpr int CACHE_A = CacheA;
    static constexpr int CACHE_B = CacheB;

    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8);
    static_assert(CacheA == 0 || CacheA == 1 || CacheA == 17);
    static_assert(CacheB == 0 || CacheB == 1 || CacheB == 17);

    static constexpr bool PIN_A = PinA;
    static constexpr bool PIN_B = PinB;
    static constexpr bool PIN_C = PinC;
    static constexpr bool FRAGMENT_ARRAYS = FragmentArrays || PinA || PinB || PinC;

    static constexpr bool XCD_ORDER = XcdOrder;
    static constexpr bool BOUNDED_OUTPUT = BoundedOutput;
    static constexpr int PUBLISH_M = PublishM;

    // Publish the next LDS tile before the last M repeat reloads B.
    static_assert(PublishM >= 0 && PublishM < BlockM / (WaveM * 16) - 1);
    static_assert(!PIN_A || E_M * 8 <= 64);
    static_assert(!PIN_B || E_N * 8 <= 64);
    static_assert(!PIN_C || 128 + E_M * E_N * 4 <= 256);

    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_OUTPUT = 8;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int SCALE_N_HALVES = (B_N + GROUP_N - 1) / GROUP_N;

    static constexpr int SWIZZLE_GROUP_M = 4;
    static constexpr int SWIZZLE_GROUP_N = GroupN;
    static_assert(GroupN == 4 || GroupN == 8 || GroupN == 16 || GroupN == 32 || GroupN == 64);
    static constexpr int SWIZZLE_MIN_N_TILES = 256;
    static constexpr int SWIZZLE_MAX_N = 2048;
    static constexpr int SWIZZLE_MIN_M = 4096;

    static constexpr int MAX_K = ScalePanel * GROUP_K;
    static constexpr int LOOP_UNROLL = 2;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int smem_padding = 32;
    static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);

    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int NUM_STAGES = 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int MATRIX_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
    static_assert(NUM_STAGES == 2);

    static constexpr int SCALE_PANEL = ScalePanel;
    static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
    static constexpr int SFA_K_PANEL = SCALE_PANEL < 32 ? SCALE_PANEL : 32;
    static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
    static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
    static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
    static constexpr int SFA_PASSES = (B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;

    static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
    static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
    static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
    static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
    static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
    static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
    static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);

    static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
    static constexpr int B_SCALE_PACKS = 1;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;

    static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
    static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
    static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
    static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
