// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

#include <opus/opus.hpp>

// M repeats share an A fragment ring, B fragments, and two LDS matrix stages.
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
    int WaveM = 4, int GroupM = 0, int CacheA = 0, int CacheB = 0,
    bool FragmentArrays = false,
    bool XcdOrder = false, bool BoundedOutput = false, int PublishM = 0, int GroupN = 8,
    bool RawScales = false>
struct opus_gemm_bpreshuffle_pingpong_lds_traits {
    // Wave64 topology and block/MFMA tile dimensions.
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
    static constexpr int A_RING = RawScales ? E_M : (E_M < 3 ? E_M : 3);

    static_assert(E_M >= 2 && B_N >= NUM_WAVES * 16);

    // K specialization, tile grouping, and matrix cache policies.
    static constexpr int FIXED_K = FixedK;
    static constexpr int GROUPED_M = GroupM;
    static constexpr int CACHE_A = CacheA;
    static constexpr int CACHE_B = CacheB;
    static constexpr bool RAW_SCALES = RawScales;

    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8);
    static_assert(CacheA == 0 || CacheA == 1 || CacheA == 17);
    static_assert(CacheB == 0 || CacheB == 1 || CacheB == 17 || (RawScales && CacheB == 2));

    // Keep fragments in separate arrays when requested by the JIT configuration.
    static constexpr bool FRAGMENT_ARRAYS = FragmentArrays;

    static constexpr bool XCD_ORDER = XcdOrder;
    static constexpr bool BOUNDED_OUTPUT = BoundedOutput;
    static constexpr int PUBLISH_M = PublishM;

    // The publication index excludes the final M repeat.
    static_assert(PublishM >= 0 && PublishM < BlockM / (WaveM * 16) - 1);

    // Vector widths and K128 block-scale groups.
    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SF = sizeof(unsigned);
    static constexpr int VEC_OUTPUT = 8;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);

    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int SCALE_N_HALVES = (B_N + GROUP_N - 1) / GROUP_N;

    // Launch-order swizzles for matrix-tile reuse.
    static constexpr int SWIZZLE_GROUP_M = 4;
    static constexpr int SWIZZLE_GROUP_N = GroupN;
    static_assert(GroupN == 4 || GroupN == 8 || GroupN == 16 || GroupN == 32 || GroupN == 64);
    static constexpr int SWIZZLE_MIN_N_TILES = 256;
    static constexpr int SWIZZLE_MAX_N = 2048;
    static constexpr int SWIZZLE_MIN_M = 4096;

    static constexpr int MAX_K = ScalePanel * GROUP_K;
    static constexpr int LOOP_UNROLL = 2;

    // Padded matrix LDS rows and two ping-pong stages.
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

    // Four M-repeat scales per A word; B packs its N128 scales into one word.
    static constexpr int SCALE_PANEL = ScalePanel;
    static constexpr int A_SCALE_PACKS = (E_M + VEC_SF - 1) / VEC_SF;
    static constexpr int SFA_M_VECTORS = A_SCALE_PACKS * T_M * VEC_SF;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_M_VECTORS;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = (SCALE_PANEL + SFA_K_COLUMNS_PER_PASS - 1) / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = A_SCALE_PACKS * T_M * W_M * VEC_SF;
    static constexpr int SFA_BYTES = RawScales ? B_M * SCALE_PANEL : SFA_PANEL_PITCH * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);

    // Scale-panel passes must cover the K columns and M rows.
    static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
    static_assert(W_M == VEC_SCALE_A && WARP_SIZE % SFA_M_VECTORS == 0);
    static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
    static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS >= SCALE_PANEL);

    static constexpr int B_SCALE_PACKS = 1;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
    static_assert(LDS_BYTES <= 160 * 1024);

    // The BF16 epilogue reuses LDS with a padded row stride.
    static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
    static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
    static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
    static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};

// Compatibility argument order for the raw-scale ping-pong policy.
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0, int WaveM = 4,
         int GroupM = 0, int CacheA = 0, int CacheB = 0, bool FragmentArrays = false,
         bool PinA = false, bool PinB = false, bool PinC = false, bool XcdOrder = false>
struct opus_gemm_bpreshuffle_lds_panel_traits
    : opus_gemm_bpreshuffle_pingpong_lds_traits<BlockM, BlockN, ScalePanel, FixedK, WaveM, GroupM,
                                                CacheA, CacheB, FragmentArrays, XcdOrder, false, 0,
                                                8, true> {
    using Base =
        opus_gemm_bpreshuffle_pingpong_lds_traits<BlockM, BlockN, ScalePanel, FixedK, WaveM, GroupM,
                                                  CacheA, CacheB, FragmentArrays, XcdOrder, false,
                                                  0, 8, true>;
    static_assert(!PinA && !PinB && !PinC, "Raw-scale ping-pong does not support AGPR pinning");
    static_assert(ScalePanel > 0 && (FixedK == 0 || FixedK >= 128));
    static constexpr bool PIN_A = false;
    static constexpr bool PIN_B = false;
    static constexpr bool PIN_C = false;
    static constexpr int C_REGS = BlockM * BlockN / Base::BLOCK_SIZE;
    static constexpr int MIN_K = 128;

    // Raw M-scale vectors are transposed into the MFMA scale words on read.
    static constexpr int SFA_K_PANEL = ScalePanel < 32 ? ScalePanel : 32;
    static constexpr int SFA_K_PASSES = ScalePanel / SFA_K_PANEL;
    static constexpr int SFA_THREADS_PER_GROUP = Base::BLOCK_SIZE / SFA_K_PANEL;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = Base::WARP_SIZE / SFA_THREADS_PER_GROUP;
    static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * Base::VEC_SCALE_A;
    static constexpr int SFA_PASSES = (BlockM + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;

    static_assert(Base::BLOCK_SIZE % ScalePanel == 0);
    static_assert(SFA_THREADS_PER_GROUP <= Base::WARP_SIZE);
    static_assert(Base::WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
    static_assert(Base::T_N * Base::T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
    static_assert(SFA_K_PANEL * SFA_K_PASSES == ScalePanel);
    static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= BlockM);
};
