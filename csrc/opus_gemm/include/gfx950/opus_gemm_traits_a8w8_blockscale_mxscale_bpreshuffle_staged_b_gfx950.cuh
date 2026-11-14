// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

template<
    int BlockM, int BlockN, int WaveM, int WaveN, int ScalePanel,
    int SplitK = 1, int CacheA = 0, int CacheB = 0, int GroupM = 0, int BlockK = 128>
struct opus_gemm_bpreshuffle_staged_b_traits {
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = WaveM * WaveN;
    static constexpr int BLOCK_SIZE = NUM_WAVES * WARP_SIZE;
    static constexpr int WG_PER_CU = 1;
    static constexpr int B_M = BlockM;
    static constexpr int B_N = BlockN;
    static constexpr int B_K = BlockK;
    static constexpr int T_M = WaveM;
    static constexpr int T_N = WaveN;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int HALF_B_M =
        (B_M * B_K + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16) * BLOCK_SIZE * 16 / B_K;
    static constexpr int HALF_B_N = B_N;
    static constexpr int E_M = B_M / (T_M * W_M);
    static constexpr int E_N = B_N / (T_N * W_N);
    static constexpr int E_K = B_K / W_K;
    static constexpr int A_RING = E_M < 3 ? E_M : 3;

    static constexpr int SPLIT_K = SplitK;
    static constexpr int CACHE_A = CacheA;
    static constexpr int CACHE_B = CacheB;
    static constexpr int GROUPED_M = GroupM;
    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SF = sizeof(unsigned);
    static constexpr int VEC_OUTPUT = (B_M * B_N / BLOCK_SIZE) % 8 == 0 ? 8 : 4;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
    static constexpr int GROUP_M = 1;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int SCALE_N_HALVES = (B_N + GROUP_N - 1) / GROUP_N;

    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    static constexpr int smem_m_rep = HALF_B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
    static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
    static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int A_RESIDENT_STAGE = HALF_B_M * B_K;
    static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
    static constexpr int RAW_SFA_ROWS = B_M;
    static constexpr int PACKED_SFA_PITCH = ((E_M + 3) / 4) * T_M * W_M * 4;
    static constexpr int SFA_PANEL_PITCH = B_M <= 32 && B_N <= 128
        ? RAW_SFA_ROWS : PACKED_SFA_PITCH;
    static constexpr int NUM_STAGES =
        B_M == 16 && B_N == 32 && B_K == 128 && T_M == 1 && T_N == 2 && ScalePanel == 128 &&
            SplitK == 1 && CacheA == 0 && CacheB == 0 && GroupM == 0 ? 4 :
        B_M == 32 && B_N == 64 && B_K == 128 &&
            8 * (A_STAGE + B_STAGE) + ScalePanel * (SFA_PANEL_PITCH + 4) <= 160 * 1024 ? 8 :
        3 * (A_STAGE + B_STAGE) + ScalePanel * (
            SFA_PANEL_PITCH + sizeof(unsigned)) <= 160 * 1024 ? 3 : 2;
    static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
    static constexpr int MATRIX_VMEM_INSTRUCTIONS =
        HALF_B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);

    // Four M-repeat scales form one word for the MFMA byte selectors.
    static constexpr int SCALE_PANEL = ScalePanel;
    static constexpr int A_SCALE_PACKS = (E_M + VEC_SF - 1) / VEC_SF;
    static constexpr int B_SCALE_PACKS = 1;
    static constexpr int SFA_M_VECTORS = A_SCALE_PACKS * T_M * VEC_SF;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_M_VECTORS;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL =
        (SCALE_PANEL + SFA_K_COLUMNS_PER_PASS - 1) / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_BYTES = SFA_PANEL_PITCH * SCALE_PANEL;
    static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
    static constexpr int SCALE_BYTES = SFA_BYTES + SFB_BYTES;
    static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
    static constexpr int C_BYTES = SplitK == 1 ? B_M * C_LDS_ROW_STRIDE_ELEMS * 2 : 0;
    static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
    static constexpr int MAX_K = SplitK * ScalePanel * GROUP_K;
    static constexpr int RESIDENT_LIMIT =
        B_M == 32 && B_N == 64 && B_K == 128 && T_M == 1 && T_N == 2 && ScalePanel == 8 &&
            SplitK == 1 && CacheA == 1 && CacheB == 17 && GroupM == 0 ? 6 : B_M <= 32
        ? (B_K == 1024 ? 2 : (ScalePanel <= 4 ? 3
            : (ScalePanel <= 8 ? (B_K == 128 ? 8 : (B_K == 256 ? 3 : 2))
                : (B_K == 512 && ScalePanel <= 16 ? 3
                    : (B_K == 128 ? 16 : (B_K == 256 ? 8 : 4))))))
        : 8;
    static constexpr int RESIDENT_PANELS = B_M <= 32
        ? (RESIDENT_LIMIT < ScalePanel / E_K ? RESIDENT_LIMIT : ScalePanel / E_K)
        : ((B_M == 64 || B_M == 96) && T_M == 2 && B_N == 64 && B_K == 128 && T_N == 2 && ScalePanel == 4 ? 3 :
            (B_M == 48 && B_N <= 64 && B_K == 128 && T_M == 1 && T_N == 2 && ScalePanel == 8 ? 6 : 8));
    static constexpr int RESIDENT_CAPACITY =
        (160 * 1024 - SCALE_BYTES) / (A_STAGE + B_STAGE) < RESIDENT_PANELS
            ? (160 * 1024 - SCALE_BYTES) / (A_STAGE + B_STAGE) : RESIDENT_PANELS;
    static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SCALE_BYTES;

    __host__ __device__ static constexpr int resident_capacity(int m) {
        const int legacy_limit = (160 * 1024 - SCALE_BYTES) / (A_STAGE + B_STAGE) < 8
            ? (160 * 1024 - SCALE_BYTES) / (A_STAGE + B_STAGE) : 8;
        const int normal_limit = legacy_limit < RESIDENT_CAPACITY ? legacy_limit : RESIDENT_CAPACITY;
        if (B_M == 16 && B_N == 32 && B_K == 128 && T_M == 1 && T_N == 2 && SCALE_PANEL == 8 && m > 0 && m <= 32)
            return RESIDENT_CAPACITY < 6 ? RESIDENT_CAPACITY : 6;
        return B_M <= 32 && m > 0 && m <= 32 ? RESIDENT_CAPACITY : normal_limit;
    }
    __host__ __device__ static constexpr bool short_k(int k, int m = 0) {
        return k / (SPLIT_K * B_K) <= resident_capacity(m);
    }
    __host__ __device__ static constexpr bool ring8_eligible(int m, int k) {
        const int panels = k / (SPLIT_K * B_K);
        return B_K <= 512 && m > 0 && m <= 32 && B_M == 16 && B_N == 16 && NUM_WAVES == 1 &&
            panels >= 8 && panels <= SCALE_PANEL &&
            8 * (A_STAGE + B_STAGE) + SCALE_BYTES <= 160 * 1024;
    }
    __host__ __device__ static constexpr int normal_sfa_bytes(int k) {
        return B_M == 128 && B_N == 128 && T_M == 4 && T_N == 2 && !short_k(k)
            ? k / (SPLIT_K * GROUP_K) * SFA_PANEL_PITCH : SFA_BYTES;
    }
    __host__ __device__ static constexpr int normal_sfb_bytes(int k) {
        return B_M == 128 && B_N == 128 && T_M == 4 && T_N == 2 && !short_k(k)
            ? k / (SPLIT_K * GROUP_K) * sizeof(unsigned) : SFB_BYTES;
    }
    __host__ __device__ static constexpr int normal_stage_count(int k) {
        if (B_M == 32 && B_N == 32 && B_K == 256 && T_M == 2 && T_N == 1 &&
            SCALE_PANEL == 128 && SPLIT_K == 1 && CACHE_A == 1 && CACHE_B == 17 && GROUPED_M == 0)
            return 4;
        if (B_M == 128 && B_N == 128 && T_M == 4 && T_N == 2 &&
            !short_k(k) && k / (SPLIT_K * B_K) <= 16) return 2;
        return B_M == 128 && B_N == 128 && T_M == 4 && T_N == 2 && !short_k(k) &&
            3 * (A_STAGE + B_STAGE) + normal_sfa_bytes(k) + normal_sfb_bytes(k) <= 160 * 1024
            ? 3 : NUM_STAGES;
    }
    __host__ __device__ static constexpr int lds_bytes(int k, int m = 0) {
        const int panels = k / (SPLIT_K * B_K);
        const int stages = short_k(k, m) ? panels : normal_stage_count(k);
        const int a_stage = !short_k(k, m) && ((B_M == 32 && B_N == 64 && B_K == 128 &&
            T_M == 1 && T_N == 4 && SCALE_PANEL == 256 && SPLIT_K == 1 &&
            CACHE_A == 17 && CACHE_B == 1 && GROUPED_M == 0) ||
            (B_M == 32 && B_N == 32 && B_K == 256 && T_M == 2 && T_N == 1 &&
             SCALE_PANEL == 128 && SPLIT_K == 1 && CACHE_A == 0 && CACHE_B == 0 && GROUPED_M == 0) ||
            (B_M == 80 && T_M == 1 && T_N == 4) ||
            (B_M == 96 && B_N == 64) || (B_M == 128 && B_N == 128 && T_M == 2 && T_N == 4))
            ? HALF_B_M * B_K : A_STAGE;
        const int resident_a_stage = B_M > 32 && short_k(k, m) ? A_RESIDENT_STAGE : a_stage;
        const int scale_bytes = B_M <= 32 && short_k(k, m) ? 0 : normal_sfa_bytes(k) + normal_sfb_bytes(k);
        const int bytes = stages * (resident_a_stage + B_STAGE) + scale_bytes;
        return bytes > C_BYTES ? bytes : C_BYTES;
    }
    __host__ __device__ static constexpr int resident_a_lds_bytes(int k, int m = 0) { return lds_bytes(k, m); }
    __host__ __device__ static constexpr int ring8_lds_bytes(int k) {
        const int bytes = 8 * (A_STAGE + B_STAGE) + SCALE_BYTES;
        return bytes > C_BYTES ? bytes : C_BYTES;
    }
    static constexpr int RING8_BLOCK_M = 16;
    static constexpr int RING8_BLOCK_N = 16;
    static constexpr int RING8_BLOCK_SIZE = 64;

    static_assert(B_K == 128 || B_K == 256 || B_K == 512 || B_K == 1024);
    static_assert(NUM_WAVES > 0 && NUM_WAVES <= 8);
    static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
    static_assert(E_M >= 1 && E_N >= 1 && B_N <= 256);
    static_assert(HALF_B_M * B_K % (BLOCK_SIZE * VEC_A) == 0);
    static_assert(B_N * B_K % (BLOCK_SIZE * VEC_B) == 0);
    static_assert(WARP_SIZE % SFA_M_VECTORS == 0);
    static_assert(ScalePanel > 0 && BLOCK_SIZE % ScalePanel == 0);
    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4);
    static_assert(CacheA == 0 || CacheA == 1 || CacheA == 17);
    static_assert(CacheB == 0 || CacheB == 1 || CacheB == 2 || CacheB == 17);
    static_assert(GroupM == 0 || GroupM == 4);
    static_assert(RESIDENT_CAPACITY >= 2 && LDS_BYTES <= 160 * 1024 && C_BYTES <= 160 * 1024);
    static_assert(4 * E_M * E_N + E_K * (8 * (A_RING + E_N) + 2 * A_SCALE_PACKS) + 48 <= 512);
};
