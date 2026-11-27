// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kernel arguments of the gfx950 bf16 grouped GEMMs (M-grouped NT / NN, K-grouped TN).
#pragma once

#include "opus_gemm_traits_a16w16_gfx950.cuh"

// Every contiguous group (M-grouped rows, K-grouped rows) starts at a multiple of this many rows.
constexpr int OPUS_A16W16_GROUP_ALIGN = 128;

// 4 waves (2 x 2), B_M x B_N x 64 tile, MFMA 16x16x32 bf16. B_T: B [K, N] (NN); K_GROUPED: TN; C_F32: fp32 C.
template<int B_M_, int B_N_, bool B_T_, bool K_GROUPED_ = false, bool C_F32_ = false>
struct opus_gemm_a16w16_grouped_traits_gfx950 {
    using D_A = opus::bf16_t;
    using D_B = opus::bf16_t;
    using D_C = std::conditional_t<C_F32_, opus::fp32_t, opus::bf16_t>;
    using D_ACC = opus::fp32_t;
    using D_OUT = D_C;
    static constexpr bool K_GROUPED = K_GROUPED_;
    static constexpr bool C_F32 = C_F32_;
    static constexpr bool A_T = K_GROUPED;
    static constexpr bool B_T = B_T_;
    static_assert(!K_GROUPED || B_T, "K-grouped: TN");
    static_assert(!C_F32 || K_GROUPED, "fp32 C: K-grouped only");
    // launch_bounds hint: 1 lets the pinned operands / C use the whole AGPR file; the occupancy then
    // follows the real register / LDS use (small tiles still get 2-3 workgroups per CU).
    static constexpr int WG_PER_CU = 1;

    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;              // 4

    static constexpr int B_M = B_M_;
    static constexpr int B_N = B_N_;
    static constexpr int B_K = 64;                                        // bf16 elements (128 bytes: one LDS row)

    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static_assert(NUM_WAVES == T_M * T_N * T_K);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;                                        // 16x16x32 bf16 MFMA

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;
    static constexpr int E_M = HALF_B_M / (W_M * T_M);                    // 1, 2, 3, 4
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static_assert(E_M >= 1 && E_N >= 1 && E_M <= 4 && E_N <= 4 && (E_N & (E_N - 1)) == 0
                  && HALF_B_M == 32 * E_M && HALF_B_N == 32 * E_N,
                  "tile = 2 halves x 2 waves x E x 16, E in {1, 2, 3, 4}");
    // E_M = 3 (B_M = 192): the M/N-contiguous read swizzle (mn_swz) needs a power of two, A stays K-contiguous.
    static_assert((E_M & (E_M - 1)) == 0 || (E_M == 3 && !A_T), "B_M = 192: A [M, K] only");

    // Copies are 16 bytes per lane (buffer_load ... lds); one wave issue = 1 KB. Sizes below are in elements.
    static constexpr int VEC_A = 8;                                       // bf16 elements (16 bytes per lane)
    static constexpr int VEC_B = 8;
    static constexpr int VEC_TR = 4;                                      // ds_read_b64_tr_b16: 4 bf16 per lane
    // K steps (16x16x32 MFMAs) per K tile: B_K / W_K.
    static constexpr int E_C = 2;
    static constexpr int VEC_C = 4;

    static constexpr int smem_padding = 0;
    static constexpr int smem_linear_wave_a = WARP_SIZE * VEC_A;          // 512 bf16 = 1 KB chunk
    static constexpr int smem_linear_wave_b = WARP_SIZE * VEC_B;
    static constexpr int smem_sub_a = smem_linear_wave_a / B_K;           // 8 rows per chunk
    static constexpr int smem_sub_b = smem_linear_wave_b / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub_a;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub_b;
    // buffer_load ... lds per wave per half tile per K tile (both layouts move HALF x B_K).
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);   // E_M
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);   // E_N

    // LDS: 2 stages x 2 halves of A and B; the C staging of the epilogue reuses it. C is staged as one
    // D_OUT image of the tile (rows padded by 8 elements) when it fits, else (256 x 256 fp32) quadrant by
    // quadrant in 2 alternating regions of HALF_B_M x (HALF_B_N + 8).
    static constexpr int SMEM_A_BYTES =
        smem_m_rep * (smem_linear_wave_a + smem_padding) * 4 * static_cast<int>(sizeof(D_A));
    static constexpr int SMEM_B_BYTES =
        smem_n_rep * (smem_linear_wave_b + smem_padding) * 4 * static_cast<int>(sizeof(D_B));
    static constexpr int C_IMAGE_BYTES = B_M * (B_N + 8) * static_cast<int>(sizeof(D_OUT));
    static constexpr bool C_STAGE_FULL = C_IMAGE_BYTES <= 163840;
    static constexpr int C_STAGE_BYTES =
        C_STAGE_FULL ? C_IMAGE_BYTES : 2 * HALF_B_M * (HALF_B_N + 8) * static_cast<int>(sizeof(D_OUT));
    static constexpr int LDS_BYTES =
        (SMEM_A_BYTES + SMEM_B_BYTES) > C_STAGE_BYTES ? (SMEM_A_BYTES + SMEM_B_BYTES) : C_STAGE_BYTES;
    static_assert(LDS_BYTES <= 163840);

    // Schedule: LDS reads per half-operand (2 per K-contiguous operand, 4 per TR16 one), at most CAP per MFMA pair.
    static constexpr int SCHED_C00CAP = 2;
    static constexpr int SCHED_C01CAP = 1;
    static constexpr int SCHED_C11CAP = 1;
};

// M-grouped kernel arguments (strides in elements); ptr_rows / ptr_first: rows / first row per group (from the
// plan kernel), ptr_first nullptr: masked (ptr_rows = masked_m).
struct opus_gemm_a16w16_grouped_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    const int* __restrict__ ptr_rows;
    const int* __restrict__ ptr_first;
    long long stride_b_group;
    int m;
    int n;
    int k;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_c_batch;
    int groups;                         // G
};
static_assert(sizeof(opus_gemm_a16w16_grouped_kargs_gfx950) == 88);

// M-grouped plan kernel arguments: grouped_layout [m] -> rows [groups] / first [groups].
struct opus_gemm_a16w16_m_grouped_plan_kargs_gfx950 {
    const int* __restrict__ layout;
    int* __restrict__ rows;
    int* __restrict__ first;
    int m;
    int groups;
};
static_assert(sizeof(opus_gemm_a16w16_m_grouped_plan_kargs_gfx950) == 32);
constexpr int OPUS_M_GROUPED_PLAN_THREADS = 1024;
constexpr int OPUS_M_GROUPED_PLAN_MAX_GROUPS = 4096;

// K-grouped kernel arguments: C[g] = (src[g] +) A_g^T B_g, group g in rows of A [R][M] / B [R][N]; the segments
// of a tile add in order through turn (bit-reproducible); table from the plan kernel. Strides in elements.
struct opus_gemm_a16w16_k_grouped_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    const void* __restrict__ ptr_src;
    const int* __restrict__ table;      // [items][4]
    int* __restrict__ turn;             // [G][tiles]
    long long stride_c_group;
    int m;
    int n;
    int stride_a;
    int stride_b;
    int stride_c;
    int write_empty;                    // 1: an empty group still writes C_g = src_g (or 0)
};
static_assert(sizeof(opus_gemm_a16w16_k_grouped_kargs_gfx950) == 80);

// K-grouped plan kernel arguments: grouped_layout [groups] -> table, zeroed turn.
struct opus_gemm_a16w16_k_grouped_plan_kargs_gfx950 {
    const int* __restrict__ grouped_layout;   // [G] rows per group
    int* __restrict__ table;            // [full_items + G][4]
    int* __restrict__ turn;             // [turn_count], zeroed
    int groups;                         // G
    int seg_len;                        // rows per full segment (multiple of OPUS_A16W16_GROUP_ALIGN)
    int rows;                           // R
    int full_items;                     // full-segment slots (>= the number of full segments)
    int turn_count;
    int max_rows;                       // the most rows a segment may have (32-bit offsets); more: trap
};
static_assert(sizeof(opus_gemm_a16w16_k_grouped_plan_kargs_gfx950) == 48);
constexpr int OPUS_K_GROUPED_PLAN_THREADS = 1024;
