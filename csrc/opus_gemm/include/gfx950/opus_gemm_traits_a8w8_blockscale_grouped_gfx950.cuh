// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits of the gfx950 FP8 blockscale grouped GEMMs: FP8 A / B and E8M0 scales as stored, bf16 (or fp32) C.
// A_T / B_T: operand stored [K, M] / [K, N]; SFB_COL: B scales 1x128 per column (wgrad); C_F32: fp32 C.
#pragma once

#include <cstdint>
#include <type_traits>
#include <opus/opus.hpp>
#include "../opus_gemm_utils.cuh"

// Every contiguous group (M-grouped rows, K-grouped rows) starts at a multiple of this many rows.
constexpr int OPUS_A8W8_BS_GROUP_ALIGN = 128;

template<bool A_T_, bool B_T_, int BM = 256, int BN = 256, bool SFB_COL_ = false, bool C_F32_ = false>
struct opus_gemm_a8w8_blockscale_grouped_tile_traits {
    static_assert(BM == 256 && (BN == 128 || BN == 256), "FP8 grouped tiles are 256x256 or 256x128");
    static_assert(!SFB_COL_ || BN == 256, "per-column B scales: 256-wide tiles");
    static constexpr bool A_T = A_T_, B_T = B_T_;
    static constexpr bool SFB_COL = SFB_COL_;
    static constexpr bool C_F32 = C_F32_;
    using D_C = std::conditional_t<C_F32, opus::fp32_t, opus::bf16_t>;

    static constexpr int BLOCK_SIZE = 256;
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int B_M = BM, B_N = BN, B_K = 128;
    // HALF_B_* is the quadrant edge; H_* counts quadrants per tile side.
    static constexpr int HALF_B_M = 128, HALF_B_N = 128;
    static constexpr int H_M = B_M / HALF_B_M, H_N = B_N / HALF_B_N;
    static constexpr int T_M = 2, T_N = 2, T_K = 1;
    static constexpr int E_M = 4, E_N = 4, E_K = 1;
    static constexpr int W_M = 16, W_N = 16, W_K = 128;
    // Global-to-LDS copies use 16 FP8 bytes per lane for both operands.
    static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
    static constexpr int GROUP_M = 1, GROUP_N = SFB_COL ? 1 : 128, GROUP_K = 128;
    static constexpr int VEC_SF = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SCALE_SF = 4;
    // 128x128 B scale: one per 128 N columns, i.e. per N quadrant.
    static constexpr int SCALE_N_HALVES = B_N / 128;
    // K128 columns of scales cached in LDS: a per-column B panel halves it (LDS).
    static constexpr int SCALE_PANEL_K_CAPACITY = SFB_COL ? 32 : 64;
    // A lane loads VEC_SCALE_A contiguous M scales, so B_M / VEC_SCALE_A lanes cover a K column.
    static constexpr int SFA_LANES_PER_K_COLUMN = B_M / VEC_SCALE_A;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_LANES_PER_K_COLUMN;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = B_M;
    static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
    static constexpr int SFB_PANEL_PITCH = SFB_COL ? B_N : SCALE_N_HALVES * VEC_SF;
    static constexpr int SFB_PANEL_BYTES = SFB_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
    // One packed A / B scale dword per quadrant row / column: bytes read per K tile.
    static constexpr int SFA_READ_BYTES = H_M * VEC_SF;
    static constexpr int SFB_READ_BYTES = H_N * VEC_SF;

    // A wave slot is one 16-byte-per-lane LDS-DMA pass (1024 bytes) plus padding.
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
    static constexpr int smem_padding = 32;
    // Rows of B_K bytes per wave slot (M rows for a [M,K] slot).
    static constexpr int smem_sub = smem_linear_wave / B_K;
    // Slots per quadrant operand: 128 x B_K bytes / 1024.
    static constexpr int smem_m_rep = HALF_B_M * B_K / smem_linear_wave;
    static constexpr int smem_n_rep = HALF_B_N * B_K / smem_linear_wave;
    static_assert(NUM_WAVES == T_M * T_N * T_K && NUM_WAVES == 4);
    static_assert(HALF_B_M == E_M * T_M * W_M && HALF_B_N == E_N * T_N * W_N);

    // LDS reads: TR8 returns 8 FP8 bytes per lane, a plain read 16.
    static constexpr int VEC_LDS_A = A_T ? 8 : 16;
    static constexpr int VEC_LDS_B = B_T ? 8 : 16;
    // Each quadrant operand occupies sixteen 1024-byte wave slots, with 32-byte padding.
    static constexpr int SMEM_HALF_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding);
    static constexpr int SMEM_HALF_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding);
    // Two LDS stages of H_M A quadrants and H_N B quadrants.
    static constexpr int SMEM_A_ELEMS = SMEM_HALF_A_ELEMS * 2 * H_M;
    static constexpr int SMEM_B_ELEMS = SMEM_HALF_B_ELEMS * 2 * H_N;
    static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
    static_assert(LDS_BYTES <= 163840);
};

// M-grouped kernel arguments (strides in elements); ptr_rows / ptr_first: rows / first row per group (from the
// plan kernel), ptr_first nullptr: masked (ptr_rows = masked_m). Scales: A 1x128 [K/128][M], B 128x128.
struct opus_gemm_a8w8_blockscale_grouped_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    const int* __restrict__ ptr_rows;
    const int* __restrict__ ptr_first;
    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    long long stride_b_group;
    long long stride_a_batch;
    long long stride_c_batch;
    int m;
    int n;
    int k;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_sfa;
    int stride_sfa_batch;
    int stride_sfb;
    int stride_sfb_group;
    int groups;
};
static_assert(sizeof(opus_gemm_a8w8_blockscale_grouped_kargs_gfx950) == 128);

// M-grouped plan kernel arguments: grouped_layout [m] -> rows [groups] / first [groups].
struct opus_gemm_a8w8_blockscale_m_grouped_plan_kargs_gfx950 {
    const int* __restrict__ layout;
    int* __restrict__ rows;
    int* __restrict__ first;
    int m;
    int groups;
};
static_assert(sizeof(opus_gemm_a8w8_blockscale_m_grouped_plan_kargs_gfx950) == 32);
constexpr int OPUS_A8W8_BS_M_GROUPED_PLAN_THREADS = 1024;
constexpr int OPUS_A8W8_BS_M_GROUPED_PLAN_MAX_GROUPS = 4096;

// K-grouped kernel arguments: C[g] = (src[g] +) A_g^T B_g, group g in rows of A [R][M] / B [R][N]; the segments
// of a tile add in order through turn (bit-reproducible); table from the plan kernel. Scales 128x1 along the
// rows: [R/128][M / N].
struct opus_gemm_a8w8_blockscale_k_grouped_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    const void* __restrict__ ptr_src;
    const int* __restrict__ table;      // [items][4]
    int* __restrict__ turn;             // [G][tiles]
    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    long long stride_c_group;
    int m;
    int n;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_sfa;
    int stride_sfb;
    int write_empty;
};
static_assert(sizeof(opus_gemm_a8w8_blockscale_k_grouped_kargs_gfx950) == 104);

// K-grouped plan kernel arguments: grouped_layout [groups] -> table, zeroed turn.
struct opus_gemm_a8w8_blockscale_k_grouped_plan_kargs_gfx950 {
    const int* __restrict__ grouped_layout;   // [G] rows per group
    int* __restrict__ table;            // [full_items + G][4]
    int* __restrict__ turn;             // [turn_count], zeroed
    int groups;                         // G
    int seg_len;                        // rows per full segment (multiple of OPUS_A8W8_BS_GROUP_ALIGN)
    int rows;                           // R
    int full_items;                     // full-segment slots (>= the number of full segments)
    int turn_count;
    int max_rows;                       // the most rows a segment may have (32-bit offsets); more: trap
};
static_assert(sizeof(opus_gemm_a8w8_blockscale_k_grouped_plan_kargs_gfx950) == 48);
constexpr int OPUS_A8W8_BS_K_GROUPED_PLAN_THREADS = 1024;
