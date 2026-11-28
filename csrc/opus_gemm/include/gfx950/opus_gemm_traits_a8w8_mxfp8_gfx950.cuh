// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kargs for the a8w8_mxfp8 pipeline (MXFP8 x MXFP8, 1x32 UE8M0 scales, NT / NN / TT / TN plain and NT
// bpreshuffle, 2x2 4-wave, one tile per block).
#pragma once

// Only opus is needed (not opus_gemm_utils.cuh): the kernel is built offline with the pin toolchain into a .co, and
// launchers only need the kargs struct; the full HIP runtime headers do not build with that toolchain on the host side.
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#include <opus/hip_minimal.hpp>
#endif
#include <opus/opus.hpp>

// Kernel arguments.  Units: fp8 tensors in bytes (= elements), C in elements, scales in bytes.  stride_a / stride_b
// are the row strides of the stored A / B: A [M, K] -> K, A_T [K, M] -> M; B [N, K] -> K, B_T [K, N] -> N (B_SHUF:
// B is the aiter shuffle_weight(w, layout=(16, 16)) of B [N, K], stride_b still K).  SFA/SFB are the native
// row-major [rows, K/32] UE8M0 tensors in every direction; stride_sfa/sfb is their row stride (K/32 when dense).
struct opus_gemm_mxfp8_kargs_gfx950 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int stride_sfa;
    int stride_sfb;
    int stride_sfa_batch;
    int stride_sfb_batch;
    int grid_blocks;             // MODE 0 (persistent): number of launched blocks
    int split_k;                 // MODE 2 (split-K): number of K segments S (grid = tiles * S); otherwise unused
    void* __restrict__ ptr_ws;   // MODE 2, S > 1: fp32 partials [S][batch][padded_M][padded_N] (padded to whole tiles)
};

__host__ __device__ inline int opus_mxfp8_ceil_div(int a, int b) { return (a + b - 1) / b; }

// MXFP8 x MXFP8, 4 waves (2x2), tile 256x256x128 (elements), 1x32 UE8M0; A_T / B_T: A stored [K, M] / B stored
// [K, N] (NT = <false, false>, NN = <false, true>, TT = <true, false>, TN = <true, true>); B_SHUF_: B is the
// shuffle_weight(w, (16, 16)) of B [N, K] (NT only, aiter gemm_a8w8_mxfp8_bpreshuffle), else B is plain row-major.
// 128 bytes per row per K tile in global memory and LDS.
// One fp8 MFMA covers K = W_K = 128 = the whole K tile (E_K = 1): lane group
// g = lane / 16 holds K [32g, +32) = 32 bytes = E_C = 2 chunks of 16 bytes.
// aiter form: <BLOCK_SIZE, seq<B_M, B_N, B_K>, tuple<D_A, D_B, D_C, D_ACC, D_SF>, seq<VEC_A, VEC_B, VEC_C>, seq<GROUP_M, GROUP_N, GROUP_K>>.
// Supported: BLOCK_SIZE 256, B_M / B_N in {256, 192, 128}, B_K 128, VEC <16, 16, 4>, GROUP <1, 1, 32>; K % 32 (BOUNDARY kids without
// split-K; the others K % 256), the host zeroing the scales of the 1x32 groups past K of the last K tile pair.
// MODE_: 0 = persistent (grid = min(tiles, CUs)), 1 = one tile per block (grid = tiles), 2 = split-K (one tile x K segment
// per block, fp32 partials in a workspace, summed into C by splitk_reduce_kernel).
// BOUNDARY_: M / N boundary handling (any M, N % 16; A_T: M % 16, B_T: N % 16); false needs M % B_M == 0 and N % B_N == 0.
template<int BLOCK_SIZE_, typename BLOCK_, typename DTYPE_, typename VEC_, typename GROUP_, bool A_T_ = false, bool B_T_ = false,
         int MODE_ = 1, bool BOUNDARY_ = false, bool B_SHUF_ = false>
struct opus_gemm_a8w8_mxfp8_traits_gfx950 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using GROUP = opus::remove_cvref_t<GROUP_>;
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr bool A_T = A_T_, B_T = B_T_, B_SHUF = B_SHUF_;
    static_assert(!(B_SHUF_ && (A_T_ || B_T_)), "the shuffled B is the NT bpreshuffle input only");
    static constexpr int MODE = MODE_;
    static constexpr bool PERSIST = MODE_ == 0, SPLITK = MODE_ == 2, BOUNDARY = BOUNDARY_;
    static_assert(MODE_ >= 0 && MODE_ <= 2);
    static_assert(!(MODE_ == 2 && opus::get<0>(BLOCK{}) == 256 && opus::get<1>(BLOCK{}) == 256),
                  "split-K 256x256 runs out of registers; use a smaller tile");
    static_assert(BLOCK_SIZE == 256, "4 waves");
    static_assert((opus::get<0>(BLOCK{}) == 256 || opus::get<0>(BLOCK{}) == 192 || opus::get<0>(BLOCK{}) == 128) &&
                  (opus::get<1>(BLOCK{}) == 256 || opus::get<1>(BLOCK{}) == 192 || opus::get<1>(BLOCK{}) == 128));
    static_assert(opus::get<2>(BLOCK{}) == 128, "B_K is one 128-element K tile");
    static_assert(opus::get<0>(VEC{}) == 16 && opus::get<1>(VEC{}) == 16 && opus::get<2>(VEC{}) == 4);
    static_assert(opus::get<0>(GROUP{}) == 1 && opus::get<1>(GROUP{}) == 1 && opus::get<2>(GROUP{}) == 32, "1x32 scales");
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;   // 4

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});          // fp8 elements
    static constexpr int ELEMS_PER_BYTE = 1;
    static constexpr int B_K_BYTES = B_K / ELEMS_PER_BYTE;     // 128

    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static_assert(NUM_WAVES == T_M * T_N * T_K);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;                            // fp8 elements per MFMA
    static constexpr int W_K_BYTES = W_K / ELEMS_PER_BYTE;     // 128

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;
    static constexpr int E_M = HALF_B_M / (W_M * T_M);         // 4 (256), 3 (192), 2 (128)
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);              // 1 MFMA step per K tile
    static_assert(E_K == 1);

    static constexpr int VEC_A = opus::get<0>(VEC{});          // bytes
    static constexpr int VEC_B = opus::get<1>(VEC{});          // bytes
    static constexpr int VEC_A_ELEMS = 16;                     // fp8 elements per 16-byte copy
    static constexpr int VEC_B_ELEMS = 16;
    static_assert(VEC_A * ELEMS_PER_BYTE == VEC_A_ELEMS && VEC_B * ELEMS_PER_BYTE == VEC_B_ELEMS);
    static_assert(VEC_A <= 16 && VEC_B <= 16, "buffer_load ... lds moves at most 16 bytes per lane");
    // One MFMA operand per lane = K / 4 lane groups = 32 bytes = E_C chunks of 16 bytes.
    static constexpr int E_C = W_K / (WARP_SIZE / W_M) / VEC_A_ELEMS;   // 2
    static_assert(E_C == 2);
    static constexpr int VEC_C = opus::get<2>(VEC{});

    // LDS: one wave issue = 64 lanes x 16 bytes = 1024 bytes = 8 rows.
    static constexpr int smem_padding = 0;
    static constexpr int smem_linear_wave_a = WARP_SIZE * VEC_A;            // 1024
    static constexpr int smem_linear_wave_b = WARP_SIZE * VEC_B;
    static constexpr int smem_sub_a = smem_linear_wave_a / B_K_BYTES;       // 8 rows per chunk
    static constexpr int smem_sub_b = smem_linear_wave_b / B_K_BYTES;
    // A_T / B_T: a half tile always takes 16 chunks (128 k rows x 128 B) whatever B_M / B_N; the slots past
    // HALF_B_M / HALF_B_N are filled with copies of the tile's own data (see prefetch_a / prefetch_b).
    static constexpr int smem_m_rep = A_T ? 16 : HALF_B_M / smem_sub_a;     // 16 chunks per half (256)
    // buffer_load ... lds (b128) per wave per half tile per K tile: HALF_B_M rows x 128 B / (256 lanes x 16 B) = E_M.
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K_BYTES / (BLOCK_SIZE * VEC_A);   // 4 (256)
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K_BYTES / (BLOCK_SIZE * VEC_B);
    static_assert(a_buffer_load_insts == E_M && b_buffer_load_insts == E_N);
    static constexpr int smem_n_rep = B_T ? 16 : HALF_B_N / smem_sub_b;
    // LDS-DMA issues per wave per half tile per K tile as actually copied (A_T / B_T: 4, i.e. the 256 x 256 copy).
    static constexpr int a_copy_insts = A_T ? 4 : a_buffer_load_insts;
    static constexpr int b_copy_insts = B_T ? 4 : b_buffer_load_insts;

    // Scales: native row-major [rows, K/32] UE8M0, 4 columns per K tile.
    static constexpr int GROUP_K = opus::get<2>(GROUP{});
    static constexpr int NUM_KGROUPS = B_K / GROUP_K;                       // 4
    // Global -> LDS: one b128 per lane = 16 columns of one row (2 lanes per row, 4 waves x 64 lanes = 128 rows per
    // issue).  A panel row holds 32 columns and serves SCALE_PANEL_TILES K tiles.  A panel always holds 256 rows
    // (LDS image [row 0..255][32 B]): with B_M / B_N < 256 the rows past the tile are copied but never used, and the
    // scale reads of half h (rows h * HALF + 32 g + ..., g = lane / 16 in 0..3) stay inside it.
    static constexpr int VEC_GLOBAL_SCALE = 16;
    static constexpr int SCALE_PANEL_ROWS = 256;
    static_assert(SCALE_PANEL_ROWS == 2 * NUM_WAVES * WARP_SIZE / 2 && HALF_B_M + 128 <= SCALE_PANEL_ROWS &&
                  HALF_B_N + 128 <= SCALE_PANEL_ROWS);
    static constexpr int SCALE_PANEL_GROUPS = 2 * VEC_GLOBAL_SCALE;             // 32 bytes / row: 2 lanes per row
    static constexpr int SCALE_PANEL_TILES = 8;                                 // tiles served per panel copy
    static_assert(SCALE_PANEL_TILES * NUM_KGROUPS <= SCALE_PANEL_GROUPS);
    static constexpr int SFA_PANEL_BYTES = SCALE_PANEL_ROWS * SCALE_PANEL_GROUPS;   // 8192
    static constexpr int SFB_PANEL_BYTES = SCALE_PANEL_ROWS * SCALE_PANEL_GROUPS;
    // Two panel buffers: panel P + 1 is copied at t = 8P, 8P + 1 into the buffer of P - 1.
    static constexpr int SCALE_PANEL_BUFFERS = 2;
    // Consumer: per tile one dword per half per tensor = the tile's 4 columns.
    static constexpr int VEC_LDS_SCALE = NUM_KGROUPS;                           // 4 bytes

    static constexpr int SMEM_A_BYTES = smem_m_rep * (smem_linear_wave_a + smem_padding) * 4;  // 65536 (256)
    static constexpr int SMEM_B_BYTES = smem_n_rep * (smem_linear_wave_b + smem_padding) * 4;
    static constexpr int LDS_BYTES = SMEM_A_BYTES + SMEM_B_BYTES + SCALE_PANEL_BUFFERS * (SFA_PANEL_BYTES + SFB_PANEL_BYTES);
    static_assert(LDS_BYTES <= 163840);
};
// Shorthand: opus_mxfp8_traits<B_M, B_N, A_T, B_T, MODE, BOUNDARY, B_SHUF>.
template<int BM, int BN, bool A_T = false, bool B_T = false, int MODE = 1, bool BOUNDARY = false, bool B_SHUF = false>
using opus_mxfp8_traits = opus_gemm_a8w8_mxfp8_traits_gfx950<256, opus::seq<BM, BN, 128>,
    opus::tuple<opus::fp8_t, opus::fp8_t, opus::bf16_t, opus::fp32_t, opus::e8m0_t>, opus::seq<16, 16, 4>, opus::seq<1, 1, 32>, A_T, B_T,
    MODE, BOUNDARY, B_SHUF>;
