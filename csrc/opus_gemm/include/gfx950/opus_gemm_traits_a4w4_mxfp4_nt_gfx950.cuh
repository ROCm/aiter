// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits for the a4w4_mxfp4_nt pipeline: MXFP4 x MXFP4, 1x32 UE8M0 scales, NT, 2x2 4-wave, with all inputs in
// their plain row-major layouts (no shuffle_weight, no e8m0_shuffle):
//   A   [M, K/2]  fp4x2, row-major (stride_a = K/2 bytes)
//   B   [N, K/2]  fp4x2, row-major (stride_b = K/2 bytes)
//   SFA [M, K/32] UE8M0, row-major (stride_sfa = K/32 bytes)
//   SFB [N, K/32] UE8M0, row-major (stride_sfb = K/32 bytes)
//   C   [M, N]    bf16
#pragma once

// Only opus is needed (not opus_gemm_utils.cuh): the kernel is built offline with the pin toolchain into a .co, and
// launchers only need the kargs struct; the full HIP runtime headers do not build with that toolchain on the host side.
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#include <opus/hip_minimal.hpp>
#endif
#include <opus/opus.hpp>

// Kernel arguments.  Units: fp4 tensors in BYTES (fp4x2 per byte, even k in the low nibble), C in elements, scales
// in bytes.  stride_sfa / stride_sfb: row stride of the row-major scales (K/32 bytes).  Tile base offsets are 64-bit
// (batch strides are long long, row / column offsets are widened), so an operand may exceed 2^31 elements; the buffer
// ranges only span one tile, so no 32-bit buffer offset or range exceeds 4 GiB.
struct opus_gemm_mxfp4_kargs_gfx950 {
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
    long long stride_a_batch;
    long long stride_b_batch;
    long long stride_c_batch;
    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int stride_sfa;
    int stride_sfb;
    long long stride_sfa_batch;
    long long stride_sfb_batch;
    int grid_blocks;          // persistent kernel: number of launched blocks (avoids reading the dispatch packet)
};

__host__ __device__ inline int opus_mxfp4_ceil_div(int a, int b) { return (a + b - 1) / b; }
__host__ __device__ inline int opus_mxfp4_min(int a, int b) { return a < b ? a : b; }

// aiter form: <BLOCK_SIZE, seq<B_M, B_N, B_K>, tuple<D_A, D_B, D_C, D_ACC, D_SF>, seq<VEC_A, VEC_B, VEC_C>, seq<GROUP_M, GROUP_N, GROUP_K>>.
// Supported: BLOCK_SIZE 256, tiles 256x256 / 256x128 / 128x256 / 128x128 / 192x256 / 256x192, B_K 256, VEC <16, 16, 4>,
// GROUP <1, 1, 32>.
// MODE_: 0 = persistent (grid = min(tiles, CUs), next tile prefetched in the last K tile, bottom half of C stored in the
//        next tile's prologue); 1 = one tile per block (grid = tiles, no cross-tile code).  (Split-K: not yet.)
// BOUNDARY_: false drops the M / N boundary handling (M % B_M == 0 and N % B_N == 0 required).
template<int BLOCK_SIZE_, typename BLOCK_, typename DTYPE_, typename VEC_, typename GROUP_, int MODE_ = 0, bool BOUNDARY_ = true>
struct opus_gemm_a4w4_mxfp4_nt_traits_gfx950 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using GROUP = opus::remove_cvref_t<GROUP_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr int MODE = MODE_;
    static constexpr bool PERSIST = MODE_ == 0, SPLITK = false, BOUNDARY = BOUNDARY_;
    static_assert(MODE_ == 0 || MODE_ == 1, "a4w4 pipeline: persistent (0) or one tile per block (1)");
    static_assert(BLOCK_SIZE == 256, "4 waves");
    static_assert(opus::get<2>(BLOCK{}) == 256, "B_K is one 256-element K tile");
    static_assert(opus::get<0>(VEC{}) == 16 && opus::get<1>(VEC{}) == 16 && opus::get<2>(VEC{}) == 4);
    static_assert(opus::get<0>(GROUP{}) == 1 && opus::get<1>(GROUP{}) == 1 && opus::get<2>(GROUP{}) == 32, "1x32 scales");

    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;   // 4

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static_assert((B_M == 256 || B_M == 192 || B_M == 128) && (B_N == 256 || B_N == 192 || B_N == 128));
    static_assert(!(B_M == 192 && B_N != 256) && !(B_N == 192 && B_M != 256), "192 tiles: 192x256 / 256x192");
    static constexpr int B_K = 256;                            // fp4 elements
    static constexpr int ELEMS_PER_BYTE = 2;
    static constexpr int B_K_BYTES = B_K / ELEMS_PER_BYTE;     // 128

    static constexpr int T_M = 2;
    static constexpr int T_N = 2;
    static constexpr int T_K = 1;
    static_assert(NUM_WAVES == T_M * T_N * T_K);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;                            // fp4 elements per MFMA
    static constexpr int W_K_BYTES = W_K / ELEMS_PER_BYTE;     // 64

    static constexpr int HALVES = 2;                           // the block tile is 2 x 2 half tiles
    static constexpr int HALF_B_M = B_M / HALVES;
    static constexpr int HALF_B_N = B_N / HALVES;
    static constexpr int E_M = HALF_B_M / (W_M * T_M);         // 4 / 3 / 2
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);              // 2
    static_assert(E_K == 2);
    // The scale LDS image holds 4 repeats per (half, wave) (one byte each after the 4x4 transpose); E <= 4.
    static_assert(E_M <= 4 && E_N <= 4);

    static constexpr int VEC_A = 16;                           // bytes
    static constexpr int VEC_B = 16;                           // bytes
    static constexpr int VEC_A_ELEMS = 32;                     // fp4 elements
    static constexpr int VEC_B_ELEMS = 32;
    static_assert(VEC_A * ELEMS_PER_BYTE == VEC_A_ELEMS && VEC_B * ELEMS_PER_BYTE == VEC_B_ELEMS);
    static_assert(VEC_A_ELEMS == W_K / (WARP_SIZE / W_M));     // one b128 = one fp4 MFMA operand
    static constexpr int VEC_C = 4;

    // LDS (A and B identical, row-major K tiles): one wave issue = 64 lanes x 16 B = 1 KB = 8 rows; chunk stride 1088 B
    // (64 B pad) with the 16 B XOR swizzle of m7 / m8, bank-conflict free ds_read_b128.
    static constexpr int smem_padding = 64;
    static constexpr int smem_linear_wave_a = WARP_SIZE * VEC_A;            // 1024
    static constexpr int smem_linear_wave_b = WARP_SIZE * VEC_B;
    static constexpr int smem_sub_a = smem_linear_wave_a / B_K_BYTES;       // 8 rows per chunk
    static constexpr int smem_sub_b = smem_linear_wave_b / B_K_BYTES;
    static constexpr int ROW_PAIR = W_M / smem_sub_a;                       // 2 interleaved 8-row chunks per 16-row block
    static constexpr int smem_m_rep = HALF_B_M / smem_sub_a;                // chunks per half
    static constexpr int smem_n_rep = HALF_B_N / smem_sub_b;
    // buffer_load ... lds (b128) per wave per half tile per K tile.
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K_BYTES / (BLOCK_SIZE * VEC_A);   // 4 / 3 / 2
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K_BYTES / (BLOCK_SIZE * VEC_B);
    static_assert(a_buffer_load_insts >= 2 && b_buffer_load_insts >= 2);
    // An odd issue count per wave (B_M / B_N = 192) splits a chunk pair (one 16-row block) across two waves.
    static constexpr bool A_SPLIT_PAIR = a_buffer_load_insts % ROW_PAIR != 0, B_SPLIT_PAIR = b_buffer_load_insts % ROW_PAIR != 0;

    // Top-half C copies per thread (bf16 x 8 per copy) and whether the C staging row swizzle applies (B_N = 256 only).
    static constexpr bool C_SWZ = (B_N == 256);

    // Scales: row-major [rows, K/32] UE8M0, 8 groups per K tile.  Global -> VGPR: writer wave (half h, w), lane 4 r + e
    // reads 16 B (16 groups = a panel of 2 K tiles) of row h * HALF_B + e * 32 + w * 16 + r (e >= E: rows of no use, their
    // bytes are never selected); a 4x4 byte transpose inside each lane quad makes byte e = repeat e; written to LDS as
    // [K tile][half][w][slot 64][8 B]: 4 waves x 64 slots x 8 B = 2 KB per K tile, whatever the tile size.
    static constexpr int GROUP_K = 32;
    static constexpr int NUM_KGROUPS = B_K / GROUP_K;                       // 8
    static constexpr int VEC_GLOBAL_SCALE = 16;
    static constexpr int SCALE_PANEL_GROUPS = VEC_GLOBAL_SCALE;                 // 16
    static constexpr int SCALE_PANEL_TILES = SCALE_PANEL_GROUPS / NUM_KGROUPS;  // 2
    static constexpr int VEC_LDS_SCALE = NUM_KGROUPS;                           // 8 bytes per (slot, K tile)
    static constexpr int SFA_PANEL_BYTES = SCALE_PANEL_TILES * NUM_WAVES * WARP_SIZE * VEC_LDS_SCALE;   // 4096
    static constexpr int SFB_PANEL_BYTES = SFA_PANEL_BYTES;
    static constexpr int SCALE_PANEL_BUFFERS = 3;

    static constexpr int SMEM_A_BYTES = smem_m_rep * (smem_linear_wave_a + smem_padding) * 4;
    static constexpr int SMEM_B_BYTES = smem_n_rep * (smem_linear_wave_b + smem_padding) * 4;
    static constexpr int LDS_BYTES = SMEM_A_BYTES + SMEM_B_BYTES + SCALE_PANEL_BUFFERS * (SFA_PANEL_BYTES + SFB_PANEL_BYTES);
    static_assert(LDS_BYTES <= 163840);
};

// Shorthand used by gen_co/co_kernels_gfx950_a4w4.json: opus_mxfp4_nt_traits<B_M, B_N, MODE, BOUNDARY>.
template<int BM, int BN, int MODE = 0, bool BOUNDARY = true>
using opus_mxfp4_nt_traits = opus_gemm_a4w4_mxfp4_nt_traits_gfx950<256, opus::seq<BM, BN, 256>,
    opus::tuple<opus::fp4_t, opus::fp4_t, opus::bf16_t, opus::fp32_t, opus::e8m0_t>, opus::seq<16, 16, 4>, opus::seq<1, 1, 32>, MODE, BOUNDARY>;
