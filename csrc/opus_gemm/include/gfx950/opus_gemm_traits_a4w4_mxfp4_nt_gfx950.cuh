// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kargs for the a4w4_mxfp4_nt pipeline (MXFP4 x MXFP4, 1x32 UE8M0 scales, NT, 2x2 4-wave, persistent).
#pragma once

// Only opus is needed (not opus_gemm_utils.cuh): the kernel is built offline with the pin toolchain into a .co, and
// launchers only need the kargs struct; the full HIP runtime headers do not build with that toolchain on the host side.
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#include <opus/hip_minimal.hpp>
#endif
#include <opus/opus.hpp>

// Kernel arguments.  Units: fp4 tensors in BYTES (fp4x2 per byte, even k in
// the low nibble), C in fp32 elements, scales in bytes.  A is row-major [M, K/2]; B is in the aiter
// shuffle_weight(w, layout=(16, 16)) layout (stride_b stays K/2, the row-major row stride).  SFA/SFB are UE8M0 in
// the aiter gfx950 e8m0_shuffle / shuffle_scale layout (rows padded to 32,
// K/32 padded to 8, [rows/32][K/256][k_lane 4][row 16][kk 2][row-pair 2]);
// stride_sfa/sfb is the byte stride of one 32-row block (32 * K/32 padded to 8).
struct opus_gemm_mxfp4_scale_kargs_gfx950 {
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
    int grid_blocks;          // persistent kernel: number of launched blocks (avoids reading the dispatch packet)
    int split_k;              // MODE 2 (split-K): number of K segments S (grid = tiles * S); otherwise unused
    void* __restrict__ ptr_ws;   // MODE 2: workspace = [tile counters (int, padded to 256 B)][tiles][S][B_M * B_N] fp32 partials
};

__host__ __device__ inline int opus_mxfp4_ceil_div(int a, int b) { return (a + b - 1) / b; }

// MXFP4 x MXFP4 NT, 4 waves (2x2), tile 256x256x256 (elements), 1x32 UE8M0.
// K tile = 256 fp4 = 128 bytes per row: the gmem/LDS geometry of the fp8
// 4-wave kernel.  One fp4 MFMA covers K = W_K = 128, so a K tile is
// E_K = 2 MFMAs along K (kk = 0: bytes [16g, +16) = K [32g, +32);
// kk = 1: bytes [64+16g, +16) = K [128+32g, +32), g = lane / 16).
// aiter form: <BLOCK_SIZE, seq<B_M, B_N, B_K>, tuple<D_A, D_B, D_C, D_ACC, D_SF>, seq<VEC_A, VEC_B, VEC_C>, seq<GROUP_M, GROUP_N, GROUP_K>>.
// Supported: BLOCK_SIZE 256, B_M / B_N in {256, 128}, B_K 256, VEC <16, 16, 4>, GROUP <1, 1, 32>.
// MODE_: 0 = persistent (grid = min(tiles, CUs), next tile prefetched in the last K tile, bottom half of C stored in the
//        next tile's prologue); 1 = one tile per block (grid = tiles, no cross-tile code); 2 = split-K, one (tile, K segment)
//        per block, fp32 partials in a workspace, the last-arriving block of a tile reduces them and writes C.
// BOUNDARY_: false drops the M / N boundary handling (M % B_M == 0 and N % B_N == 0 required).
template<int BLOCK_SIZE_, typename BLOCK_, typename DTYPE_, typename VEC_, typename GROUP_, int MODE_ = 0, bool BOUNDARY_ = true>
struct opus_gemm_a4w4_mxfp4_nt_traits_gfx950 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using GROUP = opus::remove_cvref_t<GROUP_>;
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr int MODE = MODE_;
    static constexpr bool PERSIST = MODE_ == 0, SPLITK = MODE_ == 2, BOUNDARY = BOUNDARY_;
    static_assert(MODE_ >= 0 && MODE_ <= 2);
    static_assert(!(MODE_ == 2 && opus::get<0>(BLOCK{}) == 256 && opus::get<1>(BLOCK{}) == 256), "split-K 256x256 runs out of VGPRs (86 spilled); use a smaller tile");
    static_assert(BLOCK_SIZE == 256, "4 waves");
    static_assert(opus::get<2>(BLOCK{}) == 256, "B_K is one 256-element K tile");
    static_assert(opus::get<0>(VEC{}) == 16 && opus::get<1>(VEC{}) == 16 && opus::get<2>(VEC{}) == 4);
    static_assert(opus::get<0>(GROUP{}) == 1 && opus::get<1>(GROUP{}) == 1 && opus::get<2>(GROUP{}) == 32, "1x32 scales");
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;   // 4

    static constexpr int B_M = opus::get<0>(BLOCK{});   // 256 or 128
    static constexpr int B_N = opus::get<1>(BLOCK{});   // 256 or 128
    static_assert((B_M == 256 || B_M == 192 || B_M == 128) && (B_N == 256 || B_N == 192 || B_N == 128));
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

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;
    static constexpr int E_M = HALF_B_M / (W_M * T_M);         // 4
    static constexpr int E_N = HALF_B_N / (W_N * T_N);         // 4
    static constexpr int E_K = B_K / (W_K * T_K);              // 2
    static_assert(E_K == 2);

    static constexpr int VEC_A = 16;                           // bytes
    static constexpr int VEC_B = 16;                           // bytes
    static constexpr int VEC_A_ELEMS = 32;                     // fp4 elements
    static constexpr int VEC_B_ELEMS = 32;
    static_assert(VEC_A * ELEMS_PER_BYTE == VEC_A_ELEMS && VEC_B * ELEMS_PER_BYTE == VEC_B_ELEMS);
    static_assert(VEC_A <= 16 && VEC_B <= 16, "buffer_load ... lds moves at most 16 bytes per lane");
    static_assert(VEC_A_ELEMS == W_K / (WARP_SIZE / W_M));     // one b128 = one fp4 MFMA operand
    static constexpr int VEC_C = 4;

    // LDS: one wave issue = 64 lanes x 16 bytes = 1024 bytes = 8 rows, +32 pad.
    static constexpr int smem_padding = 64;   // s1: chunk stride 1088 B; with the 16 B swizzle the A/B ds_read_b128 are bank-conflict free
    static constexpr int smem_linear_wave_a = WARP_SIZE * VEC_A;            // 1024
    static constexpr int smem_linear_wave_b = WARP_SIZE * VEC_B;
    static constexpr int smem_sub_a = smem_linear_wave_a / B_K_BYTES;       // 8 rows per chunk
    static constexpr int smem_sub_b = smem_linear_wave_b / B_K_BYTES;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub_a;                // 16 chunks per half
    // buffer_load ... lds (b128) per wave per half tile per K tile: 128 rows x 128 B / (256 lanes x 16 B).
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K_BYTES / (BLOCK_SIZE * VEC_A);   // 4
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K_BYTES / (BLOCK_SIZE * VEC_B);   // 4
    static constexpr int smem_n_rep = HALF_B_N / smem_sub_b;
    // Per-wave LDS-DMA issues per half tile (each issue = 64 lanes x 16 B = 8 rows of A / half a 16-row B block).
    static_assert(a_buffer_load_insts >= 2 && b_buffer_load_insts >= 2);
    // Odd issue counts (B_M / B_N = 192: 3 per half) use explicit per-issue chunk addressing (dma_odd in the pipeline).
    static constexpr bool A_ODD = a_buffer_load_insts % 2 == 1, B_ODD = b_buffer_load_insts % 2 == 1;
    // Scale panels: 32-row blocks, 2 blocks (2 x 512 B = one 1 KB DMA) per wave; with fewer blocks than 2 x NUM_WAVES the
    // remaining waves repeat the copies of the first ones (same data to the same LDS bytes), so every wave issues one DMA.
    static constexpr int SF_WAVES_A = B_M / 64;
    static constexpr int SF_WAVES_B = B_N / 64;
    // Top-half C copies per thread (bf16 x 8 per copy) and whether the C staging row swizzle applies (B_N = 256 only).
    static constexpr int C_COPIES = HALF_B_M * B_N / (BLOCK_SIZE * 8);
    static constexpr bool C_SWZ = (B_N == 256);

    // Scales: native row-major [rows, K/32] UE8M0, 8 columns per K tile.
    static constexpr int GROUP_K = 32;
    static constexpr int NUM_KGROUPS = B_K / GROUP_K;                       // 8
    // Global -> LDS: one b128 per lane = 16 columns of one row (lane = row,
    // 4 waves x 64 lanes = 256 rows).  Panel = 16 columns = 2 K tiles, double
    // buffered; LDS image [row 0..255][16 B].
    static constexpr int VEC_GLOBAL_SCALE = 16;
    static constexpr int SCALE_PANEL_GROUPS = VEC_GLOBAL_SCALE;                 // 16
    static constexpr int SCALE_PANEL_TILES = SCALE_PANEL_GROUPS / NUM_KGROUPS;  // 2
    static constexpr int SFA_PANEL_BYTES = B_M * SCALE_PANEL_GROUPS;            // 4096
    static constexpr int SFB_PANEL_BYTES = B_N * SCALE_PANEL_GROUPS;
    // Three panel buffers: with the rotated main loop the scales of tile t+1 are read
    // in the middle of iteration t, so a panel is copied about two tiles ahead.
    static constexpr int SCALE_PANEL_BUFFERS = 3;
    // Consumer: per tile one b64 per half per tensor = the tile's 8 columns
    // = [kk0 dword, kk1 dword] (the tile's offset in the panel row is (tile % 2) * 8).
    static constexpr int VEC_LDS_SCALE = NUM_KGROUPS;                           // 8 bytes

    static constexpr int SMEM_A_BYTES = smem_m_rep * (smem_linear_wave_a + smem_padding) * 4;  // 67584
    static constexpr int SMEM_B_BYTES = smem_n_rep * (smem_linear_wave_b + smem_padding) * 4;
    static constexpr int LDS_BYTES = SMEM_A_BYTES + SMEM_B_BYTES + SCALE_PANEL_BUFFERS * (SFA_PANEL_BYTES + SFB_PANEL_BYTES);
    static_assert(LDS_BYTES <= 163840);   // s1: padding 64 -> exactly 160 KB
};
// Shorthand used by the dev harness: opus_mxfp4_nt_traits<B_M, B_N>.
template<int BM, int BN, int MODE = 0, bool BOUNDARY = true>
using opus_mxfp4_nt_traits = opus_gemm_a4w4_mxfp4_nt_traits_gfx950<256, opus::seq<BM, BN, 256>,
    opus::tuple<opus::fp4_t, opus::fp4_t, opus::bf16_t, opus::fp32_t, opus::e8m0_t>, opus::seq<16, 16, 4>, opus::seq<1, 1, 32>, MODE, BOUNDARY>;
// Split-K workspace: a fixed 4 KB counter block (one int per tile, at most 1024 tiles; zeroed once, each tile's last block
// resets its counter) followed by tiles * S partial tiles of B_M * B_N fp32.  The fixed size keeps the counters in place
// whatever the shape, so a reused workspace never needs re-zeroing.
constexpr int OPUS_MXFP4_SK_MAX_TILES = 1024;
__host__ __device__ inline long opus_mxfp4_sk_counter_bytes(int /*tiles*/) { return OPUS_MXFP4_SK_MAX_TILES * 4; }
__host__ __device__ inline long opus_mxfp4_sk_ws_bytes(int tiles, int split_k, int b_m, int b_n) {
    return opus_mxfp4_sk_counter_bytes(tiles) + static_cast<long>(tiles) * split_k * b_m * b_n * 4;
}
