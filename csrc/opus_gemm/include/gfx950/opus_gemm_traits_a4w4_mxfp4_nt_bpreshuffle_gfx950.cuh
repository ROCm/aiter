// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits for the a4w4_mxfp4_nt_bpreshuffle pipeline: MXFP4 x MXFP4, 1x32 UE8M0 scales, NT, 2x2 4-wave, with the inputs
// of aiter gemm_a4w4(bpreshuffle=True):
//   A   [M, K/2]  fp4x2, row-major (stride_a = K/2 bytes)
//   B   shuffle_weight([N, K/2], layout=(16, 16)) fp4x2 (stride_b = K/2 bytes, the row-major row stride)
//   SFA e8m0_shuffle([M, K/32]) UE8M0 (stride_sfa = byte stride of one 32-row block)
//   SFB e8m0_shuffle([N, K/32]) UE8M0 (stride_sfb = byte stride of one 32-row block)
//   C   [M, N]    bf16
#pragma once

// Only opus is needed (not opus_gemm_utils.cuh): the kernel is built offline with the pin toolchain into a .co, and
// launchers only need the kargs struct; the full HIP runtime headers do not build with that toolchain on the host side.
#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#include <opus/hip_minimal.hpp>
#endif
#include <opus/opus.hpp>

// Kernel arguments.  Units: fp4 tensors in BYTES (fp4x2 per byte, even k in the low nibble), C in elements, scales
// in bytes.  e8m0_shuffle scales: rows padded to 32, K/32 padded to 8, [rows/32][K/256][k_lane 4][row 16][kk 2][row-pair 2];
// stride_sfa / stride_sfb: byte stride of one 32-row block (32 * K/32 padded to 8).  Tile base offsets are 64-bit
// (batch strides are long long, row / column offsets are widened), so an operand may exceed 2^31 elements; the buffer
// ranges only span one tile, so no 32-bit buffer offset or range exceeds 4 GiB.
struct opus_gemm_mxfp4_bpreshuffle_kargs_gfx950 {
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
    int split_k;              // MODE 2 (split-K): number of K segments S (grid = tiles * S); otherwise unused
    void* __restrict__ ptr_ws;   // MODE 2: workspace = [tile counters (int, padded to 256 B)][tiles][S][B_M * B_N] fp32 partials
};

__host__ __device__ inline int opus_mxfp4_bpreshuffle_ceil_div(int a, int b) { return (a + b - 1) / b; }
__host__ __device__ inline int opus_mxfp4_bpreshuffle_min(int a, int b) { return a < b ? a : b; }

// aiter form: <BLOCK_SIZE, seq<B_M, B_N, B_K>, tuple<D_A, D_B, D_C, D_ACC, D_SF>, seq<VEC_A, VEC_B, VEC_C>, seq<GROUP_M, GROUP_N, GROUP_K>>.
// Supported: BLOCK_SIZE 256, tiles 256x256 / 256x128 / 128x256 / 128x128 / 192x256 / 256x192, B_K 256, VEC <16, 16, 4>,
// GROUP <1, 1, 32>.
// MODE_: 0 = persistent (grid = min(tiles, CUs), next tile prefetched in the last K tile, bottom half of C stored in the
//        next tile's prologue); 1 = one tile per block (grid = tiles, no cross-tile code); 2 = split-K, one (tile, K segment)
//        per block, fp32 partials in a workspace, the last-arriving block of a tile reduces them and writes C.
// BOUNDARY_: false drops the M / N boundary handling (M % B_M == 0 and N % B_N == 0 required).
template<int BLOCK_SIZE_, typename BLOCK_, typename DTYPE_, typename VEC_, typename GROUP_, int MODE_ = 0, bool BOUNDARY_ = true>
struct opus_gemm_a4w4_mxfp4_nt_bpreshuffle_traits_gfx950 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using GROUP = opus::remove_cvref_t<GROUP_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr int MODE = MODE_;
    static constexpr bool PERSIST = MODE_ == 0, SPLITK = MODE_ == 2, BOUNDARY = BOUNDARY_;
    static_assert(MODE_ >= 0 && MODE_ <= 2, "a4w4 bpreshuffle pipeline: persistent (0), one tile per block (1) or split-K (2)");
    static_assert(!(MODE_ == 2 && opus::get<0>(BLOCK{}) == 256 && opus::get<1>(BLOCK{}) == 256), "split-K 256x256 runs out of VGPRs (86 spilled); use a smaller tile");
    static_assert(BLOCK_SIZE == 256, "4 waves");
    static_assert(opus::get<2>(BLOCK{}) == 256, "B_K is one 256-element K tile");
    static_assert(opus::get<0>(VEC{}) == 16 && opus::get<1>(VEC{}) == 16 && opus::get<2>(VEC{}) == 4);
    static_assert(opus::get<0>(GROUP{}) == 1 && opus::get<1>(GROUP{}) == 1 && opus::get<2>(GROUP{}) == 32, "1x32 scales");

    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;   // 4

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
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

    static constexpr int HALVES = 2;                           // the block tile is 2 x 2 half tiles
    static constexpr int HALF_B_M = B_M / HALVES;
    static constexpr int HALF_B_N = B_N / HALVES;
    static constexpr int E_M = HALF_B_M / (W_M * T_M);         // 4 / 3 / 2
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);              // 2
    static_assert(E_K == 2);

    static constexpr int VEC_A = 16;                           // bytes
    static constexpr int VEC_B = 16;                           // bytes
    static constexpr int VEC_A_ELEMS = 32;                     // fp4 elements
    static constexpr int VEC_B_ELEMS = 32;
    static_assert(VEC_A * ELEMS_PER_BYTE == VEC_A_ELEMS && VEC_B * ELEMS_PER_BYTE == VEC_B_ELEMS);
    static_assert(VEC_A_ELEMS == W_K / (WARP_SIZE / W_M));     // one b128 = one fp4 MFMA operand
    static constexpr int VEC_C = 4;

    // LDS: one wave issue = 64 lanes x 16 B = 1 KB; chunk stride 1088 B (64 B pad).  A: 8 rows per chunk with the 16 B
    // XOR swizzle, bank-conflict free ds_read_b128; B (shuffle_weight): half a 16-row block (one K half) per chunk, read
    // linearly.
    static constexpr int smem_padding = 64;
    static constexpr int smem_linear_wave_a = WARP_SIZE * VEC_A;            // 1024
    static constexpr int smem_linear_wave_b = WARP_SIZE * VEC_B;
    static constexpr int smem_sub_a = smem_linear_wave_a / B_K_BYTES;       // 8 rows per chunk
    static constexpr int smem_sub_b = smem_linear_wave_b / B_K_BYTES;
    static constexpr int ROW_PAIR = W_M / smem_sub_a;                       // 2 chunks per 16-row block (A: interleaved rows, B: K halves)
    static constexpr int smem_m_rep = HALF_B_M / smem_sub_a;                // chunks per half
    static constexpr int smem_n_rep = HALF_B_N / smem_sub_b;
    // buffer_load ... lds (b128) per wave per half tile per K tile.
    static constexpr int a_buffer_load_insts = HALF_B_M * B_K_BYTES / (BLOCK_SIZE * VEC_A);   // 4 / 3 / 2
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K_BYTES / (BLOCK_SIZE * VEC_B);
    static_assert(a_buffer_load_insts >= 2 && b_buffer_load_insts >= 2);
    // An odd issue count per wave (B_M / B_N = 192) splits a chunk pair (one 16-row block) across two waves.
    static constexpr bool A_SPLIT_PAIR = a_buffer_load_insts % ROW_PAIR != 0, B_SPLIT_PAIR = b_buffer_load_insts % ROW_PAIR != 0;

    // Top-half C copies per thread (bf16 x 8 per copy) and whether the C staging row swizzle applies (B_N = 256 only).
    static constexpr int C_COPIES = HALF_B_M * B_N / (BLOCK_SIZE * 8);
    static constexpr bool C_SWZ = (B_N == 256);

    // Scales: e8m0_shuffle, 256 B per 32-row block and K tile; lane l of a block holds the dword l = [kk0 r, kk0 r + 16,
    // kk1 r, kk1 r + 16] (r = l % 16), the MFMA lane layout, so the panel is copied global -> LDS verbatim.  A panel is
    // 2 K tiles x B_M rows = B_M / 32 runs of 512 B, 2 blocks (one 1 KB LDS-DMA) per wave; with fewer blocks than
    // 2 x NUM_WAVES the remaining waves repeat the copies of the first ones (same data to the same LDS bytes).
    static constexpr int GROUP_K = 32;
    static constexpr int NUM_KGROUPS = B_K / GROUP_K;                       // 8
    static constexpr int VEC_GLOBAL_SCALE = 16;
    static constexpr int SCALE_PANEL_GROUPS = VEC_GLOBAL_SCALE;                 // 16
    static constexpr int SCALE_PANEL_TILES = SCALE_PANEL_GROUPS / NUM_KGROUPS;  // 2
    static constexpr int SF_WAVES_A = B_M / 64;
    static constexpr int SF_WAVES_B = B_N / 64;
    static constexpr int SFA_PANEL_BYTES = B_M * SCALE_PANEL_GROUPS;            // 4096
    static constexpr int SFB_PANEL_BYTES = B_N * SCALE_PANEL_GROUPS;
    static constexpr int SCALE_PANEL_BUFFERS = 3;

    static constexpr int SMEM_A_BYTES = smem_m_rep * (smem_linear_wave_a + smem_padding) * 4;
    static constexpr int SMEM_B_BYTES = smem_n_rep * (smem_linear_wave_b + smem_padding) * 4;
    static constexpr int LDS_BYTES = SMEM_A_BYTES + SMEM_B_BYTES + SCALE_PANEL_BUFFERS * (SFA_PANEL_BYTES + SFB_PANEL_BYTES);
    static_assert(LDS_BYTES <= 163840);
};

// Shorthand used by gen_co/co_kernels_gfx950_a4w4_bpreshuffle.json: opus_mxfp4_nt_bpreshuffle_traits<B_M, B_N, MODE, BOUNDARY>.
template<int BM, int BN, int MODE = 0, bool BOUNDARY = true>
using opus_mxfp4_nt_bpreshuffle_traits = opus_gemm_a4w4_mxfp4_nt_bpreshuffle_traits_gfx950<256, opus::seq<BM, BN, 256>,
    opus::tuple<opus::fp4_t, opus::fp4_t, opus::bf16_t, opus::fp32_t, opus::e8m0_t>, opus::seq<16, 16, 4>, opus::seq<1, 1, 32>, MODE, BOUNDARY>;

// Split-K workspace: a fixed 4 KB counter block (one int per tile, at most 1024 tiles) followed by tiles * S partial
// tiles of B_M * B_N fp32.  The launcher zeroes the counter block before every split-K launch (the workspace may be a
// fresh torch.empty); each tile's last block also resets its own counter after the reduction.
constexpr int OPUS_MXFP4_BPRESHUFFLE_SK_MAX_TILES = 1024;
__host__ __device__ inline long opus_mxfp4_bpreshuffle_sk_counter_bytes(int /*tiles*/) { return OPUS_MXFP4_BPRESHUFFLE_SK_MAX_TILES * 4; }
__host__ __device__ inline long opus_mxfp4_bpreshuffle_sk_ws_bytes(int tiles, int split_k, int b_m, int b_n) {
    return opus_mxfp4_bpreshuffle_sk_counter_bytes(tiles) + static_cast<long>(tiles) * split_k * b_m * b_n * 4;
}
