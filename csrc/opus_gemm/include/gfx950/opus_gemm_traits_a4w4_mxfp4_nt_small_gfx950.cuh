// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kargs of the shuffle small-tile MXFP4 x MXFP4 NT pipeline (gfx950):
// opus_gemm_pipeline_a4w4_mxfp4_nt_small_gfx950.cuh (1x4 waves).  Inputs as aiter gemm_a4w4(bpreshuffle=True):
//   A [batch, M, K/2] fp4x2 row-major;  B shuffle_weight(16, 16);  scaleA / scaleB e8m0_shuffle (1x32 UE8M0);
//   C [batch, M, N] bf16.  Any M, N % 16 == 0, K % 256 == 0.  Non-persistent: grid = (N/B_N, M/B_M, batch * split_k).
#pragma once

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#include <opus/hip_minimal.hpp>
#endif
#include <opus/opus.hpp>

#ifndef OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
#define OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
// Indirection slot for the split-K fp32 workspace pointer (same struct as opus_gemm_traits_a16w16_gfx950.cuh): captured
// HIP graphs hold the slot address, so a later workspace grow does not leave a dangling pointer in the graph.
struct opus_splitk_ws_handle {
    void*         ptr;    // current backing workspace; null until first grow
    unsigned long bytes;  // current capacity in bytes
};
#endif

// Units: A / B in bytes (fp4x2), C in elements, scales in bytes.
// split_k = S > 1: block z = batch * S + k split writes its fp32 partial tile to ws_handle->ptr, laid out as
// [S][batch][padded_M][padded_N] (padded_M / padded_N = M / N rounded up to B_M / B_N), and splitk_reduce_kernel
// (splitk_reduce_gfx950.cuh) sums the S partials into C.  S = 1: C is written directly, ws_handle is not read.
struct opus_gemm_mxfp4_small_kargs_gfx950 {
    const void* ptr_a; const void* ptr_b; void* ptr_c; const void* ptr_sfa; const void* ptr_sfb;
    int m, n, k, batch;
    int stride_a, stride_c;                 // bytes per A row (K/2); elements per C row
    int stride_sfa, stride_sfb;             // bytes per 32-row block of the e8m0_shuffle scales (32 * K/32 padded to 8)
    long long stride_a_batch, stride_b_batch, stride_c_batch, stride_sfa_batch, stride_sfb_batch;   // bytes (C: elements)
    int split_k;                            // S >= 1 (grid.z = batch * S)
    const opus_splitk_ws_handle* ws_handle; // S > 1: fp32 partials (see above)
};

// Tiles B_M 16/32/64 x B_N 32/64/128/256, K tile B_K = 256.  STAGES (D) = ring depth: ring step t lives in slot t % D
// (an LDS stage for A, a register slot for each wave's B and scales).  KS = 1: the 4 waves split B_N (1x4); KS = 2 / 4:
// the waves split KS ways along K and 4/KS ways along N (tiles too narrow to give every wave 16 columns), and their
// partial tiles are summed through LDS once at the end.
template<int BM_, int BN_, int D_, int KS_>
struct opus_gemm_a4w4_mxfp4_nt_small_traits_gfx950 {
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = 256;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;   // 4

    static constexpr int B_M = BM_;
    static constexpr int B_N = BN_;
    static constexpr int B_K = 256;                            // fp4 elements per K tile
    static constexpr int ELEMS_PER_BYTE = 2;
    static constexpr int B_K_BYTES = B_K / ELEMS_PER_BYTE;     // 128
    static constexpr int STAGES = D_;                          // ring depth
    static constexpr int KS = KS_;                             // in-block K split of the waves

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WAVES / KS;                 // waves along N
    static constexpr int T_K = KS;                             // waves along K
    static_assert(NUM_WAVES == T_M * T_N * T_K);

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;                            // fp4 elements per MFMA
    static constexpr int W_K_BYTES = W_K / ELEMS_PER_BYTE;     // 64

    static constexpr int E_M = B_M / (W_M * T_M);              // M repeats per wave
    static constexpr int E_N = B_N / (W_N * T_N);              // N repeats per wave
    static constexpr int E_K = B_K / W_K;                      // 2 MFMAs along K per K tile
    static constexpr int WAVE_N = B_N / T_N;                   // columns per wave

    static constexpr int VEC_A = 16;                           // bytes = 32 fp4 = one MFMA operand
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;                            // fp32 per lane per 16x16 C fragment
    static constexpr int VEC_SF = 4;                           // one raw e8m0_shuffle dword per lane

    // A: global -> LDS by LDS-DMA, one issue = 64 lanes x 16 B = 8 rows x 128 B.  An 8-row chunk takes
    // smem_linear_wave_a + smem_padding = 1088 B (row r at r * 128, 16 B piece c at slot c ^ ((r >> 1) & 3)).
    static constexpr int smem_linear_wave_a = WARP_SIZE * VEC_A;             // 1024
    static constexpr int smem_padding = 64;
    static constexpr int smem_sub_a = smem_linear_wave_a / B_K_BYTES;        // 8 rows per chunk
    static constexpr int A_CHUNK_BYTES = smem_linear_wave_a + smem_padding;  // 1088
    static constexpr int A_CHUNKS = B_M / smem_sub_a;                        // chunks per K tile
    static constexpr int A_TILE_BYTES = A_CHUNKS * A_CHUNK_BYTES;
    static constexpr int STAGE_BYTES = KS * A_TILE_BYTES;                    // one ring step = KS K tiles of A
    static constexpr int A_DMA_CHUNKS = KS * A_CHUNKS;
    // LDS-DMA issues per wave per ring step; with fewer chunks than waves the last waves repeat the last chunk.
    static constexpr int a_buffer_load_insts = (A_DMA_CHUNKS + NUM_WAVES - 1) / NUM_WAVES;
    static constexpr bool A_DMA_EXACT = A_DMA_CHUNKS % NUM_WAVES == 0;

    // B: shuffle_weight(16, 16) = 1 KB per 16 columns x 128 K, global -> registers, E_N x E_K b128 per wave per K tile.
    static constexpr int B_BLOCK_K_BYTES = W_N * W_K_BYTES;                  // 1024 (one 16-col x 128-K block)
    static constexpr int b_buffer_load_insts = E_N * E_K;

    // Scales: e8m0_shuffle, 256 B per K tile in each 32-row block; lane l's dword = rows l % 16 and l % 16 + 16 of the
    // block x [kk 0, kk 1] (bytes [kk0 r, kk0 r+16, kk1 r, kk1 r+16]), so one b32 per lane per 32-row block per K tile.
    static constexpr int SF_ROWS = 32;
    static constexpr int SF_K_TILE_BYTES = WARP_SIZE * VEC_SF;               // 256
    static constexpr int SFA_LOADS = (B_M + SF_ROWS - 1) / SF_ROWS;          // b32 per wave per K tile
    static constexpr int SFB_LOADS = (E_N + 1) / 2;
    static constexpr bool SFA_HALF = B_M == 16;                              // one 16-row M block: half of a 32-row block
    static constexpr bool SFB_HALF = E_N == 1;
    static_assert(E_N % 2 == 0 || E_N == 1, "a wave covers one 16-column block or whole 32-row scale blocks");

    // VMEM instructions per wave per ring step (the counted vmcnt waits are in units of this).
    static constexpr int VMEM_PER_STEP = a_buffer_load_insts + SFA_LOADS + b_buffer_load_insts + SFB_LOADS;

    // KS > 1: groups 1..KS-1 park their partial tiles in LDS, group 0 adds them.
    static constexpr int RED_BYTES = KS > 1 ? (KS - 1) * T_N * E_M * E_N * WARP_SIZE * VEC_C * 4 : 0;
    static constexpr int LDS_BYTES = STAGES * STAGE_BYTES + RED_BYTES;

    static_assert(B_M == 16 || B_M == 32 || B_M == 64 || B_M == 128);
    static_assert(B_N == 32 || B_N == 64 || B_N == 128 || B_N == 256);
    static_assert(KS == 1 || KS == 2 || KS == 4, "KS = 1 / 2 / 4");
    static_assert(E_N >= 1 && WAVE_N % W_N == 0);
    static_assert(STAGES >= 2, "ring depth >= 2");
    static_assert(STAGES * STAGE_BYTES <= 160 * 1024, "LDS ring too deep for this tile");
    static_assert((STAGES - 1) * VMEM_PER_STEP <= 63, "vmcnt range: ring too deep for this tile");
    static_assert(VEC_A * ELEMS_PER_BYTE == W_K / (WARP_SIZE / W_M), "one b128 = one fp4 MFMA operand");
};
