// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kargs of the MXFP4 x MXFP4 NT small-tile pipeline (any M, N % 8 == 0, K % 256 == 0).
#pragma once

#if defined(__HIP_DEVICE_COMPILE__) || defined(__HIPCC_RTC__)
#include <opus/hip_minimal.hpp>
#endif
#include <opus/opus.hpp>

#ifndef OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
#define OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
// Split-K workspace slot (as a16w16): HIP graphs hold the slot, so a workspace grow keeps them valid.
struct opus_splitk_ws_handle {
    void*         ptr;    // current backing workspace; null until first grow
    unsigned long bytes;  // current capacity in bytes
};
#endif

// Units: A / B / scales in bytes, C in elements.  S > 1: fp32 partials [S][batch][padded_M][padded_N] in the workspace.
struct opus_gemm_mxfp4_small_kargs_gfx950 {
    const void* ptr_a; const void* ptr_b; void* ptr_c; const void* ptr_sfa; const void* ptr_sfb;
    int m, n, k, batch;
    int stride_a, stride_b, stride_c;       // bytes per A / B row (K/2); elements per C row
    int stride_sfa, stride_sfb;             // bytes per scale row (K/32)
    long long stride_a_batch, stride_b_batch, stride_c_batch, stride_sfa_batch, stride_sfb_batch;   // bytes (C: elements)
    int split_k;                            // S >= 1 (grid.z = batch * S)
    const opus_splitk_ws_handle* ws_handle; // S > 1: fp32 partials (see above)
};

// B_M 16/32/64 x B_N 16..256 x B_K 256; STAGES-deep ring of KS K tiles per step; waves T_N along N x KS along K.
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

    static constexpr int E_M = B_M / (W_M * T_M);              // M repeats per wave
    static constexpr int E_N = B_N / (W_N * T_N);              // N repeats per wave
    static constexpr int E_K = B_K / W_K;                      // 2 MFMAs along K per K tile
    static constexpr int WAVE_N = B_N / T_N;                   // columns per wave

    static constexpr int VEC_A = 16;                           // bytes = 32 fp4 = one MFMA operand
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;                            // fp32 per lane per 16x16 C fragment

    // A / B: one buffer_load ... lds = an 8-row x 128 B chunk (1088 B in LDS with padding).
    static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;               // 1024
    static constexpr int smem_padding = 64;
    static constexpr int smem_sub = smem_linear_wave / B_K_BYTES;            // 8 rows per chunk
    static constexpr int CHUNK_BYTES = smem_linear_wave + smem_padding;      // 1088
    static constexpr int A_CHUNKS = B_M / smem_sub;                          // chunks per K tile
    static constexpr int B_CHUNKS = B_N / smem_sub;
    static constexpr int A_TILE_BYTES = A_CHUNKS * CHUNK_BYTES;
    static constexpr int B_TILE_BYTES = B_CHUNKS * CHUNK_BYTES;

    // Scales: a panel of STAGES * KS K tiles per ring cycle, [rows][SF_ROW_BYTES], double-buffered.
    static constexpr int SF_GROUP = 32;                                      // fp4 elements per scale
    static constexpr int SF_K_TILE_BYTES = B_K / SF_GROUP;                   // 8
    static constexpr int VEC_SF = SF_K_TILE_BYTES;                           // one ds_read_b64 per 16-row block
    static constexpr int VEC_SF_DMA = 4;                                     // bytes per lane of the scale copy
    static constexpr int SF_PANEL_TILES = STAGES * KS;
    static constexpr int SF_ROW_BYTES = SF_K_TILE_BYTES * SF_PANEL_TILES;    // one row of the panel
    static constexpr int SF_DMA_BYTES = WARP_SIZE * VEC_SF_DMA;              // 256 per copy
    static constexpr int SF_PANEL_ROWS = SF_DMA_BYTES / SF_ROW_BYTES;        // rows per copy
    static constexpr int SFA_PANEL_DMAS = B_M / SF_PANEL_ROWS;
    static constexpr int SFB_PANEL_DMAS = B_N / SF_PANEL_ROWS;
    static constexpr int sfa_buffer_load_insts = SFA_PANEL_DMAS / NUM_WAVES;   // per wave per panel
    static constexpr int sfb_buffer_load_insts = SFB_PANEL_DMAS / NUM_WAVES;
    static constexpr int sf_panel_insts = sfa_buffer_load_insts + sfb_buffer_load_insts;
    static constexpr int SFA_PANEL_BYTES = B_M * SF_ROW_BYTES;
    static constexpr int SFB_PANEL_BYTES = B_N * SF_ROW_BYTES;
    static constexpr int SCALE_PANEL_BUFFERS = 2;

    // Ring stage = KS K tiles: [A tiles][B tiles].
    static constexpr int STAGE_B_OFF = KS * A_TILE_BYTES;
    static constexpr int STAGE_BYTES = STAGE_B_OFF + KS * B_TILE_BYTES;
    // Copies per wave per ring step (chunks split evenly over the waves).
    static constexpr int a_buffer_load_insts = KS * A_CHUNKS / NUM_WAVES;
    static constexpr int b_buffer_load_insts = KS * B_CHUNKS / NUM_WAVES;

    // VMEM instructions per wave per ring step (A / B).
    static constexpr int VMEM_PER_STEP = a_buffer_load_insts + b_buffer_load_insts;
    // LDS reads per wave per ring step.
    static constexpr int DS_READS_PER_STEP = E_K * (E_M + E_N) + E_M + E_N;

    // KS > 1: partial tiles of groups 1..KS-1 (in the ring stages after the loop).
    static constexpr int RED_BYTES = KS > 1 ? (KS - 1) * T_N * E_M * E_N * WARP_SIZE * VEC_C * 4 : 0;
    static constexpr int LDS_BYTES = STAGES * STAGE_BYTES + SCALE_PANEL_BUFFERS * (SFA_PANEL_BYTES + SFB_PANEL_BYTES);

    static_assert(B_M == 16 || B_M == 32 || B_M == 64);
    static_assert(B_N == 16 || B_N == 32 || B_N == 64 || B_N == 128 || B_N == 256);
    static_assert(KS == 1 || KS == 2 || KS == 4, "KS = 1 / 2 / 4");
    static_assert(E_N >= 1 && WAVE_N % W_N == 0);
    static_assert(STAGES >= 2 && STAGES % 2 == 0, "ring depth even (operands double-buffered by step parity)");
    static_assert(RED_BYTES <= STAGES * STAGE_BYTES, "KS partial tiles fit in the ring");
    static_assert(LDS_BYTES <= 160 * 1024, "LDS ring too deep for this tile");
    static_assert((STAGES - 1) * VMEM_PER_STEP + sf_panel_insts <= 63, "vmcnt range: ring too deep for this tile");
    static_assert(KS * A_CHUNKS % NUM_WAVES == 0 && a_buffer_load_insts >= 1 && A_CHUNKS % a_buffer_load_insts == 0,
                  "A chunks split evenly over the waves");
    static_assert(KS * B_CHUNKS % NUM_WAVES == 0 && b_buffer_load_insts >= 1 && B_CHUNKS % b_buffer_load_insts == 0,
                  "B chunks split evenly over the waves");
    static_assert(SF_PANEL_ROWS >= 1 && B_M % SF_PANEL_ROWS == 0 && B_N % SF_PANEL_ROWS == 0, "scale panel rows");
    static_assert(SFA_PANEL_DMAS % NUM_WAVES == 0 && SFB_PANEL_DMAS % NUM_WAVES == 0, "scale panel copies split evenly over the waves");
    static_assert(VEC_A * ELEMS_PER_BYTE == W_K / (WARP_SIZE / W_M), "one b128 = one fp4 MFMA operand");
};
