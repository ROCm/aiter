// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 a16w16 shape-heuristic: (M, N, K, has_bias) -> kid. Pure integer
// mapping (no launcher symbols) so it can be included by the dispatcher TU
// without dragging in the lookup macros.
//
// All gfx1250 kids are cluster/TDM split-K (workspace + reduce). The kernel
// requires M % B_M == 0 and N % B_N == 0 (ragged M/N is not supported; ragged
// K is, via the TDM k_extent clamp). The heuristic therefore picks the largest
// tile from the kid set whose B_M divides M and B_N divides N, preferring the
// B_M=16 "tileN" family for small M and the "tileM" family for larger M.
//
// MUST stay in sync with opus_gemm_common.py :: gfx1250_kernels_list and
// HEURISTIC_DEFAULT_KIDS_GFX1250.
#pragma once

// Kid map (B_K=128 chosen here; tuner explores B_K 256/512 + the P/wg space).
// Tiles whose per-TDM direct-copy request count (rows*B_K*2/256) hits the 256
// SIMD-pair limit on some operand are NOT generated (e.g. 32x256x128) so the
// heuristic must not return them. All returned kids are no-cluster prefetch-3.
//   tileN (B_M=16): 20000=16x32, 20024=16x64, 20032=16x128
//   tileM (B_M=32): 20040=32x32, 20048=32x64, 20056=32x128
// MUST stay in sync with opus_gemm_common.py :: gfx1250_kernels_list and
// GFX1250_BASE_KIDS (regenerated; kid numbers are NOT contiguous).
inline int opus_a16w16_heuristic_kid_gfx1250(int M, int N, int K, bool has_bias)
{
    (void)K;
    (void)has_bias;  // bias is folded by the reduce kernel for every kid.

    // M >= 32 (and M % 32 == 0) -> tileM (B_M=32); widest B_N that divides N.
    // (32x256 is unavailable -- per-TDM B req = 256 hits the direct-copy limit;
    // fall through to the B_M=16 tileN family for N % 256 == 0.)
    if (M % 32 == 0)
    {
        if (N % 128 == 0) return 20056;  // 32x128x128
        if (N % 64 == 0)  return 20048;  // 32x64x128
        if (N % 32 == 0)  return 20040;  // 32x32x128
    }

    // Small M (or N not tileM-friendly) -> tileN family (B_M=16). Ragged M/N is
    // handled by the TDM row/col clamp + padded workspace, so the smallest
    // 16x32 tile is always a valid fallback.
    if (N % 128 == 0) return 20032;  // 16x128x128
    if (N % 64 == 0)  return 20024;  // 16x64x128
    return 20000;                    // 16x32x128
}
