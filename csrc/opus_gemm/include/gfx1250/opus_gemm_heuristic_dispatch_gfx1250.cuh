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

// Kid map (see opus_gemm_common.py; B_K=128 picks here, tuner explores 256/512):
//   tileN (B_M=16, B_N=32): 20000 (B_K=128)
//   tileM (B_M=32)        : 20010=32x32, 20011=32x64, 20012=32x128, 20013=32x256
inline int opus_a16w16_heuristic_kid_gfx1250(int M, int N, int K, bool has_bias)
{
    (void)K;
    (void)has_bias;  // bias is folded by the reduce kernel for every kid.

    // M >= 32 (and M % 32 == 0) -> tileM (B_M=32); widest B_N that divides N.
    if (M % 32 == 0)
    {
        if (N % 256 == 0) return 20013;  // 32x256
        if (N % 128 == 0) return 20012;  // 32x128
        if (N % 64 == 0)  return 20011;  // 32x64
        if (N % 32 == 0)  return 20010;  // 32x32
    }

    // Small M (M % 16 == 0) -> tileN 16x32 (requires N % 32 == 0). Otherwise
    // the launcher rejects the shape (ragged M/N unsupported on this family).
    return 20000;  // 16x32x128
}
