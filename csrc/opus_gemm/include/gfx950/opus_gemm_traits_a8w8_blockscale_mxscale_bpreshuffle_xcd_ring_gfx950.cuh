// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

// A/B LDS ring with resident scales. SplitK=1 writes C directly; split-K
// uses the paired xcd_ring grid/workspace contract for XCD-local publication.
template<int BM,
          int BN,
          int WN,
          int BK,
          int Stages,
          int TotalK,
          int SplitK,
          int WM            = 1,
          bool CompactScale = false,
          int BCache        = 0>
struct opus_gemm_bpreshuffle_xcd_ring_traits
{
    // Launch geometry.
    static constexpr int BLOCK_SIZE = WM * WN * 64;
    static constexpr int WG_PER_CU = 1;

    // Matrix, wave, and MFMA tiles.
    static constexpr int B_M = BM;
    static constexpr int B_N = BN;
    static constexpr int B_K = BK;
    static constexpr int T_M = WM;
    static constexpr int T_N = WN;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int E_M = BM / (16 * WM);
    static constexpr int E_N = BN / (16 * WN);

    // K partition and the resident scale panel.
    static constexpr int SPLIT_K = SplitK;
    static constexpr int TOTAL_K = TotalK;
    static constexpr int K = TotalK / SplitK;
    static constexpr int PANELS = K / BK;
    static constexpr int KS = BK / 128;
    static constexpr int SC = K / 128;

    // Ring scheduling.
    static constexpr int STAGES = Stages;
    static constexpr int INITIAL = Stages - 1;
    static constexpr int B_CACHE = BCache;

    // Matrix copies and LDS stages.
    static constexpr int A_BYTES = BM * BK;
    static constexpr int B_BYTES = BN * BK;
    static constexpr int STAGE_BYTES = A_BYTES + B_BYTES;
    static constexpr int A_COPIES = A_BYTES / (BLOCK_SIZE * 16);
    static constexpr int B_COPIES = B_BYTES / (BLOCK_SIZE * 16);
    static constexpr int MATRIX_ISSUES = A_COPIES + B_COPIES;

    // Compact copies group four M scales; the host enforces their alignment.
    static constexpr int SCALE_COPY = CompactScale ? 4 : 1;
    static constexpr int SA_COPIES = (BM * SC + BLOCK_SIZE * SCALE_COPY - 1) / (BLOCK_SIZE * SCALE_COPY);
    static constexpr int SA_BYTES = SA_COPIES * BLOCK_SIZE * 4;

    // LDS offsets and total allocation.
    static constexpr int OFF_SA = STAGES * STAGE_BYTES;
    static constexpr int OFF_SB = OFF_SA + SA_BYTES;
    static constexpr int LDS_BYTES = OFF_SB + BLOCK_SIZE * 4;

    static constexpr int wait(int t)
    {
        int issued = INITIAL * MATRIX_ISSUES + SA_COPIES + 1;
        int ready  = (t + 1) * MATRIX_ISSUES + SA_COPIES + 1;
        for (int i = 0; i < t; ++i)
        {
            if (i + INITIAL < PANELS)
            {
                issued += MATRIX_ISSUES;
                if (i + INITIAL == t)
                    ready = issued;
            }
        }
        return issued - ready;
    }

    // Keep each B tile inside one 128-column scale group.
    static_assert(WM > 0 && WM * WN <= 8);
    static_assert(BM > 0 && BM % (WM * 16) == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(BCache == 0 || BCache == 1 || BCache == 2 || BCache == 17);
    static_assert(BK == 128 || BK == 256);
    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
    static_assert(TotalK > 0 && TotalK % (SplitK * BK) == 0);
    static_assert(A_BYTES % (BLOCK_SIZE * 16) == 0 && B_BYTES % (BLOCK_SIZE * 16) == 0);
    static_assert(BN <= 128 && 128 % BN == 0 && SC <= BLOCK_SIZE);
    static_assert(Stages >= 2 && PANELS >= INITIAL);
    static_assert(LDS_BYTES + sizeof(unsigned) <= 160 * 1024);
};
