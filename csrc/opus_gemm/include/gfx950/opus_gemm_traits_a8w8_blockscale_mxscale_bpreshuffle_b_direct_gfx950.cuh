// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

template<class Base>
struct opus_gemm_bpreshuffle_bounded_thread_traits : Base {
    static constexpr bool THREAD_RANGE_KNOWN = true;
    static_assert(Base::BLOCK_SIZE > 0 && Base::BLOCK_SIZE % 64 == 0);
};

// Use the minimum number of valid rows in a partial M tile.
template<class Base>
struct opus_gemm_bpreshuffle_simple_tail_traits : Base {
    static constexpr bool SIMPLE_TAIL_ROWS = true;
    static_assert(Base::BLOCK_SIZE > 0 && Base::BLOCK_SIZE % 64 == 0);
};

constexpr int opus_gemm_blockscale_mxscale_bpreshuffle_output_rows(int bm, int bn,
                                                                   int pipeline_bytes) {
    for (int rows = bm; rows >= 16; rows -= 16)
        if (bm % rows == 0 && rows * (bn + 8) * 2 <= pipeline_bytes)
            return rows;
    return 16;
}

template<int BM,
         int BN,
         int NB,
         int BD,
         int BC = 0,
         bool XCD = false,
         int FixedK = 7168,
         int WN = 4,
         int WGPerCU = 1,
         int ACache = 0,
         int StoreCache = 2,
         int Output = 0,
         int BlockK = 256,
         bool Contiguous = false,
         bool PinA = false,
         bool PinC = false,
         bool Grid2D = false,
         int SplitK = 1,
         bool ByteScale = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_traits {
    // Launch and cache policy.
    static constexpr int BLOCK_SIZE = WN * 64;
    static constexpr int WG_PER_CU = WGPerCU;
    static constexpr int CACHECTL_A = ACache;
    static constexpr int B_CACHE = BC;
    static constexpr int STORE_CACHE = StoreCache;

    // Matrix and wave tiles.
    static constexpr int B_M = BM;
    static constexpr int B_N = BN;
    static constexpr int B_K = BlockK;
    static constexpr int T_M = 1;
    static constexpr int T_N = WN;
    static constexpr int T_K = 1;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;

    // MFMA repeats.
    static constexpr int MR = BM / 16;
    static constexpr int NR = BN / (WN * 16);
    static constexpr int E_M = MR;
    static constexpr int E_N = NR;

    // K partition.
    static constexpr int TOTAL_K = FixedK;
    static constexpr int SPLIT_K = SplitK;
    static constexpr int K = FixedK / SplitK;
    static constexpr int PANELS = K / B_K;
    static constexpr int SCALE_COLS = K / 128;

    // A stages and the direct-B register ring.
    static constexpr int STAGES = NB;
    static constexpr int B_AHEAD = BD;
    static constexpr int B_SETS = BD + 1;

    // Addressing, output, and register pinning.
    static constexpr int OUTPUT = Output;
    static constexpr bool CONTIGUOUS = Contiguous;
    static constexpr bool PIN_A = PinA;
    static constexpr bool PIN_C = PinC;
    static constexpr bool GRID_2D = Grid2D;
    static constexpr bool XCD_ORDER = XCD;
    static constexpr bool SB_SLOTS = SCALE_COLS % 4 != 0;
    static constexpr bool C_AGPR = false;
    static constexpr bool PERSTAGE = false;
    static constexpr bool BYTE_SCALE = ByteScale;

    // A and resident scales share LDS with the output tile.
    static constexpr int A_BYTES = BM * B_K;
    static constexpr int STAGE_BYTES = A_BYTES;
    static constexpr int A_COPIES = A_BYTES / (BLOCK_SIZE * 16);
    static constexpr int SA_ROWS = (BM + 15) / 16 * 16;

    // Byte scale copies occupy one dword slot per lane in LDS.
    static constexpr int SA_SCALE = ByteScale ? 4 : 1;
    static constexpr int SA_COPY_ELEMS = ByteScale ? BLOCK_SIZE : BLOCK_SIZE * 16;
    static constexpr int SA_COPIES = (SA_ROWS * SCALE_COLS + SA_COPY_ELEMS - 1) / SA_COPY_ELEMS;
    static constexpr int SA_BYTES = SA_COPIES * BLOCK_SIZE * (ByteScale ? 4 : 16);
    static constexpr int SB_ROWS = (BN + 127) / 128;
    static constexpr int SB_BYTES = BLOCK_SIZE * 4;

    // LDS offsets and total allocation.
    static constexpr int OFF_SA = NB * STAGE_BYTES;
    static constexpr int OFF_SB = OFF_SA + SA_BYTES;
    static constexpr int PIPELINE_BYTES = OFF_SB + SB_BYTES;
    static constexpr int C_ROWS =
        opus_gemm_blockscale_mxscale_bpreshuffle_output_rows(BM, BN, PIPELINE_BYTES);
    static constexpr int LDS_BYTES = PIPELINE_BYTES;

    // Async issue counts for the prologue and steady loop.
    static constexpr int INITIAL_A = NB - 1;
    static constexpr int INITIAL_B = BD;
    static constexpr int STEADY = PANELS - INITIAL_A;
    static constexpr int B_ISSUES = NR * 2 * (B_K / 128);
    static constexpr int SCALE_ISSUES = SA_COPIES + 1;

    // Counts follow the actual prologue: B, first A, scales, remaining A.
    static constexpr int wait_a(int t) {
        int issued = INITIAL_B * B_ISSUES;
        int ready = -1;
        issued += A_COPIES;
        if (t == 0)
            ready = issued;
        issued += SCALE_ISSUES;
        // K0 consumes the resident scales as well as A0. The first barrier
        // must publish those async LDS writes before any scale LDS reads.
        if (t == 0)
            ready = issued;
        for (int stage = 1; stage < INITIAL_A; ++stage) {
            issued += A_COPIES;
            if (stage == t)
                ready = issued;
        }
        for (int step = 0; step < t; ++step) {
            if (step + BD < PANELS)
                issued += B_ISSUES;
            if (step + NB - 1 < PANELS) {
                issued += A_COPIES;
                if (step + NB - 1 == t)
                    ready = issued;
            }
        }
        return issued - ready;
    }
    static constexpr int wait_b(int t) {
        int issued = 0, ready = -1;
        for (int set = 0; set < INITIAL_B; ++set) {
            issued += B_ISSUES;
            if (set == t)
                ready = issued;
        }
        issued += INITIAL_A * A_COPIES + SCALE_ISSUES;
        for (int step = 0; step <= t; ++step) {
            if (step + BD < PANELS) {
                issued += B_ISSUES;
                if (step + BD == t)
                    ready = issued;
            }
            if (step + NB - 1 < PANELS)
                issued += A_COPIES;
        }
        return issued - ready;
    }

    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4);
    static_assert(FixedK % (SplitK * B_K) == 0);
    static_assert(SplitK == 1 || Grid2D);
    static_assert(!Grid2D || !XCD);
    // Retain parameter positions for existing configuration tuples.
    static_assert(!PinA && !PinC, "Small-shape direct-B kernels do not support AGPR pinning");
    static_assert(Output == 0 || Output == 2);
    static_assert(BM > 0 && BM % 16 == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(NB >= 2 && BD >= 1 && BD < NB);
    static_assert((BC == 0 || BC == 1 || BC == 2 || BC == 17) && K > 0 && K % B_K == 0);
    static_assert(ACache == 0 || ACache == 1 || ACache == 17);
    static_assert(BN <= 256 && (WGPerCU == 1 || WGPerCU == 2));
    static_assert(B_K == 128 || B_K == 256);
    static_assert(A_BYTES % (BLOCK_SIZE * 16) == 0 && PANELS >= INITIAL_A);
    static_assert(SB_ROWS * (SB_SLOTS ? SCALE_COLS : SCALE_COLS / 4) <= BLOCK_SIZE);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(SplitK > 1 || Output != 0 || C_ROWS * (BN + 8) * 2 <= LDS_BYTES,
                  "Direct-B output stripes must fit the reused LDS allocation");
    static_assert(MR * NR * 4 <= 256);
};
