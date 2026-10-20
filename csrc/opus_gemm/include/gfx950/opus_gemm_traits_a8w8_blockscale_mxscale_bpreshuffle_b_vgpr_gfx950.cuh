// SPDX-License-Identifier: Apache-2.0
#pragma once

// Tile geometry, operand-ring sizes, cache policies and resource limits.
template<int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false,
    int FixedK = 7168, int WN = 4, int WGPerCU = 1, int ACache = 0, int StoreCache = 2,
    int Output = 0, int BlockK = 256, bool Contiguous = false,
    bool PinA = false, bool PinC = false, int GroupM = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits {
    static constexpr int BLOCK_SIZE = WN * 64;
    static constexpr int WG_PER_CU = WGPerCU;

    static constexpr int CACHECTL_A = ACache;
    static constexpr int STORE_CACHE = StoreCache;

    static constexpr int B_M = BM;
    static constexpr int B_N = BN;
    static constexpr int B_K = BlockK;

    static constexpr int T_M = 1;
    static constexpr int T_N = WN;
    static constexpr int T_K = 1;

    static constexpr int K = FixedK;
    static constexpr int PANELS = K / B_K;
    static constexpr int SCALE_COLS = K / 128;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;

    static constexpr int MR = BM / 16;
    static constexpr int NR = BN / (WN * 16);
    static constexpr int OUTPUT = Output;
    static constexpr int E_M = MR;
    static constexpr int E_N = NR;

    static constexpr bool CONTIGUOUS = Contiguous;
    static constexpr bool PIN_A = PinA;
    static constexpr bool PIN_C = PinC;
    static constexpr int GROUPED_M = GroupM;

    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8);
    static_assert(!PinA || (B_K / 128) * MR * 8 <= 128);
    static_assert(!PinC || MR * NR * 4 <= 128);
    static_assert(Output == 0 || Output == 2);

    static constexpr int STAGES = NB;
    static constexpr int B_FRAGMENTS = (B_K / 128) * NR;
    // Keep cross-K prefetch within the wave's register budget. Each B fragment
    // costs eight dwords; reserve space for A, C, packed scales and addressing.
    static constexpr int REGISTER_BUDGET = WGPerCU == 2 ? 256 : 512;
    static constexpr int B_REGISTER_ROOM =
        REGISTER_BUDGET - MR * NR * 4 - (B_K / 128) * MR * 8 - 48;
    static constexpr int B_MAX_AHEAD = B_REGISTER_ROOM / 8 - 1 > 0
        ? B_REGISTER_ROOM / 8 - 1 : 1;
    static constexpr bool ROTATE_B = WGPerCU == 2 || (B_K / 128) * MR >= 16;
    static constexpr int B_SCHEDULE_AHEAD = ROTATE_B ? (NB == 2 ? 1 : 4)
        : (B_K == 128 && NR >= 4 && PinA && FixedK >= 1536 ? 3 : B_MAX_AHEAD);
    static constexpr int B_AHEAD = BD * B_FRAGMENTS < B_SCHEDULE_AHEAD
        ? BD * B_FRAGMENTS : B_SCHEDULE_AHEAD;
    static constexpr int B_SETS = B_AHEAD + 1;
    static constexpr int B_CACHE = BC;
    static constexpr bool XCD_ORDER = XCD;
    static constexpr bool SB_SLOTS = SCALE_COLS % 4 != 0;

    // The A ring and resident byte-scale panels share LDS with the output tile.
    static constexpr int A_BYTES = BM * B_K;
    static constexpr int STAGE_BYTES = A_BYTES;
    static constexpr int A_COPIES = A_BYTES / (BLOCK_SIZE * 16);
    static constexpr int SA_ROWS = (BM + 15) / 16 * 16;
    static constexpr int SA_COPIES = (SA_ROWS * SCALE_COLS + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
    static constexpr int SA_BYTES = SA_COPIES * BLOCK_SIZE * 16;
    static constexpr int SB_ROWS = (BN + 127) / 128;
    static constexpr int SB_BYTES = BLOCK_SIZE * 4;
    static constexpr int OFF_SA = NB * STAGE_BYTES;
    static constexpr int OFF_SB = OFF_SA + SA_BYTES;
    static constexpr int PIPELINE_BYTES = OFF_SB + SB_BYTES;
    static constexpr int LDS_BYTES = PIPELINE_BYTES;

    static constexpr int INITIAL_A = NB - 1;

    static_assert(BM > 0 && BM % 16 == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(NB >= 2 && BD >= 1 && BD < NB);
    static_assert((BC == 0 || BC == 1 || BC == 17) && K > 0 && K % B_K == 0);
    static_assert(ACache == 0 || ACache == 1 || ACache == 17);
    static_assert(BN <= 256 && (WGPerCU == 1 || WGPerCU == 2));
    static_assert(B_K == 128 || B_K == 256);
    static_assert(A_BYTES % (BLOCK_SIZE * 16) == 0 && PANELS >= INITIAL_A);
    static_assert(SB_ROWS * (SB_SLOTS ? SCALE_COLS : SCALE_COLS / 4) <= BLOCK_SIZE);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(MR * NR * 4 <= 256);
};
