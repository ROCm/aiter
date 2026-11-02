// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

template<int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false,
    int FixedK = 7168, int WN = 4, int WGPerCU = 1, int ACache = 0, int StoreCache = 2,
    int Output = 0, int BlockK = 256, bool Contiguous = false,
    bool PinA = false, bool PinC = false, int GroupM = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits {
    // Workgroup launch and cache policy.
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = WN * WARP_SIZE;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int WG_PER_CU = WGPerCU;

    static constexpr int CACHECTL_A = ACache;
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
    static constexpr int W_K = 128;

    // Operand vectors, scale-byte copies, and BF16 output vectors.
    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = 16;
    static constexpr int VEC_SCALE_B = 4;
    static constexpr int VEC_SF = sizeof(unsigned);
    static constexpr int VEC_OUTPUT = 8;
    static constexpr int C_PADDING = 8;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);

    // Fixed-K panel and scale counts.
    static constexpr int K = FixedK;
    static constexpr int PANELS = K / B_K;
    static constexpr int SCALE_COLS = K / GROUP_K;

    // MFMA repeats; output 0 uses LDS stripes and output 2 uses packed stores.
    static constexpr int MR = BM / W_M;
    static constexpr int NR = BN / (WN * W_N);
    static constexpr int OUTPUT = Output;
    static constexpr int E_M = MR;
    static constexpr int E_N = NR;

    // Operand storage, pinning, and grouped tile order.
    static constexpr bool CONTIGUOUS = Contiguous;
    static constexpr bool PIN_A = PinA;
    static constexpr bool PIN_C = PinC;
    static constexpr int GROUPED_M = GroupM;

    // Bound tile grouping and pinned fragments to the supported AGPR ranges.
    static_assert(GroupM == 0 || GroupM == 4 || GroupM == 8);
    static_assert(!PinA || (B_K / 128) * MR * 8 <= 128);
    static_assert(!PinC || MR * NR * 4 <= 128);
    static_assert(Output == 0 || Output == 2);

    // A stages and the B fragment prefetch queue.
    static constexpr int STAGES = NB;
    static constexpr int B_FRAGMENTS = (B_K / W_K) * NR;
    // Keep cross-K prefetch within the wave's register budget. Each B fragment
    // costs eight dwords; reserve space for A, C, packed scales and addressing.
    static constexpr int REGISTER_BUDGET = WGPerCU == 2 ? 256 : 512;
    static constexpr int B_REGISTER_ROOM =
        REGISTER_BUDGET - MR * NR * 4 - (B_K / 128) * MR * 8 - 48;
    static constexpr int B_MAX_AHEAD = B_REGISTER_ROOM / 8 - 1 > 0
        ? B_REGISTER_ROOM / 8 - 1 : 1;
    static constexpr bool STREAM_A = false;
    static constexpr bool STATIC_B_RING = false;
    static constexpr bool COMPACT_B_RING = false;
    static constexpr bool ROTATE_B = WGPerCU == 2 || (B_K / 128) * MR >= 16;
    static constexpr int B_SCHEDULE_AHEAD = ROTATE_B ? (NB == 2 ? 1 : 4)
        : (B_K == 128 && NR >= 4 && PinA && FixedK >= 1536 ? 3 : B_MAX_AHEAD);
    static constexpr int B_AHEAD = BD * B_FRAGMENTS < B_SCHEDULE_AHEAD
        ? BD * B_FRAGMENTS : B_SCHEDULE_AHEAD;
    static constexpr int B_SETS = B_AHEAD + 1;
    static constexpr int B_CACHE = BC;
    static constexpr bool XCD_ORDER = XCD;
    static constexpr bool SB_SLOTS = SCALE_COLS % VEC_SCALE_B != 0;

    // The A ring and byte-scale panels share LDS; output 0 reuses it for BF16 stripes.
    static constexpr int A_BYTES = BM * B_K;
    static constexpr int STAGE_BYTES = A_BYTES;
    static constexpr int A_COPIES = A_BYTES / (BLOCK_SIZE * VEC_A);
    static constexpr int SA_ROWS = (BM + VEC_SCALE_A - 1) / VEC_SCALE_A * VEC_SCALE_A;
    static constexpr int SA_COPIES = (SA_ROWS * SCALE_COLS + BLOCK_SIZE * VEC_SCALE_A - 1) / (BLOCK_SIZE * VEC_SCALE_A);
    static constexpr int SA_BYTES = SA_COPIES * BLOCK_SIZE * VEC_SCALE_A;
    static constexpr int SFA_PACKED_M_GROUPS = (MR + VEC_SF - 1) / VEC_SF;
    static constexpr int SFA_M_VECTORS = SFA_PACKED_M_GROUPS * VEC_SF;
    static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_M_VECTORS;
    static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
    static constexpr int SFA_PASSES_PER_CACHE_PANEL = (SCALE_COLS + SFA_K_COLUMNS_PER_PASS - 1) / SFA_K_COLUMNS_PER_PASS;
    static constexpr int SFA_PANEL_PITCH = SFA_PACKED_M_GROUPS * W_M * VEC_SF;
    static constexpr int SB_ROWS = (BN + GROUP_N - 1) / GROUP_N;
    static constexpr int SB_BYTES = BLOCK_SIZE * VEC_SCALE_B;
    static constexpr int OFF_SA = NB * STAGE_BYTES;
    static constexpr int OFF_SB = OFF_SA + SA_BYTES;
    static constexpr int PIPELINE_BYTES = OFF_SB + SB_BYTES;
    static constexpr int LDS_BYTES = PIPELINE_BYTES;

    static constexpr int INITIAL_A = NB - 1;

    // Require legal tile divisibility, scale loads, and per-CU LDS usage.
    static_assert(BM > 0 && BM % 16 == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(NB >= 2 && BD >= 1 && BD < NB);
    static_assert((BC == 0 || BC == 1 || BC == 17) && K > 0 && K % B_K == 0);
    static_assert(ACache == 0 || ACache == 1 || ACache == 17);
    static_assert(BN <= 256 && (WGPerCU == 1 || WGPerCU == 2));
    static_assert(B_K == 128 || B_K == 256);
    static_assert(A_BYTES % (BLOCK_SIZE * 16) == 0 && PANELS >= INITIAL_A);
    static_assert(W_M == VEC_SCALE_A && SFA_PANEL_PITCH * SCALE_COLS <= SA_BYTES);
    static_assert(SB_ROWS * (SB_SLOTS ? SCALE_COLS : SCALE_COLS / 4) <= BLOCK_SIZE);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(MR * NR * 4 <= 256);
};

template<class Base, int Ahead>
struct opus_gemm_bpreshuffle_b_queue_traits : Base {
    static_assert(Ahead >= 1 && Ahead <= Base::B_MAX_AHEAD);
    static_assert(Ahead < Base::PANELS * Base::B_FRAGMENTS);
    static constexpr int B_AHEAD = Ahead;
    static constexpr int B_SETS = Ahead + 1;
};

template<class Base, bool StreamA, bool StaticRing, int Output, bool Compact>
struct opus_gemm_bpreshuffle_b_pipeline_traits : Base {
    static constexpr bool STREAM_A = StreamA;
    static constexpr bool STATIC_B_RING = StaticRing;
    static constexpr bool COMPACT_B_RING = Compact;
    static constexpr int OUTPUT = Output < 0 ? Base::OUTPUT : Output;
    static_assert(OUTPUT == 0 || OUTPUT == 2);
};
