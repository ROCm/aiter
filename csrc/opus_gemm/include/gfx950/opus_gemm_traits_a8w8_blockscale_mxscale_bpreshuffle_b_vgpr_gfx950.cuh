// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"

template<int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false,
    int ScalePanelCols = 64, int WN = 4, int WGPerCU = 1, int ACache = 0,
    int Output = 0, int BlockK = 256, bool Contiguous = false,
    bool PinA = false, bool PinC = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits {
    template<int Columns>
    using scale_panel_traits = opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits<
        BM, BN, NB, BD, BC, XCD, Columns, WN, WGPerCU, ACache,
        Output, BlockK, Contiguous, PinA, PinC>;
    // Workgroup launch and cache policy.
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = WN * WARP_SIZE;
    static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
    static constexpr int WG_PER_CU = WGPerCU;

    static constexpr int CACHECTL_A = ACache;
    static constexpr int STORE_CACHE = 2;

    // Matrix and wave tiles.
    static constexpr int B_M = BM;
    static constexpr int B_N = BN;
    static constexpr int B_K = BlockK;

    static constexpr int T_M = 1;
    static constexpr int T_N = WN;

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

    // A bounded scale panel is reused while the runtime K loop advances.
    static constexpr int SCALE_COLS = ScalePanelCols;

    // MFMA repeats; output 0 uses LDS stripes and output 2 uses packed stores.
    static constexpr int MR = BM / W_M;
    static constexpr int NR = BN / (WN * W_N);
    static constexpr int OUTPUT = Output;

    // Operand storage and pinning.
    static constexpr bool CONTIGUOUS = Contiguous;
    static constexpr bool PIN_A = PinA;
    static constexpr bool PIN_C = PinC;

    // Bound pinned fragments to the supported AGPR ranges.
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
    // Keep the full VGPR accumulator tile from sharing an oversized B live set.
    static constexpr int B_SCHEDULE_AHEAD = ROTATE_B ? (NB == 2 ? 1 : 4)
        : (B_K == 128 && NR >= 4 && (PinA || (!PinC && MR * NR * 4 >= 128))
            ? 3 : B_MAX_AHEAD);
    static constexpr int B_AHEAD = BD * B_FRAGMENTS < B_SCHEDULE_AHEAD
        ? BD * B_FRAGMENTS : B_SCHEDULE_AHEAD;
    static constexpr int B_CACHE = BC;
    static constexpr bool XCD_ORDER = XCD;
    static constexpr bool SB_SLOTS = ((BN + GROUP_N - 1) / GROUP_N) * SCALE_COLS <= BLOCK_SIZE;

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


    // Require legal tile divisibility, scale loads, and per-CU LDS usage.
    static_assert(BM > 0 && BM % 16 == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert(WN == 1 || WN == 2 || WN == 4 || WN == 8);
    static_assert(NB >= 2 && BD >= 1 && BD < NB);
    static_assert(BC == 0 || BC == 1 || BC == 17);
    static_assert(SCALE_COLS > 0 && SCALE_COLS % (B_K / GROUP_K) == 0);
    static_assert(ACache == 0 || ACache == 1 || ACache == 17);
    static_assert(BN <= 256 && (WGPerCU == 1 || WGPerCU == 2));
    static_assert(B_K == 128 || B_K == 256);
    static_assert(A_BYTES % (BLOCK_SIZE * 16) == 0);
    static_assert(W_M == VEC_SCALE_A && SFA_PANEL_PITCH * SCALE_COLS <= SA_BYTES);
    static_assert(SB_ROWS * (SB_SLOTS ? SCALE_COLS : SCALE_COLS / 4) <= BLOCK_SIZE);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(MR * NR * 4 <= 256);
};

template<int BM, int BN, int NB, int BD, int BC = 0, bool XCD = false,
    int ScalePanelCols = 64, int WN = 4, int WGPerCU = 1, int ACache = 0,
    int Output = 0, int BlockK = 256, bool Contiguous = false,
    bool PinA = false, bool PinC = false, int PublishM = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_overlap_traits
    : opus_gemm_blockscale_mxscale_bpreshuffle_prefetch_b_vgpr_traits<
        BM, BN, NB, BD, BC, XCD, ScalePanelCols, WN, WGPerCU, ACache,
        Output, BlockK, Contiguous, PinA, PinC> {
    // Select the M repeat that issues the next A-stage prefetch.
    static constexpr int PUBLISH_M = PublishM;

    // Publish before the final M repeat without changing the fragment queue.
    static_assert(PublishM >= 0 && PublishM < BM / 16 - 1);
};

template<class Base, int Ahead>
struct opus_gemm_bpreshuffle_b_queue_traits : Base {
    static_assert(Ahead >= 1 && Ahead <= Base::B_MAX_AHEAD);
    static constexpr int B_AHEAD = Ahead;
};

template<class Base, bool StreamA, bool StaticRing, bool Compact>
struct opus_gemm_bpreshuffle_b_pipeline_traits : Base {
    static constexpr bool STREAM_A = StreamA;
    static constexpr bool STATIC_B_RING = StaticRing;
    static constexpr bool COMPACT_B_RING = Compact;
};

// A short runtime loop uses one bounded scale panel and a smaller B queue.
// Pinning, tile geometry, and the pipeline's execution route stay identical.
template<class Base>
struct opus_gemm_bpreshuffle_b_short_traits : opus_gemm_bpreshuffle_b_pipeline_traits<
    typename Base::template scale_panel_traits<(Base::SCALE_COLS < 16 ? Base::SCALE_COLS : 16)>,
    Base::STREAM_A, Base::STATIC_B_RING, Base::COMPACT_B_RING> {
    static constexpr int LIMIT = Base::PIN_C && Base::NR < 4 ? 2 : 3;
    static constexpr int B_AHEAD = Base::B_AHEAD < LIMIT ? Base::B_AHEAD : LIMIT;
};

// Derive the execution schedule from tile, wave, and operand storage layouts.
// Runtime M/N/K and the scale-panel position are handled by the pipeline.
template<class T, int PublishM, bool Short, bool SingleCache = false>
struct opus_gemm_bpreshuffle_b_schedule {
private:
    static constexpr int BM = T::B_M;
    static constexpr int BN = T::B_N;
    static constexpr int BK = T::B_K;
    static constexpr int MR = T::MR;
    static constexpr int NR = T::NR;
    static constexpr int WAVES_N = T::T_N;
    static constexpr int K_SLICES = BK / T::W_K;
    static constexpr bool PIN_A_ONLY = T::PIN_A && !T::PIN_C;
    static constexpr bool PIN_C_ONLY = !T::PIN_A && T::PIN_C;
    static constexpr bool UNPINNED = !T::PIN_A && !T::PIN_C;

public:
    // Compare operand traffic with the work performed by the MFMA tile.
    static constexpr bool MEMORY_HEAVY =
        (BM + BN) * BK >= 64 * K_SLICES * MR * NR * 32;
    static constexpr bool STREAM_A = T::ROTATE_B || T::STREAM_A;

private:
    // Reading A in smaller pieces limits the live fragment set.
    static constexpr bool NORMAL_PIN_C_LEAD_THREE = !Short && K_SLICES == 1
        && MR == 6 && NR == 4 && T::STAGES == 4 && WAVES_N == 4
        && PIN_C_ONLY && T::CONTIGUOUS && PublishM == 0;
    static constexpr bool NORMAL_PINNED_PARTIAL = !Short && K_SLICES == 1
        && T::STAGES == 4 && WAVES_N == 4 && (MR == 6 || MR == 8)
        && (NR == 2 || NR == 4) && (T::PIN_A || T::PIN_C);
    static constexpr bool NORMAL_PIN_A_PARTIAL = !Short && PIN_A_ONLY
        && K_SLICES == 1 && MR == 8 && NR == 2 && T::STAGES == 3
        && WAVES_N == 4 && PublishM == 0;
    static constexpr bool SHORT_PIN_A_PARTIAL = Short && K_SLICES == 1
        && PIN_A_ONLY && (NR >= 4
            || ((MR == 4 || MR == 8) && NR == 2 && T::STAGES == 4));
    static constexpr bool SHORT_PIN_AC_PARTIAL = Short && K_SLICES == 1
        && T::PIN_A && T::PIN_C && MEMORY_HEAVY
        && MR == 6 && NR == 2 && T::STAGES == 4;
    static constexpr bool SHORT_PIN_C_PARTIAL = Short && K_SLICES == 1
        && PIN_C_ONLY && MR == 8 && NR == 4 && T::STAGES == 4;
    static constexpr bool SHORT_A_LEAD_THREE = Short && T::PIN_A
        && K_SLICES == 1 && MR == 6 && T::STAGES == 4
        && ((T::PIN_C && NR == 2) || (!T::PIN_C && NR == 4));

public:
    static constexpr bool PARTIAL_A = NORMAL_PINNED_PARTIAL || NORMAL_PIN_A_PARTIAL
        || SHORT_PIN_A_PARTIAL || SHORT_PIN_AC_PARTIAL || SHORT_PIN_C_PARTIAL;
    static constexpr int A_READ_LEAD = []() constexpr {
        if constexpr (NORMAL_PIN_C_LEAD_THREE) return 3;
        if constexpr (NORMAL_PINNED_PARTIAL || NORMAL_PIN_A_PARTIAL) return 4;
        if constexpr (SHORT_A_LEAD_THREE || PublishM >= 0) return 3;
        return 2;
    }();
    static constexpr bool NARROW_VGPR = !Short && MEMORY_HEAVY && BK == 256
        && (PublishM >= 0
            || (MR == 4 && BN == 64 && T::STAGES == 4 && WAVES_N == 2))
        && !STREAM_A && UNPINNED && T::OUTPUT == 0;
    static constexpr bool THREE_A = NARROW_VGPR && MR == 4 && BN == 64
        && T::STAGES == 4 && (WAVES_N == 2 || WAVES_N == 4);
    static constexpr bool LATE_A = !Short && PublishM == 2 && MEMORY_HEAVY
        && BK == 256 && MR == 4 && BN == 64 && T::STAGES == 4
        && !T::ROTATE_B && UNPINNED && !T::CONTIGUOUS && T::OUTPUT == 0
        && ((WAVES_N == 2 && T::STREAM_A && T::STATIC_B_RING
                && T::COMPACT_B_RING && T::B_AHEAD == 7)
            || (WAVES_N == 4 && !T::STREAM_A
                && !T::STATIC_B_RING && !T::COMPACT_B_RING));

private:
    static constexpr bool ALIGNED_A_QUEUE = NARROW_VGPR && MR == 6 && BN == 64
        && T::STAGES == 3 && WAVES_N == 4;

public:
    static constexpr int PUBLISH_A = []() constexpr {
        if constexpr (!Short && MR == 4 && BN == 128 && BK == 256
            && WAVES_N == 4 && T::STAGES == 4 && PublishM == 2
            && UNPINNED && !T::STREAM_A && !T::ROTATE_B) return 0;
        if constexpr (SingleCache && NARROW_VGPR && MR == 4 && PublishM == 1) return 0;
        if constexpr (ALIGNED_A_QUEUE) return 1;
        if constexpr (LATE_A) return T::STREAM_A ? 0 : 1;
        return PublishM;
    }();
    static constexpr int B_AHEAD = []() constexpr {
        // Four B slots match early A publication and the rotating K256 ring.
        if constexpr (SingleCache && !Short && !T::ROTATE_B && !STREAM_A
            && BM == 64 && BN == 128 && BK == 256 && WAVES_N == 2
            && T::STAGES == 3 && PublishM < 2 && UNPINNED && T::OUTPUT == 0) return 3;
        if constexpr (SingleCache && !Short && T::ROTATE_B && !T::STATIC_B_RING
            && !MEMORY_HEAVY && BM == 64 && BN == 128 && BK == 256
            && WAVES_N == 2 && T::STAGES == 4 && UNPINNED && T::OUTPUT == 0) return 3;
        // Six B slots align the queue with the three A stages.
        if constexpr (ALIGNED_A_QUEUE) return 5;
        if constexpr (NARROW_VGPR) {
            if constexpr (MR == 6 && BN == 128 && T::STAGES == 3
                && (WAVES_N == 4 || WAVES_N == 8)) return 3;
            if constexpr (MR == 4 && BN == 128 && T::STAGES == 4
                && WAVES_N == 4) return 3;
            if constexpr (MR == 4 && BN == 128 && T::STAGES == 3
                && WAVES_N == 4) return 7;
            if constexpr (THREE_A) return WAVES_N == 2 ? 3 : 7;
        }
        if constexpr (!Short && !STREAM_A && UNPINNED
            && BK == 256 && MR == 4 && NR == 4 && PublishM >= 0) return 7;
        if constexpr (SingleCache && LATE_A && T::STREAM_A && WAVES_N == 2) return 5;
        return T::B_AHEAD;
    }();
    static constexpr int B_SLOTS = B_AHEAD + 1;

private:
    // Compute-heavy layouts need less A prefetch; pinned and narrow layouts
    // retain three A slots to match their B queue.
    static constexpr bool ROTATING_COMPUTE_A = !Short && T::ROTATE_B
        && !T::STATIC_B_RING && !MEMORY_HEAVY && BM == 64 && BN == 128
        && BK == 256 && WAVES_N == 2 && T::STAGES == 4 && UNPINNED
        && (PublishM < 0 || T::B_CACHE == 1);
    static constexpr bool STATIC_COMPUTE_A = !Short && !MEMORY_HEAVY
        && T::ROTATE_B && T::STATIC_B_RING && !T::STREAM_A && !T::COMPACT_B_RING
        && BM == 64 && BN == 128 && BK == 256 && WAVES_N == 2
        && T::STAGES == 4 && UNPINNED && T::OUTPUT == 0;
    static constexpr bool PINNED_A_RING = !Short && PIN_A_ONLY && !T::ROTATE_B
        && BK == 128 && MR == 6 && NR == 2 && WAVES_N == 4
        && T::STAGES == 4 && T::OUTPUT == 0 && PublishM < 0;

public:
    static constexpr int A_STAGES = []() constexpr {
        if constexpr (ROTATING_COMPUTE_A || STATIC_COMPUTE_A) return 2;
        if constexpr (PINNED_A_RING || THREE_A || LATE_A) return 3;
        if constexpr (T::ROTATE_B || MEMORY_HEAVY) return T::STAGES;
        return 2;
    }();
    static constexpr int INITIAL_A = A_STAGES - 1;
    static constexpr int SCALE_K_TILES = T::SCALE_COLS / K_SLICES;
    static constexpr bool ROTATE_RING = T::ROTATE_B && !T::STATIC_B_RING;
    static constexpr bool STAGED_A = NARROW_VGPR && (MR == 4 || MR == 6)
        && BN == 128 && NR == 2
        && (T::STAGES == 3 || (MR == 4 && T::STAGES == 4)) && WAVES_N == 4;


    // Compile-time eligibility; the pipeline checks the runtime K capacity.
    static constexpr bool SMALL_SCALE_CACHE = Short && T::SCALE_COLS == 16 && T::PIN_A && !T::PIN_C && T::B_K == 128 && T::MR == 6 && T::NR == 4 && T::STAGES == 4 && WAVES_N == 4 && PublishM == 0;

    static constexpr bool PINNED_SCALE_CACHE = !Short && !SingleCache && T::B_K == 128 && T::STAGES == 4
        && WAVES_N == 4 && (T::B_M == 96 || T::B_M == 128)
        && (T::B_N == 128 || T::B_N == 256) && T::PIN_A && !T::PIN_C;

    static constexpr bool W2_SCALE_CACHE = !Short && !SingleCache && T::B_M == 64 && T::B_N == 128
        && T::B_K == 256 && WAVES_N == 2
        && (T::STAGES == 3 || T::STAGES == 4)
        && !T::PIN_A && !T::PIN_C && T::OUTPUT == 0;

    static constexpr bool UNPINNED_SINGLE_PANEL = !T::PIN_A && !T::PIN_C
        && !T::ROTATE_B && !T::STREAM_A && T::OUTPUT == 0 && T::B_K == 256
        && T::B_M == 64
        && ((T::T_N == 4 && PublishM == 1
                && ((T::B_N == 64 && T::STAGES == 4)
                    || (T::B_N == 128 && T::STAGES == 3)))
            || (T::T_N == 2 && PublishM == 0
                && T::B_N == 128 && T::STAGES == 2)
            || (T::T_N == 2 && PublishM == 0
                && T::B_N == 64 && T::STAGES == 4));

    static constexpr bool OVERLAP_SCALE_CACHE = (T::B_M == 128 && T::B_N == 128 && T::B_K == 128
        && T::T_N == 4 && T::STAGES == 3 && PublishM == 0
        && T::PIN_A && !T::PIN_C)
        || (T::B_M == 64 && T::B_N == 128 && T::B_K == 256
            && T::T_N == 4 && T::STAGES == 4
            && !T::PIN_A && !T::PIN_C && !T::STREAM_A && !T::ROTATE_B)
        || (T::B_M == 64 && T::B_N == 64 && T::B_K == 256
            && T::T_N == 2 && T::STAGES == 4 && PublishM == 2
            && !T::ROTATE_B && T::STREAM_A && T::STATIC_B_RING
            && T::COMPACT_B_RING && T::B_AHEAD == 7
            && !T::PIN_A && !T::PIN_C && T::OUTPUT == 0)
        || (!T::PIN_A && !T::PIN_C && !T::ROTATE_B && !T::STREAM_A
            && T::OUTPUT == 0 && T::B_K == 256 && T::T_N == 4
            && ((T::B_M == 64 && T::B_N == 64 && T::STAGES == 4 && PublishM == 2)
                || (T::B_M == 96 && T::B_N == 64 && T::STAGES == 3 && PublishM >= 0)))
        || UNPINNED_SINGLE_PANEL;


    // B descriptor bounds and output cache policy use the same layout rules.
    static constexpr bool SHORT_COALESCED_OUTPUT = Short && T::PIN_A && !T::PIN_C
        && K_SLICES == 1 && (MR == 4 || (MR == 8 && PublishM < 0)) && NR == 2 && WAVES_N == 4 && T::STAGES == 4;
    static constexpr bool SHORT_SMALL_SCALE_OUTPUT = Short && T::STORE_CACHE == 2
        && K_SLICES == 1 && T::STAGES == 4 && WAVES_N == 4
        && ((PublishM >= 0 && T::SCALE_COLS <= 4 && T::PIN_A && !T::PIN_C && MR == 6 && NR == 4)
            || (!T::PIN_A && T::PIN_C && MR == 8 && NR == 4));
    static constexpr int C_STORE_CACHE = (SHORT_COALESCED_OUTPUT || SHORT_SMALL_SCALE_OUTPUT) ? 17 : T::STORE_CACHE;
    static constexpr bool ZERO_UNUSED_B_PREFETCH = Short && MEMORY_HEAVY && (MR <= 4 || (T::PIN_A && MR == 8 && K_SLICES == 1 && T::STAGES == 4 && PublishM < 0)) && NR <= 2 && !T::PIN_C;
};
