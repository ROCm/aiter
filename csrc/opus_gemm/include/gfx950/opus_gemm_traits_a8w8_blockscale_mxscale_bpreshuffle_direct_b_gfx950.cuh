// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_gfx950.cuh"
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include <type_traits>

namespace opus_gemm_bpreshuffle_direct_b_resources {
__host__ __device__ constexpr int gcd(int a, int b) {
    while (b) {
        int remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}
__host__ __device__ constexpr int ceil(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

__host__ __device__ constexpr int window(int bm, int threads, int matrix, int vec, int groups) {
    int columns = bm <= 32 ? 64 : 32;
    while (columns * groups > threads)
        columns /= 2;
    while (columns > 4) {
        const int sa = ceil(bm * columns, threads * vec) * threads * (vec < 4 ? 4 : vec);
        if (matrix + sa + threads * 4 <= 80 * 1024)
            break;
        columns /= 2;
    }
    return columns;
}

__host__ __device__ constexpr int rows(int bm, int bn, int bytes) {
    for (int r = bm; r >= 16; r -= 16)
        if (bm % r == 0 && r * (bn + 8) * 2 <= bytes)
            return r;
    return 16;
}
} // namespace opus_gemm_bpreshuffle_direct_b_resources

template<int BM, int BN, int BK, int WM, int WN, int Stages, int Ahead,
    int WGPerCU, int ACache, int BCache, int ScaleVec, int Output, int SplitK = 1>
struct opus_gemm_bpreshuffle_direct_b_traits {
    static constexpr int WARP_SIZE = 64;
    static constexpr int NUM_WAVES = WM * WN;
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 128;
    static constexpr int VEC_A = 16;
    static constexpr int VEC_B = 16;
    static constexpr int VEC_C = 4;
    static constexpr int VEC_SCALE_A = ScaleVec;
    static constexpr int VEC_SCALE_B = sizeof(unsigned);
    static constexpr int VEC_SF = 1;
    static constexpr int VEC_OUTPUT = 8;
    static constexpr int C_PADDING = 8;
    static constexpr int GROUP_N = 128;
    static constexpr int GROUP_K = 128;
    static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
    static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);


    static constexpr int B_M = BM;
    static constexpr int B_N = BN;
    static constexpr int B_K = BK;
    static constexpr int T_M = WM;
    static constexpr int T_N = WN;
    static constexpr int BLOCK_SIZE = NUM_WAVES * WARP_SIZE;
    static constexpr int WG_PER_CU = WGPerCU;
    static constexpr int MR = BM / (WM * W_M);
    static constexpr int NR = BN / (WN * W_N);
    static constexpr int KS = BK / W_K;
    static constexpr int SPLIT_K = SplitK;
    static constexpr int CACHECTL_A = ACache;
    static constexpr int B_CACHE = BCache;
    static constexpr int OUTPUT = Output;
    static constexpr int STAGES = Stages;
    static constexpr int B_AHEAD = Ahead;
    static constexpr int B_SETS = Ahead + 1;
    static constexpr int INITIAL_A = Stages - 1;
    static constexpr int INITIAL_B = Ahead;

    static constexpr int A_REPEATS = (BM * BK + BLOCK_SIZE * VEC_A - 1) / (BLOCK_SIZE * VEC_A);
    static constexpr int A_BYTES = A_REPEATS * BLOCK_SIZE * VEC_A;
    static constexpr int A_COPY_THREADS = BLOCK_SIZE;
    static constexpr int A_COPY_N = T_N;
    static constexpr int A_COPY_M = T_M;
    static constexpr int SHORT_A_BYTES = BM * BK;
    static constexpr int STAGE_BYTES = A_BYTES;
    static constexpr int B_ISSUES = NR * B_CHUNKS_PER_FRAGMENT * KS;

    static constexpr int SA_SPACING = ScaleVec == 1 ? 4 : 1;
    static constexpr int SB_ROWS = (BN + GROUP_N - 1) / GROUP_N;
    static constexpr bool UNIFORM_SHORT_QUEUE = BM == 64 && BN == 64 && BK == 128 && WM == 1 && WN == 4 &&
            Stages == 4 && Ahead == 2 && WGPerCU == 1 && ACache == 0 && BCache == 2 && ScaleVec == 4;
    static constexpr int STORE_CACHE = UNIFORM_SHORT_QUEUE ? 0 : 2;
    static constexpr bool RESIDENT_K128_SIX = BM == 64 && BN == 128 && BK == 128 && WM == 1 && WN == 4 &&
        Stages == 6 && Ahead == 1 && WGPerCU == 1 && ScaleVec == 4 &&
        ((ACache == 0 && BCache == 2) || (ACache == 17 && BCache == 1));
    static constexpr int SCALE_COLS =
        (BM == 128 && BN == 256 && BK == 128 && WM == 1 && WN == 4 &&
            Stages == 3 && Ahead == 2 && WGPerCU == 1 && ACache == 1 && BCache == 17 && ScaleVec == 16) ? 64 :
        (BM == 48 && BN == 128 && BK == 256 && WM == 1 && WN == 4 &&
            Stages == 3 && Ahead == 2 && WGPerCU == 2 && ACache == 17 && BCache == 1 && ScaleVec == 4) ? 64 :
        (BM == 96 && BN == 128 && BK == 128 && WM == 1 && WN == 4 &&
            Stages == 4 && Ahead == 2 && WGPerCU == 1 && ScaleVec == 16) ? 64 :
        ((BM == 96 && BN == 128 && BK == 256 && WM == 1 && WN == 4 &&
            Stages == 2 && Ahead == 1 && WGPerCU == 2 && ACache == 0 && BCache == 0 && ScaleVec == 16) ||
         (BM == 64 && BN == 128 && BK == 256 && WM == 1 && WN == 4 &&
            Stages == 3 && Ahead == 2 && WGPerCU == 2 && ACache == 0 && BCache == 0 &&
            (ScaleVec == 4 || ScaleVec == 16))) ? 64 :
        (BM == 64 && BN == 128 && BK == 256 && WM == 1 && WN == 8 &&
            Stages == 3 && Ahead == 1 && WGPerCU == 1 && ScaleVec == 4) ? 64 :
        (UNIFORM_SHORT_QUEUE || RESIDENT_K128_SIX) ? 8 :
        (BM == 32 && BN == 128 && BK == 128 && WM == 1 && WN == 4 &&
            Stages == 3 && Ahead == 1 && WGPerCU == 1 && ACache == 17 && BCache == 1 && ScaleVec == 1) ? 8 :
        opus_gemm_bpreshuffle_direct_b_resources::window(
            BM, BLOCK_SIZE, Stages * A_BYTES, ScaleVec, SB_ROWS);
    static constexpr int SA_PASSES = (BM * SCALE_COLS + BLOCK_SIZE * ScaleVec - 1) /
        (BLOCK_SIZE * ScaleVec);
    static constexpr int SA_BYTES = SA_PASSES * BLOCK_SIZE * (ScaleVec < 4 ? 4 : ScaleVec);
    static constexpr int SB_BYTES = BLOCK_SIZE * VEC_SCALE_B;
    static constexpr int OFF_SA = Stages * A_BYTES;
    static constexpr int OFF_SB = OFF_SA + SA_BYTES;
    static constexpr int LDS_BYTES = OFF_SB + SB_BYTES;
    static constexpr int C_ROWS =
        opus_gemm_bpreshuffle_direct_b_resources::rows(BM, BN, LDS_BYTES);

    static constexpr int RESIDENT_PANELS = [] {
        if (RESIDENT_K128_SIX) return Stages;
        if (UNIFORM_SHORT_QUEUE) return Stages - 1;
        if (BM == 32 && BN == 128 && BK == 128 && WM == 1 && WN == 4 &&
            Stages == 3 && Ahead == 1 && WGPerCU == 1 && ACache == 17 && BCache == 1 && ScaleVec == 1) return Stages;
        if (BM == 64 && BN == 128 && BK == 128 && WM == 1 && WN == 4 &&
            Stages == 3 && Ahead == 1 && ACache == 0 && BCache == 0 &&
            ((WGPerCU == 1 && ScaleVec == 4) || (WGPerCU == 2 && ScaleVec == 16))) return 3;
        int panels = 72 * 1024 / A_BYTES;
        if (panels < Stages) panels = Stages;
        return panels < 8 ? panels : 8;
    }();
    static constexpr int SHORT_COLUMNS = BM == 64 && BN == 128 && BK == 128 && RESIDENT_PANELS == 3 ?
        4 : RESIDENT_PANELS * KS;
    static constexpr int SHORT_SA_PASSES = (BM * SHORT_COLUMNS + BLOCK_SIZE * ScaleVec - 1) /
        (BLOCK_SIZE * ScaleVec);
    static constexpr int SHORT_SA_BYTES = SHORT_SA_PASSES * BLOCK_SIZE * (ScaleVec < 4 ? 4 : ScaleVec);
    static constexpr int SHORT_LDS_BYTES = RESIDENT_PANELS * SHORT_A_BYTES + SHORT_SA_BYTES + SB_BYTES;
    static constexpr int RESIDENT_BYTES = RESIDENT_PANELS * A_BYTES;

    __host__ __device__ static constexpr bool short_k(int k) {
        return k / (SplitK * BK) <= RESIDENT_PANELS;
    }
    __host__ __device__ static constexpr int lds_bytes(int k) {
        if constexpr (UNIFORM_SHORT_QUEUE) return 0;
        if (!short_k(k)) return LDS_BYTES;
        if constexpr (UNIFORM_SHORT_QUEUE) return SHORT_LDS_BYTES;
        const int arena = k / (SplitK * BK) * SHORT_A_BYTES + SHORT_SA_BYTES + SB_BYTES;
        const int c_arena = BM * (BN + C_PADDING) * sizeof(opus::bf16_t);
        return arena > c_arena ? arena : c_arena;
    }
    __host__ __device__ static constexpr int off_sa(int k) {
        return short_k(k) ? (UNIFORM_SHORT_QUEUE ? RESIDENT_PANELS * SHORT_A_BYTES :
            k / (SplitK * BK) * SHORT_A_BYTES) : OFF_SA;
    }
    __host__ __device__ static constexpr int off_sb(int k) {
        return short_k(k) ? off_sa(k) + SHORT_SA_BYTES : OFF_SB;
    }


    static_assert(BM > 0 && BM % (WM * 16) == 0 && BN > 0 && BN % (WN * 16) == 0);
    static_assert((WM == 1 || WM == 2) && (WN == 1 || WN == 2 || WN == 4 || WN == 8));
    static_assert(WM * WN <= 8 && BN <= 256 && (BN >= 128 ? BN % 128 == 0 : 128 % BN == 0));
    static_assert(BK == 128 || BK == 256 || BK == 512);
    static_assert(Stages >= 2 && Stages <= 8 && Ahead >= 1 && Ahead < Stages);
    static_assert(WGPerCU == 1 || WGPerCU == 2);
    static_assert(ACache == 0 || ACache == 1 || ACache == 17);
    static_assert(BCache == 0 || BCache == 1 || BCache == 2 || BCache == 17);
    static_assert(ScaleVec == 1 || ScaleVec == 4 || ScaleVec == 16);
    static_assert(Output == 0);
    static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4);
    static_assert(SCALE_COLS % KS == 0 && SB_ROWS * SCALE_COLS <= BLOCK_SIZE);
    static_assert(RESIDENT_PANELS >= Stages - 1 && SB_ROWS * SHORT_COLUMNS <= BLOCK_SIZE);
    static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(((SHORT_LDS_BYTES + 1279) / 1280 * 1280) * WGPerCU <= 160 * 1024);
    static_assert(4 * MR * NR + 8 * KS * (MR + B_SETS * NR) + 48 <=
        (WGPerCU == 2 ? 256 : 512));
    static_assert(C_ROWS * (BN + 8) * 2 <= LDS_BYTES);
};

template<class Base>
struct opus_gemm_bpreshuffle_direct_b_short_traits : Base {
    static constexpr int A_COPY_THREADS =
        opus_gemm_bpreshuffle_direct_b_resources::gcd(
            Base::B_M * Base::B_K / Base::VEC_A, Base::BLOCK_SIZE);
    static constexpr int A_COPY_N = A_COPY_THREADS / Base::WARP_SIZE < Base::T_N ?
        A_COPY_THREADS / Base::WARP_SIZE : Base::T_N;
    static constexpr int A_COPY_M = A_COPY_THREADS / Base::WARP_SIZE / A_COPY_N;
    static constexpr int A_BYTES = Base::SHORT_A_BYTES;
    static constexpr int STAGE_BYTES = A_BYTES;
    static constexpr int A_REPEATS = A_BYTES / (A_COPY_THREADS * Base::VEC_A);
    static constexpr int RESIDENT_BYTES = Base::RESIDENT_PANELS * A_BYTES;
    static constexpr int STAGES = Base::RESIDENT_PANELS + 1;
    static constexpr int INITIAL_A = Base::RESIDENT_PANELS;
    static constexpr int SCALE_COLS = Base::SHORT_COLUMNS;
    static constexpr int SA_PASSES = Base::SHORT_SA_PASSES;
    static constexpr int OFF_SA = RESIDENT_BYTES;
    static constexpr int OFF_SB = OFF_SA + Base::SHORT_SA_BYTES;
    static constexpr int LDS_BYTES = Base::SHORT_LDS_BYTES;
    static constexpr int C_ROWS =
        opus_gemm_bpreshuffle_direct_b_resources::rows(Base::B_M, Base::B_N, LDS_BYTES);
    __host__ __device__ static constexpr int off_sa(int k) { return Base::off_sa(k); }
    __host__ __device__ static constexpr int off_sb(int k) { return Base::off_sb(k); }
};
