// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: the topk-prefill-avo repo. Regenerate with
//   python3 scripts/export_aiter_op.py --aiter <this aiter checkout>
// and verify an existing tree with the same command plus --check.
//
// This is benchmark_topk.hip.cpp up to its AITER_EXPORT_END marker (the kernels
// and their dispatch) followed by csrc/topk_aiter_entry.inc.hip (the aiter op
// entry). The harness half of that file -- CPU/GPU verification oracles, timing,
// CLI -- is deliberately not here. The source repo indents at 2; what you are
// reading was reformatted to aiter's .clang-format on the way in, so this file
// does not line up line-for-line with the source.
//
// Formatted by: AMD clang-format version 22.0.0git

#pragma once

#include "topk_common.hip.hpp"

constexpr int SAMPLE_S_MIN = 4096;
constexpr int R_TARGET     = 179;
// small_n stages the WHOLE row in LDS, so N sets its LDS footprint directly and
// the boundary is a measured occupancy tradeoff, not a capacity one.
//
// Crossover against the coop pipeline, run_perftest, both entries, small_n
// time over sampled time (324 points at N = 8193..16384):
//   N <= 9000              0.73-0.99 at every M; 0.71-0.75 at M = 32768..65536,
//                          where topk_select reaches it
//   N = 11000              0.80-0.95 at M <= 768, 0.91-1.17 above
//   N = 12001..16384       at M <= 256 small_n wins 53 of 96 points; above 256
//                          it is 0.99 at best (N=12289, M=300..512), up to 1.76;
//                          at M <= 4 it is 0.95-1.30, rising with N and falling
//                          with M (18 N in that range)
// Short rows favour small_n: one LDS pass against three kernels and the
// candidate traffic. Past 9000 a row leaves 3 blocks per CU, which is one round
// only up to M = 768. At M <= 256 every row has its own CU and small_n holds to
// 16384, except at M <= 4, where a single block per row loses to the pipeline's
// G blocks past 12000. The sampled path is bumpy in N there (its sample size
// follows N's divisors). At N=32768 small_n loses at every M.
constexpr int N_LDS_MAX           = 9000;  // small_n at any M
constexpr int N_LDS_MAX_ONE_ROUND = 12000; // ... up to here while M <= 768
constexpr int N_LDS_ONE_ROUND_M   = 768;
constexpr int N_LDS_MAX_SMALL_M   = 16384; // ... and here while 4 < M <= 256
constexpr int N_LDS_SMALL_M_FLOOR = 4;
constexpr int N_LDS_SMALL_M_LIMIT = 256;

// Hardware cap on the dynamic LDS one block may request. Asking for more makes
// the LAUNCH fail, and a failed launch is not slow -- it is instant and wrong:
// forcing small_n at N=65536 (256 KB) reported 4.40 us with garbage output.
constexpr int LDS_BYTES_PER_BLOCK_MAX = 160 * 1024;

constexpr int PHASE_C_CAP_MAX = 8192;
// How full of candidates Phase C's area may be PLANNED to get. The margin clamp
// in derive_shape_params has always used this fraction; naming it lets the
// sample-count search use the same one, which is the whole of the fix for the
// N=524288 fallbacks. Filling the cap to its brim is what leaves a row nowhere
// to go, and a row with nowhere to go costs a flat ~350 us in the exact
// fallback.
constexpr double CAP_SAFE_FILL = 0.85;
// Per-row reservation counters (cand_reserved, cand_bad) sit one 128-byte line
// apart, in uints. Packed, 32 rows share a line and phase_b's atomicAdds from
// different rows serialize on it: ATT at M=4 N=131072 puts that atomic at 26.8%
// of a phase_b block.
#ifndef CTR_STRIDE_OVERRIDE
#define CTR_STRIDE_OVERRIDE 32
#endif
constexpr int CTR_STRIDE = CTR_STRIDE_OVERRIDE;
// phase_b_filter_coop's per-wave staging entries (dynamic LDS, so only that
// kernel pays). A wave drains to global once more than CAP - 256 are staged,
// and each drain waits on a global atomicAdd in the middle of the stream: a
// +inf threshold (nothing passes) prices candidate handling at 7.5 us of
// phase_b at M=512 N=131072 (coop_g=2, ~181 candidates per wave), of which the
// epilogue copy is 0.7 us.
// 576 x 8 B x 8 waves = 36.9 KB keeps 4 512-thread blocks per CU inside 160 KB.
#ifndef WSTAGE_CAP_COOP_OVERRIDE
#define WSTAGE_CAP_COOP_OVERRIDE 576
#endif
constexpr int WSTAGE_CAP_COOP = WSTAGE_CAP_COOP_OVERRIDE;
// phase_b's block size, a compile-time constant inside the kernel so its
// prologue reads neither blockDim nor the hidden workgroup-size argument. The
// coop_g table was fitted at this size.
constexpr int PB_BLOCK = 512;

// The K the GEOMETRY has to serve on a ragged launch, which is not the caller's
// K. A ragged row ranks min(K, row_len) <= min(K, N) elements and pads the rest
// of its output with -1, so asking derive_shape_params for a K above the pitch
// requests a candidate capacity that cannot exist and gets the shape refused.
// aiter's own top_k_per_row_prefill has no k <= stride0 guard and emits the
// identity for such rows (topk_per_row_kernels.cu:398, :2241), so refusing here
// would be this op declining a shape the one it replaces accepts.
//
// One definition, used by both the harness dispatcher and the aiter entry: a
// second copy is a way for the two to disagree about which shapes are servable.
__host__ inline int geometry_k_ragged(int K, int N) { return K < N ? K : N; }

// gfx950 occupancy inputs for the small_n launch geometry. SMALL_N_STATIC_LDS is
// the kernel's static __shared__ footprint, read off .group_segment_fixed_size
// with the dynamic row buffer excluded.
constexpr int LDS_BYTES_PER_CU    = 160 * 1024;
constexpr int CU_COUNT            = 256;
constexpr int TARGET_WAVES_PER_CU = 32;
// phase_b blocks resident across the GPU at once: its staging LDS and its waves
// each allow four per CU.
constexpr int PB_RESIDENT_BLOCKS =
    CU_COUNT * std::min(LDS_BYTES_PER_CU / ((PB_BLOCK / WAVE_SIZE) * WSTAGE_CAP_COOP * 8),
                        TARGET_WAVES_PER_CU / (PB_BLOCK / WAVE_SIZE));

// Static __shared__ footprints, read off .group_segment_fixed_size with the
// dynamic buffer excluded. Used only to estimate LDS-limited residency.
constexpr int SMALL_N_STATIC_LDS = 5280;
constexpr int PHASE_A_STATIC_LDS = 5144;
constexpr int PHASE_C_STATIC_LDS = 5408;

static inline int grid_blocks_per_cu(int M) { return std::max(1, (M + CU_COUNT - 1) / CU_COUNT); }

// Whether a select kernel can take block_select_lds_wide's buffers (a 12-bit
// digit on the filtered passes, one pass fewer) without losing residency: the
// blocks this grid actually stacks on a CU must still fit with them. Standalone
// A/B, k=2048 gaussian seed 0, with the buffers in STATIC LDS so that every
// launch paid for them, three-kernel device total:
//   m=128  n=131072  -2.98us   m=256 n=262144  -1.64   m=64 n=1048577 -1.24
//   m=1024 n=262144 +11.56us   m=4096 n=131072 +30.56  m=4096 n=1048576 +30.38
// The losses are residency, and this is the condition that avoids them.
static inline bool wide_select_fits(int M, int lds_bytes, int wide_bytes)
{
    const int resident = std::min(grid_blocks_per_cu(M), std::max(1, LDS_BYTES_PER_CU / lds_bytes));
    return resident * (lds_bytes + wide_bytes) <= LDS_BYTES_PER_CU;
}

constexpr int LDS_ALLOC_GRANULE = 1280;
static inline int lds_allocated_bytes(int bytes)
{ return ((bytes + LDS_ALLOC_GRANULE - 1) / LDS_ALLOC_GRANULE) * LDS_ALLOC_GRANULE; }
static inline bool wide_select_fits_allocated(int M, int lds_bytes, int wide_bytes)
{
    const int base_alloc = lds_allocated_bytes(lds_bytes);
    const int wide_alloc = lds_allocated_bytes(lds_bytes + wide_bytes);
    const int resident =
        std::min(grid_blocks_per_cu(M), std::max(1, LDS_BYTES_PER_CU / base_alloc));
    return resident * wide_alloc <= LDS_BYTES_PER_CU;
}

static inline int ilog2_floor(int v)
{
    int r = 0;
    while(v > 1)
    {
        v >>= 1;
        r++;
    }
    return r;
}

// Block size for the one-block-per-row select kernels, from measured occupancy
// behaviour rather than a fixed constant.
//
// The cost of these kernels is dominated by a fixed 3-4 pass x ~4 barrier radix
// select, not by their loads, so what matters is how many WAVES are resident per
// CU. Blocks per CU is capped by the kernel's LDS (which scales with S or cap),
// so at large N only a bigger block can supply enough waves, while at small N
// the cap is loose and a smaller block has cheaper barriers.
//
// Measured at phase_a/phase_c (warmup 15, iters 60, repeats 3), 1024 vs 512:
//   N=1048576 (2 blocks/CU): 1024 wins at every M from 64 to 1024 (-1.1..-7.8%)
//   N=131072  (4 blocks/CU): 1024 wins to M=256 (-3.4..-9.4%), ties at M=512,
//                            loses at M>=1024 (+7.5% at M=1024, +8.7% at M=4096)
// A pure-M rule cannot express that; residency can, and reproduces every point.
//
// load_cap_waves > 0 additionally bounds the waves by what the row's loads can
// occupy; pass 0 where the kernel is not load-shaped. It is ignored while the
// grid cannot even cover the CUs, since there the extra waves are free latency
// hiding rather than added barrier cost.
// Measured best wave count for phase_small_n_topk over the customer pow2 grid
// (13 M values x 3 N values x 5 block sizes, warmup 60 / iters 300 / repeats 5,
// 2026-09-17, MI355X). Rows are log2(M) for M = 1..4096, columns log2(N)-11 for
// N = 2048, 4096, 8192.
//
// The occupancy formula below is within 1% on 38 of these 39 points, but it is
// wrong by a reproducible +12.1% at M=2048 N=4096 (256 threads: 41.7 us,
// 512 threads: 37.3 us, three runs each). The structure is not something the
// formula can express: at M=2048 the best block is 256 for N=2048 but 512 for
// N=4096 and N=8192, while at M=4096 it is 256 for N=4096 and 512 for N=8192.
// Raising TARGET_WAVES_PER_CU to cover M=2048 N=4096 costs +29% at
// M=4096 N=8192, so no single constant satisfies both.
//
// Note the entries of 12 waves: the measured optimum is not always a power of
// two (M=512 N=8192 wants 768 threads, 21.2 us, against 23.0 us at 512), which
// the pow2-rounding formula cannot produce at all.
constexpr int SMALLN_TAB_M       = 13;
constexpr int SMALLN_TAB_N       = 3;
constexpr int SMALLN_N_LOG2_BASE = 11;

static const signed char kSmallNWaves[SMALLN_TAB_M][SMALLN_TAB_N] = {
    /* M=1    */ {16, 16, 16},
    /* M=2    */ {16, 16, 16},
    /* M=4    */ {16, 16, 16},
    /* M=8    */ {16, 16, 16},
    /* M=16   */ {16, 16, 16},
    /* M=32   */ {16, 16, 16},
    /* M=64   */ {16, 16, 16},
    /* M=128  */ {16, 16, 16},
    /* M=256  */ {16, 16, 16},
    /* M=512  */ {8, 12, 12},
    /* M=1024 */ {6, 8, 8},
    /* M=2048 */ {4, 8, 8},
    /* M=4096 */ {4, 4, 8},
};

// Returns threads, or -1 when the shape is off the table. Shapes between grid
// points round DOWN on both axes. Every entry is >= 4 waves because
// block_find_pivot_bucket indexes the 256 radix buckets by threadIdx.x and
// silently drops the upper ones below 256 threads.
static inline int small_n_threads_from_table(int M, int N)
{
    if(M < 1 || N < (1 << SMALLN_N_LOG2_BASE))
        return -1;
    const int mi = ilog2_floor(M);
    const int ni = ilog2_floor(N) - SMALLN_N_LOG2_BASE;
    if(mi >= SMALLN_TAB_M || ni < 0 || ni >= SMALLN_TAB_N)
        return -1;
    return (int)kSmallNWaves[mi][ni] * WAVE_SIZE;
}

static inline int occupancy_block_threads(int M, int lds_per_block, int load_cap_waves)
{
    const int lds_blocks = std::max(1, LDS_BYTES_PER_CU / std::max(1, lds_per_block));
    const int g          = grid_blocks_per_cu(M);
    const int bound      = std::min(lds_blocks, g);
    // Ceiling division, then round UP to a power of two. Truncating twice is what
    // made this formula undershoot: at bound=7 the integer divide turns 4.57 into
    // 4, which is already a power of two, so a later round-up cannot recover it
    // (measured +2.2% at M=2048 N=49152).
    int waves = std::max(1, (TARGET_WAVES_PER_CU + bound - 1) / bound);
    if(load_cap_waves > 0 && g > 1)
        waves = std::min(waves, load_cap_waves);
    // Round the wave target UP to a power of two, not down.
    //
    // Rounding down loses up to 40% of the target whenever the quotient is not
    // itself a power of two, and phase_a hits exactly those cases as soon as S
    // leaves {4096, 8192, 16384} -- which only non-pow2 N do. Measured best block
    // against what rounding down picks, at M=1024..4096:
    //   lds_blocks=7 (S=4096)          down 256, best 512    -0.0% .. -2.0%
    //   lds_blocks=6 (S=4608..5504)    down 256, best 512    -0.1% .. -1.1%
    //   lds_blocks=5 (S=5568..6848)    down 256, best 512    +0.1% .. -3.0%
    //   lds_blocks=4 (S=6912..8896)    down 512, best 512     agree
    //   lds_blocks=3 (S=8960..12352)   down 512, best 1024   -0.3% .. -3.0%
    //   lds_blocks=2 (S=12416+)        down 1024, best 1024   agree
    // Rounding up reproduces the measured optimum in all six classes, which a
    // per-class table would also do -- but the formula then keeps working for the
    // S values nobody measured, and a table would not.
    int pow2 = 1;
    while(pow2 < waves)
        pow2 *= 2;
    return std::max(4, std::min(16, pow2)) * WAVE_SIZE;
}

// Over-collection factor. The candidate count is the number of row elements
// above the rank-R value of S samples, so its spread is that estimator's noise,
// std ~ count/sqrt(R), and a row undershoots K (paying the exact fallback) when
// count < K.
//
// Uses the margin-FREE rank R0 = K*S/N, which over-states the estimator's noise
// because the estimator really runs at R = margin*K*S/N. That looks like a bug
// and it is NOT: the conservatism is load-bearing.
//
// Solving the self-consistent fixed point instead
// (x^2 - (3/sqrt(c))x - 1 = 0 with c = K*S/N, margin = x^2) gives a smaller,
// statistically "correct" margin -- and it FAILS the gate. Measured with it:
// M=4096 N=1048576 margin 2.129 -> 1.839 produced under_K=1, a row short of K,
// where the form below gives under_K=0.
//
// The reason the 3-sigma constant is not enough: the gate is "no row of M
// undershoots", i.e. a maximum over M draws, so the sigma that matters grows
// with M. Back-solved from the measured spread at M=4096 the deepest row sits
// 3.4-3.5 sigma below the mean, not 3.0. The margin-free R0 happens to absorb
// that M-dependence; a formula that removes the slack has to put the
// M-dependence back explicitly, and nothing here does.
//
// A statistically tighter margin was tried and fails the correctness gate.
static float auto_margin(int K, int S, int N)
{
    const double r0 = (double)K * S / (double)N;
    if(r0 < 9.0)
        return 4.0f; // too few samples for the 3-sigma rule to mean anything
    const double m = 1.0 / (1.0 - 3.0 / std::sqrt(r0));
    return (float)std::min(std::max(m, 1.4), 3.0);
}

// Upper edge of the same 3-sigma window, in candidates.
static inline double candidate_hi(int K, int S, int N, double margin)
{
    const double R = std::max(1.0, margin * (double)K * (double)S / (double)N);
    return margin * (double)K * (1.0 + 3.0 / std::sqrt(R));
}

enum TopkPath : int
{
    PATH_AUTO    = 0,
    PATH_SMALL_N = 1,
    PATH_PREFILL = 2,
    PATH_DECODE  = 3,
};

struct ShapeParams
{
    TopkPath path;
    int S;
    float margin;
    int rank;
    int cap;
    int coop_g;
    bool keys_only_c;
    bool geom_ok;
};

static inline int align_sample_s(int s)
{
    s = std::max(SAMPLE_S_MIN, std::min(SAMPLE_S_MAX, s));
    return ((s + SAMPLE_CHUNK_ELEMS - 1) / SAMPLE_CHUNK_ELEMS) * SAMPLE_CHUNK_ELEMS;
}

static inline bool sampling_geometry_ok(int N, int S)
{
    if(S > SAMPLE_S_MAX || S % SAMPLE_CHUNK_ELEMS != 0)
        return false;
    // No N % FP32_EPT check. It used to be here because the loads are dwordx4 and
    // the row base is `input + row * pitch`, which an odd pitch misaligns -- but
    // gfx950 serves a 4-byte-aligned dwordx4 natively (v5 Stage 2 isolated the
    // HIP 700 to the tail over-read, not the misalignment, and load_row_f4 clamps
    // that tail). sample_chunk_stride still masks the spacing to a multiple of 4,
    // so chunk starts stay 4-aligned RELATIVE to the base whatever the base is.
    const int chunks = S / SAMPLE_CHUNK_ELEMS;
    return sample_chunk_stride(N, chunks) >= SAMPLE_CHUNK_ELEMS;
}

// Whether the chunk spacing divides N exactly, i.e. whether sample_chunk_stride()
// had to mask anything off.
//
// This is a preference, not a servability test -- the mask is always safe. It is
// kept separate because the two questions were once the same function, and
// answering "should we pick this S" with the relaxed rule changed which S the
// pow2 grid picks: shapes that used to be bumped up to SAMPLE_S_MAX by the
// repair below kept the smaller S the law asks for instead, and the decode
// geomean went 30.89 -> 31.15 us (+0.82%, reproduced on 3 consecutive outer-tier
// runs, per-point sd 0.04-0.08%). The S the old rule forced was simply the
// better one, so an exact stride still decides the choice and the mask only
// widens what can be served at all.
static inline bool sample_stride_exact(int N, int S)
{
    if(S > SAMPLE_S_MAX || S % SAMPLE_CHUNK_ELEMS != 0)
        return false;
    const int chunks = S / SAMPLE_CHUNK_ELEMS;
    const int stride = N / chunks;
    return stride >= SAMPLE_CHUNK_ELEMS && stride % FP32_EPT == 0;
}

// How S is chosen: 0 = constant R_TARGET, 1 = derive S from the acceptance
// window; which one is per region.
//
// Rule 1 is FALSIFIED as a GLOBAL replacement and always was: it regresses
// M=64 N=262144 and M=256 N=262144 hard. But it is right in a region, and the
// region is larger than the v3-era note claimed ("wins 5-7.5% at M <= 8").
// Re-measured on g_22, rule 1 against rule 0, warmup 20 / iters 100 / repeats 9:
//
//   M       N=131072   N=262144   N=524288
//   1         -5.8%      -6.9%      -6.1%
//   2         -3.0%      -7.7%      -6.2%
//   4         -1.2%      -7.2%      -5.9%
//   8         -1.6%      -7.7%      -7.7%
//   16        -1.5%     -10.8%      -6.8%
//   32        -2.8%      -9.5%      -4.0%
//   64        -1.7%     +17.2%      -3.3%
//   128       +0.6%     +19.6%      +5.6%
//   256       +1.3%     +23.9%      +4.0%
//   1024      +3.6%      +5.3%      +1.9%
//   4096      +1.6%      +3.4%      +0.9%
//
// So the boundary is M, and M=64 is where it turns: it wins at two of the three
// N and loses 17.2% at the third, so it stays on rule 0. M <= 32 takes rule 1.
//
// Why there is a boundary at all: a smaller S forces a larger margin and a
// larger cap, trading phase_a work for candidate volume. At small M phase_a's
// single-block cost dominates and the trade pays; at large M the extra
// candidates Phase B writes and Phase C selects cost more than phase_a saves.
//
// The v3-era note also reported the anchor at +3.9% under rule 1; it measures
// +1.7% on g_22. Either way the anchor keeps rule 0.
//
// v8 re-measured the small-M region on the current kernels (standalone, inputs
// rotated, rule 1 time / rule 0 time, M = 1 / 8 / 32, plain and ragged alike):
// equal (0.97-1.01) up to N = 327683, rule 1 better at N = 196611 (0.90-0.95),
// mixed at N = 393219 (0.98 / 1.03 / 1.03), and rule 0 better from N = 458755
// up (1.03-1.10): rule 1's margin of 2.46 there costs phase_c more candidates
// than rule 0's S = 16384 costs phase_a. Rule 1 keeps N below 393216.
constexpr int S_RULE1_M_MAX = 32;
constexpr int S_RULE1_N_MAX = 393215;

static inline int effective_s_rule(int M, int N)
{ return M <= S_RULE1_M_MAX && N <= S_RULE1_N_MAX ? 1 : 0; }

// Smallest sample count whose 3-sigma candidate window still fits under the
// largest cap Phase C can hold.
//
// Replaces `S = R_TARGET * N / (margin * K)` with R_TARGET = 179, a constant
// reverse-engineered from ONE shape (M=4096 N=131072, where cap/K = 2). It does
// not transfer: at N=1048576 the window is cap/K = 4, so it demanded 43000
// samples, got clamped to SAMPLE_S_MAX, and left phase_a doing twice the work it
// needed. Deriving the requirement instead makes the constant unnecessary.
//
// Only powers of two are reachable: the sampling geometry needs
// (N / (S/64)) % 4 == 0, so for a power-of-two N the chunk count must also be a
// power of two.
static inline int derive_sample_s_for_n(int M, int N, int K, float margin_unused)
{
    (void)margin_unused;
    if(N < SAMPLE_S_MIN)
    {
        const int chunks = std::max(1, N / SAMPLE_CHUNK_ELEMS);
        return align_sample_s(chunks * SAMPLE_CHUNK_ELEMS);
    }
    if(effective_s_rule(M, N) == 0)
    {
        const float m  = auto_margin(K, SAMPLE_S_MAX, N);
        const double s = R_TARGET * (double)N / ((double)m * (double)K);
        return align_sample_s((int)std::lround(s));
    }
    // Steps by SAMPLE_CHUNK_ELEMS, not by doubling. Doubling only ever considers
    // powers of two, and a non-pow2 N has no exact stride at those, so the loop
    // walked all the way to SAMPLE_S_MAX and rule 1 ended up asking for MORE
    // sampling than rule 0 -- the opposite of its purpose. Measured at M=1 with
    // the doubling form: N=65532 4160 -> 16384 (+21.5%), N=131068 8256 -> 16384
    // (+13.2%), N=32832 4608 -> 8192 (+8.1%), against -4.6% to -7.3% everywhere
    // it actually reduced S. Same fix as the exact-stride repair in v5 Stage 3;
    // that one was applied to the repair and this search was left behind.
    for(int S = SAMPLE_S_MIN; S <= SAMPLE_S_MAX; S += SAMPLE_CHUNK_ELEMS)
    {
        if(!sample_stride_exact(N, S))
            continue;
        const double m = auto_margin(K, S, N);
        // candidate_hi is the THREE-SIGMA upper edge, so accepting it at the cap
        // accepts a plan that overflows on any row past 3 sigma -- and this loop
        // takes the FIRST S that fits, which is the one with the least headroom
        // there is. It cost exactly that: at N=524288 it chose S=5440, putting the
        // edge at 99.1% of the cap, 3.09 sigma of room, and 1.2% of rows then
        // overflowed into phase_d_fallback at a flat ~350 us each. Measured at
        // N=1048576 it is 99.7% and 3.04 sigma; every other width and both of
        // rule 0's was 5.7 to 15.2 sigma.
        //
        // CAP_SAFE_FILL is the fraction derive_shape_params already treats as the
        // safe level for the same cap when it clamps the margin, reused rather than
        // fitted. It moves N=524288 to S=6528 (edge at 84.9%, 4.94 sigma) and
        // N=1048576 to S=13056, and leaves N <= 262144 -- where the edge was
        // already 51.9% and 72.6% -- untouched.
        if(candidate_hi(K, S, N, m) <= CAP_SAFE_FILL * (double)PHASE_C_CAP_MAX)
            return S;
    }
    return SAMPLE_S_MAX;
}

static inline int derive_cap(int K, float margin, int S, int N)
{
    const double hi = candidate_hi(K, S, N, margin);
    int cap         = PHASE_C_CAP;
    while(cap < (int)hi && cap < PHASE_C_CAP_MAX)
        cap *= 2;
    return std::min(cap, PHASE_C_CAP_MAX);
}

// Snap to a power of two. This used to skip G=2 and G=8 entirely, as a
// workaround for what was recorded as "coop_g=2 is broken". That diagnosis was
// wrong: the real fault was an unbounded LDS staging buffer in
// phase_b_filter_coop, which corrupted counts at
// any G once a wave produced more passers than its staging held. With that fixed,
// every G from 1 to 256 gives identical, correct candidate counts, so the
// restriction is gone and G is a free tuning knob again.
//
// Clamp first, then snap: phase_b takes G as a shift, so a G that is not a power
// of two runs on its lowest set bit only (G=9 is one working block and eight idle).
static inline int snap_coop_g(int g, int max_g)
{
    g     = std::max(1, std::min(g, max_g));
    int p = 1;
    while(p * 2 <= g)
        p *= 2;
    return p;
}

// Base log2(coop_g) per cell; coop_g_from_table moves a shape to G/2 or 2G when
// that costs fewer resident rounds. Both were fitted in v8 on the acceptance
// metric itself -- run_perftest device time, cold inputs, both entries -- from G
// curves measured at 1270 (M, N) points: per row M = 2^r and 1.25 / 1.5 / 1.75 x
// 2^r, per column its lower edge and an odd midpoint, every G within two steps
// of the previous table's timed against it. Per cell,
// the base with the smallest worst-point regret among those within 1% of the
// best geomean. Minimax
// because a cell serves EVERY shape in it: argmin at one N is how M=1024 col11
// once came out as G=1, which led G=16 by 0.2% at N=786432 and lost 12.6% at
// N=1048572. Against the best measured G: 0.25% geomean, p90 1.0%, worst 6.2%;
// fitted on half the M samples and scored on the other half 0.67% (p90
// 2.4-2.9%), where a table without the rounds step reads 1.2-1.4% (p90 4.6-5.0%).
//
// This is a table and not a formula on purpose. The best G falls roughly as
// M^-0.3 and saturates differently per N, which no simple closed form
// reproduces: the best two-parameter fit over the v4 sweep left +27% worst
// case, and the rule it replaced (target 256 total blocks) +77% -- measured at
// M=128 N=1048576, where it picked G=2 for 225 us against 127 us at G=16. The
// rounds cost alone leaves 0.7% (p90 3.1%), and 9-11% at M >= 4096, where it
// always takes G=1 and the measured best is 2-8.
//
// Rows are log2(M) for M = 1..4096; M above 4096 takes the last row (fitted at
// M = 4096..7168, and below N = 131072 also at 12288..32768, where topk_select
// reaches it). Columns are HALF-octaves of N from 16384:
// column 2i is [2^k, 1.5 * 2^k) and column 2i+1 is [1.5 * 2^k, 2^(k+1)), with
// k = 14 + i. N <= 8192 takes the small_n path and never reaches here.
//
// A full octave per column is measurably too coarse. Refitting the v4 sweep
// with one column per octave cost up to **+5.42%** (M=64 over [16384, 32768))
// and more than 1% on 12 of the (M, octave) pairs, worst in
// [524288, 1048576) at large M -- which is exactly where the octave table put
// M=1024 N=1048572 on G=1 and paid +8.2%.
constexpr int COOP_TAB_M       = 13;
constexpr int COOP_TAB_N       = 13;
constexpr int COOP_N_LOG2_BASE = 14;

static const signed char kCoopLog2G[COOP_TAB_M][COOP_TAB_N] = {
    /* M=1    */ {2, 1, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6},
    /* M=2    */ {2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6},
    /* M=4    */ {2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6},
    /* M=8    */ {2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 6},
    /* M=16   */ {2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5},
    /* M=32   */ {2, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5},
    /* M=64   */ {2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4},
    /* M=128  */ {2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4},
    /* M=256  */ {1, 1, 1, 1, 2, 1, 2, 2, 2, 3, 3, 3, 3},
    /* M=512  */ {0, 0, 2, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3},
    /* M=1024 */ {0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 5},
    /* M=2048 */ {0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 5, 4},
    /* M=4096 */ {2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 4, 5, 5},
};

// Half-octave column for N: 0 = [16384, 24576), 1 = [24576, 32768),
// 2 = [32768, 49152), 3 = [49152, 65536), and so on.
static inline int coop_bucket_of(int N)
{
    const int k  = ilog2_floor(N);
    const int lo = 1 << k;
    return 2 * (k - COOP_N_LOG2_BASE) + (N >= lo + (lo >> 1) ? 1 : 0);
}

// Off-grid fallback, fitted to the same sweep: worst +27%, mean +6.1%.
constexpr int COOP_TARGET_BLOCKS      = 1024;
constexpr int COOP_MIN_VEC4_PER_BLOCK = 256;

// phase_b's time steps with the rounds of PB_RESIDENT_BLOCKS its grid takes, so
// the best G is periodic in M with period PB_RESIDENT_BLOCKS / G, finer than any
// table row. The table's G is a base; of G/2, G and 2G the shape runs the one
// with the cheapest rounds x (a block's fixed cost + its chunk of float4 loads),
// plus a fifth of a chunk once blocks start staggered over several rounds.
constexpr int COOP_BLOCK_COST_N4   = 1000;
constexpr double COOP_STAGGER_TAIL = 0.2;
static inline double coop_rounds_cost(int M, int n4, int g)
{
    const long rounds = ((long)M * g + PB_RESIDENT_BLOCKS - 1) / PB_RESIDENT_BLOCKS;
    const long chunk  = (n4 + g - 1) / g;
    return (double)(rounds * (COOP_BLOCK_COST_N4 + chunk)) +
           (rounds > 1 ? COOP_STAGGER_TAIL * chunk : 0.0);
}

static inline int coop_g_from_table(int M, int N, int max_g)
{
    if(M < 1 || N < (1 << COOP_N_LOG2_BASE))
        return -1;
    const int mi = std::min(ilog2_floor(M), COOP_TAB_M - 1);
    int ni       = coop_bucket_of(N);
    if(mi >= COOP_TAB_M || ni < 0)
        return -1;
    if(ni >= COOP_TAB_N)
        ni = COOP_TAB_N - 1;
    const int base = snap_coop_g(1 << (int)kCoopLog2G[mi][ni], max_g);
    const int n4   = N / FP32_EPT;
    int g          = base;
    for(const int c : {base / 2, base * 2})
        if(c >= 1 && c <= max_g && coop_rounds_cost(M, n4, c) < coop_rounds_cost(M, n4, g))
            g = c;
    return g;
}

static inline int choose_coop_g(int M, int N, int n4_per_row, int block, int override_g)
{
    const int max_g = std::max(1, n4_per_row / block);
    if(override_g > 0)
        return snap_coop_g(override_g, max_g);
    if(N <= N_LDS_MAX)
        return 1;
    int g;
    const int t = coop_g_from_table(M, N, max_g);
    if(t > 0)
    {
        g = t;
    }
    else
    {
        const int by_target = (M <= COOP_TARGET_BLOCKS) ? (COOP_TARGET_BLOCKS / M) : 1;
        const int by_work   = std::max(1, n4_per_row / COOP_MIN_VEC4_PER_BLOCK);
        g                   = snap_coop_g(std::min(by_target, by_work), max_g);
    }
    // At 2048 <= M <= 4095 below N = 131072 the G = 4 this picks is slower than
    // G = 2: 5-6% at M = 2816 N = 75369, 3-5% at M = 3840 N = 108137.
    if(M >= 2048 && M <= 4095 && N < 131072 && g > 2)
        g = 2;
    return g;
}

// A per-region radix scan form (rep vs wave0) was tried and is
// FALSIFIED: once coop_g > 1 reaches the anchor band, the two forms are
// indistinguishable.

static inline ShapeParams derive_shape_params(int M,
                                              int N,
                                              int K,
                                              float margin_override,
                                              int sample_s_override,
                                              int coop_g_override,
                                              TopkPath path_override)
{
    ShapeParams p{};
    p.path = path_override;
    const bool small_n_fits =
        (N * (int)sizeof(uint32_t)) <= (LDS_BYTES_PER_BLOCK_MAX - SMALL_N_STATIC_LDS);
    const bool small_n_wins =
        N <= N_LDS_MAX || (N <= N_LDS_MAX_ONE_ROUND && M <= N_LDS_ONE_ROUND_M) ||
        (N <= N_LDS_MAX_SMALL_M && M > N_LDS_SMALL_M_FLOOR && M <= N_LDS_SMALL_M_LIMIT);
    if(small_n_fits && (small_n_wins || path_override == PATH_SMALL_N) &&
       (path_override == PATH_AUTO || path_override == PATH_SMALL_N))
    {
        p.path        = PATH_SMALL_N;
        p.S           = 0;
        p.margin      = 1.f;
        p.rank        = 0;
        p.cap         = N;
        p.coop_g      = 1;
        p.keys_only_c = false;
        p.geom_ok     = (K <= N);
        return p;
    }
    if(path_override == PATH_SMALL_N && !small_n_fits)
    {
        // Refuse rather than launch a kernel that cannot start.
        p.path    = PATH_SMALL_N;
        p.geom_ok = false;
        return p;
    }

    float margin = margin_override;
    if(margin <= 0.f)
    {
        int s0 = sample_s_override > 0 ? sample_s_override : derive_sample_s_for_n(M, N, K, 1.4f);
        margin = auto_margin(K, s0, N);
    }
    int S = sample_s_override > 0 ? align_sample_s(sample_s_override)
                                  : derive_sample_s_for_n(M, N, K, margin);
    if(!sample_stride_exact(N, S))
    {
        const int chunks   = std::max(1, N / SAMPLE_CHUNK_ELEMS);
        const int repaired = align_sample_s(chunks * SAMPLE_CHUNK_ELEMS);
        // Take the repair when it buys an exact stride, which is every shape the
        // old rule served and is why those keep their measured behaviour. When it
        // does not -- N = 131328 repairs to 256 chunks of stride 513, still not a
        // multiple of 4 -- the repair only inflates S for nothing, and the masked
        // stride serves the S the law asked for. That shape used to be refused.
        //
        // But the repair had no cost cap, and buying exactness is not worth any
        // price. At N = 2^k + 64 -- aiter's own num_prefix + num_rows pattern -- it
        // moves S from 4096 to 16384, four times the phase_a sampling for 0.2% more
        // data: M=4096 N=32768 243.0 us against N=32832 313.6 us (+29%), M=1024
        // +32.5%, M=256 +18.3%, while N=33024 keeps S=4096 and costs 253.3 us.
        // 1.35% of all N in [32768, 1048576] are inflated >= 2x this way.
        //
        // The repair only ever tried ONE candidate -- N/64 chunks, which
        // align_sample_s then clamps to SAMPLE_S_MAX -- so at N = 2^k + 64 the only
        // exact choice on offer was the largest one. Searching instead finds a much
        // closer exact stride: N=32832 takes S=8192 (128 chunks of 256) rather than
        // 16384, and N=92332 takes 5952 (93 chunks of 992) rather than 16384.
        //
        // Searching beats capping the growth. A growth cap keeps the law's S with a
        // MASKED stride, and that is not free either: at M=4096 N=65600 it produced
        // under_K=760 on --dist inf where the uncapped S gives 0, while the
        // neighbouring pow2 N=65536 at the SAME S=4096 also gives 0. The difference
        // is the masked stride, not the sample count, so the right move is to keep
        // exactness and pay only the growth that exactness actually costs.
        int best = 0;
        for(int cand = S; cand <= SAMPLE_S_MAX; cand += SAMPLE_CHUNK_ELEMS)
        {
            if(sample_stride_exact(N, cand))
            {
                best = cand;
                break;
            }
        }
        if(best > 0)
            S = best;
        else if(sample_stride_exact(N, repaired) || !sampling_geometry_ok(N, S))
            S = repaired;
    }
    margin = margin_override > 0.f ? margin_override : auto_margin(K, S, N);
    // At the largest row grid, N=524288 seed-0 has one candidate count at 2031
    // with the 1.600 estimator margin. That row takes the exact full-row path
    // inside Phase C. A 1.625 floor moves the expected boundary above K while
    // leaving N=1048576 unchanged (its derived margin is already 2.129).
    if(margin_override <= 0.f && M >= 4096 && N >= 524288)
        margin = std::max(margin, 1.625f);
    const double cap_margin = CAP_SAFE_FILL * (double)PHASE_C_CAP_MAX / (double)K;
    const double eff_margin = std::min((double)margin, cap_margin);
    // Gate self-test only: a scale below 1 lifts the sampled threshold so gaussian
    // rows undershoot K and every one of them takes phase_c's fallback.
#ifndef FB_FORCE_RANK_SCALE
#define FB_FORCE_RANK_SCALE 1
#endif
    const int rank =
        std::max(1, (int)(eff_margin * FB_FORCE_RANK_SCALE * (double)K * (double)S / (double)N));
    const int cap = derive_cap(K, margin, S, N);

    p.S      = S;
    p.margin = margin;
    p.rank   = rank;
    p.cap    = cap;
    // Indices stay in LDS whenever phase_c runs one block per CU: keys-only
    // re-reads every index from global inside the gather, which ATT puts at
    // 22-25% of a small-M phase_c block.
    // cap 8192 with indices is 64 KB + static + two wide buffers ~ 106 KB.
    p.keys_only_c = cap > PHASE_C_CAP && grid_blocks_per_cu(M) > 1;
    p.geom_ok     = sampling_geometry_ok(N, S) && K <= cap;

    const int n4 = N / FP32_EPT;
    p.coop_g     = choose_coop_g(M, N, n4, PB_BLOCK, coop_g_override);

    if(path_override == PATH_AUTO)
    {
        if(p.coop_g > 1 && M <= 256)
            p.path = PATH_DECODE;
        else
            p.path = PATH_PREFILL;
    }
    else
    {
        p.path = path_override;
        if(p.path == PATH_DECODE && p.coop_g <= 1)
            p.coop_g = choose_coop_g(M, N, n4, PB_BLOCK, 64);
        if(p.path == PATH_PREFILL)
            p.coop_g = 1;
    }
    return p;
}
