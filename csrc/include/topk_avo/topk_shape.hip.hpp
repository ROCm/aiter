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
// Measured at N=16384 (64 KB per row, 2 blocks/CU), small_n vs the sampled path:
//   M=1 -1.8%, M=8 -7.3%, M=64 -15.1%, M=256 -18.0%   <- small_n wins
//   M=1024 +17.1%, M=4096 +18.7%                      <- sampled path wins
// At small M there are too few blocks for occupancy to bind, so skipping three
// kernel launches and all the candidate traffic is free. At large M the
// 2-blocks/CU ceiling costs more than the pipeline does.
//
// At N=32768 (128 KB per row) small_n loses at EVERY M (+3.8% to +45.7%), so
// 16384 is the end of it, not a point on a continuing trend.
constexpr int N_LDS_MAX           = 8192;  // small_n at any M
constexpr int N_LDS_MAX_SMALL_M   = 16384; // small_n only while M is small
constexpr int N_LDS_SMALL_M_LIMIT = 256;

// Hardware cap on the dynamic LDS one block may request. Asking for more makes
// the LAUNCH fail, and a failed launch is not slow -- it is instant and wrong:
// forcing small_n at N=65536 (256 KB) reported 4.40 us with garbage output.
constexpr int LDS_BYTES_PER_BLOCK_MAX = 160 * 1024;

constexpr int PHASE_C_CAP_MAX = 8192;
constexpr int WSTAGE_WAVES    = 8;
constexpr int WSTAGE_CAP      = 320;

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

// Static __shared__ footprints, read off .group_segment_fixed_size with the
// dynamic buffer excluded. Used only to estimate LDS-limited residency.
constexpr int SMALL_N_STATIC_LDS = 5280;
constexpr int PHASE_A_STATIC_LDS = 5144;
constexpr int PHASE_C_STATIC_LDS = 5408;

static inline int grid_blocks_per_cu(int M) { return std::max(1, (M + CU_COUNT - 1) / CU_COUNT); }

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
    int waves            = std::max(1, TARGET_WAVES_PER_CU / std::min(lds_blocks, g));
    if(load_cap_waves > 0 && g > 1)
        waves = std::min(waves, load_cap_waves);
    int pow2 = 1;
    while(pow2 * 2 <= waves)
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
// See knowledge/known_bad.md ("A statistically tighter margin fails the gate").
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
    if(N % FP32_EPT != 0)
        return false;
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
    if(N % FP32_EPT != 0)
        return false;
    const int chunks = S / SAMPLE_CHUNK_ELEMS;
    const int stride = N / chunks;
    return stride >= SAMPLE_CHUNK_ELEMS && stride % FP32_EPT == 0;
}

// 0 = constant R_TARGET (shipped), 1 = derive S from the acceptance window.
//
// Rule 1 is FALSIFIED as a global replacement and kept only so the measurement
// can be reproduced: it wins 5-7.5% at M <= 8 but regresses the anchor
// M=4096 N=131072 by +3.9% (615.5 -> 639.4 us, over the 620 us limit) and
// M=64 N=262144 by +17.4%, for an overall geomean of just -0.91%.
// Why: a smaller S forces a larger margin and a larger cap, so it trades
// phase_a work for candidate volume. At small M phase_a's single-block cost
// dominates and the trade pays; at large M the extra 52% of candidates that
// Phase B writes and Phase C selects costs more than phase_a saves.
static int g_s_rule = 0;

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
static inline int derive_sample_s_for_n(int N, int K, float margin_unused)
{
    (void)margin_unused;
    if(N < SAMPLE_S_MIN)
    {
        const int chunks = std::max(1, N / SAMPLE_CHUNK_ELEMS);
        return align_sample_s(chunks * SAMPLE_CHUNK_ELEMS);
    }
    if(g_s_rule == 0)
    {
        const float m  = auto_margin(K, SAMPLE_S_MAX, N);
        const double s = R_TARGET * (double)N / ((double)m * (double)K);
        return align_sample_s((int)std::lround(s));
    }
    for(int S = SAMPLE_S_MIN; S <= SAMPLE_S_MAX; S *= 2)
    {
        if(!sample_stride_exact(N, S))
            continue;
        const double m = auto_margin(K, S, N);
        if(candidate_hi(K, S, N, m) <= (double)PHASE_C_CAP_MAX)
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
// phase_b_filter_coop (see knowledge/known_bad.md), which corrupted counts at
// any G once a wave produced more than WSTAGE_CAP passers. With that fixed,
// every G from 1 to 256 gives identical, correct candidate counts, so the
// restriction is gone and G is a free tuning knob again.
static inline int snap_coop_g(int g, int max_g)
{
    int p = 1;
    while(p * 2 <= g)
        p *= 2;
    return std::max(1, std::min(p, max_g));
}

// Measured best log2(coop_g) over the customer pow2 grid: full sweeps at
// M=1..128 (2026-09-17) and M=256..4096 x N=131072..1048576 (v4, 2026-09-17).
//
// This is a table and not a formula on purpose. The best G falls roughly as
// M^-0.3 and saturates differently per N, which no simple closed form
// reproduces: the best two-parameter fit over this same data still leaves
// +27% worst case, and the rule it replaces (target 256 total blocks) leaves
// +77% -- measured at M=128 N=1048576, where it picked G=2 for 225 us against
// 127 us at G=16.
//
// Rows are log2(M) for M = 1..4096. Columns are log2(N) for N = 16384..1048576;
// N <= 8192 takes the small_n path and never reaches here. For M >= 256 the
// ni <= 2 columns (N <= 65536) are 0 (coop_g=1): the v4 sweep only measured
// N >= 131072 and M=256 N=32768 has no win at any G.
constexpr int COOP_TAB_M       = 13;
constexpr int COOP_TAB_N       = 7;
constexpr int COOP_N_LOG2_BASE = 14;

static const signed char kCoopLog2G[COOP_TAB_M][COOP_TAB_N] = {
    /* M=1    */ {2, 4, 5, 6, 6, 6, 7},
    /* M=2    */ {2, 4, 4, 5, 6, 6, 6},
    /* M=4    */ {3, 4, 4, 5, 5, 5, 6},
    /* M=8    */ {3, 3, 4, 4, 5, 5, 6},
    /* M=16   */ {3, 3, 3, 4, 5, 5, 5},
    /* M=32   */ {3, 3, 3, 3, 4, 4, 4},
    /* M=64   */ {3, 3, 3, 3, 3, 4, 4},
    /* M=128  */ {0, 3, 3, 3, 3, 3, 4},
    /* M=256  */ {0, 0, 0, 3, 3, 3, 4},
    /* M=512  */ {0, 0, 0, 1, 2, 3, 4},
    /* M=1024 */ {0, 0, 0, 1, 3, 3, 4},
    /* M=2048 */ {0, 0, 0, 3, 3, 3, 4},
    /* M=4096 */ {0, 0, 0, 3, 3, 3, 4},
};

// Off-grid fallback, fitted to the same sweep: worst +27%, mean +6.1%.
constexpr int COOP_TARGET_BLOCKS      = 1024;
constexpr int COOP_MIN_VEC4_PER_BLOCK = 256;

// Shapes between grid points round DOWN on both axes, which keeps the value on
// the conservative side of the measured optimum.
static inline int coop_g_from_table(int M, int N, int max_g)
{
    if(M < 1 || M > 4096 || N < (1 << COOP_N_LOG2_BASE))
        return -1;
    const int mi = ilog2_floor(M);
    int ni       = ilog2_floor(N) - COOP_N_LOG2_BASE;
    if(mi >= COOP_TAB_M || ni < 0)
        return -1;
    if(ni >= COOP_TAB_N)
        ni = COOP_TAB_N - 1;
    return std::max(1, std::min(1 << (int)kCoopLog2G[mi][ni], max_g));
}

static inline int choose_coop_g(int M, int N, int n4_per_row, int block, int override_g)
{
    const int max_g = std::max(1, n4_per_row / block);
    if(override_g > 0)
        return snap_coop_g(override_g, max_g);
    if(N <= N_LDS_MAX)
        return 1;
    const int t = coop_g_from_table(M, N, max_g);
    if(t > 0)
        return t;
    const int by_target = (M <= COOP_TARGET_BLOCKS) ? (COOP_TARGET_BLOCKS / M) : 1;
    const int by_work   = std::max(1, n4_per_row / COOP_MIN_VEC4_PER_BLOCK);
    return snap_coop_g(std::min(by_target, by_work), max_g);
}

// A per-region radix scan form (rep vs wave0) was tried in v4 Stage 2 and is
// FALSIFIED: once coop_g > 1 reaches the anchor band, the two forms are
// indistinguishable. See knowledge/known_bad.md.

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
        N <= N_LDS_MAX || (N <= N_LDS_MAX_SMALL_M && M <= N_LDS_SMALL_M_LIMIT);
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
        p.geom_ok     = (N % FP32_EPT == 0 && K <= N);
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
        int s0 = sample_s_override > 0 ? sample_s_override : derive_sample_s_for_n(N, K, 1.4f);
        margin = auto_margin(K, s0, N);
    }
    int S = sample_s_override > 0 ? align_sample_s(sample_s_override)
                                  : derive_sample_s_for_n(N, K, margin);
    if(!sample_stride_exact(N, S))
    {
        const int chunks   = std::max(1, N / SAMPLE_CHUNK_ELEMS);
        const int repaired = align_sample_s(chunks * SAMPLE_CHUNK_ELEMS);
        // Take the repair when it buys an exact stride, which is every shape the
        // old rule served and is why those keep their measured behaviour. When it
        // does not -- N = 131328 repairs to 256 chunks of stride 513, still not a
        // multiple of 4 -- the repair only inflates S for nothing, and the masked
        // stride serves the S the law asked for. That shape used to be refused.
        if(sample_stride_exact(N, repaired) || !sampling_geometry_ok(N, S))
            S = repaired;
    }
    margin                  = margin_override > 0.f ? margin_override : auto_margin(K, S, N);
    const double cap_margin = 0.85 * (double)PHASE_C_CAP_MAX / (double)K;
    const double eff_margin = std::min((double)margin, cap_margin);
    const int rank          = std::max(1, (int)(eff_margin * (double)K * (double)S / (double)N));
    const int cap           = derive_cap(K, margin, S, N);

    p.S           = S;
    p.margin      = margin;
    p.rank        = rank;
    p.cap         = cap;
    p.keys_only_c = cap > PHASE_C_CAP;
    p.geom_ok     = sampling_geometry_ok(N, S) && K <= cap;

    const int n4 = N / FP32_EPT;
    p.coop_g     = choose_coop_g(M, N, n4, 512, coop_g_override);

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
            p.coop_g = choose_coop_g(M, N, n4, 512, 64);
        if(p.path == PATH_PREFILL)
            p.coop_g = 1;
    }
    return p;
}
