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

// benchmark_topk.hip.cpp — fp32 per-row top-k indices for prefill.
// Contract: input fp32 [M,N], output int32 indices [M,K]. Target M=4096 N=131072 K=2048.
//
// Fused pipeline (default, --pipeline fused), 4 kernel launches per call:
//   A  phase_a_threshold : one block per row, samples SAMPLE_S contiguous-chunk elements,
//                          selects rank R entirely in LDS -> per-row threshold.
//   B  phase_b_filter    : the whole budget. Streams the row with dwordx4, ONE integer
//                          compare per element, wave64 ballot compression, one global
//                          atomic per wave, appends (key,index) to a per-row candidate area.
//   C  phase_c_select    : one block per row, loads the candidate set into LDS and does the
//                          exact 4x8-bit radix select + tie-correct gather in LDS.
//   D  phase_d_fallback  : rows whose candidate set is unusable (too few OR overflowed)
//                          are recomputed exactly by streaming the full row.
//
// --pipeline direct runs D over every row. That is the independent full-row oracle and is
// the same code the fallback uses, so both timed paths are separately verifiable.

#include "topk_sampled/topk_common.hip.hpp"
#include "topk_sampled/topk_shape.hip.hpp"
#include <string>
#include <type_traits>

// ---- AVO variation surface -------------------------------------------------
static int g_sample_rank = 0;    // 0 => derive from margin
static int g_sample_s    = 0;    // 0 => derive from shape (R_TARGET law)
static float g_margin    = 0.0f; // 0 => derive from the estimator's own noise

// The candidate count is the number of row elements above the rank-R value of S
// samples, so its spread is that estimator's noise: std ~ count/sqrt(R). A row
// undershoots K (and pays the exact fallback) when count < K, so the required
// over-collection factor is set by how noisy R is, NOT by a constant.
//
// Measured at S=8192, margin 1.4, N=131072 (min/mean, rows under K out of 4096):
//   K=2048 R=179  0.74  0 rows
//   K=1024 R= 89  0.67  4 rows
//   K= 512 R= 44  0.52 82 rows
// i.e. a fixed 1.4 is overfitted to K=2048. Requiring mean*(1 - 3/sqrt(R0)) > K
// with R0 = K*S/N (the margin-free rank) reproduces 1.36 at K=2048 and demands
// 2.13 at K=512, which is what the data shows.
// auto_margin() lives in topk_generalize.hip.hpp
// Distinct from g_use_nt_load above: that one flips a __constant__ read inside
// load_f4, so it applies to every phase and costs a branch in the innermost
// load. This one is a template parameter on phase_b only.
static int g_nt_load = -1; // -1 = the size gate below, 0 = never, 1 = always
// log2(M * pitch) at which phase_b's row loads, and separately its candidate
// stores, go non-temporal. Stores only ever go NT when loads do.
#ifndef NT_LOAD_LOG2
#define NT_LOAD_LOG2 25
#endif
#ifndef NT_STORE_LOG2
#define NT_STORE_LOG2 27
#endif
static int g_cf_block    = 512; // Phase B block size
static int g_cf_gx       = 16;  // Phase B blocks per row (grid.x)
static int g_use_nt_load = 0;   // non-temporal streaming loads in Phase B
// Phase B implementation: 3 = wave-private regions writing straight to global,
// 4 = same but passers staged in LDS and flushed in full contiguous bursts.
// Variants 0-2 (per-wave global atomic, block-aggregated atomic, LDS counter)
// were measured and superseded; see knowledge/known_bad.md and git history.
static int g_phase_b       = 4;
static int g_phase_c_block = 0; // 0 => derive from occupancy
static int g_phase_a_block = 0; // 0 => derive from occupancy
// 3 passes give bit-identical candidate counts to 4 (the 4th byte never moves the
// bucket at fp32 precision) and 3 is safer than 2, whose spread ran to max=3919
// against C_alloc=4096.
static int g_phase_a_passes = 3;
// FALSIFIED as a perf change, kept only so the measurement can be reproduced:
// compacting the active set is correct (same pivot, same candidate counts on all
// five distributions) but buys nothing -- phase_a 65.7 -> 65.4 us and wall time
// WORSE on three of four shapes. The discarded reads were never the cost; the
// barriers are. See knowledge/known_bad.md.
static int g_phase_a_compact = 0;
// block_select_lds_wide in phase_a / phase_c: -1 = wide_select_fits, 0 off, 1 on.
static int g_wide_a = -1;
static int g_wide_c = -1;
// Phase A sample-key residency: -1 = auto (reg when S%a_block==0 and KPT in
// {4,8,16}), 0 = force LDS keys, 1 = force register keys (errors if ineligible).
static int g_pa_keys         = -1;
static int g_dump_thresholds = 0; // 1 => write threshold+cand_count after run
static std::string g_dump_thresholds_path;

// Folds the HIST_REP reduction into the pivot scan, taking a radix pass from 5
// block barriers to 4. Shipped on; build with `-DSELECT_FUSED_REDUCE=0` to A/B
// against the separate-reduction form.
//
// Build-time and not a runtime knob because it changes the barrier structure of
// every caller of block_select_lds at once, and a runtime branch around a
// barrier would not be a fair comparison. Measured wall time, 3 runs each at
// warmup 20 / iters 100 / repeats 9, 0 -> 1:
//   M=4096 N=131072  0.6132 -> 0.6090 ms  (-0.68%, the anchor)
//   M=4096 N=1048576 3.1741 -> 3.1680 ms  (-0.19%)
//   M=1    N=1048576 0.0316 -> 0.0309 ms  (-2.2%)
//   M=1024 N=65536   0.0793 -> 0.0781 ms  (-1.5%)
//   M=4096 N=8192    0.0995 -> 0.0947 ms  (-4.8%, small_n)
//   M=2048 N=4096    0.0372 -> 0.0355 ms  (-4.6%, small_n)
// The gain tracks how much of the kernel is the select, which is what the
// barrier-bound reading of phase_a/phase_c predicts.
#ifndef SELECT_FUSED_REDUCE
#define SELECT_FUSED_REDUCE 1
#endif

// Second barrier removal on the same lever: the scan re-zeroes each histogram
// bucket as it reads it, so the per-pass clear loop and its barrier disappear
// and a radix pass goes from 4 barriers to 3. Costs no LDS, unlike
// double-buffering the histogram (+4 KB, which would cut phase_a from 4 to 3
// blocks/CU at S=8192). Requires SELECT_FUSED_REDUCE.
//
// Shipped on, but it is a REGIME TRADE, not a free win, so read this before
// moving it. Wall time, 3 runs each at warmup 20 / iters 100 / repeats 9:
//   M=4096 N=131072  0.6090 -> 0.6069 ms  (-0.35%, the anchor)
//   M=1    N=1048576 0.0308 -> 0.0306 ms  (-0.8%)
//   M=4096 N=1048576 3.1658 -> 3.1652 ms  (neutral)
//   M=1024 N=65536   0.0781 -> 0.0782 ms  (neutral)
//   M=2048 N=4096    0.0354 -> 0.0353 ms  (neutral)
//   M=4096 N=8192    0.0948 -> 0.0965 ms  (**+1.8%, small_n**)
// Inner geomean -0.40% with decode -0.8% and prefill -0.8% against small_n
// +0.40%, which stays inside PATH_NOISE_BAND_PCT. Taken because the large-N
// paths are the target and N <= 32768 is routed to aiter's own prefill by the
// stride0 >= 32768 dispatch in aiter/ops/topk.py.
//
// [unverified hypothesis] for the small_n point: the clear now runs on the 256
// threads that also carry the wave-scan, where the old loop spread it over all
// blockDim.x threads (512 at that shape), so the work moved onto the critical
// path instead of disappearing.
#ifndef SELECT_CLEAR_ON_READ
#define SELECT_CLEAR_ON_READ 1
#endif

// Diagnostic only, PRODUCES WRONG RESULTS: drops the atomicity of the histogram
// increment so the per-element LDS atomic can be priced. Never ship non-zero.
// Measured ceiling for any aggregation of that atomic: phase_a 61.7 -> 53.0 us
// at the anchor (-14.1%), and -9.3%/-13.0% of wall on small_n M=4096 N=8192 /
// M=2048 N=4096. It says nothing about phase_c: a wrong Phase A threshold blows
// up the candidate counts and sends rows to the exact fallback, which took
// phase_c 67.3 -> 2119 us. An ablation is only a price when it leaves the path
// alone.
#ifndef ABLATE_HIST_ATOMIC
#define ABLATE_HIST_ATOMIC 0
#endif

// Rounds of wave-level aggregation before falling back to per-element atomics
// (0 = off, the shipped form). See hist_add_aggregated in topk_common.hip.hpp.
#ifndef HIST_AGG_ROUNDS
#define HIST_AGG_ROUNDS 0
#endif

// Third barrier removal: wave 0 alone scans all 256 buckets, so the cross-wave
// partial sums and their barrier disappear and a pass runs 2 barriers, not 3.
// See block_find_pivot_bucket_wave0 in topk_common.hip.hpp for why the
// everyone-scans-redundantly variant cannot reach 2 without +4 KB of LDS.
//
// Shipped on. It is close to the MIRROR of SELECT_CLEAR_ON_READ's regime trade:
// that one bought the anchor and cost small_n, this one buys small_n and the
// latency-bound small-M shapes and costs the anchor slightly.
//   M=4096 N=8192    0.0964 -> 0.0943 ms  (-2.2%, small_n)
//   M=1    N=1048576 0.0305 -> 0.0300 ms  (-1.6%)
//   M=2048 N=4096    0.0352 -> 0.0347 ms  (-1.4%)
//   M=4096 N=1048576 3.1647 -> 3.1616 ms  (neutral)
//   M=1024 N=65536   0.0776 -> 0.0776 ms  (neutral)
//   M=4096 N=131072  0.6071 -> 0.6095 ms  (**+0.4%, the anchor**)
// Inner geomean 62.66 -> 61.96/61.98 us (-1.1%, two runs) with small_n -2.3%,
// decode -1.3% and prefill neutral, so the aggregate is a clear win and the one
// regressing point sits far inside POINT_REGRESS_PCT.
#ifndef SELECT_WAVE0_SCAN
#define SELECT_WAVE0_SCAN 1
#endif
#if SELECT_CLEAR_ON_READ && !SELECT_FUSED_REDUCE
#error "SELECT_CLEAR_ON_READ needs SELECT_FUSED_REDUCE: only the fused scan clears"
#endif
static int g_pipeline_direct = 0;
static int g_inject_fault    = 0;
static int g_dump_stats      = 0;
static int g_ablate_store    = 0; // diagnostic only: produces WRONG results
// Phase C must use all 4 passes to be exact. Fewer is a TIMING ABLATION ONLY.
static int g_phase_c_passes     = RADIX_PASSES;
static int g_path_override      = PATH_AUTO;
static int g_coop_g             = 0;
static int g_fuse_ab            = 0;
static int g_use_hipgraph       = 0;
static int g_small_n_block      = 0;            // 0 => derive from the vec4 load count
static int g_small_n_passes     = RADIX_PASSES; // < 4 is a TIMING ABLATION (wrong results)
static int g_verify_sample_rows = 32;
static int g_verify_oracle_gpu  = 1;
static int g_ragged             = 0;
// Mirrors aiter's create_row_boundaries(num_rows, num_prefix): row r has extent
// num_prefix + r + 1. The prefix is what decides WHICH ragged path is exercised
// and the two are disjoint, so a gate that only runs prefix 0 tests half the
// code: at prefix 0 every extent is <= M, so with S=8192 every row is routed to
// the identity/exact path and the SAMPLER never sees a ragged row at all
// (M=512 N=131072 reported fallback_rows for all 512 rows). aiter's real
// prefill config uses prefix 131072, where every extent is long and ragged.
static int g_ragged_prefix     = 0;
static int g_row_starts_stride = 0;
static int g_values            = 0; // also emit the selected scores

// ---------------------------------------------------------------------------
// Block-wide exact radix select over keys already resident in LDS.
// On return: pivot == the K-th largest sortable key, eq_needed == how many
// elements equal to pivot must be taken (so K - eq_needed are strictly greater).
// ---------------------------------------------------------------------------
// npasses < RADIX_PASSES yields a pivot truncated to the leading 8*npasses bits.
// Phase C and the fallback MUST use all 4 (their result is the answer). Phase A
// may use fewer: its pivot is only a filter threshold, and Phase C still does
// the exact select, so the only consequence is a shift in how many candidates
// Phase B collects.
__device__ __forceinline__ void block_select_lds(const uint32_t* __restrict__ s_keys,
                                                 int c,
                                                 int K,
                                                 uint32_t* __restrict__ s_hist,
                                                 uint32_t* __restrict__ s_red,
                                                 uint32_t* __restrict__ s_scan,
                                                 uint32_t* __restrict__ s_mm,
                                                 uint32_t& pivot,
                                                 int& eq_needed,
                                                 int npasses      = RADIX_PASSES,
                                                 bool prefix_skip = false,
                                                 // The caller counted pass 0's digits into s_hist
                                                 // while it was loading the keys, and zeroed s_hist
                                                 // first, so pass 0 only has to find its bucket.
                                                 bool hist_prefilled = false)
{
    const int rep = threadIdx.x & (HIST_REP - 1);
    if(threadIdx.x == 0)
    {
        s_scan[0] = 0;
        s_scan[1] = 0;
    }

    // Phase C's candidates are all >= the Phase B threshold, so they share a high
    // prefix and the passes where min and max already agree can be skipped
    // outright. Phase A must NOT do this: its samples span the whole row, so no
    // pass is ever skipped and the min/max reduction is pure loss (measured
    // phase_c -6.0 us, phase_a +8.0 us).
    int start = 0;
    pivot     = 0;
    if(prefix_skip)
    {
        uint32_t mn, mx;
        block_minmax_lds(s_keys, c, s_mm, mn, mx);
        start = common_prefix_passes(mn, mx);
        if(start >= RADIX_PASSES)
        {
            // Every key identical: the pivot is that value and all K come from ties.
            pivot     = mn;
            eq_needed = K;
            return;
        }
        pivot = mn & ((start == 0) ? 0u : (0xFFFFFFFFu << (32 - 8 * start)));
    }

    // prefix_skip can move the first executed pass off 0, and the caller only
    // counted digits for pass 0. When it moves, the prefill is for the wrong
    // byte and s_hist has to be cleared like any other call.
    const bool use_prefill = hist_prefilled && start == 0;
    int ek                 = K;
#if SELECT_CLEAR_ON_READ
    // Zeroed once here; from then on the scan re-zeroes each bucket as it reads
    // it, so the per-pass clear loop and its barrier are gone.
    if(!use_prefill)
    {
        clear_hist(s_hist);
        __syncthreads();
    }
#endif
    for(int p = start; p < npasses; p++)
    {
        const int sh      = radix_shift(p);
        const int hshift  = sh + 8;
        const bool filter = (p > 0);
        // Pass `start` reads a histogram the caller already filled, so this
        // scan of s_keys and the wait that ends it are paid for.
        const bool skip_hist = use_prefill && p == start;
        if(!skip_hist)
        {
#if !SELECT_CLEAR_ON_READ
            clear_hist(s_hist);
            __syncthreads();
#endif
            // Do NOT add an active-set min/max here to exit early once the pivot is
            // pinned. It was tried: accumulating amn/amx in this loop (the reads are
            // already happening) and breaking when they agree made small_n 21-39%
            // SLOWER and the anchor 615.5 -> 662.8 us. The two extra barriers per pass
            // in the reduction, plus the register pressure in this loop, cost far more
            // than the single pass the exit saves. See knowledge/known_bad.md.
#if HIST_AGG_ROUNDS
            // Uniform trip count, because the aggregation ballots need every lane of the
            // wave in the same iteration; the strided form below exits at different
            // iterations per lane. Same shape as block_gather_topk's loop.
            for(int i0 = 0; i0 < c; i0 += blockDim.x)
            {
                const int i      = i0 + threadIdx.x;
                const bool live  = (i < c);
                const uint32_t k = live ? s_keys[i] : 0u;
                const bool act   = live && (!filter || (k >> hshift) == (pivot >> hshift));
                hist_add_aggregated(s_hist, (k >> sh) & 0xFFu, rep, act, HIST_AGG_ROUNDS);
            }
#else
            for(int i = threadIdx.x; i < c; i += blockDim.x)
            {
                uint32_t k = s_keys[i];
                if(!filter || (k >> hshift) == (pivot >> hshift))
#if ABLATE_HIST_ATOMIC
                    // TIMING ABLATION, WRONG RESULTS: same address pattern and LDS traffic,
                    // but no atomicity, so the delta is exactly what the atomic plus its
                    // bucket conflict costs. Prices the ceiling of any wave-aggregation.
                    s_hist[((k >> sh) & 0xFFu) * HIST_REP + rep] = 1u;
#else
                    atomicAdd(&s_hist[((k >> sh) & 0xFFu) * HIST_REP + rep], 1u);
#endif
            }
#endif
        }
        __syncthreads();
#if SELECT_WAVE0_SCAN
        block_find_pivot_bucket_wave0<SELECT_CLEAR_ON_READ != 0>(s_hist, s_scan, ek);
#elif SELECT_FUSED_REDUCE
        block_find_pivot_bucket_rep<SELECT_CLEAR_ON_READ != 0>(s_hist, s_scan, ek);
#else
        if(HIST_REP > 1)
        {
            for(int b = threadIdx.x; b < 256; b += blockDim.x)
            {
                uint32_t sum = 0;
#pragma unroll
                for(int r = 0; r < HIST_REP; r++)
                    sum += s_hist[b * HIST_REP + r];
                s_red[b] = sum;
            }
            __syncthreads();
        }
        block_find_pivot_bucket(HIST_REP > 1 ? s_red : s_hist, s_scan, ek);
#endif
        pivot |= (s_scan[0] << sh);
        ek -= (int)s_scan[1];
    }
    eq_needed = ek;
}

// block_select_lds with the filtered passes on a 12-bit digit (see
// block_find_pivot_wide_wave0): pass 0 is the same 8-bit pass on s_hist, then
// `nwide` passes of 12 bits, on bits [12, 24) and then [0, 12). nwide == 2 is
// the exact 32-bit select Phase C needs; nwide == 1 leaves a 20-bit prefix,
// which is what Phase A's threshold gets (a filter threshold, so a coarser
// prefix only admits a few more candidates, never fewer).
//
// s_wide must hold nwide * WIDE_WORDS words, all zero on entry (clear_wide
// before a barrier that precedes this call). s_hist follows block_select_lds's
// SELECT_CLEAR_ON_READ + SELECT_WAVE0_SCAN contract, which this requires.
//
// Where it runs is decided on the host (wide_select_for): the buffers live in
// dynamic LDS so a launch that does not ask for them pays no occupancy for them.
#if !(SELECT_CLEAR_ON_READ && SELECT_WAVE0_SCAN)
#error "block_select_lds_wide is written against the clear-on-read wave-0 scan of pass 0"
#endif
template <bool REUSE_WIDE = false>
__device__ __forceinline__ void block_select_lds_wide(const uint32_t* __restrict__ s_keys,
                                                      int c,
                                                      int K,
                                                      uint32_t* __restrict__ s_hist,
                                                      uint32_t* __restrict__ s_wide,
                                                      uint32_t* __restrict__ s_scan,
                                                      uint32_t* __restrict__ s_mm,
                                                      uint32_t& pivot,
                                                      int& eq_needed,
                                                      int nwide,
                                                      bool prefix_skip,
                                                      bool hist_prefilled)
{
    const int rep = threadIdx.x & (HIST_REP - 1);
    if(threadIdx.x == 0)
    {
        s_scan[0] = 0;
        s_scan[1] = 0;
    }
    // Same skip as block_select_lds, at this function's pass boundaries: bit 24
    // ends pass 0 and bit 12 ends the first wide pass.
    int start = 0;
    pivot     = 0;
    if(prefix_skip)
    {
        uint32_t mn, mx;
        block_minmax_lds(s_keys, c, s_mm, mn, mx);
        if(mn == mx)
        {
            pivot     = mn;
            eq_needed = K;
            return;
        }
        start = ((mn >> 24) != (mx >> 24)) ? 0 : ((mn >> 12) != (mx >> 12)) ? 1 : 2;
        pivot = (start == 0) ? 0u : (start == 1) ? (mn & 0xFF000000u) : (mn & 0xFFFFF000u);
    }
    int ek = K;
    if(start == 0)
    {
        if(!hist_prefilled)
        {
            clear_hist(s_hist);
            __syncthreads();
            for(int i = threadIdx.x; i < c; i += blockDim.x)
                atomicAdd(&s_hist[(s_keys[i] >> 24) * HIST_REP + rep], 1u);
        }
        __syncthreads();
        block_find_pivot_bucket_wave0<true>(s_hist, s_scan, ek);
        pivot |= (s_scan[0] << 24);
        ek -= (int)s_scan[1];
    }
    const int first_w = (start == 0) ? 0 : start - 1;
    for(int w = first_w; w < nwide; w++)
    {
        const int sh     = 12 - WIDE_BITS * w;
        const int hshift = sh + WIDE_BITS;
        if constexpr(REUSE_WIDE)
        {
            if(w != first_w)
            {
                clear_wide(s_wide, 1);
                __syncthreads();
            }
        }
        uint32_t* sw = s_wide + (REUSE_WIDE ? 0 : w * WIDE_WORDS);
        for(int i = threadIdx.x; i < c; i += blockDim.x)
        {
            const uint32_t k = s_keys[i];
            if((k >> hshift) == (pivot >> hshift))
                wide_count(sw, (k >> sh) & (WIDE_FINE - 1), rep);
        }
        __syncthreads();
        block_find_pivot_wide_wave0(sw, s_scan, ek);
        pivot |= (s_scan[0] << sh);
        ek -= (int)s_scan[1];
        // After the first wide pass the crossing fine bucket can be tiny. Staging
        // those keys and finishing bits [0,12) with a Wave0 ballot walk was priced
        // because the second wide pass costs 1.1-1.5 us (plan P3), but the restage
        // scan of all `c` candidates costs about as much as the pass it replaces
        // (same-session A/B: ballot phase_c ≈ full second-wide phase_c). Default
        // off; set -DPC_BALLOT_WALK=1 to re-measure.
#ifndef PC_BALLOT_WALK
#define PC_BALLOT_WALK 0
#endif
#ifndef PA_BALLOT_MUTANT
#define PA_BALLOT_MUTANT 0
#endif
#if PC_BALLOT_WALK
        if(w == 0 && nwide == 2)
        {
            const int cross = (int)sw[WIDE_COARSE_SLOTS + s_scan[0]];
            if(cross > 0 && cross <= 64)
            {
                // Reuse the just-consumed wide buffer as a 64-key staging area.
                __syncthreads();
                for(int j = threadIdx.x; j < 64; j += blockDim.x)
                    sw[j] = 0u;
                if(threadIdx.x == 0)
                    s_scan[0] = 0;
                __syncthreads();
                for(int i = threadIdx.x; i < c; i += blockDim.x)
                {
                    const uint32_t k = s_keys[i];
                    if((k >> 12) == (pivot >> 12))
                    {
                        const unsigned slot = atomicAdd(&s_scan[0], 1u);
                        if(slot < 64u)
                            sw[slot] = k;
                    }
                }
                __syncthreads();
                if(threadIdx.x < WAVE_SIZE)
                {
                    const int lane          = threadIdx.x;
                    const uint32_t key_bits = (lane < cross) ? sw[lane] : 0u;
                    uint64_t valid          = __ballot(lane < cross);
                    uint64_t active         = valid;
                    int rank                = ek;
#pragma unroll 1
                    for(int bit = 11; bit >= 0 && (active & (active - 1ull)) != 0ull; --bit)
                    {
                        const uint32_t bit_mask = 1u << bit;
                        // Prefer the 1-partition: larger sortable bits are larger fp32.
                        const uint64_t one_mask =
                            active & (uint64_t)__ballot((key_bits & bit_mask) != 0u);
                        const int one_count = (int)__popcll(one_mask);
                        if(rank <= one_count)
                        {
                            active = one_mask;
                        }
                        else
                        {
                            active &= ~one_mask;
                            rank -= one_count;
                        }
                    }
                    const int kth_lane = __builtin_ctzll(active);
                    const uint32_t kth_bits =
                        (uint32_t)__builtin_amdgcn_readlane((int)key_bits, kth_lane);
                    const uint64_t gt_mask = valid & (uint64_t)__ballot(key_bits > kth_bits);
                    const int n_gt         = (int)__popcll(gt_mask);
                    if(lane == 0)
                    {
#if PA_BALLOT_MUTANT
                        s_scan[0] = kth_bits ^ 1u;
#else
                        s_scan[0] = kth_bits;
#endif
                        s_scan[1] = (unsigned)(ek - n_gt);
                    }
                }
                __syncthreads();
                pivot     = s_scan[0];
                eq_needed = (int)s_scan[1];
                return;
            }
        }
#endif
        // All c keys landed in the bucket just picked, so they share every bit at
        // and above `sh`. Without a prefix skip this is where a one-value candidate
        // set (a tie-dense row) shows itself, and one min/max settles it; any other
        // row never reaches the min/max.
        if(!prefix_skip && w + 1 < nwide && (int)sw[WIDE_COARSE_SLOTS + s_scan[0]] == c)
        {
            uint32_t mn, mx;
            block_minmax_lds(s_keys, c, s_mm, mn, mx);
            if(mn == mx)
            {
                pivot = mn;
                break;
            }
        }
    }
    eq_needed = ek;
}

// Register-resident cousins of block_select_lds / block_select_lds_wide for
// Phase A. Each thread already holds its S/blockDim keys (KPT in {4,8,16}) from
// the sample loads, so the select never needs the S*4-byte keys buffer in
// dynamic LDS -- only the histogram (and the optional wide buffer) stay in LDS.
// Keys never leave the owning thread: Phase A emits a threshold, not a gather.
#ifndef PA_REG_MUTANT
#define PA_REG_MUTANT 0
#endif
template <int KPT>
__device__ __forceinline__ void block_select_reg(const uint32_t keys[KPT],
                                                 int K,
                                                 uint32_t* __restrict__ s_hist,
                                                 uint32_t* __restrict__ s_red,
                                                 uint32_t* __restrict__ s_scan,
                                                 uint32_t& pivot,
                                                 int& eq_needed,
                                                 int npasses,
                                                 bool hist_prefilled)
{
    const int rep = threadIdx.x & (HIST_REP - 1);
    if(threadIdx.x == 0)
    {
        s_scan[0] = 0;
        s_scan[1] = 0;
    }
    pivot                  = 0;
    int ek                 = K;
    const bool use_prefill = hist_prefilled;
#if SELECT_CLEAR_ON_READ
    if(!use_prefill)
    {
        clear_hist(s_hist);
        __syncthreads();
    }
#endif
    for(int p = 0; p < npasses; p++)
    {
        const int sh         = radix_shift(p);
        const int hshift     = sh + 8;
        const bool filter    = (p > 0);
        const bool skip_hist = use_prefill && p == 0;
        if(!skip_hist)
        {
#if !SELECT_CLEAR_ON_READ
            clear_hist(s_hist);
            __syncthreads();
#endif
#pragma unroll
            for(int t = 0; t < KPT; t++)
            {
#if PA_REG_MUTANT
                if(t == 0)
                    continue; // deliberate drop: threshold-equivalence must go red
#endif
                const uint32_t k = keys[t];
                if(!filter || (k >> hshift) == (pivot >> hshift))
                    atomicAdd(&s_hist[((k >> sh) & 0xFFu) * HIST_REP + rep], 1u);
            }
        }
        __syncthreads();
#if SELECT_WAVE0_SCAN
        block_find_pivot_bucket_wave0<SELECT_CLEAR_ON_READ != 0>(s_hist, s_scan, ek);
#elif SELECT_FUSED_REDUCE
        block_find_pivot_bucket_rep<SELECT_CLEAR_ON_READ != 0>(s_hist, s_scan, ek);
#else
        if(HIST_REP > 1)
        {
            for(int b = threadIdx.x; b < 256; b += blockDim.x)
            {
                uint32_t sum = 0;
#pragma unroll
                for(int r = 0; r < HIST_REP; r++)
                    sum += s_hist[b * HIST_REP + r];
                s_red[b] = sum;
            }
            __syncthreads();
        }
        block_find_pivot_bucket(HIST_REP > 1 ? s_red : s_hist, s_scan, ek);
#endif
        pivot |= (s_scan[0] << sh);
        ek -= (int)s_scan[1];
    }
    eq_needed = ek;
}

template <int KPT>
__device__ __forceinline__ void block_select_reg_wide(const uint32_t keys[KPT],
                                                      int K,
                                                      uint32_t* __restrict__ s_hist,
                                                      uint32_t* __restrict__ s_wide,
                                                      uint32_t* __restrict__ s_scan,
                                                      uint32_t& pivot,
                                                      int& eq_needed,
                                                      int nwide,
                                                      bool hist_prefilled)
{
    const int rep = threadIdx.x & (HIST_REP - 1);
    if(threadIdx.x == 0)
    {
        s_scan[0] = 0;
        s_scan[1] = 0;
    }
    pivot  = 0;
    int ek = K;
    if(!hist_prefilled)
    {
        clear_hist(s_hist);
        __syncthreads();
#pragma unroll
        for(int t = 0; t < KPT; t++)
        {
#if PA_REG_MUTANT
            if(t == 0)
                continue;
#endif
            atomicAdd(&s_hist[(keys[t] >> 24) * HIST_REP + rep], 1u);
        }
    }
    __syncthreads();
    block_find_pivot_bucket_wave0<true>(s_hist, s_scan, ek);
    pivot |= (s_scan[0] << 24);
    ek -= (int)s_scan[1];
    for(int w = 0; w < nwide; w++)
    {
        const int sh     = 12 - WIDE_BITS * w;
        const int hshift = sh + WIDE_BITS;
        uint32_t* sw     = s_wide + w * WIDE_WORDS;
#pragma unroll
        for(int t = 0; t < KPT; t++)
        {
#if PA_REG_MUTANT
            if(t == 0)
                continue;
#endif
            const uint32_t k = keys[t];
            if((k >> hshift) == (pivot >> hshift))
                wide_count(sw, (k >> sh) & (WIDE_FINE - 1), rep);
        }
        __syncthreads();
        block_find_pivot_wide_wave0(sw, s_scan, ek);
        pivot |= (s_scan[0] << sh);
        ek -= (int)s_scan[1];
    }
    eq_needed = ek;
}

// Active-set compaction variant of the select above, for Phase A only.
//
// The filter-rescan form reads all `c` keys on every pass and discards the
// ~255/256 that do not match the pivot prefix. Those later passes were measured
// at 30-34% of phase_small_n_topk while carrying almost no work (see
// knowledge/known_bad.md, "Early-exiting the radix select once the pivot is
// pinned"). This form compacts the survivors after each pass, so pass p+1 reads
// only what pass p kept: ~2c key reads over three passes instead of 3c.
//
// The MECHANISM is the point, not the saving. An earlier attempt removed the
// same passes with a block-wide active-set min/max early exit and came out
// 21-39% SLOWER, because that test costs two barriers per pass. Compaction here
// is WAVE-PRIVATE, so it adds no barrier and no LDS -- the same ownership trick
// that lets the shipped Phase B filter run with no atomic of any kind.
//
// Wave w owns [lo, lo + n_active) of s_keys and compacts its own survivors to
// the front of its own segment, carrying the count in a wave-uniform register.
// In place is safe because within an iteration every lane reads before any lane
// writes, and a survivor lands at or below the index it came from
// (wcnt <= j and popcount(ballot & lt) <= lane), so nothing is overwritten
// before it has been read. Waves own disjoint segments, so there is no
// cross-wave hazard either.
//
// No prefix_skip path: a compacted pass needs no prefix filter to begin with,
// and Phase A never asked for prefix_skip anyway (its samples span the whole
// row, so no pass is ever skippable -- measured +8.0 us when tried).
__device__ __forceinline__ void block_select_lds_compact(uint32_t* __restrict__ s_keys,
                                                         int c,
                                                         int K,
                                                         uint32_t* __restrict__ s_hist,
                                                         uint32_t* __restrict__ s_red,
                                                         uint32_t* __restrict__ s_scan,
                                                         uint32_t& pivot,
                                                         int& eq_needed,
                                                         int npasses)
{
    const int rep     = threadIdx.x & (HIST_REP - 1);
    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const int wid     = threadIdx.x / WAVE_SIZE;
    const int nwaves  = blockDim.x / WAVE_SIZE;
    const uint64_t lt = (1ull << lane) - 1ull;

    if(threadIdx.x == 0)
    {
        s_scan[0] = 0;
        s_scan[1] = 0;
    }

    const int chunk = (c + nwaves - 1) / nwaves;
    const int lo    = min(wid * chunk, c);
    int n_active    = min(lo + chunk, c) - lo;

    pivot  = 0;
    int ek = K;
    for(int p = 0; p < npasses; p++)
    {
        const int sh = radix_shift(p);
        clear_hist(s_hist);
        __syncthreads();
        for(int i = lane; i < n_active; i += WAVE_SIZE)
        {
            const uint32_t k = s_keys[lo + i];
            atomicAdd(&s_hist[((k >> sh) & 0xFFu) * HIST_REP + rep], 1u);
        }
        __syncthreads();
        if(HIST_REP > 1)
        {
            for(int b = threadIdx.x; b < 256; b += blockDim.x)
            {
                uint32_t sum = 0;
#pragma unroll
                for(int r = 0; r < HIST_REP; r++)
                    sum += s_hist[b * HIST_REP + r];
                s_red[b] = sum;
            }
            __syncthreads();
        }
        // Ends in a barrier, so every read of s_hist / s_red for this pass is done
        // before the next iteration zeroes them.
        block_find_pivot_bucket(HIST_REP > 1 ? s_red : s_hist, s_scan, ek);
        pivot |= (s_scan[0] << sh);
        ek -= (int)s_scan[1];

        if(p + 1 == npasses)
            break;
        const uint32_t want = pivot >> sh;
        int wcnt            = 0;
        for(int j = 0; j < n_active; j += WAVE_SIZE)
        {
            const int i        = j + lane;
            const bool live    = (i < n_active);
            const uint32_t k   = live ? s_keys[lo + i] : 0u;
            const bool keep    = live && ((k >> sh) == want);
            const uint64_t bal = __ballot(keep);
            if(keep)
                s_keys[lo + wcnt + __popcll(bal & lt)] = k;
            wcnt += __popcll(bal);
        }
        n_active = wcnt;
    }
    eq_needed = ek;
}

#ifndef FB_LOADS
#define FB_LOADS 4
#endif
// Loads in flight per thread in the select passes. One fallback row at M=8
// N=524288 gaussian: 320 / 294 / 284us at 1 / 2 / 4 (scripts/price_fallback.py).
// They cost registers, and the fallback shares phase_c's register allocation,
// so its VGPRs set phase_c's occupancy on every call, fallback or not; that is
// what PHASE_C_OCCUPANCY below holds.
#ifndef FB_SEL_LOADS
#define FB_SEL_LOADS 4
#endif
#ifndef FB_VEC_GATHER
#define FB_VEC_GATHER 1
#endif
#ifndef FB_EARLY_EXIT
#define FB_EARLY_EXIT 1
#endif
#ifndef FB_GATHER_FAST
#define FB_GATHER_FAST 1
#endif
#ifndef FB_WIDE_STREAM
#define FB_WIDE_STREAM 1
#endif
// prefix_skip (a min/max pass over the keys) for phase_c's wide select. Under
// the wide select pass 0 is already counted during the candidate read, so the
// skip costs a read of every key and two barriers to save one scan: phase_c at
// m=4/64/128/256/512 was 9.72/11.24/9.12/9.48/12.60us with it and
// 8.20/9.76/7.68/8.04/10.08 without (scripts/wide_ab.py arms sk1/sk0). Tracking
// the min/max during the candidate read instead cost ~1us too (9.12us at m=4).
// What the skip also bought, the exit on a one-value candidate set (m=256
// n=524288 --dist inf: 6.1us with it, 11.3 without), block_select_lds_wide now
// gets from its histogram at no cost to other rows.
#ifndef PC_WIDE_SKIP
#define PC_WIDE_SKIP 0
#endif
// Ablation: force phase_c's wide select to stop after the first 12-bit pass
// (nwide_c effective = 1). Wrong results; prices the second wide pass. Plan P3
// builds a Wave0 ballot walk only if that pass costs >= 0.7 us.
#ifndef ABLATE_PC_SECOND_WIDE
#define ABLATE_PC_SECOND_WIDE 0
#endif
#ifndef ABLATE_FB_PASSES
#define ABLATE_FB_PASSES 0
#endif
#ifndef ABLATE_FB_ATOMIC
#define ABLATE_FB_ATOMIC 0
#endif
// phase_c inlines the exact fallback, so the fallback's register demand is
// phase_c's on every call. Unpinned, the fallback below took phase_c from 46-48
// to 70-82 VGPRs (8 -> 5-7 waves/SIMD) and phase_c from 13.4 to 18.2us at m=512
// n=131072 and 92 to 149us at m=4096 n=524290 with no row falling back. Pinned to
// 8 waves/SIMD it compiles to 64 VGPRs + 20-24 B/lane of scratch, and phase_c's
// no-fallback time is back within 1.4% (scripts/wide_ab.py arms x0/x1/xs4).
#ifndef PHASE_C_WAVES
#define PHASE_C_WAVES 8
#endif
#define PHASE_C_OCCUPANCY __attribute__((amdgpu_waves_per_eu(PHASE_C_WAVES)))

// Whether every key of the row whose bits at and above `sh` match `prefix` is
// one value (returned in `value`). A full-row read; all threads must call.
template <bool RAGGED>
__device__ __forceinline__ bool row_prefix_single_value(const float* __restrict__ row,
                                                        int n4,
                                                        int len,
                                                        uint32_t prefix,
                                                        int sh,
                                                        uint32_t* __restrict__ s_amm,
                                                        uint32_t& value)
{
    const int lane   = threadIdx.x & (WAVE_SIZE - 1);
    const int wv     = threadIdx.x / WAVE_SIZE;
    const int nwaves = blockDim.x / WAVE_SIZE;
    uint32_t amn = 0xFFFFFFFFu, amx = 0u;
    for(int j = threadIdx.x; j < n4; j += blockDim.x)
    {
        const vfloat4 v = load_row_f4<RAGGED>(row, j, len);
#pragma unroll
        for(int e = 0; e < FP32_EPT; e++)
        {
            const uint32_t k = fp32_to_sortable(v[e]);
            const bool act   = (!RAGGED || j * FP32_EPT + e < len) && (k >> sh) == (prefix >> sh);
            amn              = min(amn, act ? k : 0xFFFFFFFFu);
            amx              = max(amx, act ? k : 0u);
        }
    }
#pragma unroll
    for(int off = 1; off < WAVE_SIZE; off <<= 1)
    {
        amn = min(amn, (uint32_t)__shfl_xor((int)amn, off));
        amx = max(amx, (uint32_t)__shfl_xor((int)amx, off));
    }
    if(lane == 0)
    {
        s_amm[wv]                       = amn;
        s_amm[MAX_WAVES_PER_BLOCK + wv] = amx;
    }
    __syncthreads();
    uint32_t bmn = 0xFFFFFFFFu, bmx = 0u;
    for(int w = 0; w < nwaves; w++)
    {
        bmn = min(bmn, s_amm[w]);
        bmx = max(bmx, s_amm[MAX_WAVES_PER_BLOCK + w]);
    }
    __syncthreads(); // s_amm is rewritten by the next call
    value = bmn;
    return bmn == bmx;
}

// Same select but streaming the row from global memory (used by the fallback /
// direct oracle, where the row is far too large for LDS).
template <bool RAGGED>
__device__ __forceinline__ void block_select_stream(const float* __restrict__ row,
                                                    int n4,
                                                    int len,
                                                    int K,
                                                    uint32_t* __restrict__ s_hist,
                                                    uint32_t* __restrict__ s_red,
                                                    uint32_t* __restrict__ s_scan,
                                                    uint32_t& pivot,
                                                    int& eq_needed)
{
    pivot         = 0;
    int ek        = K;
    const int rep = threadIdx.x & (HIST_REP - 1);
    if(threadIdx.x == 0)
    {
        s_scan[0] = 0;
        s_scan[1] = 0;
    }
    // A tie-dense row -- the kind that overflows the candidate area and lands
    // here -- recounts its tie block on every pass: --dist equal at N=524288 spent
    // 454us of a 603us fallback row in this select (scripts/price_fallback.py).
    // When a pass's pivot bucket took EVERY key still active, the row may have
    // collapsed to one value; one read for the min and max of those keys settles
    // it, and if they agree that value is the pivot and the passes left are
    // skipped. The test is free (the bucket count and the active count are both
    // already known), so an ordinary row never pays for the read.
    //
    // The fallback's registers are phase_c's registers on every call; see
    // PHASE_C_OCCUPANCY for what that cost before it was pinned.
    __shared__ uint32_t s_amm[2 * MAX_WAVES_PER_BLOCK];
    int active = len;
    for(int p = 0; p < RADIX_PASSES; p++)
    {
        const int sh     = radix_shift(p);
        const int hshift = (p == 0) ? 0 : sh + 8;
        clear_hist(s_hist);
        __syncthreads();
        int i = threadIdx.x;
        for(; i + (FB_SEL_LOADS - 1) * (int)blockDim.x < n4; i += FB_SEL_LOADS * (int)blockDim.x)
        {
            vfloat4 v[FB_SEL_LOADS];
#pragma unroll
            for(int u = 0; u < FB_SEL_LOADS; u++)
                v[u] = load_row_f4<RAGGED>(row, i + u * blockDim.x, len);
#pragma unroll
            for(int u = 0; u < FB_SEL_LOADS; u++)
            {
                uint32_t k[FP32_EPT] = {fp32_to_sortable(v[u][0]),
                                        fp32_to_sortable(v[u][1]),
                                        fp32_to_sortable(v[u][2]),
                                        fp32_to_sortable(v[u][3])};
#pragma unroll
                for(int e = 0; e < FP32_EPT; e++)
                {
                    const int col = (i + u * (int)blockDim.x) * FP32_EPT + e;
                    if((!RAGGED || col < len) && (p == 0 || (k[e] >> hshift) == (pivot >> hshift)))
#if ABLATE_FB_ATOMIC
                        s_hist[((k[e] >> sh) & 0xFFu) * HIST_REP + rep] = 1u;
#else
                        atomicAdd(&s_hist[((k[e] >> sh) & 0xFFu) * HIST_REP + rep], 1u);
#endif
                }
            }
        }
        for(; i < n4; i += blockDim.x)
        {
            const vfloat4 v      = load_row_f4<RAGGED>(row, i, len);
            uint32_t k[FP32_EPT] = {fp32_to_sortable(v[0]),
                                    fp32_to_sortable(v[1]),
                                    fp32_to_sortable(v[2]),
                                    fp32_to_sortable(v[3])};
#pragma unroll
            for(int e = 0; e < FP32_EPT; e++)
            {
                const int col = i * FP32_EPT + e;
                if((!RAGGED || col < len) && (p == 0 || (k[e] >> hshift) == (pivot >> hshift)))
#if ABLATE_FB_ATOMIC
                    s_hist[((k[e] >> sh) & 0xFFu) * HIST_REP + rep] = 1u;
#else
                    atomicAdd(&s_hist[((k[e] >> sh) & 0xFFu) * HIST_REP + rep], 1u);
#endif
            }
        }
        __syncthreads();
        if(HIST_REP > 1)
        {
            for(int b = threadIdx.x; b < 256; b += blockDim.x)
            {
                uint32_t sum = 0;
#pragma unroll
                for(int r = 0; r < HIST_REP; r++)
                    sum += s_hist[b * HIST_REP + r];
                s_red[b] = sum;
            }
            __syncthreads();
        }
        const uint32_t* hist = HIST_REP > 1 ? s_red : s_hist;
        block_find_pivot_bucket(hist, s_scan, ek);
        const uint32_t bucket = s_scan[0];
        pivot |= (bucket << sh);
        ek -= (int)s_scan[1];
#if ABLATE_FB_PASSES > 0
        // TIMING ABLATION, WRONG RESULTS: stop after this many passes.
        if(p + 1 >= ABLATE_FB_PASSES)
            break;
#endif
        // The histogram is not rewritten before the next pass's clear barrier.
        const int in_bucket = (int)hist[bucket];
        if(FB_EARLY_EXIT && p + 1 < RADIX_PASSES && in_bucket == active)
        {
            // keys in the bucket match the pivot above bit `sh`
            uint32_t v1;
            if(row_prefix_single_value<RAGGED>(row, n4, len, pivot, sh, s_amm, v1))
            {
                pivot = v1;
                break;
            }
        }
        active = in_bucket;
    }
    eq_needed = ek;
}

// block_select_stream on a borrowed LDS scratch of FB_SCRATCH_WORDS (the fallback
// row's block never uses the candidate area it was launched with). Two changes,
// both priced on one fallback row at M=8 N=524288 (scripts/price_fallback.py):
//   - pass 0 counts into 16 replicas per bucket instead of HIST_REP: its atomics
//     were 35us (gaussian) / 83us (equal) of a 65 / 112us pass, because a
//     sortable fp32's top byte takes only a handful of values;
//   - the filtered passes are 8 + 12 + 12 bits (block_find_pivot_wide_wave0,
//     coarse buckets in s_hist, fine ones in the scratch), three full-row reads
//     instead of four, at ~37-48us a read.
// The same collapsed-row early exit as block_select_stream.
constexpr int FB_SCRATCH_WORDS = 4096;
constexpr int FB_P0_REP        = FB_SCRATCH_WORDS / 256;
static_assert(WIDE_FINE <= FB_SCRATCH_WORDS && WIDE_COARSE_SLOTS <= HIST_SLOTS,
              "the wide passes put fine buckets in the scratch and coarse ones in s_hist");
template <bool RAGGED>
__device__ __forceinline__ void block_select_stream_wide(const float* __restrict__ row,
                                                         int n4,
                                                         int len,
                                                         int K,
                                                         uint32_t* __restrict__ s_hist,
                                                         uint32_t* __restrict__ s_red,
                                                         uint32_t* __restrict__ s_x,
                                                         uint32_t* __restrict__ s_scan,
                                                         uint32_t& pivot,
                                                         int& eq_needed)
{
    __shared__ uint32_t s_amm[2 * MAX_WAVES_PER_BLOCK];
    pivot  = 0;
    int ek = K;
    if(threadIdx.x == 0)
    {
        s_scan[0] = 0;
        s_scan[1] = 0;
    }
    const int rep16 = threadIdx.x & (FB_P0_REP - 1);
    const int rep   = threadIdx.x & (HIST_REP - 1);
    using u32x4     = __attribute__((__ext_vector_type__(4))) uint32_t;
    const u32x4 z   = {0u, 0u, 0u, 0u};

    // Pass 0: bits [24, 32), all keys, 16 replicas per bucket in the scratch.
    // Plain loops (not reference-capture lambdas): the HIP compiler materialised
    // those as 24 B/lane of scratch on phase_c (b8b89180 lesson; plan P4).
    for(int j = threadIdx.x; j < FB_SCRATCH_WORDS / 4; j += blockDim.x)
        reinterpret_cast<u32x4*>(s_x)[j] = z;
    __syncthreads();
    {
        int i = threadIdx.x;
        for(; i + (FB_SEL_LOADS - 1) * (int)blockDim.x < n4; i += FB_SEL_LOADS * (int)blockDim.x)
        {
            vfloat4 v[FB_SEL_LOADS];
#pragma unroll
            for(int u = 0; u < FB_SEL_LOADS; u++)
                v[u] = load_row_f4<RAGGED>(row, i + u * blockDim.x, len);
#pragma unroll
            for(int u = 0; u < FB_SEL_LOADS; u++)
#pragma unroll
                for(int e = 0; e < FP32_EPT; e++)
                    if(!RAGGED || (i + u * (int)blockDim.x) * FP32_EPT + e < len)
                    {
                        const uint32_t k = fp32_to_sortable(v[u][e]);
                        atomicAdd(&s_x[(k >> 24) * FB_P0_REP + rep16], 1u);
                    }
        }
        for(; i < n4; i += blockDim.x)
        {
            const vfloat4 v = load_row_f4<RAGGED>(row, i, len);
#pragma unroll
            for(int e = 0; e < FP32_EPT; e++)
                if(!RAGGED || i * FP32_EPT + e < len)
                {
                    const uint32_t k = fp32_to_sortable(v[e]);
                    atomicAdd(&s_x[(k >> 24) * FB_P0_REP + rep16], 1u);
                }
        }
    }
    __syncthreads();
    for(int b = threadIdx.x; b < 256; b += blockDim.x)
    {
        uint32_t s = 0;
#pragma unroll
        for(int r = 0; r < FB_P0_REP; r++)
            s += s_x[b * FB_P0_REP + r];
        s_red[b] = s;
    }
    __syncthreads();
    block_find_pivot_bucket(s_red, s_scan, ek);
    pivot = s_scan[0] << 24;
    ek -= (int)s_scan[1];
    int in_bucket = (int)s_red[s_scan[0]];
    int active    = len;

    // Then bits [12, 24) and [0, 12).
    for(int w = 0; w < 2; w++)
    {
        const int sh = (w == 0) ? 24 : 12; // the pivot is fixed at and above bit sh
        // Every key still active sits in the bucket just picked: the row may have
        // collapsed to one value, which one more read settles.
        if(FB_EARLY_EXIT && in_bucket == active)
        {
            uint32_t v1;
            if(row_prefix_single_value<RAGGED>(row, n4, len, pivot, sh, s_amm, v1))
            {
                pivot = v1;
                break;
            }
        }
        active        = in_bucket;
        const int dsh = 12 - WIDE_BITS * w;
        __syncthreads(); // every thread has read in_bucket out of the buffers cleared next
        for(int j = threadIdx.x; j < FB_SCRATCH_WORDS / 4; j += blockDim.x)
            reinterpret_cast<u32x4*>(s_x)[j] = z;
        for(int j = threadIdx.x; j < WIDE_COARSE_SLOTS; j += blockDim.x)
            s_hist[j] = 0u;
        __syncthreads();
        {
            int i = threadIdx.x;
            for(; i + (FB_SEL_LOADS - 1) * (int)blockDim.x < n4;
                i += FB_SEL_LOADS * (int)blockDim.x)
            {
                vfloat4 v[FB_SEL_LOADS];
#pragma unroll
                for(int u = 0; u < FB_SEL_LOADS; u++)
                    v[u] = load_row_f4<RAGGED>(row, i + u * blockDim.x, len);
#pragma unroll
                for(int u = 0; u < FB_SEL_LOADS; u++)
#pragma unroll
                    for(int e = 0; e < FP32_EPT; e++)
                        if(!RAGGED || (i + u * (int)blockDim.x) * FP32_EPT + e < len)
                        {
                            const uint32_t k = fp32_to_sortable(v[u][e]);
                            if((k >> sh) == (pivot >> sh))
                            {
                                const uint32_t d = (k >> dsh) & (WIDE_FINE - 1);
                                atomicAdd(&s_hist[(d >> 6) * HIST_REP + rep], 1u);
                                atomicAdd(&s_x[d], 1u);
                            }
                        }
            }
            for(; i < n4; i += blockDim.x)
            {
                const vfloat4 v = load_row_f4<RAGGED>(row, i, len);
#pragma unroll
                for(int e = 0; e < FP32_EPT; e++)
                    if(!RAGGED || i * FP32_EPT + e < len)
                    {
                        const uint32_t k = fp32_to_sortable(v[e]);
                        if((k >> sh) == (pivot >> sh))
                        {
                            const uint32_t d = (k >> dsh) & (WIDE_FINE - 1);
                            atomicAdd(&s_hist[(d >> 6) * HIST_REP + rep], 1u);
                            atomicAdd(&s_x[d], 1u);
                        }
                    }
            }
        }
        __syncthreads();
        block_find_pivot_wide_wave0(s_hist, s_x, s_scan, ek);
        pivot |= (s_scan[0] << dsh);
        ek -= (int)s_scan[1];
        in_bucket = (int)s_x[s_scan[0]];
    }
    eq_needed = ek;
}

// block_gather_topk over a row streamed from global memory, a vector at a time:
// the element-wise form read 12.8 GB/s on one block, 164us of a 330us fallback
// row at N=524288 (scripts/price_fallback.py, ABLATE_FB=2). Same emission rule
// and the same one atomic per wave per stream; only the load shape differs.
// Always the bounds-checked loader and `col < len`, so it reads every column of
// a plain row whatever len % FP32_EPT is.
template <bool WRITE_VALUES>
__device__ __forceinline__ void block_gather_stream(const float* __restrict__ row,
                                                    int len,
                                                    int idx_base,
                                                    uint32_t pivot,
                                                    int ngt,
                                                    int eq_needed,
                                                    int* __restrict__ out,
                                                    float* __restrict__ out_val,
                                                    unsigned* __restrict__ s_wgt,
                                                    unsigned* __restrict__ s_weq)
{
    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const uint64_t lt = (1ull << lane) - 1ull;
    const int n4      = n4_cover(len);
    // Uniform trip count across the block: the ballots need every lane of a wave
    // in the same iteration.
    const int step = FB_LOADS * (int)blockDim.x;
    for(int i0 = 0; i0 < n4; i0 += step)
    {
        // Exactly ngt keys are above the pivot, so once both quotas are reserved
        // nothing left in the row can be emitted. A tie-dense row fills its
        // eq quota within the first few thousand columns; an ordinary row finds its
        // last winner near the end and gains nothing. Per wave, no barrier: a stale
        // read only stops later.
        if(FB_GATHER_FAST)
        {
            const unsigned rg = __builtin_amdgcn_readfirstlane(*(volatile unsigned*)s_wgt);
            const unsigned re = __builtin_amdgcn_readfirstlane(*(volatile unsigned*)s_weq);
            if(rg >= (unsigned)ngt && re >= (unsigned)eq_needed)
                break;
        }
        vfloat4 v[FB_LOADS];
#pragma unroll
        for(int u = 0; u < FB_LOADS; u++)
        {
            const int i = i0 + u * (int)blockDim.x + (int)threadIdx.x;
            v[u]        = i < n4 ? load_row_f4<true>(row, i, len) : vfloat4{0.f, 0.f, 0.f, 0.f};
        }
#pragma unroll
        for(int u = 0; u < FB_LOADS; u++)
        {
            const int i     = i0 + u * (int)blockDim.x + (int)threadIdx.x;
            const bool live = i < n4;
#pragma unroll
            for(int e = 0; e < FP32_EPT; e++)
            {
                const int col     = i * FP32_EPT + e;
                const bool has    = live && col < len;
                const uint32_t k  = fp32_to_sortable(v[u][e]);
                const bool gt     = has && (k > pivot);
                const bool eq     = has && (k == pivot);
                const uint64_t bg = __ballot(gt);
                const uint64_t be = __ballot(eq);
                // Almost every slot of an ordinary row has no winner in the whole wave.
                if(FB_GATHER_FAST && (bg | be) == 0ull)
                    continue;
                unsigned baseg = 0, basee = 0;
                if(lane == 0)
                {
                    if(bg)
                        baseg = atomicAdd(s_wgt, (unsigned)__popcll(bg));
                    if(be)
                        basee = atomicAdd(s_weq, (unsigned)__popcll(be));
                }
                baseg = __shfl(baseg, 0);
                basee = __shfl(basee, 0);
                if(gt)
                {
                    const unsigned p = baseg + (unsigned)__popcll(bg & lt);
                    if(p < (unsigned)ngt)
                    {
                        out_store(out + p, idx_base + col);
                        if(WRITE_VALUES)
                            out_store(out_val + p, sortable_to_fp32(k));
                    }
                }
                if(eq)
                {
                    const unsigned p = basee + (unsigned)__popcll(be & lt);
                    if(p < (unsigned)eq_needed)
                    {
                        out_store(out + ngt + p, idx_base + col);
                        if(WRITE_VALUES)
                            out_store(out_val + ngt + p, sortable_to_fp32(k));
                    }
                }
            }
        }
    }
}

// Exact full-row select for ONE row, streaming it from global memory. Shared by
// the fallback branch inside Phase C and by the standalone Phase D oracle, so
// the two can never drift apart. All threads of the block must call.
template <bool RAGGED, bool WRITE_VALUES>
__device__ __forceinline__ void exact_row_select(const float* __restrict__ input,
                                                 int pitch,
                                                 RowExtents<RAGGED> extents,
                                                 int K,
                                                 int row,
                                                 int* __restrict__ out,
                                                 float* __restrict__ out_val,
                                                 uint32_t* __restrict__ s_hist,
                                                 uint32_t* __restrict__ s_red,
                                                 uint32_t* __restrict__ s_scan,
                                                 unsigned* __restrict__ s_wgt,
                                                 unsigned* __restrict__ s_weq,
                                                 // FB_SCRATCH_WORDS of LDS, 16-byte aligned,
                                                 // that nothing else in the block is using
                                                 uint32_t* __restrict__ s_x)
{
    const int row_start = RAGGED ? extents.row_start(row, pitch) : 0;
    const int len       = row_len_of<RAGGED>(row, pitch, extents);
    const float* rif0   = input + (size_t)row * pitch + row_start;
    (void)s_x;
    if(RAGGED && len <= K)
    {
        emit_identity_row<WRITE_VALUES>(out, out_val, rif0, row_start, len, K);
        return;
    }
    const int k_out = RAGGED ? k_take_dev(K, len) : K;
    uint32_t pivot;
    int eq_needed;
    // The gather below is given `len` and scans EVERY column, so the pivot has to
    // see every column too. The plain n4 is `pitch / FP32_EPT`, which truncates:
    // the pivot search then never counts the last one to three columns, while the
    // gather still emits them, so the "greater than pivot" bucket overruns its
    // quota by exactly the number of winners living in that tail and that many
    // genuine winners are cut instead. Measured through aiter's router on plain
    // input at m=8 K=512: placing w winners in the tail loses exactly w elements
    // per row, w = 1, 2, 3 at N % 4 = 1, 2, 3, and nothing at N % 4 == 0.
    //
    // block_select_stream's RAGGED parameter only selects load_row_f4<RAGGED> and
    // the `col < len` predicate, and load_row_f4<true> reads the final partial
    // vector element-wise. So the ragged instantiation is safe on a plain row and
    // is what makes the two halves agree.
    //
    // Prices the two halves of one fallback row. WRONG RESULTS, pricing only:
    // 1 = the select without the gather, 2 = the gather without the select.
#ifndef ABLATE_FB
#define ABLATE_FB 0
#endif
#if ABLATE_FB == 2
    pivot     = 0xFFFFFFFFu;
    eq_needed = 0;
    if(false)
#endif
        if(!RAGGED && (len % FP32_EPT) != 0)
        {
#if FB_WIDE_STREAM
            block_select_stream_wide<true>(
                rif0, n4_cover(len), len, k_out, s_hist, s_red, s_x, s_scan, pivot, eq_needed);
#else
        block_select_stream<true>(
            rif0, n4_cover(len), len, k_out, s_hist, s_red, s_scan, pivot, eq_needed);
#endif
        }
        else
        {
            const int n4 = RAGGED ? n4_cover(len) : (pitch / FP32_EPT);
#if FB_WIDE_STREAM
            block_select_stream_wide<RAGGED>(
                rif0, n4, len, k_out, s_hist, s_red, s_x, s_scan, pivot, eq_needed);
#else
        block_select_stream<RAGGED>(rif0, n4, len, k_out, s_hist, s_red, s_scan, pivot, eq_needed);
#endif
        }
    if(threadIdx.x == 0)
    {
        *s_wgt = 0;
        *s_weq = 0;
    }
    __syncthreads();
#if ABLATE_FB == 1
    return;
#endif
#if FB_VEC_GATHER
    block_gather_stream<WRITE_VALUES>(
        rif0, len, row_start, pivot, k_out - eq_needed, eq_needed, out, out_val, s_wgt, s_weq);
#else
    const float* rif = rif0;
    block_gather_topk<WRITE_VALUES>(
        len,
        pivot,
        k_out - eq_needed,
        eq_needed,
        out,
        out_val,
        s_wgt,
        s_weq,
        [&](int i) { return fp32_to_sortable(rif[i]); },
        [&](int i) { return row_start + i; });
#endif
    if(RAGGED && k_out < K)
    {
        __syncthreads();
        pad_topk_tail<WRITE_VALUES>(out, out_val, k_out, K);
    }
}

#include "topk_sampled/topk_generalize.hip.hpp"

// ---------------------------------------------------------------------------
// Phase A: per-row sampled threshold, fully in LDS, one kernel, one block/row.
// ---------------------------------------------------------------------------
// Also clears the counters the later phases accumulate into. Phase A already
// runs one block per row and completes before Phase B on the same stream, so
// this is free, where a hipMemsetAsync per counter was a full dispatch each
// (~2.6 us) -- 5 of the 9 dispatches on the decode path were memsets.
// cand_reserved / cand_bad may be null on the paths that do not reserve.
// COMPACT selects the active-set-compacting select instead of the filter-rescan
// one. It is a template parameter and not a kernarg on purpose: one unused
// kernarg on phase_small_n_topk alone cost +1.0% of its geomean
// (knowledge/known_bad.md), so an A/B knob must compile out entirely.
//
// KPT is keys-per-thread for the register-resident path (4, 8 or 16). KPT==0 is
// the LDS-keys path. COMPACT requires KPT==0 (compaction rewrites s_keys).
#ifndef PHASE_A_WAVES
#define PHASE_A_WAVES 8
#endif
#define PHASE_A_OCCUPANCY __attribute__((amdgpu_waves_per_eu(PHASE_A_WAVES)))
template <bool RAGGED, bool COMPACT = false, int KPT = 0>
__global__ __launch_bounds__(1024) PHASE_A_OCCUPANCY
    void phase_a_threshold(const float* __restrict__ input,
                           int pitch,
                           RowExtents<RAGGED> extents,
                           int rank,
                           int S,
                           int npasses,
                           int chunk_stride_host,
                           uint32_t* __restrict__ threshold,
                           float* __restrict__ threshold_f,
                           unsigned int* __restrict__ cand_reserved,
                           unsigned int* __restrict__ cand_bad,
                           int* __restrict__ fb_count,
                           int K,
                           int nwide)
{
    static_assert(!COMPACT || KPT == 0, "compact select needs LDS keys");
    const int row   = blockIdx.x;
    const int len   = row_len_of<RAGGED>(row, pitch, extents);
    const float* ri = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);

    if(threadIdx.x == 0)
    {
#if CTR_STRIDE_MUTANT
        // Gate self-test only: clears at the unpadded index. Must turn the gate red.
        if(cand_reserved)
            cand_reserved[row] = 0u;
        if(cand_bad)
            cand_bad[row] = 0u;
#else
        if(cand_reserved)
            cand_reserved[(size_t)row * CTR_STRIDE] = 0u;
        if(cand_bad)
            cand_bad[(size_t)row * CTR_STRIDE] = 0u;
#endif
        if(row == 0)
            *fb_count = 0;
    }

    // Two separate reasons a row cannot go through the sampler, and they do not
    // coincide: len <= K means every element is selected so there is nothing to
    // rank (aiter's identity case), while len < S means the sampler would read
    // past the row. Either way Phase B must collect nothing for this row, so the
    // threshold is +inf and Phase C takes it through the exact/identity path.
    if(RAGGED && (len <= K || len < S))
    {
        // +inf is the whole routing signal: Phase B keeps nothing below it, so
        // cand_count lands under k_out and Phase C takes it through the
        // exact/identity path. Deliberately NOT appended to fb_rows here -- Phase C
        // appends every row it routes, and doing it in both places counted a short
        // row twice, overflowing the M-entry fb_rows (M=512 triangular reported
        // fallback_rows=1024 and wrote 512 ints past the end of the buffer).
        if(threadIdx.x == 0)
        {
            threshold[row]   = 0u;
            threshold_f[row] = __builtin_inff();
        }
        return;
    }

    extern __shared__ uint32_t s_dyn[];
    // LDS path: [S keys][nwide * WIDE_WORDS]. Reg path: [nwide * WIDE_WORDS] only.
    uint32_t* s_keys = s_dyn;
    uint32_t* s_wide = (KPT > 0) ? s_dyn : (s_keys + S);
    __shared__ __align__(16) uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_red[256];
    __shared__ uint32_t s_scan[2];
    __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];

    const int chunks       = S / SAMPLE_CHUNK_ELEMS;
    const int chunk_stride = RAGGED ? sample_chunk_stride(len, chunks) : chunk_stride_host;
    const int rank_row = RAGGED && len != pitch ? max(1, (int)((double)rank * pitch / len)) : rank;

    const int v4_per_chunk = SAMPLE_CHUNK_ELEMS / FP32_EPT;
    const int total_v4     = S / FP32_EPT;
#ifndef PA_FOLD
#define PA_FOLD 1
#endif
#if PA_FOLD
    const int fold_rep = threadIdx.x & (HIST_REP - 1);
    clear_hist(s_hist);
    __syncthreads();
#endif
    if(nwide > 0)
        clear_wide(s_wide, wide_buffer_count(nwide, false));

    // Register-resident keys for KPT>0. Filled in the same strided v4 order the
    // LDS path uses, so each thread's keys[t] is exactly what s_keys[tid+...] held.
    uint32_t keys[KPT > 0 ? KPT : 1];
    if constexpr(KPT > 0)
    {
#pragma unroll
        for(int t = 0; t < KPT; t++)
            keys[t] = 0u;
    }
    // Convert one v4 of samples: keys into registers (slot t) or LDS (index u),
    // and pass 0's digits folded into s_hist.
    auto take_v4 = [&](const vfloat4& v, int u, int t) {
        const uint32_t k0 = fp32_to_sortable(v[0]);
        const uint32_t k1 = fp32_to_sortable(v[1]);
        const uint32_t k2 = fp32_to_sortable(v[2]);
        const uint32_t k3 = fp32_to_sortable(v[3]);
        if constexpr(KPT > 0)
        {
            keys[t * FP32_EPT + 0] = k0;
            keys[t * FP32_EPT + 1] = k1;
            keys[t * FP32_EPT + 2] = k2;
            keys[t * FP32_EPT + 3] = k3;
        }
        else
        {
            (void)t;
            const int base   = u * FP32_EPT;
            s_keys[base + 0] = k0;
            s_keys[base + 1] = k1;
            s_keys[base + 2] = k2;
            s_keys[base + 3] = k3;
        }
#if PA_FOLD
        // radix_shift(0) is 24, so pass 0's digit is the top byte.
        // PA_REG_MUTANT drops keys[0] in the select; drop the matching fold count
        // here so the prefilled hist stays consistent with the mutant select.
        const bool drop0 = (PA_REG_MUTANT && KPT > 0 && t == 0);
        if(!drop0)
            atomicAdd(&s_hist[(k0 >> 24) * HIST_REP + fold_rep], 1u);
        atomicAdd(&s_hist[(k1 >> 24) * HIST_REP + fold_rep], 1u);
        atomicAdd(&s_hist[(k2 >> 24) * HIST_REP + fold_rep], 1u);
        atomicAdd(&s_hist[(k3 >> 24) * HIST_REP + fold_rep], 1u);
#endif
    };
    auto load_v4 = [&](int u) {
        const int chunk = u / v4_per_chunk;
        const int off4  = u % v4_per_chunk;
        return *(reinterpret_cast<const vfloat4*>(ri + (size_t)chunk * chunk_stride) + off4);
    };
    // Every sample load a thread owns issues before the first is converted. The
    // plain loop kept each load beside its use: 4-5 serial global round trips
    // per thread (ISA: load, s_waitcnt vmcnt(0), convert, next load).
    if constexpr(KPT > 0)
    {
        // S == KPT * blockDim.x here, so each thread owns exactly KPT/4 v4s.
        constexpr int NV = KPT / FP32_EPT;
        vfloat4 vv[NV];
#pragma unroll
        for(int t = 0; t < NV; t++)
            vv[t] = load_v4((int)threadIdx.x + t * (int)blockDim.x);
#pragma unroll
        for(int t = 0; t < NV; t++)
            take_v4(vv[t], (int)threadIdx.x + t * (int)blockDim.x, t);
    }
    else
    {
        constexpr int NB = 4;
        const int bd     = (int)blockDim.x;
        for(int u0 = (int)threadIdx.x; u0 < total_v4; u0 += NB * bd)
        {
            vfloat4 vv[NB];
#pragma unroll
            for(int t = 0; t < NB; t++)
                if(u0 + t * bd < total_v4)
                    vv[t] = load_v4(u0 + t * bd);
#pragma unroll
            for(int t = 0; t < NB; t++)
                if(u0 + t * bd < total_v4)
                    take_v4(vv[t], u0 + t * bd, t);
        }
    }
    __syncthreads();

    uint32_t pivot;
    int eq_needed;
#ifndef ABLATE_PA
#define ABLATE_PA 0
#endif
#if ABLATE_PA
    pivot     = (KPT > 0) ? keys[0] : s_keys[threadIdx.x % S];
    eq_needed = 1;
    (void)rank_row;
    (void)npasses;
#else
    if constexpr(KPT > 0)
    {
        if(nwide > 0)
        {
            block_select_reg_wide<KPT>(
                keys, rank_row, s_hist, s_wide, s_scan, pivot, eq_needed, nwide, PA_FOLD != 0);
        }
        else
        {
            block_select_reg<KPT>(
                keys, rank_row, s_hist, s_red, s_scan, pivot, eq_needed, npasses, PA_FOLD != 0);
        }
    }
    else if constexpr(COMPACT)
    {
        block_select_lds_compact(
            s_keys, S, rank_row, s_hist, s_red, s_scan, pivot, eq_needed, npasses);
    }
    else if(nwide > 0)
    {
        block_select_lds_wide<false>(s_keys,
                                     S,
                                     rank_row,
                                     s_hist,
                                     s_wide,
                                     s_scan,
                                     s_mm,
                                     pivot,
                                     eq_needed,
                                     nwide,
                                     false,
                                     PA_FOLD != 0);
    }
    else
    {
        block_select_lds(s_keys,
                         S,
                         rank_row,
                         s_hist,
                         s_red,
                         s_scan,
                         s_mm,
                         pivot,
                         eq_needed,
                         npasses,
                         false,
                         PA_FOLD != 0);
    }
#endif
    if(threadIdx.x == 0)
    {
        threshold[row]   = pivot;
        threshold_f[row] = sortable_to_fp32(pivot);
    }
}

// Variant 3: wave-private output regions, so Phase B has NO atomic of any kind
// (variant 2 still paid ~512 LDS atomics per row on one address). Each wave
// keeps a wave-uniform register counter and writes into its own slice. Key and
// index go out as one packed 64-bit store instead of two 32-bit streams.
// Overflow of a slice is detected and sends the row to the exact fallback.
// ablate: 0 = normal, 1 = skip the candidate stores (loads/compares stay live
// via wcnt), 2 = skip the compaction entirely and only consume the loads.
// Used to attribute Phase B's gap to its own read floor.
template <int ABLATE, bool RAGGED>
__global__ void phase_b_filter_waveseg(const float* __restrict__ input,
                                       int pitch,
                                       RowExtents<RAGGED> extents,
                                       const float* __restrict__ threshold_f,
                                       uint64_t* __restrict__ cand_pack,
                                       int* __restrict__ cand_seg,
                                       unsigned int* __restrict__ cand_count,
                                       int seg_stride)
{
    const int row   = blockIdx.x;
    const int len   = row_len_of<RAGGED>(row, pitch, extents);
    const float* ri = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);
    const float th  = threshold_f[row];

    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const int wid     = threadIdx.x / WAVE_SIZE;
    const int nwaves  = blockDim.x / WAVE_SIZE;
    const uint64_t lt = (1ull << lane) - 1ull;

    uint64_t* seg = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW + (size_t)wid * seg_stride;

    const int n4     = RAGGED ? n4_cover(len) : (pitch / FP32_EPT);
    const int stride = blockDim.x;
    const int iters  = (n4 + stride - 1) / stride;

    int wcnt      = 0;
    bool overflow = false;

    for(int it = 0; it < iters; it++)
    {
        const int i     = it * stride + threadIdx.x;
        vfloat4 v       = {0.f, 0.f, 0.f, 0.f};
        const bool live = (i < n4);
        if(live)
            v = load_row_f4<RAGGED>(ri, i, len);
        const int base_idx = i * FP32_EPT;

        const uint64_t b0 = __ballot(live && !(v[0] < th) && (!RAGGED || base_idx + 0 < len));
        const uint64_t b1 = __ballot(live && !(v[1] < th) && (!RAGGED || base_idx + 1 < len));
        const uint64_t b2 = __ballot(live && !(v[2] < th) && (!RAGGED || base_idx + 2 < len));
        const uint64_t b3 = __ballot(live && !(v[3] < th) && (!RAGGED || base_idx + 3 < len));
        const int t0      = __popcll(b0);
        const int t1      = t0 + __popcll(b1);
        const int t2      = t1 + __popcll(b2);
        const int wtotal  = t2 + __popcll(b3);

        if(ABLATE == 2)
        {
            wcnt += wtotal;
            continue;
        }

        if(wtotal > 0)
        {
            if(b0 & (1ull << lane))
            {
                int p = wcnt + __popcll(b0 & lt);
                if(ABLATE == 0 && p < seg_stride)
                    seg[p] = ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
            }
            if(b1 & (1ull << lane))
            {
                int p = wcnt + t0 + __popcll(b1 & lt);
                if(ABLATE == 0 && p < seg_stride)
                    seg[p] = ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
            }
            if(b2 & (1ull << lane))
            {
                int p = wcnt + t1 + __popcll(b2 & lt);
                if(ABLATE == 0 && p < seg_stride)
                    seg[p] = ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
            }
            if(b3 & (1ull << lane))
            {
                int p = wcnt + t2 + __popcll(b3 & lt);
                if(ABLATE == 0 && p < seg_stride)
                    seg[p] = ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
            }
            wcnt += wtotal;
            if(wcnt > seg_stride)
                overflow = true;
        }
    }

    __shared__ int s_seg[MAX_WAVES_PER_BLOCK];
    if(lane == 0)
        s_seg[wid] = overflow ? -1 : wcnt;
    __syncthreads();
    if(threadIdx.x == 0)
    {
        unsigned total = 0;
        bool bad       = false;
        for(int w = 0; w < nwaves; w++)
        {
            cand_seg[(size_t)row * MAX_WAVES_PER_BLOCK + w] = s_seg[w];
            if(s_seg[w] < 0)
                bad = true;
            else
                total += (unsigned)s_seg[w];
        }
        // 0xFFFFFFFF is unconditionally > PHASE_C_CAP, so Phase C routes it to the
        // exact fallback without needing a separate flag.
        cand_count[row] = bad ? 0xFFFFFFFFu : total;
    }
}

// Variant 4: same wave-private regions, but passers are staged in LDS and
// flushed only once the wave holds at least a full wave's worth, so the global
// writes go out as ~520 B contiguous bursts instead of ~45 B fragments.
//
// Ablation on variant 3 showed the candidate stores cost 139 us while the whole
// load+compare+compact path cost only 13 us over its 349 us read floor: a wave
// produces ~5.6 passers per iteration, so each 8 B-per-passer burst touched a
// 128 B line far below full width.
//
// The flush drains the WHOLE buffer, so no remainder has to be shifted down and
// wcnt simply stays unaligned; a 520 B contiguous burst spans 5 lines instead of
// 4, which is a boundary effect rather than per-element amplification.
// WSTAGE_* constants live in topk_generalize.hip.hpp
template <bool RAGGED>
__global__
    __launch_bounds__(512) void phase_b_filter_wavestage(const float* __restrict__ input,
                                                         int pitch,
                                                         RowExtents<RAGGED> extents,
                                                         const float* __restrict__ threshold_f,
                                                         uint64_t* __restrict__ cand_pack,
                                                         int* __restrict__ cand_seg,
                                                         unsigned int* __restrict__ cand_count,
                                                         int seg_stride)
{
#if BLOCK_TSTAMP
    const unsigned long long t_begin = __builtin_amdgcn_s_memrealtime();
#endif
    const int row   = blockIdx.x;
    const int len   = row_len_of<RAGGED>(row, pitch, extents);
    const float* ri = input + (size_t)row * pitch + (RAGGED ? extents.row_start(row, pitch) : 0);
    const float th  = threshold_f[row];

    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const int wid     = threadIdx.x / WAVE_SIZE;
    const int nwaves  = blockDim.x / WAVE_SIZE;
    const uint64_t lt = (1ull << lane) - 1ull;

    __shared__ uint64_t wbuf[WSTAGE_WAVES * WSTAGE_CAP];
    uint64_t* buf = wbuf + (size_t)wid * WSTAGE_CAP;
    uint64_t* seg = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW + (size_t)wid * seg_stride;

    const int n4     = RAGGED ? n4_cover(len) : (pitch / FP32_EPT);
    const int stride = blockDim.x;
    const int iters  = (n4 + stride - 1) / stride;

    int wcnt      = 0;
    int bcnt      = 0;
    bool overflow = false;

    for(int it = 0; it < iters; it++)
    {
        const int i     = it * stride + threadIdx.x;
        vfloat4 v       = {0.f, 0.f, 0.f, 0.f};
        const bool live = (i < n4);
        if(live)
            v = load_row_f4<RAGGED>(ri, i, len);

        const int base_idx = i * FP32_EPT;
        const uint64_t b0  = __ballot(live && !(v[0] < th) && (!RAGGED || base_idx + 0 < len));
        const uint64_t b1  = __ballot(live && !(v[1] < th) && (!RAGGED || base_idx + 1 < len));
        const uint64_t b2  = __ballot(live && !(v[2] < th) && (!RAGGED || base_idx + 2 < len));
        const uint64_t b3  = __ballot(live && !(v[3] < th) && (!RAGGED || base_idx + 3 < len));
        const int t0       = __popcll(b0);
        const int t1       = t0 + __popcll(b1);
        const int t2       = t1 + __popcll(b2);
        const int wtotal   = t2 + __popcll(b3);

        if(wtotal > 0)
        {
            if(b0 & (1ull << lane))
                buf[bcnt + __popcll(b0 & lt)] =
                    ((uint64_t)__float_as_uint(v[0]) << 32) | (uint32_t)(base_idx + 0);
            if(b1 & (1ull << lane))
                buf[bcnt + t0 + __popcll(b1 & lt)] =
                    ((uint64_t)__float_as_uint(v[1]) << 32) | (uint32_t)(base_idx + 1);
            if(b2 & (1ull << lane))
                buf[bcnt + t1 + __popcll(b2 & lt)] =
                    ((uint64_t)__float_as_uint(v[2]) << 32) | (uint32_t)(base_idx + 2);
            if(b3 & (1ull << lane))
                buf[bcnt + t2 + __popcll(b3 & lt)] =
                    ((uint64_t)__float_as_uint(v[3]) << 32) | (uint32_t)(base_idx + 3);
            bcnt += wtotal;
        }

        // Wave-uniform: every lane has the same bcnt.
        if(bcnt >= WAVE_SIZE)
        {
            __builtin_amdgcn_wave_barrier();
            if(wcnt + bcnt <= seg_stride)
            {
                for(int j = lane; j < bcnt; j += WAVE_SIZE)
                    seg[wcnt + j] = buf[j];
            }
            else
            {
                overflow = true;
            }
            wcnt += bcnt;
            bcnt = 0;
        }
    }

    if(bcnt > 0)
    {
        __builtin_amdgcn_wave_barrier();
        if(wcnt + bcnt <= seg_stride)
        {
            for(int j = lane; j < bcnt; j += WAVE_SIZE)
                seg[wcnt + j] = buf[j];
        }
        else
        {
            overflow = true;
        }
        wcnt += bcnt;
    }

    __shared__ int s_seg[MAX_WAVES_PER_BLOCK];
    if(lane == 0)
        s_seg[wid] = overflow ? -1 : wcnt;
    __syncthreads();
    if(threadIdx.x == 0)
    {
        unsigned total = 0;
        bool bad       = false;
        for(int w = 0; w < nwaves; w++)
        {
            cand_seg[(size_t)row * MAX_WAVES_PER_BLOCK + w] = s_seg[w];
            if(s_seg[w] < 0)
                bad = true;
            else
                total += (unsigned)s_seg[w];
        }
        cand_count[row] = bad ? 0xFFFFFFFFu : total;
    }
#if BLOCK_TSTAMP
    __syncthreads();
    if(threadIdx.x == 0)
    {
        const unsigned long long t_end   = __builtin_amdgcn_s_memrealtime();
        d_bt[(size_t)blockIdx.x * 2]     = t_begin;
        d_bt[(size_t)blockIdx.x * 2 + 1] = t_end;
    }
#endif
}

// Phase C fed by the wave-segmented layout: gathers the variable-length
// per-wave segments into one contiguous LDS array, then selects exactly.
//
// STATIC_CAP selects where the candidate staging area lives:
//   true  -> two __shared__ arrays at PHASE_C_CAP, so both base addresses are
//            compile-time constants. This is the shape the main config uses.
//   false -> one dynamic-LDS block split at the runtime `cap`, needed when cap
//            exceeds PHASE_C_CAP.
// Sizing the arrays statically at PHASE_C_CAP_MAX instead costs 32 KB of LDS
// unconditionally and halves occupancy (measured 0.6215 -> 0.6702 ms); making
// the main config pay the runtime split instead costs the +0.6% that the
// runtime base offset adds (measured 0.6254 vs 0.6215 ms).
template <bool STATIC_CAP, bool RAGGED, bool WRITE_VALUES>
__global__ PHASE_C_OCCUPANCY void
phase_c_select_waveseg(const float* __restrict__ input,
                       int pitch,
                       RowExtents<RAGGED> extents,
                       const uint64_t* __restrict__ cand_pack,
                       const int* __restrict__ cand_seg,
                       const unsigned int* __restrict__ cand_count,
                       int seg_stride,
                       int nwaves_b,
                       int K,
                       int cap,
                       TopkOut<WRITE_VALUES> dst,
                       int* __restrict__ fb_rows,
                       int* __restrict__ fb_count,
                       int npasses,
                       int nwide)
{
    const int row            = blockIdx.x;
    const int row_start      = RAGGED ? extents.row_start(row, pitch) : 0;
    const int len            = row_len_of<RAGGED>(row, pitch, extents);
    const unsigned int c_raw = cand_count[row];
    const int k_out          = RAGGED ? k_take_dev(K, len) : K;

    extern __shared__ uint32_t s_dyn[];
    __shared__ __align__(16) uint32_t s_keys_st[STATIC_CAP ? PHASE_C_CAP : 1];
    __shared__ int s_idx_st[STATIC_CAP ? PHASE_C_CAP : 1];
    uint32_t* s_keys = STATIC_CAP ? s_keys_st : s_dyn;
    int* s_idx       = STATIC_CAP ? s_idx_st : reinterpret_cast<int*>(s_dyn + cap);
    // nwide * WIDE_WORDS words after whatever dynamic LDS the candidates use.
    uint32_t* s_wide = s_dyn + (STATIC_CAP ? 0 : 2 * cap);
    __shared__ __align__(16) uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_red[256];
    __shared__ uint32_t s_scan[2];
    __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];
    __shared__ int s_cnt[MAX_WAVES_PER_BLOCK];
    __shared__ int s_off[MAX_WAVES_PER_BLOCK];
    __shared__ unsigned s_wgt, s_weq;

    int* out_row   = dst.idx_row(row, K);
    float* val_row = dst.val_row(row, K);
    // `len <= K` is routed unconditionally rather than via cand_count, so the
    // identity emit does not depend on how many candidates Phase B happened to
    // keep: under the `inf` distribution a row of +inf values passes the +inf
    // threshold and can push cand_count above k_out, which would otherwise send
    // an identity row down the candidate path and order it differently to aiter.
    if((RAGGED && len <= K) || c_raw < (unsigned)k_out || c_raw > (unsigned)cap)
    {
        if(threadIdx.x == 0)
            fb_rows[atomicAdd(fb_count, 1)] = row;
        exact_row_select<RAGGED, WRITE_VALUES>(input,
                                               pitch,
                                               extents,
                                               K,
                                               row,
                                               out_row,
                                               val_row,
                                               s_hist,
                                               s_red,
                                               s_scan,
                                               &s_wgt,
                                               &s_weq,
                                               STATIC_CAP ? s_keys_st : s_dyn);
        return;
    }
    const int c = (int)c_raw;

    if(threadIdx.x < nwaves_b)
        s_cnt[threadIdx.x] = cand_seg[(size_t)row * MAX_WAVES_PER_BLOCK + threadIdx.x];
    __syncthreads();
    if(threadIdx.x == 0)
    {
        int t = 0;
        for(int w = 0; w < nwaves_b; w++)
        {
            s_off[w] = t;
            t += s_cnt[w];
        }
        s_wgt = 0;
        s_weq = 0;
    }
    __syncthreads();

    // Published by the barrier after the candidate copy.
    if(nwide > 0)
        clear_wide(s_wide, wide_buffer_count(nwide, false));
    const uint64_t* base = cand_pack + (size_t)row * CAND_SLOTS_PER_ROW;
    for(int w = 0; w < nwaves_b; w++)
    {
        const int cnt = s_cnt[w];
        const int off = s_off[w];
        for(int i = threadIdx.x; i < cnt; i += blockDim.x)
        {
            uint64_t p      = base[(size_t)w * seg_stride + i];
            s_keys[off + i] = fp32_to_sortable_bits((uint32_t)(p >> 32));
            s_idx[off + i]  = (int)(uint32_t)p;
        }
    }
    __syncthreads();

    uint32_t pivot;
    int eq_needed;
    if(nwide > 0)
        block_select_lds_wide<false>(s_keys,
                                     c,
                                     k_out,
                                     s_hist,
                                     s_wide,
                                     s_scan,
                                     s_mm,
                                     pivot,
                                     eq_needed,
                                     nwide,
                                     /*prefix_skip=*/PC_WIDE_SKIP != 0,
                                     false);
    else
        block_select_lds(s_keys,
                         c,
                         k_out,
                         s_hist,
                         s_red,
                         s_scan,
                         s_mm,
                         pivot,
                         eq_needed,
                         npasses,
                         /*prefix_skip=*/true);

    if constexpr(RAGGED)
    {
        block_gather_topk_ragged<WRITE_VALUES>(c,
                                               pivot,
                                               k_out - eq_needed,
                                               eq_needed,
                                               out_row,
                                               val_row,
                                               &s_wgt,
                                               &s_weq,
                                               s_keys,
                                               s_idx,
                                               row_start);
    }
    else
    {
        block_gather_topk_ptrs<WRITE_VALUES>(c,
                                             pivot,
                                             k_out - eq_needed,
                                             eq_needed,
                                             out_row,
                                             val_row,
                                             &s_wgt,
                                             &s_weq,
                                             s_keys,
                                             s_idx);
    }
    if(RAGGED && k_out < K)
    {
        __syncthreads();
        pad_topk_tail<WRITE_VALUES>(out_row, val_row, k_out, K);
    }
}

// ---------------------------------------------------------------------------
// Phase D: exact full-row select over a compacted row list. Now used only by
// --pipeline direct as the independent oracle; the fused path folds the same
// work into Phase C to save a dispatch.
// ---------------------------------------------------------------------------
template <bool RAGGED, bool WRITE_VALUES>
__global__ __launch_bounds__(1024) void phase_d_fallback(const float* __restrict__ input,
                                                         int pitch,
                                                         RowExtents<RAGGED> extents,
                                                         int K,
                                                         const int* __restrict__ fb_rows,
                                                         const int* __restrict__ fb_count,
                                                         TopkOut<WRITE_VALUES> dst)
{
    const int count = *fb_count;

    __shared__ __align__(16) uint32_t s_fbx[FB_SCRATCH_WORDS];
    __shared__ __align__(16) uint32_t s_hist[HIST_SLOTS];
    __shared__ uint32_t s_red[256];
    __shared__ uint32_t s_scan[2];
    __shared__ unsigned s_wgt;
    __shared__ unsigned s_weq;

    for(int slot = blockIdx.y; slot < count; slot += gridDim.y)
    {
        const int row = fb_rows[slot];
        const int len = row_len_of<RAGGED>(row, pitch, extents);
        exact_row_select<RAGGED, WRITE_VALUES>(input,
                                               pitch,
                                               extents,
                                               K,
                                               row,
                                               dst.idx_row(row, K),
                                               dst.val_row(row, K),
                                               s_hist,
                                               s_red,
                                               s_scan,
                                               &s_wgt,
                                               &s_weq,
                                               s_fbx);
        __syncthreads();
    }
}

__global__ void fill_identity_rows(int* rows, int* count, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < M)
        rows[i] = i;
    if(i == 0)
        *count = M;
}

__global__ void fill_random_fp32(float* data, size_t count, unsigned int seed, int mode, int N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for(size_t i = idx; i < count; i += (size_t)gridDim.x * blockDim.x)
    {
        unsigned int h = (unsigned int)i ^ seed;
        h ^= h >> 16;
        h *= 0x45d9f3bu;
        h ^= h >> 16;
        h *= 0x45d9f3bu;
        h ^= h >> 16;
        float v;
        if(mode == 1)
        {
            float u1 = (h & 0xFFFF) / 65535.f + 1e-6f;
            float u2 = ((h >> 16) & 0xFFFF) / 65535.f;
            v        = sqrtf(-2.f * logf(u1)) * cosf(2.f * 3.14159265f * u2);
        }
        else if(mode == 2)
        {
            v = 1.0f;
        }
        else if(mode == 3)
        {
            v = ((i & 0xFF) == 0) ? __int_as_float(0x7f800000) : (float)(i & 0xFFFF) * 1e-6f;
        }
        else if(mode == 4)
        {
            int col = (int)(i % (size_t)N);
            v       = (col >= N - 3000) ? 100.f + (float)(i & 0xF) : (float)(i & 0xFFFF) * 1e-6f;
        }
        else
        {
            v = ((int)(h & 0xFFFF) - 32768) / 32768.f;
        }
        data[i] = v;
    }
}

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------
struct Bufs
{
    uint32_t* threshold;
    float* threshold_f;
    uint64_t* cand_pack;
    int* cand_seg;
    unsigned int* cand_count;
    unsigned int* cand_reserved;
    unsigned int* cand_bad;
    int* fb_rows;
    int* fb_count;
    int C_alloc;
};

static void alloc_bufs(Bufs& b, int M, int K, int cap)
{
    b.C_alloc           = cap;
    const int row_slots = std::max(CAND_SLOTS_PER_ROW, cap);
    HIP_CHECK(hipMalloc(&b.threshold, (size_t)M * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&b.threshold_f, (size_t)M * sizeof(float)));
    HIP_CHECK(hipMalloc(&b.cand_pack, (size_t)M * row_slots * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&b.cand_seg, (size_t)M * MAX_WAVES_PER_BLOCK * sizeof(int)));
    HIP_CHECK(hipMalloc(&b.cand_count, (size_t)M * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&b.cand_reserved, (size_t)M * CTR_STRIDE * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&b.cand_bad, (size_t)M * CTR_STRIDE * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc(&b.fb_rows, (size_t)M * sizeof(int)));
    HIP_CHECK(hipMalloc(&b.fb_count, sizeof(int)));
    (void)K;
}

static void free_bufs(Bufs& b)
{
    (void)hipFree(b.threshold);
    (void)hipFree(b.threshold_f);
    (void)hipFree(b.cand_pack);
    (void)hipFree(b.cand_seg);
    (void)hipFree(b.cand_count);
    (void)hipFree(b.cand_reserved);
    (void)hipFree(b.cand_bad);
    (void)hipFree(b.fb_rows);
    (void)hipFree(b.fb_count);
}

// Block size for the small_n path. Sized from the dwordx4 LOAD count (n4 = N/4),
// not from N: sizing it from N gave a 16-wave block where only N/4 of the lanes
// ever issued a load (768 of 1024 idle at N=1024) and all 16 waves still paid
// the 4-pass x 4-barrier select over just 1024 elements.
//
// Lower bound 256 is a correctness constraint, not a tuning choice:
// block_find_pivot_bucket() indexes the 256 radix buckets by threadIdx.x, so a
// block under 256 threads silently drops the upper buckets and picks a wrong
// pivot.
// Sizing it from the load count (one dwordx4 per lane) is NOT optimal: measured
// M=4096 N=2048 wants 256 threads (39.7 us) where the load rule picks 512
// (50.2 us). What actually decides it is how many WAVES end up resident per CU,
// because a row's cost is dominated by the fixed 4-pass x 4-barrier select, not
// by its loads. Blocks per CU is capped by the dynamic LDS (the row itself), so
// at large N only a bigger block can supply enough waves, while at small N the
// cap is loose and the cheapest block wins.
//
// The row itself is the dynamic LDS here, and the load cap matters because a
// small row cannot occupy many waves: sizing purely from the load count picked a
// 512-thread block at M=4096 N=2048 (50.2 us) where 256 measures 39.7 us.
static int small_n_block(int M, int N)
{
    if(g_small_n_block > 0)
        return std::max(256, std::min(1024, g_small_n_block));
    const int t = small_n_threads_from_table(M, N);
    if(t > 0)
        return t;
    return occupancy_block_threads(
        M, SMALL_N_STATIC_LDS + N * (int)sizeof(uint32_t), std::max(1, (N / FP32_EPT) / WAVE_SIZE));
}

// Builds the output bundle for the requested instantiation. The WRITE_VALUES
// =false form ignores d_val, so a stray non-null pointer cannot quietly turn
// into stores the caller did not ask for.
template <bool WRITE_VALUES>
static TopkOut<WRITE_VALUES> make_topk_out(int* d_idx, float* d_val);

template <>
TopkOut<false> make_topk_out<false>(int* d_idx, float*)
{ return TopkOut<false>{d_idx}; }

template <>
TopkOut<true> make_topk_out<true>(int* d_idx, float* d_val)
{ return TopkOut<true>{d_idx, d_val}; }

template <bool RAGGED>
static RowExtents<RAGGED> make_row_extents(const int* d_starts, const int* d_ends);
template <>
RowExtents<false> make_row_extents<false>(const int*, const int*)
{ return RowExtents<false>{nullptr}; }
template <>
RowExtents<true> make_row_extents<true>(const int* d_starts, const int* d_ends)
{ return RowExtents<true>{d_starts, d_ends}; }

template <bool RAGGED, bool WRITE_VALUES>
static void topk_small_n(const float* d_in,
                         int M,
                         int pitch,
                         const int* d_row_starts,
                         const int* d_row_ends,
                         int K,
                         int* d_idx,
                         float* d_val,
                         hipStream_t s)
{
    phase_small_n_topk<RAGGED, WRITE_VALUES>
        <<<M, small_n_block(M, pitch), (size_t)pitch * sizeof(uint32_t), s>>>(
            d_in,
            pitch,
            make_row_extents<RAGGED>(d_row_starts, d_row_ends),
            K,
            make_topk_out<WRITE_VALUES>(d_idx, d_val),
            g_small_n_passes);
}

template <bool RAGGED, bool WRITE_VALUES>
static void topk_fused_impl(const float* d_in,
                            int M,
                            int pitch,
                            const int* d_row_starts,
                            const int* d_row_ends,
                            int K,
                            int* d_idx,
                            float* d_val,
                            Bufs& b,
                            const ShapeParams& sp,
                            hipStream_t s)
{
    const RowExtents<RAGGED> ext = make_row_extents<RAGGED>(d_row_starts, d_row_ends);
    const auto dst               = make_topk_out<WRITE_VALUES>(d_idx, d_val);
    const int S                  = sp.S;
    const float margin           = sp.margin;
    const int rank               = g_sample_rank > 0 ? g_sample_rank : sp.rank;
    const int cap                = sp.cap;
    const int n4                 = pitch / FP32_EPT;
    const int gx                 = std::max(1, std::min(g_cf_gx, n4 / g_cf_block));
    const int nwaves_b           = std::max(1, g_cf_block / WAVE_SIZE);
    const int seg_stride         = CAND_SLOTS_PER_ROW / nwaves_b;

#ifndef FORCE_PC_KEYS_ONLY
#define FORCE_PC_KEYS_ONLY 1
#endif
    const int wide_buf         = WIDE_WORDS * (int)sizeof(uint32_t);
    const bool baseline_wide_c = wide_select_fits(M, PHASE_C_STATIC_LDS + cap * 8, 2 * wide_buf);
    const bool compact_wide_c =
        cap == PHASE_C_CAP && wide_select_fits_allocated(M, PHASE_C_STATIC_LDS + cap * 4, wide_buf);
    const bool resource_compact_c = FORCE_PC_KEYS_ONLY && PC_WIDE_REUSE && sp.coop_g > 1 &&
                                    !sp.keys_only_c && !baseline_wide_c && compact_wide_c;
    const bool keys_only_c  = sp.keys_only_c || resource_compact_c;
    const bool reuse_wide_c = resource_compact_c;
#if PCIDX_MUTANT
    // Gate self-test only: sizes phase_c's LDS for keys alone even when the
    // kernel also stages indices. Must turn the gate red.
    const int phase_c_candidate_lds = cap * (int)sizeof(uint32_t);
#else
    const int phase_c_candidate_lds =
        cap * (keys_only_c ? (int)sizeof(uint32_t) : (int)(sizeof(uint32_t) + sizeof(int)));
#endif
    // Provisional wide/occupancy with keys in LDS (the historical default). The
    // register path below may drop S*4 from LDS and re-evaluate wide_select_fits.
    bool wide_a =
        g_wide_a < 0 ? wide_select_fits(M, PHASE_A_STATIC_LDS + S * 4, wide_buf) : (g_wide_a != 0);
    const bool wide_c   = g_wide_c < 0 ? (resource_compact_c || baseline_wide_c) : (g_wide_c != 0);
    int nwide_a         = wide_a ? 1 : 0;
    const int nwide_c   = wide_c ? (ABLATE_PC_SECOND_WIDE ? 1 : 2) : 0;
    size_t wide_a_bytes = (size_t)wide_buffer_count(nwide_a, false) * wide_buf;
    const size_t wide_c_bytes = (size_t)wide_buffer_count(nwide_c, reuse_wide_c) * wide_buf;
    int a_block =
        g_phase_a_block > 0
            ? g_phase_a_block
            : occupancy_block_threads(M, PHASE_A_STATIC_LDS + S * 4 + (int)wide_a_bytes, 0);
    const int c_block =
        g_phase_c_block > 0
            ? g_phase_c_block
            : occupancy_block_threads(
                  M, PHASE_C_STATIC_LDS + phase_c_candidate_lds + (int)wide_c_bytes, 0);

    // Register-resident sample keys. Auto picks KPT in {4,8} only: KPT=16 is
    // correct and matches the LDS path on the standalone hipcc binary, but the
    // aiter JIT build of the same source is ~3x slower on phase_a at M>=1024
    // (149 us vs 55 us at M=4096 N=131072, same-session ABBA). Until that gap is
    // attributed, auto never selects 16; --pa-keys=reg still accepts it when the
    // block divides that way. When the provisional KPT would be 16, double
    // a_block (capped at 1024) to land on KPT=8 instead of falling back to LDS.
#ifndef PA_ALLOW_KPT16
#define PA_ALLOW_KPT16 1
#endif
    int pa_kpt = 0;
    if(!g_phase_a_compact && a_block > 0 && (S % a_block) == 0)
    {
        int kpt = S / a_block;
        if(!PA_ALLOW_KPT16 && kpt == 16 && a_block * 2 <= 1024 && (S % (a_block * 2)) == 0)
        {
            a_block *= 2;
            kpt = 8;
        }
        if(kpt == 4 || kpt == 8 || (PA_ALLOW_KPT16 && kpt == 16))
            pa_kpt = kpt;
    }
    if(g_pa_keys == 0)
        pa_kpt = 0;
    if(g_pa_keys == 1 && pa_kpt == 0)
    {
        fprintf(stderr,
                "ERROR: --pa-keys=reg but S=%d a_block=%d is not KPT in {4,8,16}\n",
                S,
                a_block);
        std::abort();
    }
    if(pa_kpt > 0)
    {
        // Keys leave LDS; re-check whether the wide buffer now fits at this M.
        if(g_wide_a < 0)
        {
            wide_a       = wide_select_fits(M, PHASE_A_STATIC_LDS, wide_buf);
            nwide_a      = wide_a ? 1 : 0;
            wide_a_bytes = (size_t)wide_buffer_count(nwide_a, false) * wide_buf;
        }
    }
    const size_t pa_dyn_bytes =
        (pa_kpt > 0 ? (size_t)0 : (size_t)S * sizeof(uint32_t)) + wide_a_bytes;

    const bool coop = sp.coop_g > 1;
#if BLOCK_TSTAMP
    // One-shot per-block start/end realtime dump for phase_b_filter_coop, to
    // settle whether coop_g=1 is slower because of a ragged block-wave tail or
    // for a bandwidth reason. known_bad.md records that as unmeasured.
    static unsigned long long* bt_dev = nullptr;
    static size_t bt_slots            = 0;
    const size_t bt_need =
        (size_t)(coop ? 2 * sp.coop_g : 1) * (size_t)M * 2; // launch may double G
    if(bt_need > bt_slots)
    {
        if(bt_dev)
            hipFree(bt_dev);
        HIP_CHECK(hipMalloc(&bt_dev, bt_need * sizeof(unsigned long long)));
        bt_slots = bt_need;
        HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(d_bt), &bt_dev, sizeof(bt_dev)));
    }
#endif

    const int chunk_stride = sample_chunk_stride(pitch, S / SAMPLE_CHUNK_ELEMS);

    if(g_fuse_ab)
    {
        phase_ab_fused<RAGGED><<<M, a_block, (size_t)S * sizeof(uint32_t), s>>>(d_in,
                                                                                pitch,
                                                                                ext,
                                                                                rank,
                                                                                S,
                                                                                g_phase_a_passes,
                                                                                chunk_stride,
                                                                                seg_stride,
                                                                                b.cand_pack,
                                                                                b.cand_seg,
                                                                                b.cand_count,
                                                                                b.fb_count,
                                                                                K);
    }
    else
    {
        if(g_phase_a_compact)
        {
            phase_a_threshold<RAGGED, true, 0>
                <<<M, a_block, pa_dyn_bytes, s>>>(d_in,
                                                  pitch,
                                                  ext,
                                                  rank,
                                                  S,
                                                  g_phase_a_passes,
                                                  chunk_stride,
                                                  b.threshold,
                                                  b.threshold_f,
                                                  coop ? b.cand_reserved : nullptr,
                                                  coop ? b.cand_bad : nullptr,
                                                  b.fb_count,
                                                  K,
                                                  0);
        }
        else
        {
            auto launch_pa = [&](auto kpt_tag) {
                constexpr int KP = decltype(kpt_tag)::value;
                phase_a_threshold<RAGGED, false, KP>
                    <<<M, a_block, pa_dyn_bytes, s>>>(d_in,
                                                      pitch,
                                                      ext,
                                                      rank,
                                                      S,
                                                      g_phase_a_passes,
                                                      chunk_stride,
                                                      b.threshold,
                                                      b.threshold_f,
                                                      coop ? b.cand_reserved : nullptr,
                                                      coop ? b.cand_bad : nullptr,
                                                      b.fb_count,
                                                      K,
                                                      nwide_a);
            };
            if(pa_kpt == 4)
                launch_pa(std::integral_constant<int, 4>{});
            else if(pa_kpt == 8)
                launch_pa(std::integral_constant<int, 8>{});
            else if(pa_kpt == 16)
                launch_pa(std::integral_constant<int, 16>{});
            else
                launch_pa(std::integral_constant<int, 0>{});
        }

        if(coop)
        {
            // Stream the row data past the caches: every element is read by exactly
            // one block and never again. Priced on rotated inputs, one fresh buffer
            // per call, same-process dual-module A/B against cached loads
            // (log/v7/nt_price_load20.json, nt_region_load20.json, nt_interior_v7c1.json):
            //
            //   M*N in [2^26, 2^27)  1.032 .. 1.114   M*N = 2^24  0.987 .. 1.021
            //   M*N in [2^25, 2^26)  1.009 .. 1.068   M*N <= 2^23 0.985 .. 1.002
            //
            // The whole gain is phase_b's (phase_b 1.03-1.15, phase_c 0.97-1.02). The
            // old 2^27 gate was fitted on one reused input buffer, where anything under
            // the 256 MB MALL stayed resident between calls, and it priced loads and
            // stores together. Stores stay cached below 2^27: NT loads+stores over NT
            // loads alone is 0.970-0.988 at 2^25 and 0.981-1.011 at 2^26
            // (log/v7/nt_price_both20.json). This assumes the input is not already in
            // the MALL when the op starts. g_nt_load forces loads and stores for pricing.
            const size_t mn = (size_t)M * (size_t)pitch;
            const bool nt = g_nt_load < 0 ? (mn >= ((size_t)1 << NT_LOAD_LOG2)) : (g_nt_load != 0);
            const bool nt_st = nt && (g_nt_load >= 0 || mn >= ((size_t)1 << NT_STORE_LOG2));
            // One WSTAGE_CAP-entry staging slot per wave, and no more: see the
            // declaration in phase_b_filter_coop for why this is not a constant.
#if WSTAGE_HOST_MUTANT
            // Gate self-test only: the host sizes the old 320-entry staging while
            // the kernel indexes WSTAGE_CAP_COOP. Must turn the gate red.
            const size_t wstage_bytes =
                (size_t)(g_cf_block / WAVE_SIZE) * WSTAGE_CAP * sizeof(uint64_t);
#else
            const size_t wstage_bytes =
                (size_t)(g_cf_block / WAVE_SIZE) * WSTAGE_CAP_COOP * sizeof(uint64_t);
#endif
            // Batched filter loads only where at most two blocks share a CU: a
            // prefetch ring cost the bandwidth-bound grids 1-16% (log/v7/price_pbpf*.json).
            // Not ragged: load_row_f4's partial-vector branch waits after each load.
            // The batch never exceeds a thread's loads: dead slots cost 4-7% at 1-2.
#ifndef PB_BATCH_MAX_BLOCKS
#define PB_BATCH_MAX_BLOCKS (2 * CU_COUNT)
#endif
            auto batches = [&](int g) {
                const int it = ((n4 + g - 1) / g + g_cf_block - 1) / g_cf_block;
                return !RAGGED && !nt_st && it >= 2 &&
                       (size_t)M * (size_t)g <= (size_t)PB_BATCH_MAX_BLOCKS;
            };
            // kCoopLog2G predates the batched loads. Below M=32 at N >= 131072 one more
            // doubling of G wins where the doubled grid still batches: 1.005-1.059 at
            // pow2 N, 1.007-1.115 at 35 widths inside the buckets (log/v7/price_cgup*.json,
            // price_v7c7b_*.json). A doubling that turned batching off lost (M=1 N=262144
            // 0.98, 1024-block grids 0.86-0.99), so it is taken only where the new grid batches.
            int g = sp.coop_g;
            if(g_coop_g <= 0 && M < 32 && pitch >= 131072 &&
               2 * g <= std::max(1, n4 / g_cf_block) && batches(2 * g))
                g *= 2;
            const int pb_iters  = ((n4 + g - 1) / g + g_cf_block - 1) / g_cf_block;
            const bool pb_batch = batches(g);
            auto launch_pb      = [&](auto ntc, auto pb) {
                phase_b_filter_coop<RAGGED,
                                         decltype(ntc)::value,
                                         false,
                                    RAGGED ? 0 : decltype(pb)::value>
                    <<<dim3(g, M), g_cf_block, wstage_bytes, s>>>(d_in,
                                                                  pitch,
                                                                  ext,
                                                                  n4,
                                                                  b.threshold_f,
                                                                  b.cand_pack,
                                                                  b.cand_reserved,
                                                                  b.cand_bad,
                                                                  cap);
            };
            // NT needs M*N >= 2^25, so at <= 512 blocks a thread has >= 32 loads.
            auto launch_batched = [&](auto ntc) {
                if constexpr(decltype(ntc)::value)
                {
                    launch_pb(ntc, std::integral_constant<int, 8>{});
                }
                else
                {
                    if(pb_iters >= 8)
                        launch_pb(ntc, std::integral_constant<int, 8>{});
                    else if(pb_iters >= 4)
                        launch_pb(ntc, std::integral_constant<int, 4>{});
                    else
                        launch_pb(ntc, std::integral_constant<int, 2>{});
                }
            };
            if(nt_st)
                phase_b_filter_coop<RAGGED, true>
                    <<<dim3(g, M), g_cf_block, wstage_bytes, s>>>(d_in,
                                                                  pitch,
                                                                  ext,
                                                                  n4,
                                                                  b.threshold_f,
                                                                  b.cand_pack,
                                                                  b.cand_reserved,
                                                                  b.cand_bad,
                                                                  cap);
            else if(pb_batch && nt)
                launch_batched(std::true_type{});
            else if(pb_batch)
                launch_batched(std::false_type{});
            else if(nt)
                phase_b_filter_coop<RAGGED, true, false>
                    <<<dim3(g, M), g_cf_block, wstage_bytes, s>>>(d_in,
                                                                  pitch,
                                                                  ext,
                                                                  n4,
                                                                  b.threshold_f,
                                                                  b.cand_pack,
                                                                  b.cand_reserved,
                                                                  b.cand_bad,
                                                                  cap);
            else
                phase_b_filter_coop<RAGGED, false>
                    <<<dim3(g, M), g_cf_block, wstage_bytes, s>>>(d_in,
                                                                  pitch,
                                                                  ext,
                                                                  n4,
                                                                  b.threshold_f,
                                                                  b.cand_pack,
                                                                  b.cand_reserved,
                                                                  b.cand_bad,
                                                                  cap);
        }
        else if(g_phase_b == 4)
        {
            phase_b_filter_wavestage<RAGGED><<<M, g_cf_block, 0, s>>>(
                d_in, pitch, ext, b.threshold_f, b.cand_pack, b.cand_seg, b.cand_count, seg_stride);
        }
        else
        {
            phase_b_filter_waveseg<0, RAGGED><<<M, g_cf_block, 0, s>>>(
                d_in, pitch, ext, b.threshold_f, b.cand_pack, b.cand_seg, b.cand_count, seg_stride);
        }
    }

    if(coop)
    {
        const size_t lds_c  = (size_t)phase_c_candidate_lds;
        auto launch_phase_c = [&](auto reuse_tag) {
            constexpr bool REUSE = decltype(reuse_tag)::value;
            phase_c_select_contig<RAGGED, WRITE_VALUES, REUSE>
                <<<M, c_block, lds_c + wide_c_bytes, s>>>(d_in,
                                                          pitch,
                                                          ext,
                                                          b.cand_pack,
                                                          b.cand_reserved,
                                                          b.cand_bad,
                                                          b.cand_count,
                                                          cap,
                                                          K,
                                                          dst,
                                                          b.fb_rows,
                                                          b.fb_count,
                                                          g_phase_c_passes,
                                                          keys_only_c,
                                                          nwide_c);
        };
        if(reuse_wide_c)
            launch_phase_c(std::true_type{});
        else
            launch_phase_c(std::false_type{});
    }
    else if(cap <= PHASE_C_CAP)
    {
        phase_c_select_waveseg<true, RAGGED, WRITE_VALUES>
            <<<M, c_block, wide_c_bytes, s>>>(d_in,
                                              pitch,
                                              ext,
                                              b.cand_pack,
                                              b.cand_seg,
                                              b.cand_count,
                                              seg_stride,
                                              nwaves_b,
                                              K,
                                              cap,
                                              dst,
                                              b.fb_rows,
                                              b.fb_count,
                                              g_phase_c_passes,
                                              nwide_c);
    }
    else
    {
        const size_t lds_c = (size_t)cap * (sizeof(uint32_t) + sizeof(int));
        phase_c_select_waveseg<false, RAGGED, WRITE_VALUES>
            <<<M, c_block, lds_c + wide_c_bytes, s>>>(d_in,
                                                      pitch,
                                                      ext,
                                                      b.cand_pack,
                                                      b.cand_seg,
                                                      b.cand_count,
                                                      seg_stride,
                                                      nwaves_b,
                                                      K,
                                                      cap,
                                                      dst,
                                                      b.fb_rows,
                                                      b.fb_count,
                                                      g_phase_c_passes,
                                                      nwide_c);
    }

#if BLOCK_TSTAMP
    {
        static int bt_dumped = 0;
        if(bt_dumped++ == 2)
        { // third call: past warmup
            HIP_CHECK(hipDeviceSynchronize());
            std::vector<unsigned long long> h(bt_need);
            HIP_CHECK(hipMemcpy(
                h.data(), bt_dev, bt_need * sizeof(unsigned long long), hipMemcpyDeviceToHost));
            const char* o = getenv("BT_OUT");
            FILE* f       = fopen(o ? o : "/tmp/bt.txt", "w");
            fprintf(f, "# blocks %zu coop_g %d M %d\n", bt_need / 2, sp.coop_g, M);
            for(size_t i = 0; i < bt_need; i += 2)
                fprintf(f, "%llu %llu\n", h[i], h[i + 1]);
            fclose(f);
        }
    }
#endif
    (void)margin;
    (void)gx;
}

// RAGGED and WRITE_VALUES are both compile-time, so a runtime dispatcher has to
// pick one of four instantiations. It is expanded in ONE place, here, rather
// than at each launch site: the four phase kernels would otherwise each carry
// the same 4-way if-tree, and the launch-site version is where a missed branch
// silently runs the wrong instantiation.
template <bool RAGGED, bool WRITE_VALUES>
static void topk_indices_inst(const float* d_in,
                              int M,
                              int pitch,
                              const int* d_row_starts,
                              const int* d_row_ends,
                              int K,
                              int* d_idx,
                              float* d_val,
                              Bufs& b,
                              const ShapeParams& sp,
                              hipStream_t s)
{
    if(sp.path == PATH_SMALL_N)
    {
        topk_small_n<RAGGED, WRITE_VALUES>(
            d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, s);
        return;
    }
    topk_fused_impl<RAGGED, WRITE_VALUES>(
        d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, sp, s);
}

static void topk_indices(const float* d_in,
                         int M,
                         int pitch,
                         const int* d_row_starts,
                         const int* d_row_ends,
                         int K,
                         int* d_idx,
                         float* d_val,
                         Bufs& b,
                         int smc,
                         hipStream_t s)
{
    const int k_geom = g_ragged ? geometry_k_ragged(K, pitch) : K;
    ShapeParams sp   = derive_shape_params(
        M, pitch, k_geom, g_margin, g_sample_s, g_coop_g, (TopkPath)g_path_override);
    g_sample_s = sp.S > 0 ? sp.S : g_sample_s;
    if(g_ragged)
    {
        if(d_val)
            topk_indices_inst<true, true>(
                d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, sp, s);
        else
            topk_indices_inst<true, false>(
                d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, nullptr, b, sp, s);
    }
    else
    {
        if(d_val)
            topk_indices_inst<false, true>(
                d_in, M, pitch, nullptr, nullptr, K, d_idx, d_val, b, sp, s);
        else
            topk_indices_inst<false, false>(
                d_in, M, pitch, nullptr, nullptr, K, d_idx, nullptr, b, sp, s);
    }
    (void)smc;
}

static void topk_fused(const float* d_in,
                       int M,
                       int pitch,
                       const int* d_row_starts,
                       const int* d_row_ends,
                       int K,
                       int* d_idx,
                       float* d_val,
                       Bufs& b,
                       int smc,
                       hipStream_t s)
{ topk_indices(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, smc, s); }

static void topk_direct(const float* d_in,
                        int M,
                        int pitch,
                        const int* d_row_starts,
                        const int* d_row_ends,
                        int K,
                        int* d_idx,
                        float* d_val,
                        Bufs& b,
                        hipStream_t s)
{
    fill_identity_rows<<<(M + 255) / 256, 256, 0, s>>>(b.fb_rows, b.fb_count, M);
    const dim3 g(1, FB_GRID);
    if(g_ragged)
    {
        const RowExtents<true> ext = make_row_extents<true>(d_row_starts, d_row_ends);
        if(d_val)
            phase_d_fallback<true, true><<<g, 1024, 0, s>>>(
                d_in, pitch, ext, K, b.fb_rows, b.fb_count, make_topk_out<true>(d_idx, d_val));
        else
            phase_d_fallback<true, false><<<g, 1024, 0, s>>>(
                d_in, pitch, ext, K, b.fb_rows, b.fb_count, make_topk_out<false>(d_idx, nullptr));
    }
    else
    {
        const RowExtents<false> ext = make_row_extents<false>(nullptr, nullptr);
        if(d_val)
            phase_d_fallback<false, true><<<g, 1024, 0, s>>>(
                d_in, pitch, ext, K, b.fb_rows, b.fb_count, make_topk_out<true>(d_idx, d_val));
        else
            phase_d_fallback<false, false><<<g, 1024, 0, s>>>(
                d_in, pitch, ext, K, b.fb_rows, b.fb_count, make_topk_out<false>(d_idx, nullptr));
    }
}

static void run_topk(const float* d_in,
                     int M,
                     int pitch,
                     const int* d_row_starts,
                     const int* d_row_ends,
                     int K,
                     int* d_idx,
                     float* d_val,
                     Bufs& b,
                     int smc,
                     hipStream_t s)
{
    if(g_pipeline_direct)
        topk_direct(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, s);
    else
        topk_fused(d_in, M, pitch, d_row_starts, d_row_ends, K, d_idx, d_val, b, smc, s);
}

// aiter op entry for the AVO fp32 per-row top-k kernels.
//
// scripts/export_aiter_op.py appends this file verbatim after the exported
// kernel region, so the entry and the kernels it calls live in one repo and
// move in one diff. Do not add anything here that the benchmark harness needs:
// this side is only reachable from aiter.
//
// The contract is mirrored from top_k_per_row_prefill
// (aiter csrc/kernels/topk_per_row_kernels.cu): host-side shape decisions key
// off `stride0`, the row pitch, because the per-row extents live on the device
// in rowEnds and the dispatcher has to pick a path, a block size and a grid
// without reading device memory back.

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include <optional>

namespace sampled {

// One caller-provided buffer, carved the way alloc_bufs() carves its separate
// hipMallocs. This is the single definition of that layout: workspace_bytes()
// publishes the total to Python and bind_bufs() reads the offsets, so the size
// Python allocates and the size the kernels address cannot disagree.
struct WsLayout
{
    size_t threshold, threshold_f, cand_pack, cand_seg;
    size_t cand_count, cand_reserved, cand_bad, fb_rows, fb_count;
    size_t total;
};

static inline WsLayout ws_layout(int M, int cap)
{
    // 256 B keeps every field on a dwordx4-friendly boundary regardless of M.
    auto align256          = [](size_t x) { return (x + 255u) & ~(size_t)255u; };
    const size_t row_slots = (size_t)std::max(CAND_SLOTS_PER_ROW, cap);
    size_t o               = 0;
    auto take              = [&](size_t bytes) {
        const size_t here = o;
        o                 = align256(o + bytes);
        return here;
    };
    WsLayout L{};
    L.threshold   = take((size_t)M * sizeof(uint32_t));
    L.threshold_f = take((size_t)M * sizeof(float));
    L.cand_pack   = take((size_t)M * row_slots * sizeof(uint64_t));
    L.cand_seg    = take((size_t)M * MAX_WAVES_PER_BLOCK * sizeof(int));
    L.cand_count  = take((size_t)M * sizeof(unsigned int));
#if CTR_ALLOC_MUTANT
    // Gate self-test only: allocates the counters unpadded while the kernels
    // index them padded. Must turn the gate red.
    L.cand_reserved = take((size_t)M * sizeof(unsigned int));
    L.cand_bad      = take((size_t)M * sizeof(unsigned int));
#else
    L.cand_reserved = take((size_t)M * CTR_STRIDE * sizeof(unsigned int));
    L.cand_bad      = take((size_t)M * CTR_STRIDE * sizeof(unsigned int));
#endif
    L.fb_rows  = take((size_t)M * sizeof(int));
    L.fb_count = take(sizeof(int));
    L.total    = o;
    return L;
}

// Derived without consulting the harness globals, and in particular without
// topk_indices(), which assigns the derived S back into g_sample_s. In a
// library that write is a bug: the next call on a different shape would be
// handed the previous shape's S as an override and derive different parameters,
// making the result depend on call order.
// Every call here is ragged, so the geometry is sized by geometry_k_ragged()
// (topk_shape.hip.hpp) rather than the caller's k.
static inline ShapeParams params_for(int M, int N, int K)
{ return derive_shape_params(M, N, geometry_k_ragged(K, N), 0.0f, 0, 0, PATH_AUTO); }

static inline Bufs bind_bufs(void* ws, const WsLayout& L, int cap)
{
    char* base = static_cast<char*>(ws);
    Bufs b{};
    b.threshold     = reinterpret_cast<uint32_t*>(base + L.threshold);
    b.threshold_f   = reinterpret_cast<float*>(base + L.threshold_f);
    b.cand_pack     = reinterpret_cast<uint64_t*>(base + L.cand_pack);
    b.cand_seg      = reinterpret_cast<int*>(base + L.cand_seg);
    b.cand_count    = reinterpret_cast<unsigned int*>(base + L.cand_count);
    b.cand_reserved = reinterpret_cast<unsigned int*>(base + L.cand_reserved);
    b.cand_bad      = reinterpret_cast<unsigned int*>(base + L.cand_bad);
    b.fb_rows       = reinterpret_cast<int*>(base + L.fb_rows);
    b.fb_count      = reinterpret_cast<int*>(base + L.fb_count);
    b.C_alloc       = cap;
    return b;
}

} // namespace sampled

// Published to Python so the caller allocates the scratch (aiter's rule: host
// code here never allocates device memory). Plain scratch, not zeroed: Phase A
// clears the reservation counters and fb_count before any consumer reads them,
// Phase B assigns cand_count rather than accumulating into it, and the small_n
// path touches none of them.
int64_t topk_sampled_workspace_size(int64_t numRows, int64_t stride0, int64_t k)
{
    if(numRows <= 0)
        return 1;
    const int M   = static_cast<int>(numRows);
    const int N   = static_cast<int>(stride0);
    const int K   = static_cast<int>(k);
    const auto sp = sampled::params_for(M, N, K);
    const int cap = sp.path == PATH_SMALL_N ? CAND_SLOTS_PER_ROW : sp.cap;
    return static_cast<int64_t>(sampled::ws_layout(M, cap).total);
}

// Reports whether this op can serve the shape at all, so Python can route
// around it instead of taking an AITER_CHECK abort. Kept in C++ because every
// term it tests (LDS residency, dwordx4 geometry, the Phase C LDS cap) is a
// property of these kernels, not of the caller.
bool topk_sampled_supports(int64_t numRows, int64_t stride0, int64_t k)
{
    if(numRows <= 0 || stride0 <= 0 || k <= 0)
        return false;
    if(k > PHASE_C_CAP_MAX)
        return false;
    // stride0 need not be a multiple of FP32_EPT. This entry always
    // instantiates RAGGED=true, where the vector count is n4_cover(len) and
    // load_row_f4 loads the final partial vector element-wise, so an odd pitch
    // only makes the row base unaligned -- which gfx950 serves natively.
    return sampled::params_for(
               static_cast<int>(numRows), static_cast<int>(stride0), static_cast<int>(k))
        .geom_ok;
}

// rowStarts[row] and rowEnds[row] bound [start, end); indices are absolute columns.
void top_k_per_row_prefill_sampled(
    const aiter_tensor_t& logits,
    const aiter_tensor_t& rowStarts,
    const aiter_tensor_t& rowEnds,
    aiter_tensor_t& indices,
    std::optional<aiter_tensor_t> values,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    int64_t k                               = 2048,
    std::optional<aiter_tensor_t> workspace = std::nullopt,
    // Whether the caller actually gave per-row bounds. The default keeps
    // every existing caller on the path they have today.
    //
    // topk_select synthesises rowStarts=0 and rowEnds=N when the caller
    // passes no `end`, so the entry cannot tell a genuinely ragged batch
    // from a plain [M, N] one, and it has always assumed the first. The
    // RAGGED=true kernels then bounds-check every element against a limit
    // that is the row length. Measured through aiter at k=2048 --dist
    // gaussian, the same instantiation in both builds:
    //
    //   m=2048 n=131072  phase_b 202.96 -> 193.14us, phase_a 38.31 ->
    //   37.44, phase_c 40.80 -> 41.09; per call 282.07 -> 271.67, -3.7%
    //
    // The shape plan is deliberately left alone -- params_for still sizes
    // by geometry_k_ragged -- so this changes which kernel runs and
    // nothing else.
    bool ragged = true)
{
    if(numRows <= 0)
        return;

    AITER_CHECK(logits.dtype() == AITER_DTYPE_fp32,
                "top_k_per_row_prefill_sampled: logits must be fp32");
    AITER_CHECK(stride1 == 1, "top_k_per_row_prefill_sampled: logits inner stride must be 1");
    AITER_CHECK(indices.dtype() == AITER_DTYPE_i32,
                "top_k_per_row_prefill_sampled: indices must be int32");
    AITER_CHECK(indices.numel() >= static_cast<size_t>(numRows) * static_cast<size_t>(k),
                "top_k_per_row_prefill_sampled: indices must hold numRows*k entries");
    if(values.has_value())
    {
        AITER_CHECK(values.value().dtype() == AITER_DTYPE_fp32,
                    "top_k_per_row_prefill_sampled: values must be fp32");
        AITER_CHECK(values.value().numel() >= static_cast<size_t>(numRows) * static_cast<size_t>(k),
                    "top_k_per_row_prefill_sampled: values must hold numRows*k entries");
    }
    AITER_CHECK(rowStarts.numel() >= static_cast<size_t>(numRows) &&
                    rowEnds.numel() >= static_cast<size_t>(numRows),
                "top_k_per_row_prefill_sampled: rowStarts/rowEnds must have numRows entries");
    AITER_CHECK(topk_sampled_supports(numRows, stride0, k),
                "top_k_per_row_prefill_sampled: unsupported shape (numRows=%ld stride0=%ld k=%ld); "
                "ask topk_sampled_supports() first",
                (long)numRows,
                (long)stride0,
                (long)k);
    AITER_CHECK(workspace.has_value(),
                "top_k_per_row_prefill_sampled requires a caller-provided workspace "
                "(see top_k_per_row_prefill_sampled in topk.py)");

    const int M = static_cast<int>(numRows);
    const int N = static_cast<int>(stride0);
    const int K = static_cast<int>(k);

    // The non-ragged kernels take n4 as `pitch / FP32_EPT`, which TRUNCATES,
    // while the ragged ones take `n4_cover(len)`, which rounds up and then
    // predicates the tail on `< len`. The two agree only when the row width is
    // a multiple of FP32_EPT; below that the plain path never reads the last
    // one to three columns, and measurably worse -- at m=8 N=131077 it returns
    // 72.5% of the wrong elements, which is more than the dropped tail can
    // explain and is not yet understood.
    //
    // The fused path has THREE phase_b kernels and only one of them reads the
    // tail. phase_b_filter_coop does; phase_b_filter_wavestage and
    // phase_b_filter_waveseg, which topk_fused_impl launches when coop_g == 1,
    // still take the truncating n4 and never look at the last one to three
    // columns. An earlier version of this gate lifted the restriction for the
    // whole fused path on the strength of the coop kernel alone, which was
    // wrong and shipped: measured through aiter's router on plain [M, N] input,
    // 124 cells returned wrong results -- M 128 to 4096, N in 2^k+1/+2/+3 for
    // k 13 to 16, K 128 / 512 / 2048, and none at N % 4 == 0. Every one of them
    // returned ZERO indices in the truncated tail against 16 to 3071 expected,
    // and the wrong-row count tracked the expected tail hits (m=4096 N=8193:
    // 1067 wrong rows against 1023.9 expected).
    //
    // So the gate is lifted exactly where the coop kernel runs, and nowhere
    // else. topk_small_n truncates the same way with no tail handling and keeps
    // the gate outright.
    HipDeviceGuard device_guard(logits.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const ShapeParams sp = sampled::params_for(M, N, K);

    const bool tail_cols    = (N % FP32_EPT) != 0;
    const bool ragged_fused = ragged || (tail_cols && sp.coop_g <= 1);
    const bool ragged_small = ragged || tail_cols;
    const float* in         = static_cast<const float*>(logits.data_ptr());
    const int* row_starts   = static_cast<const int*>(rowStarts.data_ptr());
    const int* row_ends     = static_cast<const int*>(rowEnds.data_ptr());
    int* idx                = static_cast<int*>(indices.data_ptr());
    float* val = values.has_value() ? static_cast<float*>(values.value().data_ptr()) : nullptr;

    if(sp.path == PATH_SMALL_N)
    {
        if(ragged_small)
        {
            if(val)
                topk_small_n<true, true>(in, M, N, row_starts, row_ends, K, idx, val, stream);
            else
                topk_small_n<true, false>(in, M, N, row_starts, row_ends, K, idx, nullptr, stream);
        }
        else
        {
            if(val)
                topk_small_n<false, true>(in, M, N, row_starts, row_ends, K, idx, val, stream);
            else
                topk_small_n<false, false>(in, M, N, row_starts, row_ends, K, idx, nullptr, stream);
        }
        return;
    }

    const auto L = sampled::ws_layout(M, sp.cap);
    AITER_CHECK(workspace.value().numel() * workspace.value().element_size() >= L.total,
                "top_k_per_row_prefill_sampled: workspace is %zu B, needs %zu B",
                workspace.value().numel() * workspace.value().element_size(),
                L.total);
    Bufs b = sampled::bind_bufs(workspace.value().data_ptr(), L, sp.cap);
    if(ragged_fused)
    {
        if(val)
            topk_fused_impl<true, true>(in, M, N, row_starts, row_ends, K, idx, val, b, sp, stream);
        else
            topk_fused_impl<true, false>(
                in, M, N, row_starts, row_ends, K, idx, nullptr, b, sp, stream);
    }
    else
    {
        if(val)
            topk_fused_impl<false, true>(
                in, M, N, row_starts, row_ends, K, idx, val, b, sp, stream);
        else
            topk_fused_impl<false, false>(
                in, M, N, row_starts, row_ends, K, idx, nullptr, b, sp, stream);
    }
}
