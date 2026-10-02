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

// benchmark_topk.hip.cpp -- fp32 per-row top-k indices for prefill.
// Contract: input fp32 [M,N], output int32 indices [M,K].
//
// Three kernels per call:
//   A  phase_a_threshold     one block per row samples S contiguous-chunk elements
//                            and selects their rank-R key: the row's threshold T.
//   B  phase_b_filter_coop   coop_g blocks per row stream the row with dwordx4 and
//                            append every (key, index) not below T to the row's
//                            candidate area through per-wave LDS staging.
//   C  phase_c_select_contig one block per row selects the exact K-th key of the
//                            candidates in LDS and gathers the winners; a row whose
//                            candidates are unusable (too few, or over cap) runs
//                            the streamed radix fallback over the full row instead.
//
// Everything above AITER_EXPORT_END ships in aiter (scripts/export_aiter_op.py).
// The harness below it adds --pipeline direct, which runs phase_d_fallback's
// exact full-row select over every row as an independent oracle.

#include "topk_sampled/topk_common.hip.hpp"
#include "topk_sampled/topk_shape.hip.hpp"
#include <string>
#include <type_traits>

// log2(M * pitch) at which phase_b's row loads, and separately its candidate
// stores, go non-temporal. Stores only ever go NT when loads do.
#ifndef NT_LOAD_LOG2
#define NT_LOAD_LOG2 24
#endif
#ifndef NT_STORE_LOG2
#define NT_STORE_LOG2 27
#endif

// Every radix pass below scans with wave 0 alone and clears each bucket as it
// reads it, so a pass costs two block barriers. The alternatives (a separate
// HIST_REP reduction, a clear loop, an all-wave scan) were measured and each
// lost in some regime.

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
    // Zeroed once here; from then on the scan re-zeroes each bucket as it reads
    // it, so the per-pass clear loop and its barrier are gone.
    if(!use_prefill)
    {
        clear_hist(s_hist);
        __syncthreads();
    }
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
            // Do NOT add an active-set min/max here to exit early once the pivot is
            // pinned. It was tried: accumulating amn/amx in this loop (the reads are
            // already happening) and breaking when they agree made small_n 21-39%
            // SLOWER and the anchor 615.5 -> 662.8 us. The two extra barriers per pass
            // in the reduction, plus the register pressure in this loop, cost far more
            // than the single pass the exit saves.
            for(int i = threadIdx.x; i < c; i += blockDim.x)
            {
                uint32_t k = s_keys[i];
                if(!filter || (k >> hshift) == (pivot >> hshift))
                    atomicAdd(&s_hist[((k >> sh) & 0xFFu) * HIST_REP + rep], 1u);
            }
        }
        __syncthreads();
        block_find_pivot_bucket_wave0<true>(s_hist, s_scan, ek);
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
// contract: wave 0 scans the buckets and clears each one as it reads it.
//
// Where it runs is decided on the host (wide_select_for): the buffers live in
// dynamic LDS so a launch that does not ask for them pays no occupancy for them.
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
        // Finishing bits [0,12) with a wave-0 ballot walk over a small crossing
        // bucket was priced: restaging the bucket costs as much as the second wide
        // pass it replaces.
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
    if(!use_prefill)
    {
        clear_hist(s_hist);
        __syncthreads();
    }
    for(int p = 0; p < npasses; p++)
    {
        const int sh         = radix_shift(p);
        const int hshift     = sh + 8;
        const bool filter    = (p > 0);
        const bool skip_hist = use_prefill && p == 0;
        if(!skip_hist)
        {
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
        block_find_pivot_bucket_wave0<true>(s_hist, s_scan, ek);
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

#ifndef FB_LOADS
#define FB_LOADS 4
#endif
// Loads in flight per thread in the select passes. One fallback row at M=8
// N=524288 gaussian: 320 / 294 / 284us at 1 / 2 / 4.
// They cost registers, and the fallback shares phase_c's register allocation,
// so its VGPRs set phase_c's occupancy on every call, fallback or not; that is
// what PHASE_C_OCCUPANCY below holds.
#ifndef FB_SEL_LOADS
#define FB_SEL_LOADS 4
#endif
// prefix_skip (a min/max pass over the keys) for phase_c's wide select. Under
// the wide select pass 0 is already counted during the candidate read, so the
// skip costs a read of every key and two barriers to save one scan: phase_c at
// m=4/64/128/256/512 was 9.72/11.24/9.12/9.48/12.60us with it and
// 8.20/9.76/7.68/8.04/10.08 without. Tracking
// the min/max during the candidate read instead cost ~1us too (9.12us at m=4).
// What the skip also bought, the exit on a one-value candidate set (m=256
// n=524288 --dist inf: 6.1us with it, 11.3 without), block_select_lds_wide now
// gets from its histogram at no cost to other rows.
// Ablation: force phase_c's wide select to stop after the first 12-bit pass
// (nwide_c effective = 1). Wrong results; prices the second wide pass. Plan P3
// builds a Wave0 ballot walk only if that pass costs >= 0.7 us.
// phase_c inlines the exact fallback, so the fallback's register demand is
// phase_c's on every call. Unpinned, the fallback below took phase_c from 46-48
// to 70-82 VGPRs (8 -> 5-7 waves/SIMD) and phase_c from 13.4 to 18.2us at m=512
// n=131072 and 92 to 149us at m=4096 n=524290 with no row falling back. Pinned to
// 8 waves/SIMD it compiles to 64 VGPRs + 20-24 B/lane of scratch, and phase_c's
// no-fallback time is back within 1.4%.
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
    block_minmax(amn, amx, s_amm);
    __syncthreads(); // s_amm is rewritten by the next call
    value = amn;
    return amn == amx;
}

// block_gather_topk over a row streamed from global memory, a vector at a time:
// the element-wise form read 12.8 GB/s on one block, 164us of a 330us fallback
// row at N=524288. Same emission rule
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
        const unsigned rg = __builtin_amdgcn_readfirstlane(*(volatile unsigned*)s_wgt);
        const unsigned re = __builtin_amdgcn_readfirstlane(*(volatile unsigned*)s_weq);
        if(rg >= (unsigned)ngt && re >= (unsigned)eq_needed)
            break;
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
                if((bg | be) == 0ull)
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
                        out[p] = idx_base + col;
                        if(WRITE_VALUES)
                            out_val[p] = sortable_to_fp32(k);
                    }
                }
                if(eq)
                {
                    const unsigned p = basee + (unsigned)__popcll(be & lt);
                    if(p < (unsigned)eq_needed)
                    {
                        out[ngt + p] = idx_base + col;
                        if(WRITE_VALUES)
                            out_val[ngt + p] = sortable_to_fp32(k);
                    }
                }
            }
        }
    }
}

// Fallback-row LDS scratch, in words: phase_c lends its candidate area.
constexpr int FB_SCRATCH_WORDS = 4096;

// phase_c's fallback for a row whose sampled threshold missed: the helpers
// below and radix_fallback_row, which uses them.
// Gate self-test only: collects nothing from the crossing bucket but still
// reports it collected. Must turn the gate red.
#ifndef FB_BAND_MUTANT
#define FB_BAND_MUTANT 0
#endif
// Pricing / census only: prints every fallback row's outcome from phase_c.
#ifndef FB_BAND_PROBE
#define FB_BAND_PROBE 0
#endif
constexpr int FB_BAND_SH = 32 - WIDE_BITS;
static_assert(WIDE_FINE <= FB_SCRATCH_WORDS, "the band histogram lives in the fallback scratch");

// One block streams the whole row here, so these passes are bound by VALU work
// per key, not by the read: exact_row_select's pass 0 is ~6 VALU a key. A
// wave-level vote per key (a leader election, or a ballot per key slot) cost
// 1.6-3x of one of its passes.
//
// Histogram: each lane counts a run of keys in one bucket and adds the run when
// the bucket changes. A tie-dense or locally sorted row then costs a compare and
// an add per key instead of every atomic of the row queueing on one LDS word.
// SINK (an overflowed row): every bucket below sink + 1, phase_a's threshold
// bucket, is counted as bucket sink, so the background keys that cannot win
// form one long run per lane instead of an atomic each.
template <bool LR, bool SINK>
__device__ __forceinline__ void band_hist_vec(uint32_t* __restrict__ s_fine,
                                              const vfloat4& v,
                                              int i,
                                              int n4,
                                              int len,
                                              uint32_t sink,
                                              uint32_t& run_d,
                                              uint32_t& run_n)
{
#pragma unroll
    for(int e = 0; e < FP32_EPT; e++)
    {
        if(i < n4 && (!LR || i * FP32_EPT + e < len))
        {
            uint32_t d = fp32_to_sortable(v[e]) >> FB_BAND_SH;
            if(SINK)
                d = max(d, sink);
            if(d != run_d)
            {
                if(run_n)
                    atomicAdd(&s_fine[run_d], run_n);
                run_d = d;
                run_n = 0u;
            }
            run_n++;
        }
    }
}

// One histogram pass over the row: main trips while every lane of this wave has
// a vector in every slot, so the loads go out unpredicated; the rest of the row
// in live-predicated trips.
template <bool LR, bool SINK>
__device__ __forceinline__ void band_hist_pass(const float* __restrict__ row,
                                               int n4,
                                               int len,
                                               uint32_t sink,
                                               uint32_t* __restrict__ s_fine,
                                               uint32_t& run_d,
                                               uint32_t& run_n)
{
    const int step      = FB_SEL_LOADS * (int)blockDim.x;
    const int wave_last = (int)threadIdx.x | (WAVE_SIZE - 1);
    int i0              = 0;
    for(; i0 + wave_last + (FB_SEL_LOADS - 1) * (int)blockDim.x < n4; i0 += step)
    {
        vfloat4 v[FB_SEL_LOADS];
#pragma unroll
        for(int u = 0; u < FB_SEL_LOADS; u++)
            v[u] = load_row_f4<LR>(row, i0 + u * (int)blockDim.x + (int)threadIdx.x, len);
#pragma unroll
        for(int u = 0; u < FB_SEL_LOADS; u++)
            band_hist_vec<LR, SINK>(s_fine,
                                    v[u],
                                    i0 + u * (int)blockDim.x + (int)threadIdx.x,
                                    n4,
                                    len,
                                    sink,
                                    run_d,
                                    run_n);
    }
    for(; i0 < n4; i0 += step)
    {
        vfloat4 v[FB_SEL_LOADS];
#pragma unroll
        for(int u = 0; u < FB_SEL_LOADS; u++)
        {
            const int i = i0 + u * (int)blockDim.x + (int)threadIdx.x;
            v[u]        = i < n4 ? load_row_f4<LR>(row, i, len) : vfloat4{0.f, 0.f, 0.f, 0.f};
        }
#pragma unroll
        for(int u = 0; u < FB_SEL_LOADS; u++)
            band_hist_vec<LR, SINK>(s_fine,
                                    v[u],
                                    i0 + u * (int)blockDim.x + (int)threadIdx.x,
                                    n4,
                                    len,
                                    sink,
                                    run_d,
                                    run_n);
    }
}

// Appends the keys of one vector whose prefix key >> csh is at least cthr to the
// candidate records. A few thousand keys of the row qualify, so each reserves its
// own slot.
template <bool LR>
__device__ __forceinline__ void band_take_vec(uint64_t* __restrict__ cand_w,
                                              unsigned* __restrict__ s_cnt,
                                              const vfloat4& v,
                                              int i,
                                              int n4,
                                              int len,
                                              uint32_t csh,
                                              uint32_t cthr,
                                              int cap)
{
#pragma unroll
    for(int e = 0; e < FP32_EPT; e++)
    {
        const int col    = i * FP32_EPT + e;
        const uint32_t d = fp32_to_sortable(v[e]) >> csh;
        if(i < n4 && (!LR || col < len) && (FB_BAND_MUTANT == 1 ? d > cthr : d >= cthr))
        {
            const unsigned p = atomicAdd(s_cnt, 1u);
            if(p < (unsigned)cap)
                cand_w[p] = ((uint64_t)__float_as_uint(v[e]) << 32) | (uint32_t)col;
        }
    }
}

// Histogram of digit (key >> dsh) & dmask over the keys with key >> fsh == fpfx,
// for refining inside a crossing bucket too full to collect, plus those keys'
// min and max: a bucket of one value is answered without another read. Most keys
// fail the prefix test, so this costs about what a filtered radix pass does.
//
// The refine and collect passes re-read a row the histogram pass just read, and
// load it non-temporal; the histogram pass does not. Non-temporal there too was
// 5-8% slower at M <= 64, where those re-reads hit what it left in cache.
#ifndef FB_REFINE_LOADS
#define FB_REFINE_LOADS 1
#endif
template <bool LR>
__device__ __forceinline__ void band_refine_pass(const float* __restrict__ row,
                                                 int n4,
                                                 int len,
                                                 uint32_t fsh,
                                                 uint32_t fpfx,
                                                 uint32_t dsh,
                                                 uint32_t dmask,
                                                 uint32_t* __restrict__ s_fine,
                                                 uint32_t& kmin,
                                                 uint32_t& kmax)
{
    const int step = FB_REFINE_LOADS * (int)blockDim.x;
    uint32_t run_d = 0xFFFFFFFFu, run_n = 0u;
    for(int i0 = 0; i0 < n4; i0 += step)
    {
        vfloat4 v[FB_REFINE_LOADS];
#pragma unroll
        for(int u = 0; u < FB_REFINE_LOADS; u++)
        {
            const int i = i0 + u * (int)blockDim.x + (int)threadIdx.x;
            v[u]        = i < n4 ? load_row_f4<LR, true>(row, i, len) : vfloat4{0.f, 0.f, 0.f, 0.f};
        }
#pragma unroll
        for(int u = 0; u < FB_REFINE_LOADS; u++)
        {
            const int i = i0 + u * (int)blockDim.x + (int)threadIdx.x;
#pragma unroll
            for(int e = 0; e < FP32_EPT; e++)
            {
                const uint32_t key = fp32_to_sortable(v[u][e]);
                if(i < n4 && (!LR || i * FP32_EPT + e < len) && (key >> fsh) == fpfx)
                {
                    const uint32_t d = (key >> dsh) & dmask;
                    if(d != run_d)
                    {
                        if(run_n)
                            atomicAdd(&s_fine[run_d], run_n);
                        run_d = d;
                        run_n = 0u;
                    }
                    run_n++;
                    kmin = min(kmin, key);
                    kmax = max(kmax, key);
                }
            }
        }
    }
    if(run_n)
        atomicAdd(&s_fine[run_d], run_n);
}

// Coarse buckets from the fine ones, then the fine bucket holding rank `rank`
// (s_scan[0], or 0xFFFFFFFF when the histogram holds fewer keys) and the count
// above it (s_scan[1]). Ends with a barrier.
__device__ __forceinline__ void band_find(uint32_t* __restrict__ s_hist,
                                          uint32_t* __restrict__ s_x,
                                          uint32_t* __restrict__ s_scan,
                                          int rank)
{
    // Coarse bucket cb is the sum of fine buckets [64 cb, 64 cb + 64), written to
    // replica 0; 16 lanes of 4 fine buckets each.
    for(int t0 = 0; t0 < WIDE_FINE / 4; t0 += blockDim.x)
    {
        const int t = t0 + (int)threadIdx.x;
        uint32_t s  = 0u;
        if(t < WIDE_FINE / 4)
        {
            const opus::u32x4_t f = reinterpret_cast<const opus::u32x4_t*>(s_x)[t];
            s                     = f[0] + f[1] + f[2] + f[3];
        }
        s += (uint32_t)__shfl_xor((int)s, 1);
        s += (uint32_t)__shfl_xor((int)s, 2);
        s += (uint32_t)__shfl_xor((int)s, 4);
        s += (uint32_t)__shfl_xor((int)s, 8);
        if(t < WIDE_FINE / 4 && (t & 15) == 0)
            s_hist[(t >> 4) * HIST_REP] = s;
    }
    __syncthreads();
    block_find_pivot_wide_wave0(s_hist, s_x, s_scan, rank);
}

// Clears the histogram buffers and the search result. Every wave must have read
// the previous search result before this runs.
__device__ __forceinline__ void
band_clear(uint32_t* __restrict__ s_hist, uint32_t* __restrict__ s_x, uint32_t* __restrict__ s_scan)
{
    const opus::u32x4_t z = {0u, 0u, 0u, 0u};
    for(int j = threadIdx.x; j < WIDE_FINE / 4; j += blockDim.x)
        reinterpret_cast<opus::u32x4_t*>(s_x)[j] = z;
    for(int j = threadIdx.x; j < WIDE_COARSE_SLOTS; j += blockDim.x)
        s_hist[j] = 0u;
    if(threadIdx.x == 0)
    {
        s_scan[0] = 0xFFFFFFFFu;
        s_scan[1] = 0u;
    }
    __syncthreads();
}

// The whole fallback of phase_c for one row (every thread of the block calls):
// an MSD radix select streamed from the row, 12 + 12 + 8 bits, that stops as soon
// as it can.
//   level 1: histogram of the top 12 bits of every key (an overflowed row,
//            tmin > 0, counts the buckets below phase_a's threshold as one);
//   level 2: bits [8, 20) of the keys in the crossing bucket only;
//   level 3: bits [0, 8) of the keys in the crossing 24-bit bucket only.
// After each level: if the keys at and above the crossing bucket fit cap they
// are collected into the row's candidate records and phase_c's own LDS select
// finishes the row (returns their count); if the crossing bucket is one value,
// or level 3 has pinned the k-th key, block_gather_stream emits the row
// (returns -1). Every level drops only keys that cannot be among the top k_out,
// so the row is exact on every exit; at most four reads of the row.
template <bool LR, bool WRITE_VALUES>
__device__ __forceinline__ int radix_fallback_row(const float* __restrict__ row,
                                                  int n4,
                                                  int len,
                                                  int row_start,
                                                  int k_out,
                                                  int cap,
                                                  uint32_t tmin,
                                                  uint64_t* __restrict__ cand_w,
                                                  int* __restrict__ out,
                                                  float* __restrict__ out_val,
                                                  uint32_t* __restrict__ s_hist,
                                                  uint32_t* __restrict__ s_x,
                                                  uint32_t* __restrict__ s_scan,
                                                  uint32_t* __restrict__ s_amm,
                                                  unsigned* __restrict__ s_wgt,
                                                  unsigned* __restrict__ s_weq)
{
    const int step = FB_SEL_LOADS * (int)blockDim.x;
    // The collect pass trips the same way as band_hist_pass.
    const int wave_last = (int)threadIdx.x | (WAVE_SIZE - 1);

    // Level 1. The merged bucket stands for every key below phase_a's threshold;
    // landing in it means the k-th key is below that threshold after all (only a
    // row full of negative NaNs, which phase_b counts), and the row is counted
    // again without it. Straight-line on purpose: a loop around these passes kept
    // their state live across the back edge and cost phase_c 36-104 B/lane scratch.
    const uint32_t tb = tmin >> FB_BAND_SH;
    bool sink         = tb > 0u;
    uint32_t bstar    = 0xFFFFFFFFu;
    int above = 0, in_b = 0;
    if(sink)
    {
        band_clear(s_hist, s_x, s_scan);
        uint32_t run_d = 0xFFFFFFFFu, run_n = 0u;
        band_hist_pass<LR, true>(row, n4, len, tb - 1u, s_x, run_d, run_n);
        if(run_n)
            atomicAdd(&s_x[run_d], run_n);
        __syncthreads();
        band_find(s_hist, s_x, s_scan, k_out);
        bstar = s_scan[0];
        above = (int)s_scan[1];
        in_b  = bstar != 0xFFFFFFFFu ? (int)s_x[bstar] : 0;
        // Every wave reads the search result before any wave moves on: the next pass
        // clears s_scan and s_x, and a wave that read them late took a different
        // branch from the rest of its block (wrong rows and memory faults at M >= 512,
        // where blocks share a CU).
        __syncthreads();
        sink = bstar != 0xFFFFFFFFu && bstar != tb - 1u;
    }
    if(!sink)
    {
        band_clear(s_hist, s_x, s_scan);
        uint32_t run_d = 0xFFFFFFFFu, run_n = 0u;
        band_hist_pass<LR, false>(row, n4, len, 0u, s_x, run_d, run_n);
        if(run_n)
            atomicAdd(&s_x[run_d], run_n);
        __syncthreads();
        band_find(s_hist, s_x, s_scan, k_out);
        bstar = s_scan[0];
        above = (int)s_scan[1];
        in_b  = bstar != 0xFFFFFFFFu ? (int)s_x[bstar] : 0;
        __syncthreads(); // as above
    }
    // Pricing only, WRONG RESULTS: 1 stops after the level-1 pass.
    // Every key of the row was counted and len > k_out, so rank k_out exists.
    if(bstar == 0xFFFFFFFFu)
        __builtin_trap();

    uint32_t csh = FB_BAND_SH, cthr = bstar; // collect keys with key >> csh >= cthr
    int ngt        = above;                  // keys known to rank above the current prefix
    uint32_t pivot = 0u;
    bool emit      = false;
    if(above + in_b > cap)
    {
        // The whole row in one bucket: a min/max read settles one value for less
        // than a refine pass, whose prefix test every key of such a row passes.
        emit = above == 0 && in_b == len &&
               row_prefix_single_value<LR>(
                   row, n4, len, bstar << FB_BAND_SH, FB_BAND_SH, s_amm, pivot);
        uint32_t fsh = FB_BAND_SH, pfx = bstar; // keys with key >> fsh == pfx are refined
#pragma unroll 1
        for(int lvl = 0; lvl < 2 && !emit; lvl++)
        {
            const uint32_t dsh   = lvl == 0 ? 8u : 0u;
            const uint32_t dmask = lvl == 0 ? 0xFFFu : 0xFFu;
            band_clear(s_hist, s_x, s_scan);
            uint32_t kmin = 0xFFFFFFFFu, kmax = 0u;
            band_refine_pass<LR>(row, n4, len, fsh, pfx, dsh, dmask, s_x, kmin, kmax);
            block_minmax(kmin, kmax, s_amm);
            if(kmin == kmax)
            {
                // Every key of the prefix is one value: ngt keys above it, the rest copies.
                pivot = kmin;
                emit  = true;
                break;
            }
            band_find(s_hist, s_x, s_scan, k_out - ngt);
            const uint32_t b = s_scan[0];
            const int a      = (int)s_scan[1];
            const int in     = b != 0xFFFFFFFFu ? (int)s_x[b] : 0;
            __syncthreads(); // as at level 1: everyone has the result before the next clear
            pfx = (pfx << (fsh - dsh)) | (b & dmask);
            fsh = dsh;
            ngt += a;
            if(lvl == 0 && ngt + in <= cap)
            {
                csh  = dsh;
                cthr = pfx;
                break;
            }
            if(lvl == 1)
            {
                pivot = pfx; // all 32 bits: the k-th key itself
                emit  = true;
            }
        }
    }
    if(emit)
    {
        if(FB_BAND_MUTANT == 2)
            ngt++;
        if(threadIdx.x == 0)
        {
            *s_wgt = 0u;
            *s_weq = 0u;
        }
        __syncthreads();
        block_gather_stream<WRITE_VALUES>(
            row, len, row_start, pivot, ngt, k_out - ngt, out, out_val, s_wgt, s_weq);
        return -1;
    }

    if(threadIdx.x == 0)
        *s_wgt = 0u;
    __syncthreads();
    int i0 = 0;
    for(; i0 + wave_last + (FB_SEL_LOADS - 1) * (int)blockDim.x < n4; i0 += step)
    {
        vfloat4 v[FB_SEL_LOADS];
#pragma unroll
        for(int u = 0; u < FB_SEL_LOADS; u++)
            v[u] = load_row_f4<LR, true>(row, i0 + u * (int)blockDim.x + (int)threadIdx.x, len);
#pragma unroll
        for(int u = 0; u < FB_SEL_LOADS; u++)
            band_take_vec<LR>(cand_w,
                              s_wgt,
                              v[u],
                              i0 + u * (int)blockDim.x + (int)threadIdx.x,
                              n4,
                              len,
                              csh,
                              cthr,
                              cap);
    }
    for(; i0 < n4; i0 += step)
    {
        vfloat4 v[FB_SEL_LOADS];
#pragma unroll
        for(int u = 0; u < FB_SEL_LOADS; u++)
        {
            const int i = i0 + u * (int)blockDim.x + (int)threadIdx.x;
            v[u]        = i < n4 ? load_row_f4<LR, true>(row, i, len) : vfloat4{0.f, 0.f, 0.f, 0.f};
        }
#pragma unroll
        for(int u = 0; u < FB_SEL_LOADS; u++)
            band_take_vec<LR>(cand_w,
                              s_wgt,
                              v[u],
                              i0 + u * (int)blockDim.x + (int)threadIdx.x,
                              n4,
                              len,
                              csh,
                              cthr,
                              cap);
    }
    // phase_c reads these records back in this workgroup, on this CU, so the
    // barrier's workgroup-scope fence is enough. An agent-scope __threadfence()
    // here writes back and invalidates the XCD's L2 under every other block.
    __syncthreads();
    const int got = (int)*s_wgt;
    if(FB_BAND_MUTANT == 1)
        return got + 1;
    // Every key at or above the collect prefix was taken, and all of them were
    // written when got <= cap, so the records hold the top k_out whatever the
    // histogram counted; outside that the row changed between two reads.
    if(got < k_out || got > cap)
        __builtin_trap();
    return got;
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
//
// KPT is keys-per-thread for the register-resident path (4, 8 or 16). KPT==0 is
// the LDS-keys path.
#ifndef PHASE_A_WAVES
#define PHASE_A_WAVES 8
#endif
#define PHASE_A_OCCUPANCY __attribute__((amdgpu_waves_per_eu(PHASE_A_WAVES)))
template <bool RAGGED, int KPT = 0>
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
    __shared__ uint32_t s_scan[2];
    __shared__ uint32_t s_mm[2 * MAX_WAVES_PER_BLOCK];

    const int chunks       = S / SAMPLE_CHUNK_ELEMS;
    const int chunk_stride = RAGGED ? sample_chunk_stride(len, chunks) : chunk_stride_host;
    const int rank_row = RAGGED && len != pitch ? max(1, (int)((double)rank * pitch / len)) : rank;

    const int v4_per_chunk = SAMPLE_CHUNK_ELEMS / FP32_EPT;
    const int total_v4     = S / FP32_EPT;
    const int fold_rep     = threadIdx.x & (HIST_REP - 1);
    clear_hist(s_hist);
    __syncthreads();
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
        // radix_shift(0) is 24, so pass 0's digit is the top byte.
        // PA_REG_MUTANT drops keys[0] in the select; drop the matching fold count
        // here so the prefilled hist stays consistent with the mutant select.
        const bool drop0 = (PA_REG_MUTANT && KPT > 0 && t == 0);
        if(!drop0)
            atomicAdd(&s_hist[(k0 >> 24) * HIST_REP + fold_rep], 1u);
        atomicAdd(&s_hist[(k1 >> 24) * HIST_REP + fold_rep], 1u);
        atomicAdd(&s_hist[(k2 >> 24) * HIST_REP + fold_rep], 1u);
        atomicAdd(&s_hist[(k3 >> 24) * HIST_REP + fold_rep], 1u);
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
                keys, rank_row, s_hist, s_wide, s_scan, pivot, eq_needed, nwide, true);
        }
        else
        {
            block_select_reg<KPT>(keys, rank_row, s_hist, s_scan, pivot, eq_needed, npasses, true);
        }
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
                                     true);
    }
    else
    {
        block_select_lds(
            s_keys, S, rank_row, s_hist, s_scan, s_mm, pivot, eq_needed, npasses, false, true);
    }
#endif
    if(threadIdx.x == 0)
    {
        threshold[row]   = pivot;
        threshold_f[row] = sortable_to_fp32(pivot);
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
    unsigned int* cand_count;
    unsigned int* cand_reserved;
    unsigned int* cand_bad;
    int* fb_rows;
    int* fb_count;
    int C_alloc;
};

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
            RADIX_PASSES);
}

// phase_a's radix passes when it selects without the wide buffer. 3 give
// bit-identical candidate counts to 4 (the 4th byte never moves the bucket at
// fp32 precision); 2 spread to max=3919 against C_alloc=4096.
constexpr int PHASE_A_PASSES = 3;
#ifndef PB_BATCH_MAX_BLOCKS
#define PB_BATCH_MAX_BLOCKS (2 * CU_COUNT)
#endif

// Research-harness overrides of the launch policy; the aiter entry passes none.
struct LaunchOverrides
{
    int c_block = 0;  // > 0: phase_c block size
    int nt      = -1; // -1: the M*pitch gates; 0 / 1 force phase_b's NT loads and stores
};

// Every launch decision for one call of the three-kernel pipeline, made in one
// place. Every shape runs phase_b_filter_coop + phase_c_select_contig; g = 1
// is one block per row.
struct LaunchPlan
{
    int S, rank, cap, chunk_stride;
    int a_block, pa_kpt, nwide_a;
    size_t pa_dyn_bytes;
    int g, pb_b; // phase_b grid.x; filter loads batched per thread (0 = one at a time)
    bool nt, nt_st;
    size_t wstage_bytes;
    int c_block, nwide_c;
    bool keys_only_c, reuse_wide_c;
    size_t c_dyn_bytes;
};

static LaunchPlan plan_launch(int M,
                              int pitch,
                              bool ragged,
                              const ShapeParams& sp,
                              const LaunchOverrides& ov = LaunchOverrides{})
{
    LaunchPlan lp{};
    const int S     = sp.S;
    const int cap   = sp.cap;
    const int n4    = pitch / FP32_EPT;
    lp.S            = S;
    lp.rank         = sp.rank;
    lp.cap          = cap;
    lp.chunk_stride = sample_chunk_stride(pitch, S / SAMPLE_CHUNK_ELEMS);

    const int wide_buf         = WIDE_WORDS * (int)sizeof(uint32_t);
    const bool baseline_wide_c = wide_select_fits(M, PHASE_C_STATIC_LDS + cap * 8, 2 * wide_buf);
    const bool compact_wide_c =
        cap == PHASE_C_CAP && wide_select_fits_allocated(M, PHASE_C_STATIC_LDS + cap * 4, wide_buf);
    const bool resource_compact_c = !sp.keys_only_c && !baseline_wide_c && compact_wide_c;
    lp.keys_only_c                = sp.keys_only_c || resource_compact_c;
    lp.reuse_wide_c               = resource_compact_c;
#if PCIDX_MUTANT
    // Gate self-test only: sizes phase_c's LDS for keys alone even when the
    // kernel also stages indices. Must turn the gate red.
    const int c_cand_lds = cap * (int)sizeof(uint32_t);
#else
    const int c_cand_lds =
        cap * (lp.keys_only_c ? (int)sizeof(uint32_t) : (int)(sizeof(uint32_t) + sizeof(int)));
#endif
    lp.nwide_c                = (resource_compact_c || baseline_wide_c) ? 2 : 0;
    const size_t wide_c_bytes = (size_t)wide_buffer_count(lp.nwide_c, lp.reuse_wide_c) * wide_buf;
    lp.c_block =
        ov.c_block > 0
            ? ov.c_block
            : occupancy_block_threads(M, PHASE_C_STATIC_LDS + c_cand_lds + (int)wide_c_bytes, 0);
    lp.c_dyn_bytes = (size_t)c_cand_lds + wide_c_bytes;

    // Provisional wide/occupancy with keys in LDS; register-resident keys (KPT in
    // {4, 8, 16} when the block divides S) leave LDS, and the wide buffer is
    // re-checked without them. The block size stays the provisional one.
    bool wide_a         = wide_select_fits(M, PHASE_A_STATIC_LDS + S * 4, wide_buf);
    size_t wide_a_bytes = (size_t)wide_buffer_count(wide_a ? 1 : 0, false) * wide_buf;
    lp.a_block = occupancy_block_threads(M, PHASE_A_STATIC_LDS + S * 4 + (int)wide_a_bytes, 0);
    if(lp.a_block > 0 && (S % lp.a_block) == 0)
    {
        const int kpt = S / lp.a_block;
        if(kpt == 4 || kpt == 8 || kpt == 16)
            lp.pa_kpt = kpt;
    }
    if(lp.pa_kpt > 0)
    {
        wide_a       = wide_select_fits(M, PHASE_A_STATIC_LDS, wide_buf);
        wide_a_bytes = (size_t)wide_buffer_count(wide_a ? 1 : 0, false) * wide_buf;
    }
    lp.nwide_a      = wide_a ? 1 : 0;
    lp.pa_dyn_bytes = (lp.pa_kpt > 0 ? (size_t)0 : (size_t)S * sizeof(uint32_t)) + wide_a_bytes;

    // Stream the row data past the caches: every element is read by exactly
    // one block and never again. Priced on rotated inputs, one fresh buffer
    // per call, same-process dual-module A/B against cached loads:
    //
    //   M*N in [2^26, 2^27)  1.032 .. 1.114   M*N = 2^24  0.987 .. 1.021
    //   M*N in [2^25, 2^26)  1.009 .. 1.068   M*N <= 2^23 0.985 .. 1.002
    //
    // v8 re-measured the lower edge on the current kernels (standalone, inputs
    // rotated, NT loads from 2^22 against from 2^25): 1.012-1.034 at nine of ten
    // [2^24, 2^25) points on both entries, 1.000-1.012 at 2^23, 0.962-0.995 at
    // 2^22; run_perftest at the accept grid's eight [2^24, 2^25) cells: router
    // 1.024, plain 1.031. Loads go NT from 2^24.
    //
    // The whole gain is phase_b's (phase_b 1.03-1.15, phase_c 0.97-1.02). Stores
    // stay cached below 2^27: NT loads+stores over NT loads alone is 0.970-0.988
    // at 2^25 and 0.981-1.011 at 2^26. This
    // assumes the input is not already in the MALL when the op starts.
    const size_t mn = (size_t)M * (size_t)pitch;
    lp.nt           = ov.nt < 0 ? (mn >= ((size_t)1 << NT_LOAD_LOG2)) : (ov.nt != 0);
    lp.nt_st        = lp.nt && (ov.nt >= 0 || mn >= ((size_t)1 << NT_STORE_LOG2));
    // One staging slot per wave, and no more: see the declaration in
    // phase_b_filter_coop for why this is not a constant.
#if WSTAGE_HOST_MUTANT
    // Gate self-test only: the host sizes the old 320-entry staging while
    // the kernel indexes WSTAGE_CAP_COOP. Must turn the gate red.
    lp.wstage_bytes = (size_t)(PB_BLOCK / WAVE_SIZE) * 320 * sizeof(uint64_t);
#else
    lp.wstage_bytes = (size_t)(PB_BLOCK / WAVE_SIZE) * WSTAGE_CAP_COOP * sizeof(uint64_t);
#endif
    // Batched filter loads only where at most two blocks share a CU: a
    // prefetch ring cost the bandwidth-bound grids 1-16%.
    // The batch never exceeds a thread's loads: dead slots cost 4-7% at 1-2.
    auto batches = [&](int g) {
        const int it = ((n4 + g - 1) / g + PB_BLOCK - 1) / PB_BLOCK;
        return !lp.nt_st && it >= 2 && (size_t)M * (size_t)g <= (size_t)PB_BATCH_MAX_BLOCKS;
    };
    const int g = sp.coop_g;
    lp.g        = g;
    // NT needs M*N >= 2^24, so at <= 512 blocks a thread has >= 16 loads.
    const int pb_iters = ((n4 + g - 1) / g + PB_BLOCK - 1) / PB_BLOCK;
    if(batches(g))
        lp.pb_b = lp.nt ? 8 : pb_iters >= 8 ? 8 : pb_iters >= 4 ? 4 : 2;
    return lp;
}

template <bool RAGGED, bool NT, bool NT_ST, int PB>
static void launch_phase_b(const float* d_in,
                           int M,
                           int pitch,
                           RowExtents<RAGGED> ext,
                           Bufs& b,
                           const LaunchPlan& lp,
                           hipStream_t s)
{
    phase_b_filter_coop<RAGGED, NT, NT_ST, PB>
        <<<dim3(lp.g, M), PB_BLOCK, lp.wstage_bytes, s>>>(d_in,
                                                          pitch,
                                                          ext,
                                                          pitch / FP32_EPT,
                                                          __builtin_ctz((unsigned)lp.g),
                                                          b.threshold_f,
                                                          b.cand_pack,
                                                          b.cand_reserved,
                                                          b.cand_bad,
                                                          lp.cap);
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
                            const LaunchPlan& lp,
                            hipStream_t s)
{
    const RowExtents<RAGGED> ext = make_row_extents<RAGGED>(d_row_starts, d_row_ends);
    const auto dst               = make_topk_out<WRITE_VALUES>(d_idx, d_val);

    auto launch_pa = [&](auto kpt_tag) {
        constexpr int KP = decltype(kpt_tag)::value;
        phase_a_threshold<RAGGED, KP><<<M, lp.a_block, lp.pa_dyn_bytes, s>>>(d_in,
                                                                             pitch,
                                                                             ext,
                                                                             lp.rank,
                                                                             lp.S,
                                                                             PHASE_A_PASSES,
                                                                             lp.chunk_stride,
                                                                             b.threshold,
                                                                             b.threshold_f,
                                                                             b.cand_reserved,
                                                                             b.cand_bad,
                                                                             b.fb_count,
                                                                             K,
                                                                             lp.nwide_a);
    };
    if(lp.pa_kpt == 4)
        launch_pa(std::integral_constant<int, 4>{});
    else if(lp.pa_kpt == 8)
        launch_pa(std::integral_constant<int, 8>{});
    else if(lp.pa_kpt == 16)
        launch_pa(std::integral_constant<int, 16>{});
    else
        launch_pa(std::integral_constant<int, 0>{});

    if(lp.nt_st)
        launch_phase_b<RAGGED, true, true, 0>(d_in, M, pitch, ext, b, lp, s);
    else if(lp.nt && lp.pb_b == 8)
        launch_phase_b<RAGGED, true, false, 8>(d_in, M, pitch, ext, b, lp, s);
    else if(lp.nt)
        launch_phase_b<RAGGED, true, false, 0>(d_in, M, pitch, ext, b, lp, s);
    else if(lp.pb_b == 8)
        launch_phase_b<RAGGED, false, false, 8>(d_in, M, pitch, ext, b, lp, s);
    else if(lp.pb_b == 4)
        launch_phase_b<RAGGED, false, false, 4>(d_in, M, pitch, ext, b, lp, s);
    else if(lp.pb_b == 2)
        launch_phase_b<RAGGED, false, false, 2>(d_in, M, pitch, ext, b, lp, s);
    else
        launch_phase_b<RAGGED, false, false, 0>(d_in, M, pitch, ext, b, lp, s);

    auto launch_phase_c = [&](auto reuse_tag) {
        constexpr bool REUSE = decltype(reuse_tag)::value;
        phase_c_select_contig<RAGGED, WRITE_VALUES, REUSE>
            <<<M, lp.c_block, lp.c_dyn_bytes, s>>>(d_in,
                                                   pitch,
                                                   ext,
                                                   b.cand_pack,
                                                   b.cand_reserved,
                                                   b.cand_bad,
                                                   b.cand_count,
                                                   lp.cap,
                                                   K,
                                                   dst,
                                                   b.fb_rows,
                                                   b.fb_count,
                                                   RADIX_PASSES,
                                                   lp.keys_only_c,
                                                   lp.nwide_c,
                                                   b.threshold);
    };
    if(lp.reuse_wide_c)
        launch_phase_c(std::true_type{});
    else
        launch_phase_c(std::false_type{});
}

// aiter op entry for the sampled fp32 per-row top-k kernels.
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
    size_t threshold, threshold_f, cand_pack;
    size_t cand_count, cand_reserved, cand_bad, fb_rows, fb_count;
    size_t total;
};

static inline WsLayout ws_layout(int M, int cap)
{
    // 256 B keeps every field on a dwordx4-friendly boundary regardless of M.
    auto align256 = [](size_t x) { return (x + 255u) & ~(size_t)255u; };
    size_t o      = 0;
    auto take     = [&](size_t bytes) {
        const size_t here = o;
        o                 = align256(o + bytes);
        return here;
    };
    WsLayout L{};
    L.threshold   = take((size_t)M * sizeof(uint32_t));
    L.threshold_f = take((size_t)M * sizeof(float));
    L.cand_pack   = take((size_t)M * cap * sizeof(uint64_t));
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
// the harness's topk_fused(), which assigns the derived S back into g_sample_s. In a
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
    const int cap = sp.path == PATH_SMALL_N ? 0 : sp.cap;
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

    // A plain row whose width is not a multiple of FP32_EPT is safe on the fused
    // path: phase_b_filter_coop reads the tail, and the gate covers every N % 4
    // residue on both entries. topk_small_n truncates `pitch / FP32_EPT` with no
    // tail handling, so odd widths keep its bounds-checked instantiation.
    HipDeviceGuard device_guard(logits.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    const ShapeParams sp = sampled::params_for(M, N, K);

    // The plain kernels select K of every row, which a row of N <= K does not
    // have; the bounded ones emit such a row whole, in column order, and pad
    // with -1, as aiter's other top-k kernels do.
    if(N <= K)
        ragged = true;
    const bool ragged_small = ragged || (N % FP32_EPT) != 0;
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
    Bufs b              = sampled::bind_bufs(workspace.value().data_ptr(), L, sp.cap);
    const LaunchPlan lp = plan_launch(M, N, ragged, sp);
    if(ragged)
    {
        if(val)
            topk_fused_impl<true, true>(in, M, N, row_starts, row_ends, K, idx, val, b, lp, stream);
        else
            topk_fused_impl<true, false>(
                in, M, N, row_starts, row_ends, K, idx, nullptr, b, lp, stream);
    }
    else
    {
        if(val)
            topk_fused_impl<false, true>(
                in, M, N, row_starts, row_ends, K, idx, val, b, lp, stream);
        else
            topk_fused_impl<false, false>(
                in, M, N, row_starts, row_ends, K, idx, nullptr, b, lp, stream);
    }
}
