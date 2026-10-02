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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <numeric>
#include <vector>

#include "opus/opus.hpp"

constexpr int WAVE_SIZE    = opus::get_warp_size();
constexpr int FP32_EPT     = 4; // floats per dwordx4 load
constexpr int RADIX_PASSES = 4; // 4 x 8-bit covers all 32 sortable bits

// Replicas of each histogram bucket. A sortable fp32's top byte is sign+exponent,
// so on uniform[-1,1] data roughly half of all positive values share ONE bucket
// and the per-element LDS atomicAdd serialises hard. Lanes are spread across
// HIST_REP adjacent counters (adjacent => different LDS banks), summed before
// the scan. HIST_REP=1 restores the plain histogram.
#ifndef HIST_REPLICAS
#define HIST_REPLICAS 4
#endif
constexpr int HIST_REP   = HIST_REPLICAS;
constexpr int HIST_SLOTS = 256 * HIST_REP;

constexpr int MAX_WAVES_PER_BLOCK = 16;

// Phase A sampling geometry: NUM_CHUNKS contiguous runs of CHUNK floats each.
// Contiguous runs (not stride-1-of-32) so the DRAM traffic equals the useful
// bytes; a strided sample would fetch a whole 128 B line per useful float.
// A chunk is 64 floats = 256 B, so the DRAM traffic equals the useful bytes.
// S is a runtime knob: the spread of the resulting candidate count is the
// statistical noise of a rank-R estimator (std ~ count/sqrt(R), R = margin*K*S/N),
// and that spread is what decides whether any row needs the exact fallback.
constexpr int SAMPLE_CHUNK_ELEMS = 64;
constexpr int SAMPLE_S_MAX       = 16384;

// Spacing between the chunk starts, and the one definition of it: the sampler
// kernels index with it and sampling_geometry_ok() decides servability from it,
// so a second copy of this expression is a way for the host to accept a shape
// the kernel then reads out of bounds on.
//
// Masked to a multiple of FP32_EPT because a chunk is loaded with dwordx4 from
// row + chunk * stride, which needs 16 B alignment. Masking rather than
// rejecting the odd stride is what makes a row width that does not divide by
// `chunks` servable at all -- every N on the pow2 grid happened to divide
// evenly, so the rule that rejected them cost nothing there, but aiter's
// prefill widths are num_prefix + num_rows (131072 + 256 = 131328) and mostly
// do not. Where N/chunks was already a multiple of FP32_EPT this returns
// exactly N/chunks, so no shape that already worked changes behaviour.
//
// stride >= SAMPLE_CHUNK_ELEMS is then the only bound needed to keep the last
// chunk in the row: it reads [(chunks-1)*stride, +CHUNK), and
// (chunks-1)*(N/chunks) + CHUNK <= N - N/chunks + CHUNK <= N once N/chunks >= CHUNK.
__host__ __device__ inline int sample_chunk_stride(int N, int chunks)
{ return (N / chunks) & ~(FP32_EPT - 1); }

// Per-row window bundled like TopkOut: uniform keeps one 8 B null ends pointer;
// ragged carries starts+ends (16 B). Valid slice is [starts[row], ends[row]).
template <bool RAGGED>
struct RowExtents;
template <>
struct RowExtents<false>
{
    const int* ends;
    __device__ __forceinline__ int row_start(int, int) const { return 0; }
};
// Both accessors clamp the caller's window into [0, pitch].
//
// This is the only place that can. rowStarts/rowEnds are device pointers, so a
// host-side check would cost a D2H sync on every call; and declining the shape
// in topk_sampled_supports() does not help either, because that signature is
// (numRows, stride0, k) and never sees the extents -- a decline just routes the
// same arguments to aiter's mb/ob path, which faults on them too.
//
// What it prevents, measured by bench/stress_topk.py on the unclamped build:
// rowStarts = -8 and rowEnds = INT32_MAX each took a GPU memory fault, and
// rowEnds = pitch + 64 was worse than a fault -- it returned indices past the
// pitch with no error at all.
//
// `pitch` is a parameter rather than a member on purpose. Adding a field would
// change the kernarg layout, and one unused kernarg has already been measured
// moving the small_n geomean 1.0% in this kernel (see TopkOut below). The four
// call sites in topk_generalize.hip.hpp all have pitch in scope already, so
// passing it is free. The arithmetic is per ROW, not per element.
template <>
struct RowExtents<true>
{
    const int* starts;
    const int* ends;
    __device__ __forceinline__ int row_start(int row, int pitch) const
    {
        const int s = starts[row];
        return s < 0 ? 0 : (s > pitch ? pitch : s);
    }
    __device__ __forceinline__ int row_len(int row, int pitch) const
    {
        const int e  = ends[row];
        const int ec = e < 0 ? 0 : (e > pitch ? pitch : e);
        const int s  = row_start(row, pitch);
        return ec > s ? ec - s : 0;
    }
};

template <bool RAGGED>
__device__ __forceinline__ int row_len_of(int row, int pitch, RowExtents<RAGGED> ext)
{
    if constexpr(RAGGED)
        return ext.row_len(row, pitch);
    return pitch;
}

__device__ __forceinline__ int n4_cover(int len) { return (len + FP32_EPT - 1) / FP32_EPT; }

__device__ __forceinline__ int k_take_dev(int K, int len) { return len < K ? len : K; }

// The two output buffers, bundled so that the SIZE of the kernel argument
// depends on WRITE_VALUES.
//
// This is not tidiness. The no-values instantiation has to be kernarg-IDENTICAL
// to the version from before values existed, not merely free of the stores: one
// extra UNUSED float* kernarg on phase_small_n_topk alone, with no other change
// at all, moved the small_n geomean from 28.51/28.54 to 28.82/28.85 us (+1.0%,
// two runs each, interleaved A/B on this box), which is over the scoring gate's
// 0.5% band. small_n runs one short block per row, so its kernarg prologue is a
// real share of the kernel rather than noise.
//
// TopkOut<false> holds a single pointer, so it is 8 bytes with 8-byte alignment
// -- exactly the `int* out_idx` it replaces -- and every later argument keeps
// its old offset. Only a caller that asks for values pays the extra 8.
template <bool WRITE_VALUES>
struct TopkOut;

template <>
struct TopkOut<false>
{
    int* idx;
    __device__ __forceinline__ int* idx_row(int row, int K) const { return idx + (size_t)row * K; }
    __device__ __forceinline__ float* val_row(int, int) const { return nullptr; }
};

template <>
struct TopkOut<true>
{
    int* idx;
    float* val;
    __device__ __forceinline__ int* idx_row(int row, int K) const { return idx + (size_t)row * K; }
    __device__ __forceinline__ float* val_row(int row, int K) const
    { return val + (size_t)row * K; }
};

// The value padding is -inf, NOT 0, and that is aiter's rule rather than a
// preference (topk_per_row_kernels.cu:2249 states it): the index slot is -1, so
// its score has to sort below every real one. Logits are routinely negative, so
// a 0.0 pad outranks them, and a consumer that ranks these scores -- DCP merges
// the exchanged top-k across ranks -- would let padding steal a real
// candidate's slot.
template <bool WRITE_VALUES>
__device__ __forceinline__ void
pad_topk_tail(int* __restrict__ out, float* __restrict__ out_val, int k_take, int K)
{
    for(int i = k_take + threadIdx.x; i < K; i += blockDim.x)
    {
        out[i] = -1;
        if(WRITE_VALUES)
            out_val[i] = -__builtin_inff();
    }
}

// A row with row_len <= K has every element selected, so there is nothing to
// rank: emit the columns in index order and pad the tail with -1. This is
// aiter's own convention for the case, in both of its kernels
// (topk_per_row_kernels.cu:398 mb path, :2241 ob path), and matching it is not
// cosmetic -- which of those kernels runs is a perf heuristic on aiter's side,
// so the padding and the emit have to agree or the same call would mean
// different things at a batch-size boundary.
//
// This is the one emit path that has to READ the row to produce values: there
// is no select here, so no key is sitting in a register the way it is inside
// block_gather_topk. aiter reads the row here too.
template <bool WRITE_VALUES>
__device__ __forceinline__ void emit_identity_row(int* __restrict__ out,
                                                  float* __restrict__ out_val,
                                                  const float* __restrict__ row,
                                                  int row_start,
                                                  int len,
                                                  int K)
{
    for(int i = threadIdx.x; i < K; i += blockDim.x)
    {
        const bool live = (i < len);
        out[i]          = live ? (row_start + i) : -1;
        if(WRITE_VALUES)
            out_val[i] = live ? row[i] : -__builtin_inff();
    }
}

// LDS capacity for the Phase C candidate set (keys + indices).
constexpr int PHASE_C_CAP = 4096; // 4096 * (4+4) B = 32 KB LDS

__device__ __forceinline__ void clear_hist(uint32_t* __restrict__ s_hist)
{
    for(int i = threadIdx.x; i < HIST_SLOTS; i += blockDim.x)
        s_hist[i] = 0u;
}

// IEEE-754 fp32 -> monotone uint32. NaN lands above +INF, matching the
// "distort" trick from DeepSelect (csrc/hip_kernels/bit_utils_hip.cuh).
__host__ __device__ __forceinline__ uint32_t fp32_to_sortable_bits(uint32_t u)
{ return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u); }

__device__ __forceinline__ uint32_t fp32_to_sortable(float v)
{ return fp32_to_sortable_bits(__float_as_uint(v)); }

__device__ __forceinline__ float sortable_to_fp32(uint32_t s)
{
    uint32_t u = (s & 0x80000000u) ? (s ^ 0x80000000u) : ~s;
    return __uint_as_float(u);
}

// Native ext_vector_type: __builtin_nontemporal_load rejects HIP_vector_type.
using vfloat4 = opus::fp32x4_t;

// vec4 load at vector index `i` of a row slice that is `len` floats long, which
// never reads past the slice.
//
// n4_cover() rounds the vector count UP, so the last vector of a slice whose
// length is not a multiple of 4 reaches up to 3 floats beyond it. For an
// interior row those floats belong to the next row and are harmless -- every
// consumer predicates on `< len` -- but once nonzero rowStarts make
// `row_start + len` land within 3 floats of the END OF THE ALLOCATION it is a
// HIP 700. Measured on the shipped tree: M=256 N=131072 prefix=131072 with
// --row-starts-stride 65 faults, while the SAME strides at prefix=100000, where
// the slice ends well before the pitch, pass on every distribution. That pair is
// what isolates the cause to the over-read.
//
// It is NOT a misalignment. gfx950 serves a 4-byte-aligned global_load_dwordx4
// natively: strides 1, 3, 7 and 65 give base % 4 in {1, 3} and pass at
// prefix=100000 on both the sampled and the small_n path, with the plain 16-byte
// vfloat4 typedef. Declaring the typedef `aligned(4)` was tried and changed
// nothing, which is the other half of the same evidence.
//
// RAGGED=false cannot over-read -- the slice is the whole row, len == pitch, and
// the vector count is an exact division -- so that instantiation keeps the plain
// load and byte-identical codegen. The aiter entry always instantiates
// RAGGED=true, so the path that ships gets the clamp.
// Measured cost on the ragged path, interleaved A/B against the pre-fix binary,
// 2 rounds each. Three forms of the same clamp were tried and this one, the
// simplest, is also the cheapest:
//   this form, `e + FP32_EPT <= len` inline        +0.39% .. +0.88%
//   `i < n4_full` with n4_full hoisted per row     +0.52% .. +1.00%
//   clamp armed only on row == gridDim-1          +0.55% .. +2.63%
// So the cost is the branch existing in the loop at all, not the arithmetic
// feeding it, and paying it once per vector beats trying to predicate it away.
template <bool RAGGED, bool NT = false>
__device__ __forceinline__ vfloat4 load_row_f4(const float* __restrict__ row, int i, int len)
{
    const vfloat4* v4 = reinterpret_cast<const vfloat4*>(row);
    if constexpr(!RAGGED)
    {
        (void)len;
        // NT: every element of a row is read by exactly one block and never looked
        // at again, so a cache line buys the row data nothing and evicts what the
        // other blocks are still reading. Same bytes, same addresses. Whether that
        // helps depends on whether the input could have stayed resident at all --
        // see the gate at the phase_b launch.
        if constexpr(NT)
            return __builtin_nontemporal_load(v4 + i);
        else
            return v4[i];
    }
    else
    {
        const int e = i * FP32_EPT;
        if(e + FP32_EPT <= len)
        {
            if constexpr(NT)
                return __builtin_nontemporal_load(v4 + i);
            else
                return v4[i];
        }
        // Zero, not garbage: no consumer looks at a lane whose column is >= len, so
        // the value is unobservable, and zero keeps it that way if one ever does.
        vfloat4 v = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for(int j = 0; j < FP32_EPT; j++)
            if(e + j < len)
                v[j] = row[e + j];
        return v;
    }
}

// Block-wide min and max of each thread's (mn, mx), left on EVERY thread: an
// xor butterfly within the wave (a __shfl_down tree would leave it in lane 0
// only -- that exact trap cost 3% recall in the DeepSelect port), then the wave
// results through s_mm (2 * MAX_WAVES_PER_BLOCK words). All threads must call.
__device__ __forceinline__ void
block_minmax(uint32_t& mn, uint32_t& mx, uint32_t* __restrict__ s_mm)
{
    const int lane   = threadIdx.x & (WAVE_SIZE - 1);
    const int wv     = threadIdx.x / WAVE_SIZE;
    const int nwaves = blockDim.x / WAVE_SIZE;
#pragma unroll
    for(int off = WAVE_SIZE / 2; off > 0; off >>= 1)
    {
        mn = min(mn, (uint32_t)__shfl_xor(mn, off));
        mx = max(mx, (uint32_t)__shfl_xor(mx, off));
    }
    if(lane == 0)
    {
        s_mm[wv]                       = mn;
        s_mm[MAX_WAVES_PER_BLOCK + wv] = mx;
    }
    __syncthreads();
    mn = 0xFFFFFFFFu;
    mx = 0u;
    for(int w = 0; w < nwaves; w++)
    {
        mn = min(mn, s_mm[w]);
        mx = max(mx, s_mm[MAX_WAVES_PER_BLOCK + w]);
    }
}

// Block-wide min and max of keys already in LDS. All threads must call.
__device__ __forceinline__ void block_minmax_lds(const uint32_t* __restrict__ s_keys,
                                                 int c,
                                                 uint32_t* __restrict__ s_mm,
                                                 uint32_t& out_min,
                                                 uint32_t& out_max)
{
    uint32_t mn = 0xFFFFFFFFu, mx = 0u;
    for(int i = threadIdx.x; i < c; i += blockDim.x)
    {
        uint32_t k = s_keys[i];
        mn         = min(mn, k);
        mx         = max(mx, k);
    }
    block_minmax(mn, mx, s_mm);
    out_min = mn;
    out_max = mx;
}

// Highest radix pass at which min and max still agree can be skipped outright:
// if every key shares that byte, the pass's histogram lands entirely in one
// bucket and contributes nothing to the pivot but the byte itself.
__device__ __host__ __forceinline__ int common_prefix_passes(uint32_t mn, uint32_t mx)
{
    int start = 0;
    while(start < RADIX_PASSES)
    {
        const int sh = 24 - 8 * start;
        if(((mn >> sh) & 0xFFu) != ((mx >> sh) & 0xFFu))
            break;
        start++;
    }
    return start;
}

#ifndef BCAST_RL_MUTANT
#define BCAST_RL_MUTANT 0
#endif
// v from lane `src`, which must be the same on every lane (a constant or a
// ballot index): v_readlane instead of a ds_bpermute round trip.
__device__ __forceinline__ uint32_t wave_bcast(uint32_t v, int src)
{
#if BCAST_RL_MUTANT // gate self-test only: reads the neighbouring lane
    src = (src + 1) & (WAVE_SIZE - 1);
#endif
    return (uint32_t)__builtin_amdgcn_readlane((int)v, src);
}

// Tie-correct gather shared by Phase C and the fallback: emits the indices of
// every key strictly above the pivot, then exactly eq_needed of the keys equal
// to it. Uses one LDS atomic per wave via ballot/popcount rather than one per
// element -- the per-element form serialized ~2048 atomics onto a single
// address inside every Phase C block.
//
// KeyFn(i) -> sortable key, IdxFn(i) -> output index. All threads must call.
//
// WRITE_VALUES costs one store per emitted element and NOT a second read of the
// row: the key is already in a register here, and sortable_to_fp32() inverts
// fp32_to_sortable_bits() exactly (both are bijective bit ops, same file), so
// the selected score is recovered arithmetically. Every caller's KeyFn yields a
// sortable key -- s_keys / s_keys_ext are stored that way, and the streaming
// path applies fp32_to_sortable() in its lambda -- so this holds on all four.
template <bool WRITE_VALUES, typename KeyFn, typename IdxFn>
__device__ __forceinline__ void block_gather_topk(int c,
                                                  uint32_t pivot,
                                                  int ngt,
                                                  int eq_needed,
                                                  int* __restrict__ out,
                                                  float* __restrict__ out_val,
                                                  unsigned* __restrict__ s_wgt,
                                                  unsigned* __restrict__ s_weq,
                                                  KeyFn key_at,
                                                  IdxFn idx_at)
{
    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const uint64_t lt = (1ull << lane) - 1ull;
    for(int i0 = 0; i0 < c; i0 += blockDim.x)
    {
        const int i       = i0 + threadIdx.x;
        const bool has    = (i < c);
        const uint32_t k  = has ? key_at(i) : 0u;
        const bool gt     = has && (k > pivot);
        const bool eq     = has && (k == pivot);
        const uint64_t bg = __ballot(gt);
        const uint64_t be = __ballot(eq);
        const int tg      = __popcll(bg);
        const int te      = __popcll(be);
        unsigned baseg = 0, basee = 0;
        if(lane == 0)
        {
            if(tg)
                baseg = atomicAdd(s_wgt, (unsigned)tg);
            if(te)
                basee = atomicAdd(s_weq, (unsigned)te);
        }
        baseg = wave_bcast(baseg, 0);
        basee = wave_bcast(basee, 0);
        if(gt)
        {
            unsigned p = baseg + (unsigned)__popcll(bg & lt);
            if(p < (unsigned)ngt)
            {
                out[p] = idx_at(i);
                if(WRITE_VALUES)
                    out_val[p] = sortable_to_fp32(k);
            }
        }
        if(eq)
        {
            unsigned p = basee + (unsigned)__popcll(be & lt);
            if(p < (unsigned)eq_needed)
            {
                out[ngt + p] = idx_at(i);
                if(WRITE_VALUES)
                    out_val[ngt + p] = sortable_to_fp32(k);
            }
        }
    }
}

// Byte offset of radix pass p (MSB first).
__device__ __host__ __forceinline__ int radix_shift(int pass) { return 24 - 8 * pass; }

template <int ctrl, int row_mask, int bank_mask>
__device__ __forceinline__ uint32_t dpp_add_u32(uint32_t x)
{
    return x + opus::upd_dpp(0u,
                             x,
                             opus::number<ctrl>{},
                             opus::number<row_mask>{},
                             opus::number<bank_mask>{},
                             opus::bool_constant<false>{});
}
// Inclusive suffix sum over one fully active wave64 (lane i: lanes i..63).
// The prefix is the GCN row_shr/row_bcast DPP sequence aiter's plain top-k
// ships (topk_per_row_kernels.cu, wave_inclusive_sum_dpp_u32), pure VALU;
// the __shfl_down tree it replaces is six ds_bpermute + lgkmcnt(0) round trips.
__device__ __forceinline__ uint32_t wave_suffix_sum(uint32_t v)
{
    uint32_t x = v;
    x          = dpp_add_u32<0x111, 0xf, 0xf>(x); // row_shr:1
    x          = dpp_add_u32<0x112, 0xf, 0xf>(x); // row_shr:2
    x          = dpp_add_u32<0x114, 0xf, 0xe>(x); // row_shr:4
    x          = dpp_add_u32<0x118, 0xf, 0xc>(x); // row_shr:8
    x          = dpp_add_u32<0x142, 0xa, 0xf>(x); // row_bcast:15
#if !SCAN_DPP_MUTANT // gate self-test only: without this step lanes 32..63 miss lanes 0..31
    x = dpp_add_u32<0x143, 0xc, 0xf>(x); // row_bcast:31
#endif
    const uint32_t tot = (uint32_t)__builtin_amdgcn_readlane((int)x, WAVE_SIZE - 1);
    return tot - x + v;
}

// Single-wave form of the scan: wave 0 alone reduces the replicas, scans all 256
// buckets and publishes the pivot, so a radix pass needs 2 block barriers
// instead of 3.
//
// Why that works. A block-wide scan, one wave per 64 buckets, needs TWO
// barriers for two different reasons: one to publish `s_wavetot`, the
// cross-wave partial sums that exist
// only because 256 buckets span 4 waves, and one to publish `s_scan` to the
// block. Confine the scan to one wave and the first reason disappears entirely
// -- 256 buckets at 4 per lane fit in one wave, so the whole suffix scan is
// shuffles with no barrier. The second barrier stays, and CLEAR still rides on
// it for free, because wave 0 is now the only reader of the histogram and can
// zero each slot as it reads it.
//
// Costs no LDS and reads no slot twice; what it does do is concentrate 1024 slot
// reads onto 64 lanes (16 per lane, against 4 per lane spread over 4 waves), so
// it trades block-barrier latency for LDS-read depth on one wave. g_14 showed
// that trade can go either way by regime, so measure both ends before shipping.
//
// Rejected alternative: let EVERY wave scan the whole histogram redundantly,
// which removes both barriers. It does not help -- with all waves reading all
// slots, no wave may zero anything until all have finished, so the clear needs
// its own before-and-after barrier pair and the pass is back to 3, now with 4x
// the LDS reads. Reaching 2 that way needs a double-buffered histogram (+4 KB),
// which takes phase_a from 4 to 3 blocks/CU at S=8192 (163840/42008 vs
// 163840/37912) for a barrier that g_14 measured at -0.35% on the anchor.
template <bool CLEAR = false>
__device__ __forceinline__ void
block_find_pivot_bucket_wave0(uint32_t* __restrict__ s_hist, uint32_t* __restrict__ s_scan, int ek)
{
    constexpr int PER_LANE = 256 / WAVE_SIZE;
    if(threadIdx.x < WAVE_SIZE)
    {
        const int lane = (int)threadIdx.x;
        uint32_t v[PER_LANE];
        uint32_t tot = 0;
#pragma unroll
        for(int j = 0; j < PER_LANE; j++)
        {
            const int b = lane * PER_LANE + j;
            uint32_t s  = 0;
#pragma unroll
            for(int r = 0; r < HIST_REP; r++)
            {
                s += s_hist[b * HIST_REP + r];
                if constexpr(CLEAR)
                    s_hist[b * HIST_REP + r] = 0u;
            }
            v[j] = s;
            tot += s;
        }
        // Inclusive suffix sum of the per-lane totals, so above_lane is everything
        // in buckets above this lane's group.
        const uint32_t inc = wave_suffix_sum(tot);
        uint32_t acc       = inc - tot;
        int hit_j          = -1;
        uint32_t hit_above = 0;
#pragma unroll
        for(int j = PER_LANE - 1; j >= 0; j--)
        {
            const uint32_t nxt = acc; // suffix sum at bucket b+1
            acc += v[j];              // suffix sum at bucket b
            if(ek > 0 && acc >= (uint32_t)ek && nxt < (uint32_t)ek)
            {
                hit_j     = j;
                hit_above = nxt;
            }
        }
        // Shuffles run on every lane, outside the lane-0 store, or the ones that
        // did not elect themselves would not participate.
        const uint64_t bal       = __ballot(hit_j >= 0);
        const int src            = bal != 0ull ? __builtin_ctzll(bal) : 0;
        const int j_sel          = (int)wave_bcast((uint32_t)hit_j, src);
        const uint32_t above_sel = wave_bcast(hit_above, src);
        if(bal != 0ull && lane == 0)
        {
            s_scan[0] = (uint32_t)(src * PER_LANE + j_sel);
            s_scan[1] = above_sel;
        }
    }
    __syncthreads();
}

// 12-bit radix digit for the filtered passes, so an exact 32-bit select is
// 8 + 12 + 12 bits in three passes instead of four, and phase_a's threshold is
// 8 + 12 bits in two instead of 8 + 8 + 8 in three. A pass costs its barriers
// and its scan latency, not its data (knowledge/known_bad.md, "What actually
// sets phase A and phase C cost"), so the lever is the pass count.
//
// A flat 4096-bucket scan would give that back: one wave reading 4096 slots, or
// a block-wide scan with a third barrier. Instead every counted key bumps two
// histograms, 64 coarse buckets (the digit's top 6 bits) and 4096 fine ones,
// and wave 0 scans the coarse one (one slot per lane) and then only the 64 fine
// buckets under the coarse bucket it picked. Two slot reads per lane and the
// same 2 barriers per pass as block_find_pivot_bucket_wave0.
//
// The filtered passes count only keys whose higher digits match the pivot, so
// the second atomic per counted key is cheap. The fine buckets outside the
// picked coarse one are never read, so they are not cleared on read: every wide
// pass gets its own WIDE_WORDS buffer, zeroed once by the kernel before its
// first barrier (clear_wide).
//
// The coarse buckets carry HIST_REP replicas like s_hist does: Phase C's
// candidates sit just above one threshold, so most of them share one or two
// coarse buckets and a single counter per bucket serialises the whole wave.
constexpr int WIDE_BITS         = 12;
constexpr int WIDE_COARSE       = 64;
constexpr int WIDE_FINE         = 1 << WIDE_BITS;
constexpr int WIDE_COARSE_SLOTS = WIDE_COARSE * HIST_REP;
constexpr int WIDE_WORDS        = WIDE_COARSE_SLOTS + WIDE_FINE;
static_assert(WIDE_COARSE == WAVE_SIZE && WIDE_FINE == WIDE_COARSE * WAVE_SIZE,
              "the two-level scan puts one bucket per lane at each level");

__host__ __device__ static inline constexpr int wide_buffer_count(int nwide, bool reuse)
{ return reuse && nwide > 0 ? 1 : nwide; }

// One counted key: its coarse bucket (replica `rep`) and its fine bucket.
__device__ __forceinline__ void wide_count(uint32_t* __restrict__ s_w, uint32_t d, int rep)
{
    atomicAdd(&s_w[(d >> 6) * HIST_REP + rep], 1u);
    atomicAdd(&s_w[WIDE_COARSE_SLOTS + d], 1u);
}
static_assert(WIDE_WORDS % 4 == 0, "clear_wide stores 16 bytes per thread");

__device__ __forceinline__ void clear_wide(uint32_t* __restrict__ s_wide, int nbuf)
{
    const opus::u32x4_t z = {0u, 0u, 0u, 0u};
    for(int i = threadIdx.x; i < nbuf * WIDE_WORDS / 4; i += blockDim.x)
        reinterpret_cast<opus::u32x4_t*>(s_wide)[i] = z;
}

// Coarse buckets (HIST_REP replicas each) and fine buckets in separate arrays;
// the one-buffer form below keeps them back to back.
__device__ __forceinline__ void block_find_pivot_wide_wave0(const uint32_t* __restrict__ s_coarse,
                                                            const uint32_t* __restrict__ s_fine,
                                                            uint32_t* __restrict__ s_scan,
                                                            int ek)
{
    if(threadIdx.x < WAVE_SIZE)
    {
        const int lane = (int)threadIdx.x;
        uint32_t cv    = 0;
#pragma unroll
        for(int r = 0; r < HIST_REP; r++)
            cv += s_coarse[lane * HIST_REP + r];
        const uint32_t cinc    = wave_suffix_sum(cv);
        const uint32_t cnxt    = cinc - cv;
        const uint64_t cbal    = __ballot(ek > 0 && cinc >= (uint32_t)ek && cnxt < (uint32_t)ek);
        const int cb           = cbal != 0ull ? __builtin_ctzll(cbal) : 0;
        const uint32_t above_c = wave_bcast(cnxt, cb);

        const uint32_t fv      = s_fine[cb * WAVE_SIZE + lane];
        const uint32_t finc    = wave_suffix_sum(fv);
        const uint32_t fs      = above_c + finc;
        const uint32_t fnxt    = fs - fv;
        const uint64_t fbal    = __ballot(ek > 0 && fs >= (uint32_t)ek && fnxt < (uint32_t)ek);
        const int fb           = fbal != 0ull ? __builtin_ctzll(fbal) : 0;
        const uint32_t above_f = wave_bcast(fnxt, fb);
        if(cbal != 0ull && fbal != 0ull && lane == 0)
        {
            s_scan[0] = (uint32_t)(cb * WAVE_SIZE + fb);
            s_scan[1] = above_f;
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void
block_find_pivot_wide_wave0(const uint32_t* __restrict__ s_w, uint32_t* __restrict__ s_scan, int ek)
{ block_find_pivot_wide_wave0(s_w, s_w + WIDE_COARSE_SLOTS, s_scan, ek); }
