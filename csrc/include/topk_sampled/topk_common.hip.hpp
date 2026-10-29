// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

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

#include "dpp_scan.h"
#include "opus/opus.hpp"
#include "topk_fp32_key.h"

constexpr int WAVE_SIZE = opus::get_warp_size();
// The kernels are wave64 code tuned for gfx950, and their launch plan assumes
// its 256 CUs and 160 KiB of LDS (topk_shape.hip.hpp). A device pass for any
// other target compiles each kernel to a trap so the module that carries them
// still builds there; topk_sampled_supports() keeps them from being launched.
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx950__)
#define TOPK_SAMPLED_DEVICE 0
#else
#define TOPK_SAMPLED_DEVICE 1
#endif
constexpr int FP32_EPT     = 4; // floats per dwordx4 load
constexpr int RADIX_PASSES = 4; // 4 x 8-bit covers all 32 sortable bits

// Replicas of each histogram bucket. A sortable fp32's top byte is sign+exponent,
// so on uniform[-1,1] data roughly half of all positive values share ONE bucket
// and the per-element LDS atomicAdd serialises hard. Lanes are spread across
// HIST_REP adjacent counters (adjacent => different LDS banks), summed before
// the scan. HIST_REP=1 restores the plain histogram.
constexpr int HIST_REP   = 4;
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
{
    return (N / chunks) & ~(FP32_EPT - 1);
}

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
// What it prevents, measured on the unclamped build:
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
// two runs each, interleaved A/B). small_n runs one short block per row, so its kernarg prologue is
// a real share of the kernel rather than noise.
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
    {
        return val + (size_t)row * K;
    }
};

// The value padding is -inf, NOT 0, and that is aiter's rule rather than a
// preference (radix_topk_one_block_kernel states it): the index slot is -1, so
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
// (radix_kernel_persistent on the multi-block path, radix_topk_one_block_kernel
// on the one-block path), and matching it is not
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

// The bit-level keys -- fp32_to_sortable_bits, sortable_key_bits, NAN_HIGH_SHIFT
// -- are topk_fp32_key.h's, which the radix selector ranks by too.
__device__ __forceinline__ uint32_t fp32_to_sortable(float v)
{
    return fp32_to_sortable_bits(__float_as_uint(v));
}

// topk_select ranks a NaN of either sign above +inf, the order of its stream,
// decode and small_k selectors. top_k_per_row_prefill's radix and FlyDSL paths
// keep the sign instead: a negative NaN sits below -inf, as
// fp32_to_sortable_bits orders it. NAN_HIGH picks topk_select's order.
template <bool NAN_HIGH>
__device__ __forceinline__ uint32_t sortable_key(float v)
{
    return sortable_key_bits<NAN_HIGH>(__float_as_uint(v));
}

__device__ __forceinline__ float sortable_to_fp32(uint32_t s)
{
    return __uint_as_float(sortable_to_fp32_bits(s));
}

template <bool NAN_HIGH>
__device__ __forceinline__ float key_to_fp32(uint32_t k)
{
    return sortable_to_fp32(k + (NAN_HIGH ? NAN_HIGH_SHIFT : 0u));
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
// row starts 65 elements apart faults, while the SAME strides at prefix=100000, where
// the slice ends well before the pitch, pass on every distribution. That pair is
// what isolates the cause to the over-read.
//
// It is NOT a misalignment. gfx950 serves a 4-byte-aligned global_load_dwordx4
// natively: strides 1, 3, 7 and 65 give base % 4 in {1, 3} and pass at
// prefix=100000 on both the sampled and the small_n path, with the plain 16-byte
// vfloat4 typedef. Declaring the typedef `aligned(4)` was tried and changed
// nothing, which is the other half of the same evidence.
//
// RAGGED=false cannot over-read here -- the slice is the whole row, len == pitch
// -- so that instantiation keeps the plain load and byte-identical codegen. The
// aiter entry runs it only for rows whose pitch is their width: _sampled_run
// (topk.py) runs a row-strided call as ragged, a width that is not a multiple of
// FP32_EPT takes the bounded small_n build (ragged_small), and on the fused path
// phases B and C read a row's partial last vector themselves: phase_b_filter_coop
// picks the tail up in its last-chunk block, and phase_c_select_contig's fallback
// re-reads such a row through the bounds-checked loader. Every row that needs
// the clamp gets it only while all three hold.
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
        // see NT_LOAD_LOG2 in plan_launch.
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

// v from lane `src`, which must be the same on every lane (a constant or a
// ballot index): v_readlane instead of a ds_bpermute round trip.
__device__ __forceinline__ uint32_t wave_bcast(uint32_t v, int src)
{
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

// Inclusive suffix sum over one fully active wave64 (lane i: lanes i..63), from
// the DPP prefix scan in dpp_scan.h, pure VALU; the __shfl_down tree it replaces
// is six ds_bpermute + lgkmcnt(0) round trips.
__device__ __forceinline__ uint32_t wave_suffix_sum(uint32_t v)
{
    const uint32_t x   = aiter::wave_inclusive_sum_dpp(v);
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
// it trades block-barrier latency for LDS-read depth on one wave. That trade
// was measured to go either way by regime, so measure both ends before shipping.
//
// Rejected alternative: let EVERY wave scan the whole histogram redundantly,
// which removes both barriers. It does not help -- with all waves reading all
// slots, no wave may zero anything until all have finished, so the clear needs
// its own before-and-after barrier pair and the pass is back to 3, now with 4x
// the LDS reads. Reaching 2 that way needs a double-buffered histogram (+4 KB),
// which takes phase_a from 4 to 3 blocks/CU at S=8192 (163840/42008 vs
// 163840/37912) for a barrier measured at -0.35% at M=4096 N=131072.
template <bool CLEAR>
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
// and its scan latency, not its data, so the lever is the pass count.
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
static_assert(!TOPK_SAMPLED_DEVICE ||
                  (WIDE_COARSE == WAVE_SIZE && WIDE_FINE == WIDE_COARSE * WAVE_SIZE),
              "the two-level scan puts one bucket per lane at each level");

__host__ __device__ static inline constexpr int wide_buffer_count(int nwide, bool reuse)
{
    return reuse && nwide > 0 ? 1 : nwide;
}

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
{
    block_find_pivot_wide_wave0(s_w, s_w + WIDE_COARSE_SLOTS, s_scan, ek);
}

// The stable emit: the top k_out in ascending column order, a tie at the pivot
// going to the smaller column. That is top_k_per_row_prefill's stable=True
// contract, and topk_select's for tie='low' and deterministic=True; both make
// every tensor-parallel rank select and order the same set. block_gather_topk
// places by atomic counters, so its order and its pick among tied keys follow
// wave timing; the two emitters below place by counting instead.
//
// Both apply one rule. An element above the pivot is always selected; one equal
// to it is selected while fewer than eq_needed equal keys lie to its left. A
// selected element's slot is the number of selected elements to its left: the
// keys above the pivot there, plus the equal ones there capped at eq_needed.
//
// NAN_ONE is topk_select's order for tie='low', where every NaN is one value
// above +inf (its decode selector ranks them so). The NAN_HIGH key ranks NaNs
// by payload instead, which changes the answer only when the pivot itself is a
// NaN -- otherwise every NaN is above it under both -- so that is the one case
// remapped: no key above the NaN class, and k_out of its members taken by
// column. The caller has to show the emitter every NaN of the row then.
constexpr uint32_t NAN_ONE_KEY = sortable_key_bits<true>(0x7F800001u); // the lowest positive NaN
static_assert(NAN_ONE_KEY == sortable_key_bits<true>(0x7F800000u) + 1u &&
                  sortable_key_bits<true>(0xFFFFFFFFu) > NAN_ONE_KEY,
              "every NaN keys above +inf, the lowest positive one right above it");

template <bool NAN_ONE>
__device__ __forceinline__ void nan_one_remap(uint32_t& pivot, int& ngt, int& eq_needed)
{
    if(NAN_ONE && pivot >= NAN_ONE_KEY)
    {
        pivot = NAN_ONE_KEY;
        eq_needed += ngt;
        ngt = 0;
    }
}

template <bool NAN_ONE>
__device__ __forceinline__ uint32_t nan_one_key(uint32_t k)
{
    return NAN_ONE ? min(k, NAN_ONE_KEY) : k;
}

// One element at its slot, given the counts of selected-class keys to its left.
template <bool WRITE_VALUES>
__device__ __forceinline__ void stable_place(uint32_t k,
                                             uint32_t pivot,
                                             unsigned g,
                                             unsigned q,
                                             int eq_needed,
                                             int col,
                                             int* __restrict__ out,
                                             float* __restrict__ out_val)
{
    int pos = -1;
    if(k > pivot)
        pos = (int)(g + min(q, (unsigned)eq_needed));
    else if(k == pivot && q < (unsigned)eq_needed)
        pos = (int)(g + q);
    if(pos >= 0)
    {
        out[pos] = col;
        if(WRITE_VALUES)
            out_val[pos] = sortable_to_fp32(k);
    }
}

// Rows read in column order: small_n's keys in LDS and the fallback's row in
// global memory. Each wave owns one contiguous run of the row in whole 64-vector
// steps; a counting pass gives every run its base, and an emit pass walks the
// run again, one wave scan per step placing each lane's four columns.
// keys4(j) returns the keys of columns 4j..4j+3 and is asked only for j < n4;
// columns at or past len are masked here. s_wtot is 2 * MAX_WAVES_PER_BLOCK
// words of scratch. All threads must call.
// Two vectors in flight per lane: phase_c inlines this twice next to its other
// emitters, and four put one instantiation past its 64-VGPR budget into scratch.
constexpr int STABLE_ROW_LOADS = 2;
// The row goes by in chunks of this many vectors, each counted and then placed,
// and the emit stops at the chunk that fills both quotas: a row with a long tie
// at the pivot is done in its first chunk instead of counted whole first.
constexpr int STABLE_ROW_CHUNK_VECS = 8192;

// keys4 for a row in global memory, bounds-checked like block_gather_stream's
// loads so a row of any width is read whole.
template <bool NAN_HIGH>
__device__ __forceinline__ opus::u32x4_t row_keys4(const float* __restrict__ row, int j, int len)
{
    const vfloat4 v = load_row_f4<true>(row, j, len);
    return opus::u32x4_t{sortable_key<NAN_HIGH>(v[0]),
                         sortable_key<NAN_HIGH>(v[1]),
                         sortable_key<NAN_HIGH>(v[2]),
                         sortable_key<NAN_HIGH>(v[3])};
}

template <bool WRITE_VALUES, bool NAN_ONE, bool CHUNKED = false, typename Keys4Fn>
__device__ __forceinline__ void block_gather_in_order(int len,
                                                      int idx_base,
                                                      uint32_t pivot,
                                                      int ngt,
                                                      int eq_needed,
                                                      int* __restrict__ out,
                                                      float* __restrict__ out_val,
                                                      uint32_t* __restrict__ s_wtot,
                                                      Keys4Fn keys4)
{
    if constexpr(!CHUNKED)
    {
        static_assert(!(NAN_ONE && WRITE_VALUES), "NAN_ONE is instantiated for indices only");
        nan_one_remap<NAN_ONE>(pivot, ngt, eq_needed);
        const int lane     = threadIdx.x & (WAVE_SIZE - 1);
        const int wid      = threadIdx.x / WAVE_SIZE;
        const int nwaves   = blockDim.x / WAVE_SIZE;
        const int n4       = n4_cover(len);
        const int steps    = (n4 + WAVE_SIZE - 1) / WAVE_SIZE;
        const int run      = (steps + nwaves - 1) / nwaves * WAVE_SIZE;
        const int v0       = min(wid * run, n4);
        const int v1       = min(v0 + run, n4);
        constexpr int SPAN = STABLE_ROW_LOADS * WAVE_SIZE;

        unsigned cg = 0, ce = 0;
        for(int jb = v0; jb < v1; jb += SPAN)
        {
            opus::u32x4_t kv[STABLE_ROW_LOADS];
#pragma unroll
            for(int u = 0; u < STABLE_ROW_LOADS; u++)
            {
                const int j = jb + u * WAVE_SIZE + lane;
                kv[u]       = j < v1 ? keys4(j) : opus::u32x4_t{0u, 0u, 0u, 0u};
            }
#pragma unroll
            for(int u = 0; u < STABLE_ROW_LOADS; u++)
            {
                const int j = jb + u * WAVE_SIZE + lane;
#pragma unroll
                for(int e = 0; e < FP32_EPT; e++)
                {
                    const bool has   = j < v1 && j * FP32_EPT + e < len;
                    const uint32_t k = nan_one_key<NAN_ONE>(kv[u][e]);
                    cg += (has && k > pivot) ? 1u : 0u;
                    ce += (has && k == pivot) ? 1u : 0u;
                }
            }
        }
        // Lane 0's inclusive suffix is the wave's total.
        const uint32_t tg = wave_suffix_sum(cg);
        const uint32_t te = wave_suffix_sum(ce);
        if(lane == 0)
        {
            s_wtot[wid]                       = tg;
            s_wtot[MAX_WAVES_PER_BLOCK + wid] = te;
        }
        __syncthreads();
        unsigned bg = 0, be = 0;
        for(int w = 0; w < wid; w++)
        {
            bg += s_wtot[w];
            be += s_wtot[MAX_WAVES_PER_BLOCK + w];
        }
        __syncthreads(); // s_wtot is the caller's again

        for(int jb = v0; jb < v1; jb += SPAN)
        {
            // Wave-uniform: every selected element of this run is placed.
            if(bg >= (unsigned)ngt && be >= (unsigned)eq_needed)
                break;
            opus::u32x4_t kv[STABLE_ROW_LOADS];
#pragma unroll
            for(int u = 0; u < STABLE_ROW_LOADS; u++)
            {
                const int j = jb + u * WAVE_SIZE + lane;
                kv[u]       = j < v1 ? keys4(j) : opus::u32x4_t{0u, 0u, 0u, 0u};
            }
#pragma unroll
            for(int u = 0; u < STABLE_ROW_LOADS; u++)
            {
                const int j = jb + u * WAVE_SIZE + lane;
                uint32_t k[FP32_EPT];
                bool has[FP32_EPT];
                uint32_t lg = 0, le = 0;
#pragma unroll
                for(int e = 0; e < FP32_EPT; e++)
                {
                    has[e] = j < v1 && j * FP32_EPT + e < len;
                    k[e]   = nan_one_key<NAN_ONE>(kv[u][e]);
                    lg += (has[e] && k[e] > pivot) ? 1u : 0u;
                    le += (has[e] && k[e] == pivot) ? 1u : 0u;
                }
                // At most 4 per lane, 256 per step: both counts fit a half each. Lane 0's
                // inclusive suffix is the step's total, so total minus a lane's own suffix
                // is what the lanes to its left hold.
                const uint32_t s    = wave_suffix_sum(lg | (le << 16));
                const uint32_t tot  = (uint32_t)__builtin_amdgcn_readfirstlane((int)s);
                const uint32_t left = tot - s;
                unsigned g          = bg + (left & 0xFFFFu);
                unsigned q          = be + (left >> 16);
#pragma unroll
                for(int e = 0; e < FP32_EPT; e++)
                {
                    if(!has[e])
                        continue;
                    stable_place<WRITE_VALUES>(
                        k[e], pivot, g, q, eq_needed, idx_base + j * FP32_EPT + e, out, out_val);
                    g += k[e] > pivot ? 1u : 0u;
                    q += k[e] == pivot ? 1u : 0u;
                }
                bg += tot & 0xFFFFu;
                be += tot >> 16;
            }
        }
    }
    else
    {
        static_assert(!(NAN_ONE && WRITE_VALUES), "NAN_ONE is instantiated for indices only");
        nan_one_remap<NAN_ONE>(pivot, ngt, eq_needed);
        const int lane     = threadIdx.x & (WAVE_SIZE - 1);
        const int wid      = threadIdx.x / WAVE_SIZE;
        const int nwaves   = blockDim.x / WAVE_SIZE;
        const int n4       = n4_cover(len);
        constexpr int SPAN = STABLE_ROW_LOADS * WAVE_SIZE;
        // Block-uniform: the selected keys of the chunks already placed.
        unsigned done_g = 0, done_e = 0;
        const int chunk = CHUNKED ? STABLE_ROW_CHUNK_VECS : n4;
        for(int c0 = 0; c0 < n4; c0 += chunk)
        {
            const int c1    = min(c0 + chunk, n4);
            const int steps = (c1 - c0 + WAVE_SIZE - 1) / WAVE_SIZE;
            const int run   = (steps + nwaves - 1) / nwaves * WAVE_SIZE;
            const int v0    = min(c0 + wid * run, c1);
            const int v1    = min(v0 + run, c1);

            unsigned cg = 0, ce = 0;
            for(int jb = v0; jb < v1; jb += SPAN)
            {
                opus::u32x4_t kv[STABLE_ROW_LOADS];
#pragma unroll
                for(int u = 0; u < STABLE_ROW_LOADS; u++)
                {
                    const int j = jb + u * WAVE_SIZE + lane;
                    kv[u]       = j < v1 ? keys4(j) : opus::u32x4_t{0u, 0u, 0u, 0u};
                }
#pragma unroll
                for(int u = 0; u < STABLE_ROW_LOADS; u++)
                {
                    const int j = jb + u * WAVE_SIZE + lane;
#pragma unroll
                    for(int e = 0; e < FP32_EPT; e++)
                    {
                        const bool has   = j < v1 && j * FP32_EPT + e < len;
                        const uint32_t k = nan_one_key<NAN_ONE>(kv[u][e]);
                        cg += (has && k > pivot) ? 1u : 0u;
                        ce += (has && k == pivot) ? 1u : 0u;
                    }
                }
            }
            // Lane 0's inclusive suffix is the wave's total.
            const uint32_t tg = wave_suffix_sum(cg);
            const uint32_t te = wave_suffix_sum(ce);
            if(lane == 0)
            {
                s_wtot[wid]                       = tg;
                s_wtot[MAX_WAVES_PER_BLOCK + wid] = te;
            }
            __syncthreads();
            unsigned bg = done_g, be = done_e;
            for(int w = 0; w < nwaves; w++)
            {
                const unsigned wg = s_wtot[w], we = s_wtot[MAX_WAVES_PER_BLOCK + w];
                bg += w < wid ? wg : 0u;
                be += w < wid ? we : 0u;
                done_g += wg;
                done_e += we;
            }
            __syncthreads(); // s_wtot is the caller's again

            for(int jb = v0; jb < v1; jb += SPAN)
            {
                // Wave-uniform: every selected element of this run is placed.
                if(bg >= (unsigned)ngt && be >= (unsigned)eq_needed)
                    break;
                opus::u32x4_t kv[STABLE_ROW_LOADS];
#pragma unroll
                for(int u = 0; u < STABLE_ROW_LOADS; u++)
                {
                    const int j = jb + u * WAVE_SIZE + lane;
                    kv[u]       = j < v1 ? keys4(j) : opus::u32x4_t{0u, 0u, 0u, 0u};
                }
#pragma unroll
                for(int u = 0; u < STABLE_ROW_LOADS; u++)
                {
                    const int j = jb + u * WAVE_SIZE + lane;
                    uint32_t k[FP32_EPT];
                    bool has[FP32_EPT];
                    uint32_t lg = 0, le = 0;
#pragma unroll
                    for(int e = 0; e < FP32_EPT; e++)
                    {
                        has[e] = j < v1 && j * FP32_EPT + e < len;
                        k[e]   = nan_one_key<NAN_ONE>(kv[u][e]);
                        lg += (has[e] && k[e] > pivot) ? 1u : 0u;
                        le += (has[e] && k[e] == pivot) ? 1u : 0u;
                    }
                    // At most 4 per lane, 256 per step: both counts fit a half each. Lane 0's
                    // inclusive suffix is the step's total, so total minus a lane's own suffix
                    // is what the lanes to its left hold.
                    const uint32_t s    = wave_suffix_sum(lg | (le << 16));
                    const uint32_t tot  = (uint32_t)__builtin_amdgcn_readfirstlane((int)s);
                    const uint32_t left = tot - s;
                    unsigned g          = bg + (left & 0xFFFFu);
                    unsigned q          = be + (left >> 16);
#pragma unroll
                    for(int e = 0; e < FP32_EPT; e++)
                    {
                        if(!has[e])
                            continue;
                        stable_place<WRITE_VALUES>(k[e],
                                                   pivot,
                                                   g,
                                                   q,
                                                   eq_needed,
                                                   idx_base + j * FP32_EPT + e,
                                                   out,
                                                   out_val);
                        g += k[e] > pivot ? 1u : 0u;
                        q += k[e] == pivot ? 1u : 0u;
                    }
                    bg += tot & 0xFFFFu;
                    be += tot >> 16;
                }
            }
            if(done_g >= (unsigned)ngt && done_e >= (unsigned)eq_needed)
                break;
        }
    }
}

// phase_c's candidates arrive in reservation order, so they are placed from a
// sort. An entry is column << 1 with the low bit set for a key equal to the
// pivot; 0xFFFFFFFF marks one below it and sorts last. With no more equal keys
// than eq_needed an entry's slot is its sorted index; otherwise a scan of the
// equal flags drops the equal keys past the first eq_needed and closes the
// gaps. A bitonic sort costs log2(n) (log2(n) + 1) / 2 stages whatever the
// row's width; ranking against a column bitmap cost a pass over the candidates
// per window of columns, 3x the default emit at N=1M.
constexpr uint32_t STABLE_NONE   = 0xFFFFFFFFu;
constexpr int STABLE_RANK_DIRECT = 128;

// The list the selected keys are gathered into: s_hist's HIST_SLOTS words for
// k up to half of them, which leaves the other half for ties at the pivot, and
// `cap` words of its own above that, which no row outgrows (c <= cap). Host and
// kernel both ask, so the launch and the carve agree.
__host__ __device__ constexpr int stable_list_words(int K, int cap)
{
    return K <= HIST_SLOTS / 2 ? 0 : cap;
}

// Rows spanning few windows of columns rank their candidates through bitmaps
// instead of sorting: per window of 32 * nw columns, one bit per column above
// the pivot and one per column equal to it, then a block scan of the words'
// popcounts gives every candidate the count to its left. Two passes over the
// candidates per window against a sort's stages: 1.04x-1.94x faster than the
// sort at 2 windows, about even at 8, 0.57x-0.92x at 16 (N=1M).
// s_bm is 3 * nw words, s_wtot MAX_WAVES_PER_BLOCK words; blockDim.x must
// divide nw. A window holds at most c <= PHASE_C_CAP_MAX keys, so each count
// fits a 16-bit half. All threads must call.
constexpr int STABLE_BM_WORDS       = 2048;
constexpr int STABLE_BM_WORDS_WIDE  = 4096;
constexpr int STABLE_BM_MAX_WINDOWS = 4;

template <bool WRITE_VALUES, bool NAN_ONE, typename KeyFn, typename ColFn>
__device__ __forceinline__ void block_gather_by_column(int c,
                                                       int len,
                                                       int idx_base,
                                                       uint32_t pivot,
                                                       int eq_needed,
                                                       int* __restrict__ out,
                                                       float* __restrict__ out_val,
                                                       uint32_t* __restrict__ s_bm,
                                                       int nw,
                                                       uint32_t* __restrict__ s_wtot,
                                                       KeyFn key_at,
                                                       ColFn col_at)
{
    static_assert(!(NAN_ONE && WRITE_VALUES), "NAN_ONE is instantiated for indices only");
    uint32_t* s_gt   = s_bm;
    uint32_t* s_eq   = s_bm + nw;
    uint32_t* s_pre  = s_bm + 2 * nw;
    const int cols   = nw * 32;
    const int lane   = threadIdx.x & (WAVE_SIZE - 1);
    const int wid    = threadIdx.x / WAVE_SIZE;
    const int nwaves = blockDim.x / WAVE_SIZE;
    const int wpt    = nw / (int)blockDim.x;
    const int wb     = (int)threadIdx.x * wpt;
    auto word_counts = [&](int w) {
        return (uint32_t)__popc(s_gt[w]) | ((uint32_t)__popc(s_eq[w]) << 16);
    };
    unsigned carry_g = 0, carry_e = 0;
    // By window number: stepping a column offset past the last window would
    // overflow an int row length within `cols` of INT32_MAX.
    const int nwin = len / cols + (len % cols != 0 ? 1 : 0);
    for(int w = 0; w < nwin; w++)
    {
        const int w0 = w * cols;
        for(int i = threadIdx.x; i < 2 * nw; i += blockDim.x)
            s_bm[i] = 0u;
        __syncthreads();
        for(int i = threadIdx.x; i < c; i += blockDim.x)
        {
            const int col = col_at(i) - w0;
            if((unsigned)col < (unsigned)cols)
            {
                const uint32_t k = nan_one_key<NAN_ONE>(key_at(i));
                if(k > pivot)
                    atomicOr(&s_gt[col >> 5], 1u << (col & 31));
                else if(k == pivot)
                    atomicOr(&s_eq[col >> 5], 1u << (col & 31));
            }
        }
        __syncthreads();
        uint32_t loc = 0;
        for(int t = 0; t < wpt; t++)
            loc += word_counts(wb + t);
        const uint32_t s = wave_suffix_sum(loc);
        if(lane == 0)
            s_wtot[wid] = s;
        __syncthreads();
        uint32_t run = (uint32_t)__builtin_amdgcn_readfirstlane((int)s) - s;
        uint32_t tot = 0;
        for(int w = 0; w < nwaves; w++)
        {
            const uint32_t v = s_wtot[w];
            run += w < wid ? v : 0u;
            tot += v;
        }
        for(int t = 0; t < wpt; t++)
        {
            s_pre[wb + t] = run;
            run += word_counts(wb + t);
        }
        __syncthreads();
        for(int i = threadIdx.x; i < c; i += blockDim.x)
        {
            const int col = col_at(i) - w0;
            if((unsigned)col < (unsigned)cols)
            {
                const uint32_t k = nan_one_key<NAN_ONE>(key_at(i));
                if(k >= pivot)
                {
                    const int wd         = col >> 5;
                    const uint32_t below = (1u << (col & 31)) - 1u;
                    const uint32_t pw    = s_pre[wd];
                    const unsigned g =
                        carry_g + (pw & 0xFFFFu) + (unsigned)__popc(s_gt[wd] & below);
                    const unsigned q = carry_e + (pw >> 16) + (unsigned)__popc(s_eq[wd] & below);
                    stable_place<WRITE_VALUES>(
                        k, pivot, g, q, eq_needed, idx_base + w0 + col, out, out_val);
                }
            }
        }
        carry_g += tot & 0xFFFFu;
        carry_e += tot >> 16;
        __syncthreads(); // the next window clears what this one read
    }
}

// The windowed stable emit. The block bitmaps above walk every window in turn,
// each with all c candidates twice and a scan of its words, and the scan's
// thread-contiguous words conflict in the LDS banks: 3.6us a window at
// 1 x 524288, k=2048. Here the entries at or above the pivot are counted and
// bucketed by window of STABLE_WIN_COLS columns once, and the first nww waves
// then take the windows in parallel, each through its own 3 * STABLE_WIN_WORDS
// words of bitmaps (16-byte aligned; the first nww_first waves' in s_bm, the
// rest in s_bm_rest) read and written 64 lanes x 16 bytes at a time, with no
// block barrier between windows. s_keys: the candidate keys, rewritten in place
// as entries; s_list: the entries; s_wc: HIST_SLOTS words. All threads must call.
constexpr int STABLE_WIN_WORDS = 256;
constexpr int STABLE_WIN_COLS  = 32 * STABLE_WIN_WORDS;
constexpr int STABLE_WIN_MAX   = 512;
// A wave pays its windows one after another, the list sort a log^2 of k: past
// this many windows a wave, the sort is the cheaper of the two. Windowed over
// sort, 1 x 1M..4M, 16 waves: 1.05 / 0.99 / 0.88 at 16 windows a wave for
// k=512 / 1024 / 2048, 1.14 / 1.14 / 1.02 at 24.
__host__ __device__ constexpr int stable_win_per_wave(int K) { return 13 + (K - 512) / 155; }

__host__ __device__ constexpr int stable_win_count(int len)
{
    return len / STABLE_WIN_COLS + (len % STABLE_WIN_COLS != 0 ? 1 : 0);
}
// Replicas of each window's counter and cursor, lane & (rep - 1) picking one:
// lanes of a wave meeting at one counter serialise, as at s_hist (1x131200,
// k=2048: the count pass 1.68us with one, 0.80us with four).
__host__ __device__ constexpr int stable_win_reps(int nwin)
{
    return 2 * 4 * nwin <= HIST_SLOTS ? 4 : 2 * 2 * nwin <= HIST_SLOTS ? 2 : 1;
}
static_assert(2 * STABLE_WIN_MAX <= HIST_SLOTS, "the window counts live in s_hist");
static_assert(!TOPK_SAMPLED_DEVICE || STABLE_WIN_WORDS == 4 * WAVE_SIZE,
              "one 16-byte load per lane covers a bitmap");

template <bool WRITE_VALUES, bool NAN_ONE, typename ColFn>
__device__ __forceinline__ void block_gather_by_window(int c,
                                                       int len,
                                                       int nww,
                                                       int nww_first,
                                                       int idx_base,
                                                       uint32_t pivot,
                                                       int eq_needed,
                                                       int* __restrict__ out,
                                                       float* __restrict__ out_val,
                                                       const float* __restrict__ row,
                                                       uint32_t* __restrict__ s_keys,
                                                       uint32_t* __restrict__ s_bm,
                                                       uint32_t* __restrict__ s_bm_rest,
                                                       uint32_t* __restrict__ s_list,
                                                       uint32_t* __restrict__ s_wc,
                                                       ColFn col_at)
{
    static_assert(!(NAN_ONE && WRITE_VALUES), "NAN_ONE is instantiated for indices only");
    const int lane  = threadIdx.x & (WAVE_SIZE - 1);
    const int wid   = threadIdx.x / WAVE_SIZE;
    const int nwin  = stable_win_count(len);
    const int reps  = stable_win_reps(nwin);
    const int nslot = nwin * reps;
    const int rep   = lane & (reps - 1);
    uint32_t* s_cur = s_wc + nslot;
    auto wave_sync  = [] {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    };
    for(int w = threadIdx.x; w < nslot; w += blockDim.x)
        s_wc[w] = 0u;
    __syncthreads();
    // Per slot: above | equal << 16. A window holds at most c <= PHASE_C_CAP_MAX
    // entries, so each count, and each prefix below, fits a 16-bit half. Each key
    // becomes its entry, column << 1 | equal, or STABLE_NONE below the pivot.
    for(int i = threadIdx.x; i < c; i += blockDim.x)
    {
        const uint32_t k = nan_one_key<NAN_ONE>(s_keys[i]);
        uint32_t e       = STABLE_NONE;
        if(k >= pivot)
        {
            const int col = col_at(i);
            e             = ((uint32_t)col << 1) | (k == pivot ? 1u : 0u);
            atomicAdd(&s_wc[(col / STABLE_WIN_COLS) * reps + rep], k > pivot ? 1u : 0x10000u);
        }
        s_keys[i] = e;
    }
    __syncthreads();
    if(wid == 0)
    {
        // One wave, its loads issued together: nslot <= HIST_SLOTS / 2.
        constexpr int ROUNDS = HIST_SLOTS / 2 / WAVE_SIZE;
        uint32_t v[ROUNDS];
#pragma unroll
        for(int j = 0; j < ROUNDS; j++)
            v[j] = j * WAVE_SIZE + lane < nslot ? s_wc[j * WAVE_SIZE + lane] : 0u;
        uint32_t carry = 0;
#pragma unroll
        for(int j = 0; j < ROUNDS; j++)
        {
            if(j * WAVE_SIZE >= nslot)
                break;
            const uint32_t inc = aiter::wave_inclusive_sum_dpp(v[j]);
            const uint32_t p   = carry + inc - v[j];
            if(j * WAVE_SIZE + lane < nslot)
            {
                s_wc[j * WAVE_SIZE + lane]  = p;
                s_cur[j * WAVE_SIZE + lane] = (p & 0xFFFFu) + (p >> 16);
            }
            carry += (uint32_t)__builtin_amdgcn_readlane((int)inc, WAVE_SIZE - 1);
        }
    }
    __syncthreads();
    for(int i = threadIdx.x; i < c; i += blockDim.x)
    {
        const uint32_t e = s_keys[i];
        if(e != STABLE_NONE)
        {
            const int col     = (int)(e >> 1);
            const int w       = col / STABLE_WIN_COLS;
            const uint32_t at = atomicAdd(&s_cur[w * reps + rep], 1u);
            s_list[at]        = ((uint32_t)(col - w * STABLE_WIN_COLS) << 1) | (e & 1u);
        }
    }
    __syncthreads();
    uint32_t* s_gt  = wid < nww_first ? s_bm + wid * 3 * STABLE_WIN_WORDS
                                      : s_bm_rest + (wid - nww_first) * 3 * STABLE_WIN_WORDS;
    uint32_t* s_eq  = s_gt + STABLE_WIN_WORDS;
    uint32_t* s_pre = s_eq + STABLE_WIN_WORDS;
    for(int w = wid; wid < nww && w < nwin; w += nww)
    {
        // The scatter's last cursor ended at the window's end.
        const uint32_t pw0 = (uint32_t)__builtin_amdgcn_readfirstlane((int)s_wc[w * reps]);
        const int beg      = (int)((pw0 & 0xFFFFu) + (pw0 >> 16));
        const int end      = __builtin_amdgcn_readfirstlane((int)s_cur[w * reps + reps - 1]);
        if(beg == end)
            continue;
        for(int j = lane; j < 2 * STABLE_WIN_WORDS / 4; j += WAVE_SIZE)
            reinterpret_cast<uint4*>(s_gt)[j] = uint4{0u, 0u, 0u, 0u}; // s_gt, s_eq adjacent
        wave_sync();
        for(int e = beg + lane; e < end; e += WAVE_SIZE)
        {
            const uint32_t v = s_list[e];
            const int col    = (int)(v >> 1);
            atomicOr(&(v & 1u ? s_eq : s_gt)[col >> 5], 1u << (col & 31));
        }
        wave_sync();
        // Lane l takes words 4l .. 4l + 3: four counts, one wave scan.
        uint32_t run = pw0;
        for(int j = 0; j < STABLE_WIN_WORDS / 4; j += WAVE_SIZE)
        {
            const uint4 g      = reinterpret_cast<const uint4*>(s_gt)[j + lane];
            const uint4 e      = reinterpret_cast<const uint4*>(s_eq)[j + lane];
            const uint32_t c0  = (uint32_t)__popc(g.x) | ((uint32_t)__popc(e.x) << 16);
            const uint32_t c1  = (uint32_t)__popc(g.y) | ((uint32_t)__popc(e.y) << 16);
            const uint32_t c2  = (uint32_t)__popc(g.z) | ((uint32_t)__popc(e.z) << 16);
            const uint32_t c3  = (uint32_t)__popc(g.w) | ((uint32_t)__popc(e.w) << 16);
            const uint32_t tot = c0 + c1 + c2 + c3;
            const uint32_t inc = aiter::wave_inclusive_sum_dpp(tot);
            const uint32_t b   = run + inc - tot;
            reinterpret_cast<uint4*>(s_pre)[j + lane] =
                uint4{b, b + c0, b + c0 + c1, b + c0 + c1 + c2};
            run += (uint32_t)__builtin_amdgcn_readlane((int)inc, WAVE_SIZE - 1);
        }
        wave_sync();
        const int w0 = w * STABLE_WIN_COLS;
        for(int e = beg + lane; e < end; e += WAVE_SIZE)
        {
            const uint32_t v     = s_list[e];
            const int col        = (int)(v >> 1);
            const int wd         = col >> 5;
            const uint32_t below = (1u << (col & 31)) - 1u;
            const uint32_t pw    = s_pre[wd];
            const unsigned g     = (pw & 0xFFFFu) + (unsigned)__popc(s_gt[wd] & below);
            const unsigned q     = (pw >> 16) + (unsigned)__popc(s_eq[wd] & below);
            int pos              = -1;
            if(!(v & 1u))
                pos = (int)(g + min(q, (unsigned)eq_needed));
            else if(q < (unsigned)eq_needed)
                pos = (int)(g + q);
            if(pos >= 0)
            {
                out[pos] = idx_base + w0 + col;
                if(WRITE_VALUES)
                    out_val[pos] = row[w0 + col];
            }
        }
        wave_sync(); // the next window clears what this one read
    }
    __syncthreads();
}

// Gathers the entries at or above the pivot into s_list, up to list_cap of
// them; *s_cnt ends as their number and *s_ceq as that of the equal ones, both
// zeroed by the caller and published by the barrier at the end.
template <bool NAN_ONE, typename KeyFn, typename ColFn>
__device__ __forceinline__ void stable_collect(int c,
                                               uint32_t pivot,
                                               uint32_t* __restrict__ s_list,
                                               int list_cap,
                                               unsigned* __restrict__ s_cnt,
                                               unsigned* __restrict__ s_ceq,
                                               KeyFn key_at,
                                               ColFn col_at)
{
    const int lane    = threadIdx.x & (WAVE_SIZE - 1);
    const uint64_t lt = (1ull << lane) - 1ull;
    for(int i0 = 0; i0 < c; i0 += blockDim.x)
    {
        const int i       = i0 + threadIdx.x;
        const uint32_t k  = i < c ? nan_one_key<NAN_ONE>(key_at(i)) : 0u;
        const bool sel    = i < c && k >= pivot;
        const uint64_t bs = __ballot(sel);
        if(bs == 0ull)
            continue;
        const uint64_t be = __ballot(sel && k == pivot);
        unsigned base     = 0;
        if(lane == 0)
        {
            base = atomicAdd(s_cnt, (unsigned)__popcll(bs));
            if(be)
                atomicAdd(s_ceq, (unsigned)__popcll(be));
        }
        base                = wave_bcast(base, 0);
        const unsigned slot = base + (unsigned)__popcll(bs & lt);
        if(sel && slot < (unsigned)list_cap)
            s_list[slot] = ((uint32_t)col_at(i) << 1) | (k == pivot ? 1u : 0u);
    }
    __syncthreads();
}

// Sorts s[0, n) -- p entries and n - p STABLE_NONE -- and places the p, e_all of
// them equal to the pivot. s has room for n rounded up to a power of two;
// s_wtot is MAX_WAVES_PER_BLOCK words. All threads must call.
template <bool WRITE_VALUES>
__device__ __forceinline__ void stable_sort_place(uint32_t* __restrict__ s,
                                                  int n,
                                                  int p,
                                                  int e_all,
                                                  int eq_needed,
                                                  int idx_base,
                                                  const float* __restrict__ row,
                                                  int* __restrict__ out,
                                                  float* __restrict__ out_val,
                                                  uint32_t* __restrict__ s_wtot)
{
    const int lane          = threadIdx.x & (WAVE_SIZE - 1);
    const int wid           = threadIdx.x / WAVE_SIZE;
    const bool all_eq_taken = e_all <= eq_needed;
    // Entry v at sorted index i with eq_before equal keys left of it.
    auto place = [&](int i, uint32_t v, unsigned eq_before) {
        const bool is_eq = v & 1u;
        if(all_eq_taken || !is_eq || eq_before < (unsigned)eq_needed)
        {
            const int pos =
                all_eq_taken
                    ? i
                    : i - (int)(eq_before > (unsigned)eq_needed ? eq_before - eq_needed : 0u);
            out[pos] = idx_base + (int)(v >> 1);
            if(WRITE_VALUES)
                out_val[pos] = row[v >> 1];
        }
    };
    if(n <= STABLE_RANK_DIRECT)
    {
        // A short list ranks by comparison, n reads per entry and no barrier: a
        // sort's stages cost more here.
        for(int j = threadIdx.x; j < n; j += blockDim.x)
        {
            const uint32_t v = s[j];
            if(v == STABLE_NONE)
                continue;
            unsigned left = 0, eq_left = 0;
            for(int m = 0; m < n; m++)
            {
                const uint32_t w = s[m];
                left += w < v ? 1u : 0u;
                eq_left += (w < v && (w & 1u)) ? 1u : 0u;
            }
            place((int)left, v, eq_left);
        }
        return;
    }
    int n2 = 1;
    while(n2 < n)
        n2 <<= 1;
    for(int i = n + (int)threadIdx.x; i < n2; i += blockDim.x)
        s[i] = STABLE_NONE;
    __syncthreads();
    // A stage of stride <= WAVE_SIZE pairs entries inside one 2 * WAVE_SIZE run,
    // and every run belongs to one wave (pair t of the stage is thread t mod
    // blockDim), so between two such stages the wave only has to see its own
    // writes; any other boundary needs the block.
    for(int size = 2; size <= n2; size <<= 1)
    {
        for(int stride = size >> 1; stride > 0; stride >>= 1)
        {
            for(int t = threadIdx.x; t < n2 / 2; t += blockDim.x)
            {
                const int a      = 2 * t - (t & (stride - 1));
                const uint32_t x = s[a], y = s[a + stride];
                if((x > y) == ((a & size) == 0))
                {
                    s[a]          = y;
                    s[a + stride] = x;
                }
            }
            const int next = stride > 1 ? stride >> 1 : size;
            if(stride <= WAVE_SIZE && next <= WAVE_SIZE && next < n2)
            {
                __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
                __builtin_amdgcn_wave_barrier();
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
            }
            else
            {
                __syncthreads();
            }
        }
    }
    if(all_eq_taken)
    {
        // Every entry is selected, and the first p are in column order.
        for(int i = threadIdx.x; i < p; i += blockDim.x)
            place(i, s[i], 0u);
        return;
    }
    // More equal keys than places: each thread walks one contiguous run of the
    // sorted entries with the count of equal keys before it.
    const int run  = (p + (int)blockDim.x - 1) / (int)blockDim.x;
    const int r0   = min((int)threadIdx.x * run, p);
    const int r1   = min(r0 + run, p);
    uint32_t local = 0;
    for(int i = r0; i < r1; i++)
        local += s[i] & 1u;
    const uint32_t sw = wave_suffix_sum(local);
    if(lane == 0)
        s_wtot[wid] = sw;
    __syncthreads();
    unsigned eq_before = (uint32_t)__builtin_amdgcn_readfirstlane((int)sw) - sw;
    for(int w = 0; w < wid; w++)
        eq_before += s_wtot[w];
    for(int i = r0; i < r1; i++)
    {
        const uint32_t v = s[i];
        place(i, v, eq_before);
        eq_before += v & 1u;
    }
}
