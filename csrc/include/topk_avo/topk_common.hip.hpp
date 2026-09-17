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

#define HIP_CHECK(call)                             \
    do                                              \
    {                                               \
        hipError_t _err = (call);                   \
        if(_err != hipSuccess)                      \
        {                                           \
            fprintf(stderr,                         \
                    "HIP error %d (%s) at %s:%d\n", \
                    (int)_err,                      \
                    hipGetErrorString(_err),        \
                    __FILE__,                       \
                    __LINE__);                      \
            exit(1);                                \
        }                                           \
    } while(0)

constexpr int WAVE_SIZE    = 64;
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
    __device__ __forceinline__ int row_start(int) const { return 0; }
};
template <>
struct RowExtents<true>
{
    const int* starts;
    const int* ends;
    __device__ __forceinline__ int row_start(int row) const { return starts[row]; }
    __device__ __forceinline__ int row_len(int row) const { return ends[row] - starts[row]; }
};

__device__ __forceinline__ int row_len_dev(int row, int pitch, const int* row_ends)
{ return row_ends ? row_ends[row] : pitch; }

template <bool RAGGED>
__device__ __forceinline__ int row_len_of(int row, int pitch, RowExtents<RAGGED> ext)
{
    if constexpr(RAGGED)
        return ext.row_len(row);
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

struct GPUInfo
{
    char name[256];
    int compute_major;
    int compute_minor;
    int clock_khz;
    int sm_count;
    int mem_clock_khz;
    int mem_bus_width;
};

static inline GPUInfo get_gpu_info()
{
    int dev = 0;
    HIP_CHECK(hipGetDevice(&dev));
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, dev));
    GPUInfo g{};
    std::strncpy(g.name, p.name, sizeof(g.name) - 1);
    g.compute_major = p.major;
    g.compute_minor = p.minor;
    g.clock_khz     = p.clockRate;
    g.sm_count      = p.multiProcessorCount;
    g.mem_clock_khz = p.memoryClockRate;
    g.mem_bus_width = p.memoryBusWidth;
    return g;
}

class HipTimer
{
    public:
    HipTimer()
    {
        (void)hipEventCreate(&start_);
        (void)hipEventCreate(&stop_);
    }
    ~HipTimer()
    {
        (void)hipEventDestroy(start_);
        (void)hipEventDestroy(stop_);
    }
    void begin(hipStream_t s = 0) { (void)hipEventRecord(start_, s); }
    double end(hipStream_t s = 0)
    {
        (void)hipEventRecord(stop_, s);
        (void)hipEventSynchronize(stop_);
        float ms = 0.f;
        (void)hipEventElapsedTime(&ms, start_, stop_);
        return (double)ms;
    }

    private:
    hipEvent_t start_, stop_;
};

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

static inline uint32_t fp32_to_sortable_host(float v)
{
    uint32_t u;
    std::memcpy(&u, &v, sizeof(u));
    return fp32_to_sortable_bits(u);
}

// Selects the load flavour for the streaming filter pass. Non-temporal is the
// right hint for Phase B: pure streaming, zero reuse, so keeping the lines in
// L2 only evicts useful data.
__constant__ int d_use_nt_load = 0;

// Native ext_vector_type: __builtin_nontemporal_load rejects HIP_vector_type.
typedef float vfloat4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ vfloat4 load_f4(const vfloat4* p)
{
    if(d_use_nt_load)
        return __builtin_nontemporal_load(p);
    return *p;
}

// Block-wide min and max of keys already in LDS, using an xor butterfly so
// EVERY lane ends up holding the result (a __shfl_down tree would leave it in
// lane 0 only -- that exact trap cost 3% recall in the DeepSelect port).
// All threads must call.
__device__ __forceinline__ void block_minmax_lds(const uint32_t* __restrict__ s_keys,
                                                 int c,
                                                 uint32_t* __restrict__ s_mm,
                                                 uint32_t& out_min,
                                                 uint32_t& out_max)
{
    const int lane   = threadIdx.x & (WAVE_SIZE - 1);
    const int wv     = threadIdx.x / WAVE_SIZE;
    const int nwaves = blockDim.x / WAVE_SIZE;
    uint32_t mn = 0xFFFFFFFFu, mx = 0u;
    for(int i = threadIdx.x; i < c; i += blockDim.x)
    {
        uint32_t k = s_keys[i];
        mn         = min(mn, k);
        mx         = max(mx, k);
    }
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
        baseg = __shfl(baseg, 0);
        basee = __shfl(basee, 0);
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

// Grid.y for the fallback kernel. Blocks loop over the compacted row list, so
// any number of fallback rows is handled; this only bounds the dispatch cost.
// grid.y = M would dispatch 4096 blocks to do work for a handful of rows.
constexpr int FB_GRID = 64;

// Wave-private candidate regions (Phase B variant 3). Each wave in the row's
// block owns a fixed slice, so it needs no atomic at all -- just a wave-uniform
// register counter. The PHYSICAL stride is deliberately generous: only the
// occupied slots are ever written or read, so a wide stride costs address space
// and nothing else, while making per-wave overflow ~25 sigma away instead of
// ~0 sigma (expected passers/wave is ~178 +/- 13 at K=2048).
constexpr int CAND_SLOTS_PER_ROW = 8192;

// Turns s_hist[256] (per-bucket counts) into an INCLUSIVE SUFFIX sum in place,
// then finds the bucket where the running count from the top first reaches ek.
// Writes s_scan[0] = bucket, s_scan[1] = count strictly above that bucket.
//
// Replaces a serial 256-step walk by thread 0: that walk is a chain of
// dependent LDS reads and measured as the dominant cost of the select kernels
// (phase_a 118 us / phase_c 220 us for a few MB of traffic).
// Every thread in the block must call this (it contains barriers).
// The suffix sums stay in REGISTERS: the 256 buckets are covered by exactly
// 4 wave64s, so the intra-wave scan is 6 shuffles with no barrier, and the only
// shared state is the 4 wave totals. The neighbour value S[t+1] that the search
// needs also comes from a shuffle -- at a wave boundary S[64*(wv+1)] is exactly
// the sum of all strictly higher wave totals -- so nothing is written back to
// s_hist and no barrier is needed for it.
//
// 4 barriers per radix pass total (zero / histogram / wave totals / result),
// down from ~8. Measured per-pass fixed cost was 7-12 us across 4096 blocks.
//
// s_scan must be pre-initialised by the caller. For every path that reaches
// here on a row it will actually use, the search always finds a bucket (ek is
// never larger than the number of elements matching the fixed prefix); rows
// where that does not hold are routed to the fallback and their output is
// discarded, and the pre-initialised {0,0} keeps them in bounds regardless.
// Replica-reducing form of the scan below: folds the HIST_REP replicas of each
// bucket in as it reads them, instead of having the caller run a separate
// block-wide reduction loop into s_red first.
//
// The point is the barrier, not the arithmetic. The separate loop needs its own
// __syncthreads() before the scan may read s_red, so folding it in here takes a
// radix pass from 5 block barriers to 4, and leaves s_red unused (1 KB of LDS
// per block). The select's cost is the serial depth of one row -- passes x
// barriers over the keys in LDS -- not read throughput, which is why barrier
// count is the lever here (knowledge/known_bad.md).
__device__ __forceinline__ void block_find_pivot_bucket_rep(const uint32_t* __restrict__ s_hist,
                                                            uint32_t* __restrict__ s_scan,
                                                            int ek)
{
    const int t = threadIdx.x;
    __shared__ uint32_t s_wavetot[256 / WAVE_SIZE];
    const int lane = t & (WAVE_SIZE - 1);
    const int wv   = t / WAVE_SIZE;
    uint32_t x     = 0u;
    if(t < 256)
    {
#pragma unroll
        for(int r = 0; r < HIST_REP; r++)
            x += s_hist[t * HIST_REP + r];
#pragma unroll
        for(int off = 1; off < WAVE_SIZE; off <<= 1)
        {
            uint32_t up = __shfl_down(x, off);
            if(lane + off < WAVE_SIZE)
                x += up;
        }
        if(lane == 0)
            s_wavetot[wv] = x;
    }
    __syncthreads();

    uint32_t above_waves = 0;
    if(t < 256)
        for(int w = wv + 1; w < 256 / WAVE_SIZE; w++)
            above_waves += s_wavetot[w];
    const uint32_t s_t = x + above_waves;
    uint32_t s_next    = __shfl_down(s_t, 1);
    if(lane == WAVE_SIZE - 1)
        s_next = above_waves;
    if(t == 255)
        s_next = 0u;

    if(t < 256 && ek > 0 && s_t >= (uint32_t)ek && s_next < (uint32_t)ek)
    {
        s_scan[0] = (uint32_t)t;
        s_scan[1] = s_next;
    }
    __syncthreads();
}

__device__ __forceinline__ void
block_find_pivot_bucket(const uint32_t* __restrict__ s_hist, uint32_t* __restrict__ s_scan, int ek)
{
    const int t = threadIdx.x;
    __shared__ uint32_t s_wavetot[256 / WAVE_SIZE];
    const int lane = t & (WAVE_SIZE - 1);
    const int wv   = t / WAVE_SIZE;
    uint32_t x     = (t < 256) ? s_hist[t] : 0u;
    if(t < 256)
    {
#pragma unroll
        for(int off = 1; off < WAVE_SIZE; off <<= 1)
        {
            uint32_t up = __shfl_down(x, off);
            if(lane + off < WAVE_SIZE)
                x += up;
        }
        if(lane == 0)
            s_wavetot[wv] = x;
    }
    __syncthreads();

    uint32_t above_waves = 0;
    if(t < 256)
        for(int w = wv + 1; w < 256 / WAVE_SIZE; w++)
            above_waves += s_wavetot[w];
    const uint32_t s_t = x + above_waves;     // inclusive suffix sum at bucket t
    uint32_t s_next    = __shfl_down(s_t, 1); // suffix sum at bucket t+1
    if(lane == WAVE_SIZE - 1)
        s_next = above_waves;
    if(t == 255)
        s_next = 0u;

    // s_t is non-increasing in t, so {t : s_t >= ek} is a prefix; its last member
    // is the pivot bucket and s_next there is the count strictly above it.
    if(t < 256 && ek > 0 && s_t >= (uint32_t)ek && s_next < (uint32_t)ek)
    {
        s_scan[0] = (uint32_t)t;
        s_scan[1] = s_next;
    }
    __syncthreads();
}
