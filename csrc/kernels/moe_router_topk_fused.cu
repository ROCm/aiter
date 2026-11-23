// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused MoE router: bf16 router GEMM (x @ W^T, fp32 accumulate) + biased
// sigmoid top-k gating, one launch.
//
// It replaces the decode pair
//   gemm_a16w16 (FlyDSL gfx1250, split-K, bf16 logits) -> topk_gating_kernel_opt
// and reproduces both bit for bit:
//
// * GEMM: every logit is the same chain of v_wmma_f32_16x16x32_bf16 the FlyDSL
//   kernel runs (`kernels/gemm_a16w16_kernel_gfx1250.py`): the router weight is
//   WMMA operand A and the activation operand B, the K range is cut into the
//   same `split_k` chunks, each chunk accumulates its 32-wide K steps in order
//   from zero, and the chunks are summed in split order (p0 + p1 + ...) before
//   the round to bf16. The operand lane layout is the standard gfx1250 one the
//   FlyDSL kernel loads from LDS (lanes 0-15: K 0-7,16-23; lanes 16-31: K
//   8-15,24-31), so every WMMA sees the same operands in the same slots.
// * Gating: the top-k body is `topk_gating_kernel_opt` itself, reusing its
//   score function, sorting network and DPP argmax from
//   topk_gating_kernels.cuh, one wave per token row over the bf16 logits.
//
// Grid (1-D, WAVES waves per workgroup), two roles by block id:
//   [0, n_gemm)          GEMM: (m tile, 16*WAVES experts, K chunk); each wave
//                        runs one 16x16 chain and stores its fp32 partial to
//                        ws[split][row][expert], then the workgroup bumps the
//                        m tile's arrival counter.
//   [n_gemm, +m_tiles*16/WAVES)  gating: one wave per token row; waits for its
//                        m tile's (8/WAVES)*split_k arrivals, sums the row's
//                        partials in split order, rounds to bf16, top-k.
// Workgroups are dispatched in block-id order, so every GEMM workgroup a gating
// workgroup waits for is already resident or finished: the wait cannot block
// its producers. The last gating workgroup of an m tile resets its counters,
// so the semaphore arena is zero again for the next launch (graph replay).
// `waves == 16` selects the other layout (moe_router_topk_last_kernel): 8-wave
// workgroups, the tile's last arrival reduces and gates without polling.
//
// Status (MI455X, T = 64..256): bit exact, but 7.4-9.6 us against 6.0-7.4 us
// for the FlyDSL GEMM + topk_gating pair. The GEMM part alone is faster than
// the FlyDSL kernel (2.5-3.0 us); the cross-workgroup handoff (device-scope
// store, arrival, poll and partial loads) costs ~4.5 us, more than the kernel
// boundary it replaces. Not wired into ATOM.

#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "moe_router_topk_fused.h"
#include "topk_gating_kernels.cuh"

namespace aiter {
namespace moe_router_fused {

typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16_t __attribute__((ext_vector_type(16)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));
typedef float f32x4_t __attribute__((ext_vector_type(4)));

constexpr int kExperts  = 128; // N: router outputs (experts)
constexpr int kTile     = 16;  // WMMA M = N
constexpr int kNTiles   = kExperts / kTile;
constexpr int kWmmaK    = 32;
constexpr int kWave     = 32;
constexpr int kMaxSplit = 8;
constexpr int kSpinSleep    = 1;       // s_sleep units (64 clocks) between polls
constexpr int kSpinMaxPolls = 1 << 20; // poll cap of the gating wait

__device__ __forceinline__ bf16x16_t load_frag(const __bf16* p)
{
    // K [0,8) and [16,24) of this lane's 32-wide step (p already holds the
    // lane group's +8), concatenated lo|hi as the FlyDSL fragment shuffle does.
    const bf16x8_t lo = *reinterpret_cast<const bf16x8_t*>(p);
    const bf16x8_t hi = *reinterpret_cast<const bf16x8_t*>(p + 16);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

// One 16x16 accumulator chain over `k_steps` 32-wide K steps from zero, the
// operands read straight from global memory. KSTEPS > 0 unrolls the chunk into
// groups of DEPTH steps held in an NBUF-deep register ring: NBUF-1 groups are
// in flight ahead of the WMMAs (the whole chunk at split_k 8), and
// sched_barrier keeps the scheduler from sinking the loads to their use.
template <int KSTEPS, int DEPTH, int NBUF>
__device__ __forceinline__ f32x8_t router_chain(const __bf16* pa, const __bf16* pb, int k_steps)
{
    f32x8_t acc = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if constexpr(KSTEPS > 0)
    {
        static_assert(KSTEPS % DEPTH == 0, "chunk not a multiple of the group");
        constexpr int G = KSTEPS / DEPTH;
        bf16x16_t ga[NBUF][DEPTH], gb[NBUF][DEPTH];
#pragma unroll
        for(int g = 0; g < NBUF - 1; ++g)
        {
            if(g < G)
            {
#pragma unroll
                for(int j = 0; j < DEPTH; ++j)
                {
                    ga[g][j] = load_frag(pa + (g * DEPTH + j) * kWmmaK);
                    gb[g][j] = load_frag(pb + (g * DEPTH + j) * kWmmaK);
                }
            }
        }
#pragma unroll
        for(int g = 0; g < G; ++g)
        {
            const int lg = g + NBUF - 1;
            if(lg < G)
            {
#pragma unroll
                for(int j = 0; j < DEPTH; ++j)
                {
                    ga[lg % NBUF][j] = load_frag(pa + (lg * DEPTH + j) * kWmmaK);
                    gb[lg % NBUF][j] = load_frag(pb + (lg * DEPTH + j) * kWmmaK);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int j = 0; j < DEPTH; ++j)
                acc = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
                    false, ga[g % NBUF][j], false, gb[g % NBUF][j], static_cast<short>(0), acc, false, false);
            __builtin_amdgcn_sched_barrier(0);
        }
    }
    else
    {
        // Any chunk length (k_steps % DEPTH == 0).
        bf16x16_t fa[DEPTH], fb[DEPTH];
#pragma unroll
        for(int j = 0; j < DEPTH; ++j)
        {
            fa[j] = load_frag(pa + j * kWmmaK);
            fb[j] = load_frag(pb + j * kWmmaK);
        }
        for(int c = DEPTH; c <= k_steps; c += DEPTH)
        {
            // Past the end re-reads the last group: same addresses, unused.
            const int nc = (c < k_steps) ? c : (k_steps - DEPTH);
            bf16x16_t na[DEPTH], nb[DEPTH];
#pragma unroll
            for(int j = 0; j < DEPTH; ++j)
            {
                na[j] = load_frag(pa + (nc + j) * kWmmaK);
                nb[j] = load_frag(pb + (nc + j) * kWmmaK);
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int j = 0; j < DEPTH; ++j)
                acc = __builtin_amdgcn_wmma_f32_16x16x32_bf16(
                    false, fa[j], false, fb[j], static_cast<short>(0), acc, false, false);
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int j = 0; j < DEPTH; ++j)
            {
                fa[j] = na[j];
                fb[j] = nb[j];
            }
        }
    }
    return acc;
}

// topk_gating_kernel_opt (SCORE_SIGMOID) over one row's bf16-rounded logits
// (logit[i] = expert lane + 32*i); lane = threadIdx.x there.
template <bool NEED_RENORM>
__device__ __forceinline__ void gate_row(const float (&logit)[kExperts / kWave],
                                         const float* __restrict__ bias,
                                         const int* __restrict__ num_valid_rows,
                                         float* __restrict__ topk_weights,
                                         int* __restrict__ topk_ids,
                                         int row,
                                         int lane,
                                         int stride_tk,
                                         int topk,
                                         float routed_scaling_factor)
{
    constexpr int EPT = kExperts / kWave;
    float vals[EPT];
    float orig[EPT];
    int idxs[EPT];
#pragma unroll
    for(int i = 0; i < EPT; i++)
    {
        const int e       = lane + i * kWave;
        const float score = compute_score<SCORE_SIGMOID>(logit[i]);
        orig[i]           = ::isnan(score) ? 0.0f : score;
        vals[i]           = score;
        idxs[i]           = e;
        if(bias != nullptr)
            vals[i] += bias[e];
        vals[i] = fmaxf(vals[i], -INFINITY);
    }
    float* wrow = topk_weights + static_cast<long>(row) * stride_tk;
    int* irow   = topk_ids + static_cast<long>(row) * stride_tk;
    if(is_pad_row(num_valid_rows, row))
    {
        write_pad_row(wrow, irow, topk, lane, kWave);
    }
    else
    {
        sort_network_desc<EPT>(vals, orig, idxs);

        int cursor       = 0;
        float sum        = 0.0f;
        int topk_indice  = 0;
        float topk_value = 0.0f;
        for(int k = 0; k < topk; ++k)
        {
            float my_val = (cursor < EPT) ? vals[cursor] : -INFINITY;
            int my_idx   = (cursor < EPT) ? idxs[cursor] : 0;

            warpReduceMax_softplus(my_val, my_idx);

            bool i_won    = (cursor < EPT && idxs[cursor] == my_idx);
            float my_orig = i_won ? orig[cursor] : 0.0f;
            if(i_won)
                cursor++;

            int win_lane = my_idx & (kWave - 1);
            float weight = __builtin_bit_cast(
                float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, my_orig), win_lane));

            if(lane == k)
            {
                topk_indice = my_idx;
                topk_value  = weight;
            }
            if constexpr(NEED_RENORM)
                sum += weight;
        }
        if constexpr(NEED_RENORM)
            sum = routed_scaling_factor / fmaxf(sum, RENORM_SUM_FLOOR);
        else
            sum = routed_scaling_factor;
        if(lane < topk)
        {
            wrow[lane] = topk_value * sum;
            irow[lane] = topk_indice;
        }
    }
}

template <int WAVES, int KSTEPS, int DEPTH, int NBUF, bool NEED_RENORM>
__global__ __launch_bounds__(WAVES* kWave) __attribute__((amdgpu_waves_per_eu(1, 4))) void
moe_router_topk_kernel(const __bf16* __restrict__ x,     // [M, K] row stride lda
                       const __bf16* __restrict__ w,     // [128, K] row stride ldb
                       const float* __restrict__ bias,   // [128] fp32 or nullptr
                       const int* __restrict__ num_valid_rows,
                       float* __restrict__ ws,           // [split_k][m_tiles*16][128] fp32
                       int* __restrict__ arrived,        // [m_tiles], zero on entry and exit
                       int* __restrict__ released,       // [m_tiles], zero on entry and exit
                       float* __restrict__ topk_weights, // [M, stride_tk]
                       int* __restrict__ topk_ids,       // [M, stride_tk]
                       __bf16* __restrict__ logits_out,  // [M, 128] or nullptr
                       int M,
                       int lda,
                       int ldb,
                       int k_steps, // K chunk / 32
                       int splits,
                       int stride_tk,
                       int topk,
                       float routed_scaling_factor)
{
    static_assert(kNTiles % WAVES == 0 && kTile % WAVES == 0, "WAVES must divide 8");
    constexpr int NGROUPS = kNTiles / WAVES;        // GEMM workgroups per (m tile, split)
    constexpr int GATE_WGS = kTile / WAVES;         // gating workgroups per m tile

    const int tid     = threadIdx.x;
    const int wave    = tid / kWave;
    const int lane    = tid % kWave;
    const int m_tiles = (M + kTile - 1) / kTile;
    const int m_rows  = m_tiles * kTile;
    const int n_gemm  = m_tiles * NGROUPS * splits;
    const int bid     = blockIdx.x;

    if(bid < n_gemm)
    {
        // ---------------- GEMM: one 16x16 chain per wave over a K chunk -----
        const int split  = bid % splits;
        const int rest   = bid / splits;
        const int ngroup = rest % NGROUPS;
        const int mt     = rest / NGROUPS;
        const int ntile  = ngroup * WAVES + wave;
        const int lane16 = lane % 16;
        const int kgrp   = lane / 16;
        const long kbase = static_cast<long>(split) * k_steps * kWmmaK;
        // Rows past M read row M-1: their columns never leave the tile.
        const int tok    = min(mt * kTile + lane16, M - 1);
        const __bf16* pa = w + static_cast<long>(ntile * kTile + lane16) * ldb + kbase + kgrp * 8;
        const __bf16* pb = x + static_cast<long>(tok) * lda + kbase + kgrp * 8;
        const f32x8_t acc = router_chain<KSTEPS, DEPTH, NBUF>(pa, pb, k_steps);

        // acc[i]: expert ntile*16 + kgrp*8 + i of token row mt*16 + lane16.
        // Device-scope stores (scope:SCOPE_DEV, as the FlyDSL split-K's CPOL_DEV
        // workspace copies) and no fence: a release fence's cache write-back
        // costs more than this whole kernel.
        float* dst = ws + (static_cast<long>(split) * m_rows + mt * kTile + lane16) * kExperts +
                     ntile * kTile + kgrp * 8;
#pragma unroll
        for(int i = 0; i < 8; ++i)
            __hip_atomic_store(dst + i, acc[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
        __syncthreads();
        if(tid == 0)
            __hip_atomic_fetch_add(arrived + mt, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        return;
    }

    // ---------------- gating: one wave per token row ------------------------
    const int g  = bid - n_gemm;
    const int mt = g / GATE_WGS;
    const int r  = (g % GATE_WGS) * WAVES + wave;
    const int expected = NGROUPS * splits;
    __shared__ int timed_out;
    if(tid == 0)
    {
        // Bounded: past kSpinMaxPolls (~tens of ms) the rows are written as pad
        // rows (id -1) instead of hanging the device on a lost arrival.
        int polls = 0;
        while(__hip_atomic_load(arrived + mt, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < expected &&
              ++polls < kSpinMaxPolls)
            __builtin_amdgcn_s_sleep(kSpinSleep);
        timed_out = polls >= kSpinMaxPolls;
    }
    __syncthreads();

    const int row = mt * kTile + r;
    if(row < M && timed_out)
    {
        write_pad_row(topk_weights + static_cast<long>(row) * stride_tk,
                      topk_ids + static_cast<long>(row) * stride_tk, topk, lane, kWave);
    }
    else if(row < M)
    {
        constexpr int EPT = kExperts / kWave;
        // The row's partials, every split's loads issued before the first add.
        float part[kMaxSplit][EPT];
#pragma unroll
        for(int s = 0; s < kMaxSplit; ++s)
        {
            if(s < splits)
            {
                // Device-scope loads: the partials were stored at that scope.
                float* src = ws + (static_cast<long>(s) * m_rows + row) * kExperts;
#pragma unroll
                for(int i = 0; i < EPT; ++i)
                    part[s][i] = __hip_atomic_load(src + lane + i * kWave, __ATOMIC_RELAXED,
                                                   __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        float logit[EPT];
#pragma unroll
        for(int i = 0; i < EPT; ++i)
        {
            float total = part[0][i];
#pragma unroll
            for(int s = 1; s < kMaxSplit; ++s)
                if(s < splits)
                    total = total + part[s][i];
            const __bf16 b = static_cast<__bf16>(total);
            if(logits_out != nullptr)
                logits_out[static_cast<long>(row) * kExperts + lane + i * kWave] = b;
            logit[i] = static_cast<float>(b);
        }

        gate_row<NEED_RENORM>(logit, bias, num_valid_rows, topk_weights, topk_ids, row, lane,
                              stride_tk, topk, routed_scaling_factor);
    }

    // Every gating workgroup of the tile is past its wait once the last one
    // gets here: put both counters back to zero for the next launch.
    __syncthreads();
    if(tid == 0)
    {
        if(__hip_atomic_fetch_add(released + mt, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) ==
           GATE_WGS - 1)
        {
            __hip_atomic_store(arrived + mt, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(released + mt, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// Variant "last": a workgroup of 8 waves owns one (16-row M tile, K chunk) and
// all 128 experts; the last of the tile's split_k workgroups to arrive (the
// counter's return value, no polling) reduces the tile and gates its 16 rows,
// two per wave, from LDS.
template <int KSTEPS, int DEPTH, int NBUF, bool NEED_RENORM>
__global__ __launch_bounds__(kNTiles* kWave) __attribute__((amdgpu_waves_per_eu(1, 2))) void
moe_router_topk_last_kernel(const __bf16* __restrict__ x,
                            const __bf16* __restrict__ w,
                            const float* __restrict__ bias,
                            const int* __restrict__ num_valid_rows,
                            float* __restrict__ ws, // [split_k][m_tiles*16][128] fp32
                            int* __restrict__ arrived,
                            float* __restrict__ topk_weights,
                            int* __restrict__ topk_ids,
                            __bf16* __restrict__ logits_out,
                            int M,
                            int lda,
                            int ldb,
                            int k_steps,
                            int splits,
                            int stride_tk,
                            int topk,
                            float routed_scaling_factor)
{
    __shared__ float lg[kTile][kExperts + 1];
    __shared__ int last_flag;
    const int tid    = threadIdx.x;
    const int wave   = tid / kWave;
    const int lane   = tid % kWave;
    const int lane16 = lane % 16;
    const int kgrp   = lane / 16;
    const int m_rows = ((M + kTile - 1) / kTile) * kTile;
    const int split  = blockIdx.x % splits;
    const int mt     = blockIdx.x / splits;
    const int ntile  = wave;

    const long kbase = static_cast<long>(split) * k_steps * kWmmaK;
    const int tok    = min(mt * kTile + lane16, M - 1);
    const __bf16* pa = w + static_cast<long>(ntile * kTile + lane16) * ldb + kbase + kgrp * 8;
    const __bf16* pb = x + static_cast<long>(tok) * lda + kbase + kgrp * 8;
    const f32x8_t acc = router_chain<KSTEPS, DEPTH, NBUF>(pa, pb, k_steps);

    if(splits > 1)
    {
        float* dst = ws + (static_cast<long>(split) * m_rows + mt * kTile + lane16) * kExperts +
                     ntile * kTile + kgrp * 8;
#pragma unroll
        for(int i = 0; i < 8; ++i)
            __hip_atomic_store(dst + i, acc[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
        __syncthreads();
        if(tid == 0)
            last_flag = __hip_atomic_fetch_add(arrived + mt, 1, __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT) == splits - 1;
        __syncthreads();
        if(!last_flag)
            return;
        // Reduce this wave's 16x16 tile over the splits, in split order.
        float part[kMaxSplit][8];
#pragma unroll
        for(int s = 0; s < kMaxSplit; ++s)
        {
            if(s < splits)
            {
                float* src = ws + (static_cast<long>(s) * m_rows + mt * kTile + lane16) * kExperts +
                             ntile * kTile + kgrp * 8;
#pragma unroll
                for(int i = 0; i < 8; ++i)
                    part[s][i] = __hip_atomic_load(src + i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            float total = part[0][i];
#pragma unroll
            for(int s = 1; s < kMaxSplit; ++s)
                if(s < splits)
                    total = total + part[s][i];
            lg[lane16][ntile * kTile + kgrp * 8 + i] = static_cast<float>(static_cast<__bf16>(total));
        }
        if(tid == 0)
            __hip_atomic_store(arrived + mt, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    else
    {
#pragma unroll
        for(int i = 0; i < 8; ++i)
            lg[lane16][ntile * kTile + kgrp * 8 + i] = static_cast<float>(static_cast<__bf16>(acc[i]));
    }
    __syncthreads();
    constexpr int EPT = kExperts / kWave;
#pragma unroll
    for(int rr = 0; rr < kTile / kNTiles; ++rr)
    {
        const int r   = wave + rr * kNTiles;
        const int row = mt * kTile + r;
        if(row >= M)
            break;
        float logit[EPT];
#pragma unroll
        for(int i = 0; i < EPT; ++i)
        {
            logit[i] = lg[r][lane + i * kWave];
            if(logits_out != nullptr)
                logits_out[static_cast<long>(row) * kExperts + lane + i * kWave] =
                    static_cast<__bf16>(logit[i]);
        }
        gate_row<NEED_RENORM>(logit, bias, num_valid_rows, topk_weights, topk_ids, row, lane,
                              stride_tk, topk, routed_scaling_factor);
    }
}

} // namespace moe_router_fused

void moe_router_topk_fused(aiter_tensor_t& topk_weights,
                           aiter_tensor_t& topk_ids,
                           aiter_tensor_t& x,
                           aiter_tensor_t& w,
                           aiter_tensor_t& correction_bias,
                           aiter_tensor_t& workspace,
                           aiter_tensor_t& semaphore,
                           int split_k,
                           int waves,
                           int chunk,
                           bool need_renorm,
                           float routed_scaling_factor,
                           std::optional<aiter_tensor_t> num_valid_rows,
                           std::optional<aiter_tensor_t> logits_out)
{
    using namespace moe_router_fused;
    AITER_CHECK(x.dtype() == AITER_DTYPE_bf16 && w.dtype() == AITER_DTYPE_bf16,
                "moe_router_topk_fused: x and w must be bf16");
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32 && topk_ids.dtype() == AITER_DTYPE_i32,
                "moe_router_topk_fused: topk_weights fp32, topk_ids int32");
    AITER_CHECK(x.dim() == 2 && w.dim() == 2 && x.size(1) == w.size(1),
                "moe_router_topk_fused: x [M,K], w [E,K]");
    AITER_CHECK(w.size(0) == kExperts, "moe_router_topk_fused: E must be ", kExperts);
    AITER_CHECK(x.stride(1) == 1 && w.stride(1) == 1, "moe_router_topk_fused: K must be contiguous");
    const int M    = static_cast<int>(x.size(0));
    const int K    = static_cast<int>(x.size(1));
    const int topk = static_cast<int>(topk_ids.size(1));
    AITER_CHECK(topk >= 1 && topk <= kWave, "moe_router_topk_fused: bad topk");
    AITER_CHECK(topk_weights.stride(0) == topk_ids.stride(0), "topk_weights/topk_ids row stride");
    AITER_CHECK(split_k >= 1 && split_k <= kMaxSplit && K % (split_k * kWmmaK) == 0,
                "moe_router_topk_fused: split_k in [1, ", kMaxSplit, "] dividing K/32");
    const int k_steps = K / (split_k * kWmmaK);
    AITER_CHECK(chunk == 4 || chunk == 8, "chunk (prefetch depth) must be 4 or 8");
    AITER_CHECK(k_steps % chunk == 0, "moe_router_topk_fused: K steps per split % chunk");
    // 16 selects the 8-wave last-arriver variant.
    AITER_CHECK(waves == 1 || waves == 2 || waves == 4 || waves == 8 || waves == 16,
                "waves must be 1, 2, 4, 8 or 16");
    AITER_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && x.stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 && w.stride(0) % 8 == 0,
                "moe_router_topk_fused: x/w rows must be 16-byte aligned");
    if(M == 0)
        return;
    const int m_tiles = (M + kTile - 1) / kTile;
    AITER_CHECK(workspace.dtype() == AITER_DTYPE_fp32 &&
                    workspace.numel() >= static_cast<size_t>(split_k) * m_tiles * kTile * kExperts,
                "moe_router_topk_fused: workspace too small");
    AITER_CHECK(semaphore.dtype() == AITER_DTYPE_i32 && semaphore.numel() % 2 == 0 &&
                    semaphore.numel() / 2 >= static_cast<size_t>(m_tiles),
                "moe_router_topk_fused: semaphore too small");
    const bool has_bias = correction_bias.numel() > 0;
    if(has_bias)
        AITER_CHECK(correction_bias.dtype() == AITER_DTYPE_fp32 && correction_bias.numel() == kExperts,
                    "moe_router_topk_fused: correction_bias must be fp32 [128]");
    __bf16* lg_out = nullptr;
    if(logits_out.has_value() && logits_out->numel() > 0)
    {
        AITER_CHECK(logits_out->dtype() == AITER_DTYPE_bf16 && logits_out->is_contiguous() &&
                        logits_out->numel() == static_cast<size_t>(M) * kExperts,
                    "moe_router_topk_fused: logits_out bf16 [M,128]");
        lg_out = reinterpret_cast<__bf16*>(logits_out->data_ptr());
    }
    const int* nvr = num_valid_rows_ptr(num_valid_rows);
    int* arrived   = reinterpret_cast<int*>(semaphore.data_ptr());
    int* released  = arrived + semaphore.numel() / 2;

    HipDeviceGuard device_guard(x.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int wg_waves = waves == 16 ? kNTiles : waves;
    const int n_gemm   = m_tiles * (kNTiles / wg_waves) * split_k;
    const int n_gate   = m_tiles * (kTile / wg_waves);
    dim3 grid(n_gemm + n_gate);
    dim3 block(wg_waves * kWave);

#define ROUTER_LAUNCH(W_, KS_, D_, RN_)                                                         \
    hipLaunchKernelGGL((moe_router_topk_kernel<W_, KS_, D_, (D_ == 4 ? 3 : 4), RN_>), grid, block, 0, stream, \
                       reinterpret_cast<const __bf16*>(x.data_ptr()),                           \
                       reinterpret_cast<const __bf16*>(w.data_ptr()),                           \
                       has_bias ? reinterpret_cast<const float*>(correction_bias.data_ptr())    \
                                : nullptr,                                                      \
                       nvr,                                                                     \
                       reinterpret_cast<float*>(workspace.data_ptr()),                          \
                       arrived,                                                                 \
                       released,                                                                \
                       reinterpret_cast<float*>(topk_weights.data_ptr()),                       \
                       reinterpret_cast<int*>(topk_ids.data_ptr()),                             \
                       lg_out, M, static_cast<int>(x.stride(0)), static_cast<int>(w.stride(0)), \
                       k_steps, split_k, static_cast<int>(topk_ids.stride(0)), topk,            \
                       routed_scaling_factor)
    // Unrolled chunks for the K = 6144 split-K cuts the tuned router GEMM takes
    // (8 -> 24 steps, 6 -> 32, 4 -> 48, 3 -> 64, 2 -> 96); any other the loop.
#define ROUTER_KS(W_, D_, RN_)                           \
    switch(k_steps)                                      \
    {                                                    \
    case 24: ROUTER_LAUNCH(W_, 24, D_, RN_); break;      \
    case 32: ROUTER_LAUNCH(W_, 32, D_, RN_); break;      \
    case 48: ROUTER_LAUNCH(W_, 48, D_, RN_); break;      \
    case 64: ROUTER_LAUNCH(W_, 64, D_, RN_); break;      \
    case 96: ROUTER_LAUNCH(W_, 96, D_, RN_); break;      \
    default: ROUTER_LAUNCH(W_, 0, D_, RN_); break;       \
    }
#define ROUTER_D(W_, RN_)                                \
    if(chunk == 4)                                       \
    {                                                    \
        ROUTER_KS(W_, 4, RN_)                            \
    }                                                    \
    else                                                 \
    {                                                    \
        ROUTER_KS(W_, 8, RN_)                            \
    }
#define LAST_LAUNCH(KS_, D_, RN_)                                                               \
    hipLaunchKernelGGL((moe_router_topk_last_kernel<KS_, D_, (D_ == 4 ? 3 : 4), RN_>),          \
                       dim3(m_tiles * split_k), dim3(kNTiles * kWave), 0, stream,                \
                       reinterpret_cast<const __bf16*>(x.data_ptr()),                           \
                       reinterpret_cast<const __bf16*>(w.data_ptr()),                           \
                       has_bias ? reinterpret_cast<const float*>(correction_bias.data_ptr())    \
                                : nullptr,                                                      \
                       nvr, reinterpret_cast<float*>(workspace.data_ptr()), arrived,            \
                       reinterpret_cast<float*>(topk_weights.data_ptr()),                       \
                       reinterpret_cast<int*>(topk_ids.data_ptr()),                             \
                       lg_out, M, static_cast<int>(x.stride(0)), static_cast<int>(w.stride(0)), \
                       k_steps, split_k, static_cast<int>(topk_ids.stride(0)), topk,            \
                       routed_scaling_factor)
#define LAST_KS(D_, RN_)                                 \
    switch(k_steps)                                      \
    {                                                    \
    case 24: LAST_LAUNCH(24, D_, RN_); break;            \
    case 32: LAST_LAUNCH(32, D_, RN_); break;            \
    case 48: LAST_LAUNCH(48, D_, RN_); break;            \
    case 64: LAST_LAUNCH(64, D_, RN_); break;            \
    case 96: LAST_LAUNCH(96, D_, RN_); break;            \
    default: LAST_LAUNCH(0, D_, RN_); break;             \
    }
#define ROUTER_W(RN_)                                    \
    if(waves == 16)                                      \
    {                                                    \
        if(chunk == 4)                                   \
        {                                                \
            LAST_KS(4, RN_)                              \
        }                                                \
        else                                             \
        {                                                \
            LAST_KS(8, RN_)                              \
        }                                                \
    }                                                    \
    else if(waves == 1)                                  \
    {                                                    \
        ROUTER_D(1, RN_)                                 \
    }                                                    \
    else if(waves == 2)                                  \
    {                                                    \
        ROUTER_D(2, RN_)                                 \
    }                                                    \
    else if(waves == 4)                                  \
    {                                                    \
        ROUTER_D(4, RN_)                                 \
    }                                                    \
    else                                                 \
    {                                                    \
        ROUTER_D(8, RN_)                                 \
    }
    if(need_renorm)
    {
        ROUTER_W(true)
    }
    else
    {
        ROUTER_W(false)
    }
#undef ROUTER_W
#undef LAST_KS
#undef LAST_LAUNCH
#undef ROUTER_D
#undef ROUTER_KS
#undef ROUTER_LAUNCH
    HIP_CALL(hipGetLastError());
}

} // namespace aiter
