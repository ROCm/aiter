// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 large-message all-reduce + residual add + RMSNorm (bf16, TP2/TP4),
// for prefill-sized inputs (tens of MiB) where the Lamport kernels run out of
// inbox and RCCL is 2x off the link.
//
//   phase 1  reduce-scatter, pushed: the rows are ngpus contiguous chunks of
//            R = ceil(rows / ngpus); every rank stores the rows of chunk c into
//            slot[rank] of rank c's RS inbox (remote stores, the faster
//            direction on MI455X: ~1.4 TB/s egress per GPU against ~1.15 TB/s
//            ingress for peer loads, all four GPUs at once).
//   phase 2  the owner sums its chunk in fp32 in rank order (the bits of the
//            Lamport kernels), rounds to bf16 once, pushes the sum into
//            slot[rank] of every peer's AG inbox, and adds the residual and
//            normalizes its own rows.
//   phase 3  every rank adds the residual and normalizes the peers' rows from
//            its AG inbox.
//
// Sync: block b owns rows b, b + G, ... of every chunk on every rank, so it
// only waits for block b of its peers: a flag per (phase, sender, block),
// 128 B apart, in cached IPC memory, written with a system-scope store once
// the block's remote data stores have completed (s_wait_storecnt; a
// system-scope release fence is a full L2 writeback on gfx1250 and costs
// milliseconds), polled with system-scope loads. Data is stored and loaded
// with system-scope 128-bit accesses, so no cache maintenance is needed.
// Epochs are a per-block counter in the rank's own flag buffer; every call of
// a given size launches the same grid on every rank, so the counters of block
// b agree across ranks even when the grid changes between calls.
//
// Reuse of the inboxes across calls: a rank enters call k+1 only after its
// call k finished, and its phase 3 waited for every owner's phase-2 flag, sent
// after that owner had read its RS inbox; an AG inbox is written in call k+1
// only after the writer saw the reader's call k+1 phase-1 flag. So no barrier
// is needed at either end, eager or in a CUDA graph.
//
// out / residual_out follow ar_rmsnorm_gfx1250 and the Lamport fused kernel
// step by step (round the sum to bf16, add the residual in fp32, 32-lane
// butterfly then a serial sum over the waves of the row, x * (rsqrt * w)).
#include "ar_2shot_rms_gfx1250.h"
#include "aiter_stream.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace aiter {
namespace ar2s {

constexpr int kMaxBlocks = 1024;
constexpr int kMaxRanks  = 8;
// loads a flag poll may take before it gives up (counted in Flags::err, no hang)
constexpr int kMaxPolls = 20000000;
constexpr int kFlagPad   = 32; // uint32 a flag: 128 B apart

struct Flags
{
    uint32_t rs[kMaxRanks][kMaxBlocks * kFlagPad];
    uint32_t ag[kMaxRanks][kMaxBlocks * kFlagPad];
    uint32_t counter[kMaxBlocks];
    uint32_t err; // polls of this rank that gave up (bounded waits)
};

struct __align__(16) Peers
{
    char* data[kMaxRanks];
    Flags* flags[kMaxRanks];
};

using u32x4      = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using u32x4_gptr = __attribute__((address_space(1))) u32x4*;

__device__ __forceinline__ void st_sys(void* p, u32x4 v)
{
    __builtin_amdgcn_global_store_b128((u32x4_gptr)p, v, "");
}
__device__ __forceinline__ u32x4 ld_sys(const void* p)
{
    return __builtin_amdgcn_global_load_b128((u32x4_gptr)p, "");
}
// gfx1250 asynchronous global -> LDS copy, 16 B a lane (counted by ASYNCcnt; SYS: system scope)
typedef int i32x4v __attribute__((vector_size(16)));
template <bool SYS>
__device__ __forceinline__ void ld_async(const void* g, void* l)
{
    __builtin_amdgcn_global_load_async_to_lds_b128((__attribute__((address_space(1))) i32x4v*)g,
                                                   (__attribute__((address_space(3))) i32x4v*)l, 0, SYS ? 24 : 0);
}
__device__ __forceinline__ uint32_t f2bf(float f)
{
    uint32_t u = __float_as_uint(f);
    if((u & 0x7fffffffu) > 0x7f800000u)
        return (u >> 16) | 0x40u; // quiet NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}
__device__ __forceinline__ void unpack(u32x4 v, float* x)
{
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        x[2 * j]     = __uint_as_float(v[j] << 16);
        x[2 * j + 1] = __uint_as_float(v[j] & 0xffff0000u);
    }
}
__device__ __forceinline__ u32x4 pack(const float* x)
{
    u32x4 v;
#pragma unroll
    for(int j = 0; j < 4; ++j)
        v[j] = f2bf(x[2 * j]) | (f2bf(x[2 * j + 1]) << 16);
    return v;
}
__device__ __forceinline__ bool epoch_before(uint32_t observed, uint32_t target)
{
    return static_cast<int32_t>(observed - target) < 0;
}
__device__ __forceinline__ void signal_flag(uint32_t* p, uint32_t v)
{
    __scoped_atomic_store_n(p, v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}



// ---------------------------------------------------------------------------------------------
// The kernel (gfx1250 ISA notes):
//   - __syncthreads() is a workgroup release fence and compiles to s_wait_storecnt 0 before
//     s_barrier, which would make every row wait for its own remote stores to be acked: the
//     row-local norm-sum barrier is LDS-only, and the only store drains are the ones
//     before a flag;
//   - loads are straight-line (no per-chunk `if` around each load);
//   - pipeline: iteration i pushes step i's RS rows, reduces + AG-pushes step i - 1 once every
//     peer flagged it, copies step i - 2 once its owner flagged it, then drains its stores and
//     flags RS step i and AG step i - 1. A step is a block's whole share of a chunk (overlapped
//     smaller steps measured level or slower). A flag value counts steps, not calls: block b's
//     counter advances by its step count S_b a call (the same on every rank: it depends on b, R
//     and G only). A rank's RS inbox rows of a step are read before its AG flag of that step,
//     the last thing a peer's previous call waits for; an AG inbox row is written only after
//     the writer saw the reader's RS flag of that step in the same call;
//   - flag polls are bounded (maxit loads); a poll that gives up counts in Flags::err and the
//     block carries on (no hang; the call's output is then undefined);
//   - <= 64 VGPRs (__launch_bounds__(1024, 16)): two 768-thread blocks a CU are resident only
//     then (80 VGPRs measured one block a CU: half the grid started ~150-190 us late);
//   - phase 2 is bound by its loads' latency, ~12-16 us a row; the next row's NG + 1 packs are
//     copied into LDS asynchronously (global_load_async_to_lds_b128, system scope for the
//     inbox, s_wait_asynccnt) during this row's arithmetic, without holding VGPRs.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void lds_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
__device__ __forceinline__ void wait_flag_bounded(const uint32_t* p, uint32_t target, int maxit, uint32_t* err)
{
    for(int it = 0; it < maxit; ++it)
        if(!epoch_before(__scoped_atomic_load_n(p, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM), target))
            return;
    atomicAdd(err, 1u);
}
// block_sum with an LDS-only barrier; the serial sum over the waves keeps block_sum's order
__device__ __forceinline__ float block_sum2(float v, float (*partial)[32], int& buf)
{
    for(int off = 16; off > 0; off >>= 1)
        v += __shfl_xor(v, off);
    // wave index in an SGPR: no VGPR address kept live (or spilled) across the row
    const int wid = __builtin_amdgcn_readfirstlane(threadIdx.x / 32);
    if(__lane_id() == 0)
        partial[buf][wid] = v;
    lds_barrier();
    const int nw    = blockDim.x / 32;
    const float4* p = reinterpret_cast<const float4*>(partial[buf]);
    float r         = 0.f;
#pragma unroll
    for(int h = 0; h < 2; ++h) // two batches of 4 float4 (16 VGPRs), the waves in order
    {
        if(16 * h >= nw)
            break;
        float4 q[4];
#pragma unroll
        for(int i = 0; i < 4; ++i)
            q[i] = p[4 * h + i];
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const int w0 = 16 * h + 4 * i;
            if(w0 == 0)
                r = q[i].x;
            else if(w0 < nw)
                r += q[i].x;
            if(w0 + 1 < nw)
                r += q[i].y;
            if(w0 + 2 < nw)
                r += q[i].z;
            if(w0 + 3 < nw)
                r += q[i].w;
        }
    }
    buf ^= 1;
    return r;
}
// w: the weight pack, unpacked (and the Gemma 1 added) here
__device__ __forceinline__ void norm_packs2(
    const float* x, u32x4 wpk, float wadd, float eps, int hidden, float (*partial)[32], int& buf,
    u32x4& res_pack, u32x4& out_pack)
{
    float sq = 0.f;
#pragma unroll
    for(int j = 0; j < 8; ++j)
        sq += x[j] * x[j];
    const float inv_rms = rsqrtf(block_sum2(sq, partial, buf) / hidden + eps);
    res_pack            = pack(x);
    float w[8];
    unpack(wpk, w);
#pragma unroll
    for(int j = 0; j < 8; ++j)
        w[j] += wadd;
    float o[8];
#pragma unroll
    for(int j = 0; j < 8; ++j)
        o[j] = x[j] * (inv_rms * w[j]);
    out_pack = pack(o);
}

// LDS barrier after a flag poll; the compiler barriers keep the inbox loads below the poll
__device__ __forceinline__ void poll_barrier()
{
    asm volatile("" ::: "memory");
    lds_barrier();
    asm volatile("" ::: "memory");
}
// first row index (m) of step s: the first step has r0 rows, the others `step`
__device__ __forceinline__ int step_start(int s, int M, int r0, int step)
{
    const int m = s == 0 ? 0 : r0 + (s - 1) * step;
    return m < M ? m : M;
}

// Every per-thread address is a uniform base plus a 32-bit pack index (SGPR base + scaled VGPR
// offset), which keeps the 64-bit address arithmetic out of the VGPRs.
// Phase 2 (stage B) reads its rows through a double-buffered LDS slot of NG + 1 packs a
// thread (the NG summands and the residual, dynamic LDS), loaded asynchronously one row ahead:
// the next row's loads are in flight during this row's arithmetic without holding VGPRs.
template <int NG>
__global__ void __launch_bounds__(1024, 16) ar2s_rms_kernel(Peers peers,
                                                             int rank,
                                                             const u32x4* __restrict__ in,
                                                             const u32x4* __restrict__ res,
                                                             const u32x4* __restrict__ wgt,
                                                             u32x4* __restrict__ out,
                                                             u32x4* __restrict__ res_out,
                                                             int N,
                                                             int R,
                                                             int64_t slot_packs,
                                                             int64_t ag_off_packs,
                                                             float eps,
                                                             bool gemma,
                                                             int maxit)
{
// gfx1250-only instructions (async global -> LDS); the other device passes get an empty body.
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    __shared__ __align__(16) float partial[2][32];
    __shared__ uint32_t s_base;
    extern __shared__ __align__(16) u32x4 ar2s_pf[]; // [2][NG + 1][rp]
    int buf           = 0;
    const uint32_t t  = threadIdx.x;
    const uint32_t rp = blockDim.x; // 16B packs a row
    const int b       = blockIdx.x;
    const int G       = gridDim.x;
    const int hidden  = rp * 8;
    // pack indices are 32-bit (a launch is < 2^32 packs of every buffer): one base a buffer
    const uint32_t slot = (uint32_t)slot_packs;
    const uint32_t agof = (uint32_t)ag_off_packs;
    Flags* my_flags     = peers.flags[rank];
    const u32x4* my_data = reinterpret_cast<const u32x4*>(peers.data[rank]);
    if(t == 0)
        s_base = my_flags->counter[b];
    __syncthreads();
    const uint32_t base = s_base;

    // k = 0..NG-1 is chunk / rank (rank + k) % NG
    int rows_k[NG];
#pragma unroll
    for(int k = 0; k < NG; ++k)
    {
        int r     = N - ((rank + k) % NG) * R;
        rows_k[k] = r < 0 ? 0 : (r > R ? R : r);
    }
    int rmin = R; // rows every peer chunk has
#pragma unroll
    for(int k = 1; k < NG; ++k)
        rmin = rows_k[k] < rmin ? rows_k[k] : rmin;
    const int own_rows = rows_k[0];
    auto chunk         = [&](int k) { return (rank + k) % NG; };
    auto peer          = [&](int k) { return reinterpret_cast<u32x4*>(peers.data[(rank + k) % NG]); };
    // the step schedule (b < G <= R: M >= 1)
    const int M  = (R - b + G - 1) / G;
    const int st = M; // one step: the three phases (overlapped steps measured level or slower)
    const int r0 = st;
    const int S  = M <= r0 ? 1 : 1 + (M - r0 + st - 1) / st;

    for(int i = 0; i <= S + 1; ++i)
    {
        // ---- A: reduce-scatter push of step i, two rows of every peer chunk at a time
        if(i < S)
        {
            const int m1 = step_start(i + 1, M, r0, st);
            for(int m = step_start(i, M, r0, st); m < m1; m += 2)
            {
                const int j0      = b + m * G;
                const int j1      = j0 + G;
                const uint32_t i0 = (uint32_t)j0 * rp + t; // in a chunk / an inbox slot
                const uint32_t i1 = i0 + (uint32_t)G * rp;
                const uint32_t ds = (uint32_t)rank * slot; // slot[rank] of a peer's RS inbox
                if(m + 1 < m1 && j1 < rmin)
                {
                    u32x4 v0[NG], v1[NG];
#pragma unroll
                    for(int k = 1; k < NG; ++k)
                        v0[k] = in[(uint32_t)(chunk(k) * R) * rp + i0];
#pragma unroll
                    for(int k = 1; k < NG; ++k)
                        v1[k] = in[(uint32_t)(chunk(k) * R) * rp + i1];
#pragma unroll
                    for(int k = 1; k < NG; ++k)
                    {
                        st_sys(peer(k) + (ds + i0), v0[k]);
                        st_sys(peer(k) + (ds + i1), v1[k]);
                    }
                }
                else
                {
#pragma unroll
                    for(int h = 0; h < 2; ++h)
                    {
                        if(h && m + 1 >= m1)
                            break;
                        const int j       = h ? j1 : j0;
                        const uint32_t ii = h ? i1 : i0;
#pragma unroll
                        for(int k = 1; k < NG; ++k)
                            if(j < rows_k[k])
                                st_sys(peer(k) + (ds + ii), in[(uint32_t)(chunk(k) * R) * rp + ii]);
                    }
                }
            }
        }
        // ---- B: the own chunk's rows of step i - 1: sum in rank order (the own slot from `in`),
        // AG push of the sum, residual add + norm
        if(i >= 1 && i <= S && b + step_start(i - 1, M, r0, st) * G < own_rows)
        {
            const int s  = i - 1;
            const int m1 = step_start(s + 1, M, r0, st);
            if(t < NG && t != rank)
                wait_flag_bounded(&my_flags->rs[t][b * kFlagPad], base + s + 1, maxit, &my_flags->err);
            poll_barrier();
            const uint32_t own0 = (uint32_t)(rank * R); // first row of the own chunk
            const int mb        = step_start(s, M, r0, st);
            // row j's NG summands (SYS scope, as ld_sys) and residual into LDS slot sl
            auto issue = [&](int j, int sl) {
                const uint32_t i0 = (uint32_t)j * rp + t;
                const uint32_t io = own0 * rp + i0;
                u32x4* l          = ar2s_pf + (uint32_t)sl * (NG + 1) * rp + t;
#pragma unroll
                for(int g = 0; g < NG; ++g)
                {
                    if(g == rank)
                        ld_async<true>(in + io, l + (uint32_t)g * rp);
                    else
                        ld_async<true>(my_data + ((uint32_t)g * slot + i0), l + (uint32_t)g * rp);
                }
                ld_async<false>(res + io, l + (uint32_t)NG * rp);
            };
            issue(b + mb * G, 0);
            for(int m = mb, n = 0; m < m1; ++m, ++n)
            {
                const int j = b + m * G;
                if(j >= own_rows)
                    break;
                const uint32_t i0 = (uint32_t)j * rp + t;      // in the chunk / the slot
                const uint32_t io = own0 * rp + i0;             // in a [rows, hidden] tensor
                u32x4 got[NG], rv;
                // the next row's loads go out first; then wait for this row's NG + 1 (in order)
                asm volatile("" ::: "memory");
                if(m + 1 < m1 && j + G < own_rows)
                {
                    issue(j + G, (n + 1) & 1);
                    __builtin_amdgcn_s_wait_asynccnt(NG + 1);
                }
                else
                    __builtin_amdgcn_s_wait_asynccnt(0);
                asm volatile("" ::: "memory");
                const u32x4* l = ar2s_pf + (uint32_t)(n & 1) * (NG + 1) * rp + t;
#pragma unroll
                for(int g = 0; g < NG; ++g)
                    got[g] = l[(uint32_t)g * rp];
                rv = l[(uint32_t)NG * rp];
                float x[8]     = {};
#pragma unroll
                for(int g = 0; g < NG; ++g)
                {
                    float v[8];
                    unpack(got[g], v);
#pragma unroll
                    for(int e = 0; e < 8; ++e)
                        x[e] += v[e];
                }
                const u32x4 sum = pack(x);
                const uint32_t ag = agof + (uint32_t)rank * slot + i0; // slot[rank] of a peer's AG inbox
#pragma unroll
                for(int k = 1; k < NG; ++k)
                    st_sys(peer(k) + ag, sum);
                float r8[8];
                unpack(sum, x);
                unpack(rv, r8);
#pragma unroll
                for(int e = 0; e < 8; ++e)
                    x[e] += r8[e];
                u32x4 rpk, opk;
                norm_packs2(x, wgt[t], gemma ? 1.f : 0.f, eps, hidden, partial, buf, rpk, opk);
                res_out[io] = rpk;
                out[io]     = opk;
            }
        }
        // ---- C: the peers' rows of step i - 2 from the AG inbox: add + norm
        if(i >= 2)
        {
            const int s  = i - 2;
            const int m0 = step_start(s, M, r0, st);
            const int m1 = step_start(s + 1, M, r0, st);
            if(t > 0 && t < NG)
            {
                const int c = (rank + (int)t) % NG;
                int r       = N - c * R;
                r           = r < 0 ? 0 : (r > R ? R : r);
                if(b + m0 * G < r)
                    wait_flag_bounded(&my_flags->ag[c][b * kFlagPad], base + s + 1, maxit, &my_flags->err);
            }
            poll_barrier();
#pragma unroll
            for(int k = 1; k < NG; ++k)
            {
                const uint32_t c0 = (uint32_t)(chunk(k) * R);
                for(int m = m0; m < m1; ++m)
                {
                    const int j = b + m * G;
                    if(j >= rows_k[k])
                        break;
                    const uint32_t i0 = (uint32_t)j * rp + t;
                    const uint32_t io = c0 * rp + i0;
                    const u32x4 sv    = ld_sys(my_data + (agof + (uint32_t)chunk(k) * slot + i0));
                    const u32x4 rv    = res[io];
                    float x[8], r8[8];
                    unpack(sv, x);
                    unpack(rv, r8);
#pragma unroll
                    for(int e = 0; e < 8; ++e)
                        x[e] += r8[e];
                    u32x4 rpk, opk;
                    norm_packs2(x, wgt[t], gemma ? 1.f : 0.f, eps, hidden, partial, buf, rpk, opk);
                    res_out[io] = rpk;
                    out[io]     = opk;
                }
            }
        }
        // ---- drain, then flag RS step i and AG step i - 1 to every peer
        if(i <= S)
        {
            asm volatile("s_wait_storecnt 0x0" ::: "memory");
            lds_barrier(); // every wave's stores are acked
            if(t < NG && t != rank)
            {
                if(i < S)
                    signal_flag(&peers.flags[t]->rs[rank][b * kFlagPad], base + i + 1);
                if(i >= 1)
                    signal_flag(&peers.flags[t]->ag[rank][b * kFlagPad], base + i);
            }
        }
    }
    if(t == 0)
        my_flags->counter[b] = base + S;
#endif // defined(__gfx1250__) || defined(__gfx1250_strict__)
}

struct Ctx
{
    int rank, world, grid;
    int64_t max_bytes;
    Peers peers;
};

} // namespace ar2s

int64_t ar2s_rms_buffer_size(int64_t world_size, int64_t max_bytes)
{
    // RS inbox and AG inbox, world_size slots of a chunk each (slot = max_bytes / world)
    (void)world_size;
    return 2 * ((max_bytes + 4095) / 4096 * 4096);
}

int64_t ar2s_rms_flags_size() { return (int64_t)sizeof(ar2s::Flags); }

fptr_t init_ar2s_rms(int64_t rank,
                     const std::vector<int64_t>& data_ptrs,
                     const std::vector<int64_t>& flag_ptrs,
                     int64_t max_bytes,
                     int64_t grid)
{
    const int world = (int)data_ptrs.size();
    if((world != 2 && world != 4) || (int)flag_ptrs.size() != world)
        throw std::invalid_argument("ar2s_rms: world size must be 2 or 4");
    auto* c      = new ar2s::Ctx();
    c->rank      = (int)rank;
    c->world     = world;
    c->max_bytes = (max_bytes + 4095) / 4096 * 4096;
    c->grid      = (int)std::min<int64_t>(std::max<int64_t>(grid, 1), ar2s::kMaxBlocks);
    for(int r = 0; r < world; ++r)
    {
        c->peers.data[r]  = reinterpret_cast<char*>(data_ptrs[r]);
        c->peers.flags[r] = reinterpret_cast<ar2s::Flags*>(flag_ptrs[r]);
    }
    // epochs and flags start at 0; the caller barriers before the first call
    if(hipMemset(c->peers.flags[rank], 0, sizeof(ar2s::Flags)) != hipSuccess ||
       hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("ar2s_rms: cannot clear the flag buffer");
    return reinterpret_cast<fptr_t>(c);
}

int64_t ar2s_rms_max_rows(fptr_t ctx, int64_t hidden)
{
    auto* c = reinterpret_cast<ar2s::Ctx*>(ctx);
    return c->max_bytes / c->world / (hidden * 2) * c->world;
}

void ar2s_rms(fptr_t ctx,
              int64_t inp,
              int64_t res_inp,
              int64_t w,
              int64_t out,
              int64_t res_out,
              int64_t rows,
              int64_t hidden,
              double eps,
              bool gemma_norm)
{
    auto* c      = reinterpret_cast<ar2s::Ctx*>(ctx);
    const int rp = (int)(hidden / 8);
    if(hidden % 256 != 0 || rp > 1024)
        throw std::invalid_argument("ar2s_rms: hidden must be a multiple of 256 and at most 8192");
    if(rows <= 0)
        return;
    const int64_t R          = (rows + c->world - 1) / c->world;
    const int64_t slot_bytes = c->max_bytes / c->world / 16 * 16;
    if(R * hidden * 2 > slot_bytes)
        throw std::invalid_argument("ar2s_rms: " + std::to_string(rows) +
                                    " rows exceed the inbox; split the call (ar2s_rms_max_rows)");
    if(((inp | res_inp | w | out | res_out) & 15) != 0)
        throw std::invalid_argument("ar2s_rms: operands must be 16-byte aligned");
    // the whole grid even if the last round of rows leaves blocks idle: fewer,
    // evenly loaded blocks measured slower (each block is latency bound)
    const int G              = (int)std::min<int64_t>(c->grid, R);
    const int64_t slot_packs = slot_bytes / 16;
    const int64_t ag_off     = (int64_t)c->world * slot_packs;
    hipStream_t stream       = aiter::getCurrentHIPStream();
    // the kernel indexes with 32-bit packs: both inboxes and every [rows, hidden] operand fit
    if(2 * ag_off >= (int64_t(1) << 32) || rows * rp >= (int64_t(1) << 31))
        throw std::invalid_argument("ar2s_rms: inbox or operand past 32-bit pack indexing");
    // phase 2's prefetch: [2][NG + 1][rp] packs of dynamic LDS (122880 B at TP4, hidden 6144)
    const size_t pf_bytes = (size_t)2 * (c->world + 1) * rp * 16;
#define AR2S_LAUNCH(NG)                                                                        \
    do {                                                                                       \
        auto* kp = ar2s::ar2s_rms_kernel<NG>;                                                  \
        (void)hipFuncSetAttribute((const void*)kp, hipFuncAttributeMaxDynamicSharedMemorySize, \
                                  (int)pf_bytes);                                              \
        kp<<<G, rp, pf_bytes, stream>>>(c->peers,                                              \
                                        c->rank,                                               \
                                        (const ar2s::u32x4*)inp,                               \
                                        (const ar2s::u32x4*)res_inp,                           \
                                        (const ar2s::u32x4*)w,                                 \
                                        (ar2s::u32x4*)out,                                     \
                                        (ar2s::u32x4*)res_out,                                 \
                                        (int)rows,                                             \
                                        (int)R,                                                \
                                        slot_packs,                                            \
                                        ag_off,                                                \
                                        (float)eps,                                            \
                                        gemma_norm,                                            \
                                        ar2s::kMaxPolls);                                      \
    } while(0)
    switch(c->world)
    {
    case 2: AR2S_LAUNCH(2); break;
    case 4: AR2S_LAUNCH(4); break;
    default: throw std::invalid_argument("ar2s_rms: unsupported world size");
    }
#undef AR2S_LAUNCH
    hipError_t e = hipGetLastError();
    if(e != hipSuccess)
        throw std::runtime_error(std::string("ar2s_rms launch: ") + hipGetErrorString(e));
}

} // namespace aiter
