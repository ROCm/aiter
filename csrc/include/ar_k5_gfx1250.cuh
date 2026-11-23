// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 TP4 one-shot all-reduce + residual add + RMSNorm (+ native group-32 MXFP8) for
// decode sizes (hidden 6144, up to a few hundred rows), see ar_k5_gfx1250.h. Device code,
// included by ar_k5_gfx1250.cu (Lamport-order kernels) and ar_k5_gluon_gfx1250.cu
// (Gluon-order kernels, built with IEEE fp32 denormals as Triton builds the Gluon kernel).
//
// Protocol: the Lamport data-as-flag one-shot of lamport_ar_rmsnorm_gfx1250, on an inbox of
// its own. Each rank owns [3 buffers][4 senders][slot] 16-byte packs, armed with 0xff (the
// 0xffff bf16 sentinel). A sender rewrites its 0xffff half-words to 0x7fff and stores each
// pack (system scope, 128-bit) into slot[rank] of buffer round % 3 of every peer. A receiver
// re-arms buffer (round + 2) % 3 of its packs first (consumed in the previous call; a peer
// writes it again only in round + 2, after this rank's next call), then polls its own inbox
// until no half-word of its packs is 0xffff. The rounds are rank-local, one per 96-pack wave
// slab and per role (a send and a receive counter, as the Gluon all-reduce keeps one per
// role), advanced once by every call that covers the slab. Polls are bounded (maxit) and
// count timeouts in err.
//
// Work: a launch is rows sender workgroups, then rows receiver workgroups (the GPU starts
// workgroups in order, so a polling receiver cannot keep a sender from starting), 256
// threads each. Wave w, lane L of a row handles packs w * 96 + 32 j + L (j < 3), values
// 768 w + 256 j + 8 L + k. The epilogue keeps the bits of the path it replaces:
//   GLUON:   all_reduce_rmsnorm_per_token_quant(quant=False) / all_reduce_rmsnorm_mxfp8_quant
//            (aiter/ops/triton/_gluon_kernels/gfx1250/comms/all_reduce.py): the sum of squares
//            in the order of add_rmsnorm_quant_kernel<256, 24>, fp64 eps, (s * rstd) * (1 + w),
//            no fused multiply-adds, its E8M0 / E4M3 steps.
//   LAMPORT: lamport_ar_rmsnorm_gfx1250 (custom_all_reduce_gfx1250.cuh) and, for MXFP8,
//            per_1x32_mx_quant_hip of its bf16 output (the steps of ar2s's mxfp8_quant8).
//
// TAIL (the deferred routed-MoE tail, aiter moe_defer_tail): the input rows are not read from x
// but computed where they are loaded, by the sender (for its peers) and again by the receiver
// (its own rows), with the bits of the gather-reduce launch they replace
// (kernels/moe_gather_reduce.py, f32 route weights, split_k 1, the moeglue epilogue):
//   acc = 0; for k < 4: acc = acc + w[t, k] * (0 + bf16 g[rows[t, k]])   (no fused multiply-add;
//         a route with rows < 0 adds w * 0), x = bf16(acc); with an addend: x = bf16(x * scale)
//         when scale != 1, then x = bf16(x + addend[t]).
// That launch runs with IEEE fp32 denormals (FlyDSL's kernel descriptor), as the Gluon-order
// kernels are built; the Lamport-order kernels (flushed) switch the MODE register's fp32
// denormal field to IEEE around the tail arithmetic.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "mx_quant_utils.h"

namespace aiter {
namespace k5ar {

constexpr int kRanks     = 4;
constexpr int kBuf       = 3;
constexpr int kHidden    = 6144;
constexpr int kRowPacks  = kHidden / 8; // 768
constexpr int kWaves     = 8;           // waves of a row's sender or receiver workgroup
constexpr int kJ         = 3;           // packs a lane
constexpr int kSlabPacks = 32 * kJ;     // 96: a wave's slab, one round counter
constexpr int kThreads   = 32 * kWaves;

using u32x4   = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using u32x4_g = __attribute__((address_space(1))) u32x4*;
typedef float f32x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x8_t __attribute__((ext_vector_type(8)));
typedef int i32x2_t __attribute__((ext_vector_type(2)));

// the deferred MoE tail of a TAIL kernel (see the file comment)
struct Tail
{
    const u32x4* g;      // GEMM2 output rows [*, hidden] bf16 (the grouped output, flat)
    const int32_t* rows; // [rows, 4] the GEMM2 row of route (t, k), < 0: none (adds w * 0)
    const float* w;      // [rows, 4] route weights
    const u32x4* add;    // [rows, hidden] bf16 addend, or null
    float scale;         // applied (and rounded) only with an addend and scale != 1
    int flags;           // kTailAdd | kTailScale
};
constexpr int kTailAdd   = 1;
constexpr int kTailScale = 2;

struct Args
{
    const u32x4* x;
    const u32x4* res;
    const u32x4* w;
    u32x4* res_out;
    void* out;      // bf16 [rows, hidden], or E4M3 [rows, hidden] with fp8 out
    uint8_t* scale; // E8M0 [rows, hidden / 32], fp8 out only
    char* in0;
    char* in1;
    char* in2;
    char* in3;
    uint32_t* send_ctr;
    uint32_t* recv_ctr;
    uint32_t* err;
    long slot; // packs a sender a buffer
    double eps;
    int hidden;
    int rows;
    int rank;
    int maxit;
    int gemma;
    Tail tail;
};

__device__ __forceinline__ void st_sys(char* p, u32x4 v)
{
    __builtin_amdgcn_global_store_b128((u32x4_g)p, v, "");
}
__device__ __forceinline__ u32x4 ld_sys(const char* p)
{
    return __builtin_amdgcn_global_load_b128((u32x4_g)p, "");
}
__device__ __forceinline__ char* inbox(const Args& a, int g)
{
    return g == 0 ? a.in0 : g == 1 ? a.in1 : g == 2 ? a.in2 : a.in3;
}
// one bit per half-word that is the 0xffff sentinel
__device__ __forceinline__ uint32_t sentinel_mask(uint32_t w)
{
    const uint32_t x = ~w;
    const uint32_t t = ((x & 0x7fff7fffu) + 0x7fff7fffu) | x;
    return (t & 0x80008000u) ^ 0x80008000u;
}
__device__ __forceinline__ float bf2f(uint32_t w, int k)
{
    return __uint_as_float(k & 1 ? w & 0xffff0000u : w << 16);
}
// two fp32 -> packed bf16 bits: the gfx1250 convert (round to nearest even), the one both
// replaced paths use (Triton's fp32 -> bf16 and opus's fp32_to_bf16 on gfx1250)
__device__ __forceinline__ uint32_t f2bf2(float lo, float hi)
{
    const bf16x2 b = __builtin_convertvector((f32x2){lo, hi}, bf16x2);
    return __builtin_bit_cast(uint32_t, b);
}
// software fp32 -> bf16 (ar2s's f2bf, used by mxfp8_quant8 for the clip bounds)
__device__ __forceinline__ uint32_t f2bf_sw(float f)
{
    uint32_t u = __float_as_uint(f);
    if((u & 0x7fffffffu) > 0x7f800000u)
        return (u >> 16) | 0x40u;
    u += 0x7fffu + ((u >> 16) & 1u);
    return u >> 16;
}
// LDS-only workgroup barrier: no wait on this wave's outstanding global stores
__device__ __forceinline__ void lds_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// ---------------------------------------------------------------------------------------------
// the deferred MoE tail: the input packs of a slab, computed (see the file comment)
// ---------------------------------------------------------------------------------------------
// s_setreg / s_getreg field of the MODE register's fp32 denormal bits ([5:4])
constexpr int kModeF32Denorm = 1 | (4 << 6) | ((2 - 1) << 11);

// the arithmetic of tail_slab for one addend / scale case (uniform: one branch a slab, none a pack)
template <bool ADD, bool SCALE>
__device__ __forceinline__ void tail_math(const u32x4 (&gv)[4][kJ],
                                          const u32x4 (&av)[kJ],
                                          const float (&ww)[4],
                                          float scale,
                                          u32x4 (&x)[kJ])
{
#pragma clang fp contract(off)
#pragma unroll
    for(int j = 0; j < kJ; ++j)
    {
#pragma unroll
        for(int c = 0; c < 4; ++c)
        {
            float acc[2];
#pragma unroll
            for(int h = 0; h < 2; ++h)
            {
                float s_ = 0.0f;
#pragma unroll
                for(int k = 0; k < 4; ++k)
                {
                    const float red = bf2f(gv[k][j][c], h) + 0.0f;
                    s_              = s_ + ww[k] * red;
                }
                acc[h] = s_;
            }
            uint32_t o = f2bf2(acc[0], acc[1]);
            if constexpr(ADD)
            {
                if constexpr(SCALE)
                    o = f2bf2(bf2f(o, 0) * scale, bf2f(o, 1) * scale);
                o = f2bf2(bf2f(o, 0) + bf2f(av[j][c], 0), bf2f(o, 1) + bf2f(av[j][c], 1));
            }
            x[j][c] = o;
        }
    }
}

template <bool SWITCH_DENORM>
__device__ __forceinline__ void tail_slab(const Args& a, int r, int w, int lane, u32x4 (&x)[kJ])
{
    // every scalar of the tail read once, up front (the route of row r is workgroup-uniform)
    const u32x4* g    = a.tail.g;
    const u32x4* add  = a.tail.add;
    const int flags   = a.tail.flags;
    const float scale = a.tail.scale;
    // scalar loads (constant address space): one s_load_b128 each, no VMEM round trip + readfirstlane
    typedef int i32x4_v __attribute__((ext_vector_type(4)));
    typedef const __attribute__((address_space(4))) i32x4_v* c_i32x4_p;
    const i32x4_v rv  = *(c_i32x4_p)(a.tail.rows + 4 * (long)r);
    const i32x4_v wv  = *(c_i32x4_p)(reinterpret_cast<const int*>(a.tail.w) + 4 * (long)r);
    const int rw[4]   = {rv[0], rv[1], rv[2], rv[3]};
    const float ww[4] = {__int_as_float(wv[0]), __int_as_float(wv[1]), __int_as_float(wv[2]), __int_as_float(wv[3])};
    const long pk0 = (long)w * kSlabPacks + lane; // + 32 j: this lane's packs of a row
    u32x4 av[kJ];
    if(flags & kTailAdd)
    {
        const u32x4* src = add + (long)r * kRowPacks + pk0;
#pragma unroll
        for(int j = 0; j < kJ; ++j)
            av[j] = src[32 * j];
    }
    u32x4 gv[4][kJ];
#pragma unroll
    for(int k = 0; k < 4; ++k)
    {
        if(rw[k] >= 0)
        {
            const u32x4* src = g + (long)rw[k] * kRowPacks + pk0;
#pragma unroll
            for(int j = 0; j < kJ; ++j)
                gv[k][j] = src[32 * j];
        }
        else
        {
#pragma unroll
            for(int j = 0; j < kJ; ++j)
                gv[k][j] = u32x4{0u, 0u, 0u, 0u};
        }
    }
    uint32_t saved_mode = 0;
    if constexpr(SWITCH_DENORM)
    {
        saved_mode = __builtin_amdgcn_s_getreg(kModeF32Denorm);
        __builtin_amdgcn_s_setreg(kModeF32Denorm, 3u);
    }
    if(!(flags & kTailAdd))
        tail_math<false, false>(gv, av, ww, scale, x);
    else if(flags & kTailScale)
        tail_math<true, true>(gv, av, ww, scale, x);
    else
        tail_math<true, false>(gv, av, ww, scale, x);
    if constexpr(SWITCH_DENORM)
        __builtin_amdgcn_s_setreg(kModeF32Denorm, saved_mode);
}

// ---------------------------------------------------------------------------------------------
// protocol
// ---------------------------------------------------------------------------------------------
// sends the slab of wave w of row r: x (TAIL: computed), 0xffff -> 0x7fff, into slot[rank] of
// every peer
template <bool TAIL = false, bool SWITCH_DENORM = false>
__device__ __forceinline__ void send_slab(const Args& a, int r, int w, int lane)
{
    const int g   = r * kWaves + w;
    const long p0 = (long)r * kRowPacks + w * kSlabPacks + lane;
    u32x4 v[kJ];
    uint32_t round;
    if constexpr(TAIL)
    {
        // The round counter by a scalar load (written by the previous call, visible at dispatch):
        // a vector load + readfirstlane here would hold the tail's row loads back by a round trip.
        typedef const __attribute__((address_space(4))) uint32_t* c_u32_p;
        round = *(c_u32_p)(a.send_ctr + __builtin_amdgcn_readfirstlane(g));
        tail_slab<SWITCH_DENORM>(a, r, w, lane, v);
    }
    else
    {
        round = __builtin_amdgcn_readfirstlane(a.send_ctr[g]);
#pragma unroll
        for(int j = 0; j < kJ; ++j)
            v[j] = a.x[p0 + 32 * j];
    }
#pragma unroll
    for(int j = 0; j < kJ; ++j)
#pragma unroll
        for(int i = 0; i < 4; ++i)
            v[j][i] ^= sentinel_mask(v[j][i]);
    const long at = (((long)round * kRanks + a.rank) * a.slot + p0) * 16;
#pragma unroll
    for(int i = 1; i < kRanks; ++i)
    {
        char* dst = inbox(a, (a.rank + i) & (kRanks - 1)) + at;
#pragma unroll
        for(int j = 0; j < kJ; ++j)
            st_sys(dst + 32 * 16 * j, v[j]);
    }
    if(lane == 0)
        a.send_ctr[g] = (round + 1) % kBuf;
}

__device__ __forceinline__ void
poll_load(u32x4 (&in)[kRanks - 1][kJ], const char* const (&src)[kRanks - 1])
{
#pragma unroll
    for(int i = 0; i < kRanks - 1; ++i)
#pragma unroll
        for(int j = 0; j < kJ; ++j)
            in[i][j] = ld_sys(src[i] + 32 * 16 * j);
}
__device__ __forceinline__ bool poll_pending(const u32x4 (&in)[kRanks - 1][kJ])
{
    uint32_t m = 0;
#pragma unroll
    for(int i = 0; i < kRanks - 1; ++i)
#pragma unroll
        for(int j = 0; j < kJ; ++j)
#pragma unroll
            for(int c = 0; c < 4; ++c)
                m |= sentinel_mask(in[i][j][c]);
    return m != 0;
}

// The receiving half of a slab: re-arm, poll. Returns the packs of sender (rank + i) % 4 in
// in[i - 1] and this rank's own packs (and the residual and weight packs) loaded before the wait.
struct Slab
{
    long p0;
    u32x4 own[kJ], rs[kJ], wt[kJ];
    u32x4 in[kRanks - 1][kJ];
};

template <bool TAIL = false, bool SWITCH_DENORM = false>
__device__ __forceinline__ void receive_slab(const Args& a, int r, int w, int lane, Slab& sl)
{
    const int g          = r * kWaves + w;
    sl.p0                = (long)r * kRowPacks + w * kSlabPacks + lane;
    const uint32_t round = __builtin_amdgcn_readfirstlane(a.recv_ctr[g]);
    const int rank       = a.rank;
    char* my             = inbox(a, rank);
#pragma unroll
    for(int j = 0; j < kJ; ++j)
    {
        if constexpr(!TAIL)
            sl.own[j] = a.x[sl.p0 + 32 * j];
        sl.rs[j] = a.res[sl.p0 + 32 * j];
        sl.wt[j] = a.w[w * kSlabPacks + 32 * j + lane];
    }
    // TAIL: this rank's own rows, computed as the sender computes them (the same bits)
    if constexpr(TAIL)
        tail_slab<SWITCH_DENORM>(a, r, w, lane, sl.own);
    // re-arm before the wait, which hides the acks of these stores
    {
        const long rearm = ((long)((round + 2) % kBuf) * kRanks * a.slot + sl.p0) * 16;
        const u32x4 sentinel{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
#pragma unroll
        for(int i = 1; i < kRanks; ++i)
        {
            char* dst = my + rearm + (long)((rank + i) & (kRanks - 1)) * a.slot * 16;
#pragma unroll
            for(int j = 0; j < kJ; ++j)
                st_sys(dst + 32 * 16 * j, sentinel);
        }
        if(lane == 0)
            a.recv_ctr[g] = (round + 1) % kBuf;
    }
    const long at = ((long)round * kRanks * a.slot + sl.p0) * 16;
    const char* src[kRanks - 1];
#pragma unroll
    for(int i = 1; i < kRanks; ++i)
        src[i - 1] = my + at + (long)((rank + i) & (kRanks - 1)) * a.slot * 16;
    // wave-uniform polls: the wave leaves when one poll saw the data of all its lanes
    int it       = 0;
    bool pending = true;
    while(pending && it < a.maxit)
    {
        // without the clobber the loads leave the loop
        asm volatile("" ::: "memory");
        poll_load(sl.in, src);
        __builtin_amdgcn_sched_barrier(0);
        pending = __builtin_amdgcn_ballot_w32(poll_pending(sl.in)) != 0;
        ++it;
    }
    if(pending && lane == 0)
        atomicAdd(a.err, 1u);
}

// rank-order sources of plane j: this rank's packs at index RK
template <int RK>
__device__ __forceinline__ const u32x4& src_pack(const Slab& sl, int q, int j)
{
    const int i = (q - RK) & (kRanks - 1);
    return i == 0 ? sl.own[j] : sl.in[i - 1][j];
}

// ---------------------------------------------------------------------------------------------
// GLUON epilogue (rows <= 128 today): the bits of the Gluon one-shot kernels
// ---------------------------------------------------------------------------------------------
template <int RK>
__device__ __forceinline__ void gluon_rank_sum(const Slab& sl, float (&s)[kJ][8])
{
#pragma clang fp contract(off)
#pragma unroll
    for(int j = 0; j < kJ; ++j)
#pragma unroll
        for(int k = 0; k < 8; ++k)
        {
            float acc = bf2f(src_pack<RK>(sl, 0, j)[k / 2], k);
            acc       = acc + bf2f(src_pack<RK>(sl, 1, j)[k / 2], k);
            acc       = acc + bf2f(src_pack<RK>(sl, 2, j)[k / 2], k);
            acc       = acc + bf2f(src_pack<RK>(sl, 3, j)[k / 2], k);
            s[j][k]   = acc;
        }
}

__device__ __forceinline__ float rcp_asm(float v)
{
    float r;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(v));
    return r;
}
__device__ __forceinline__ float rsq_asm(float v)
{
    float r;
    asm volatile("v_rsq_f32 %0, %1" : "=v"(r) : "v"(v));
    return r;
}

template <bool FP8>
__device__ __forceinline__ void gluon_epilogue(const Args& a, const Slab& sl, int r, int w, int lane)
{
#pragma clang fp contract(off)
    float s[kJ][8];
    switch(a.rank)
    {
    case 0: gluon_rank_sum<0>(sl, s); break;
    case 1: gluon_rank_sum<1>(sl, s); break;
    case 2: gluon_rank_sum<2>(sl, s); break;
    default: gluon_rank_sum<3>(sl, s); break;
    }
    // the all-reduce result is bf16, the residual add fp32; squares in (j, k) order
    float sq = 0.f;
#pragma unroll
    for(int j = 0; j < kJ; ++j)
    {
        u32x4 o;
#pragma unroll
        for(int c = 0; c < 4; ++c)
        {
            const uint32_t ab = f2bf2(s[j][2 * c], s[j][2 * c + 1]);
            s[j][2 * c]       = bf2f(ab, 0) + bf2f(sl.rs[j][c], 0);
            s[j][2 * c + 1]   = bf2f(ab, 1) + bf2f(sl.rs[j][c], 1);
        }
#pragma unroll
        for(int k = 0; k < 8; ++k)
        {
            const float e = s[j][k] * s[j][k];
            sq            = (j == 0 && k == 0) ? e : sq + e;
        }
#pragma unroll
        for(int c = 0; c < 4; ++c)
            o[c] = f2bf2(s[j][2 * c], s[j][2 * c + 1]);
        a.res_out[sl.p0 + 32 * j] = o;
    }
    // lane pairs at distance 1 to 16, then wave pairs at distance 1 to 4
#pragma unroll
    for(int d = 1; d < 32; d <<= 1)
        sq = sq + __shfl_xor(sq, d, 32);
    __shared__ float part[kWaves];
    if(lane == 0)
        part[w] = sq;
    lds_barrier();
    const float total =
        ((part[0] + part[1]) + (part[2] + part[3])) + ((part[4] + part[5]) + (part[6] + part[7]));
    // rsqrt(sum * rcp(n) + eps), eps added in fp64
    const float mean   = total * rcp_asm((float)a.hidden);
    const float rstd   = rsq_asm((float)((double)mean + a.eps));
    const float offset = a.gemma ? 1.0f : 0.0f;
    float y[kJ][8];
#pragma unroll
    for(int j = 0; j < kJ; ++j)
#pragma unroll
        for(int k = 0; k < 8; ++k)
            y[j][k] = (s[j][k] * rstd) * (bf2f(sl.wt[j][k / 2], k) + offset);
    const int col0 = w * 768 + 8 * lane; // + 256 j
    if constexpr(!FP8)
    {
        u32x4* out = reinterpret_cast<u32x4*>(a.out);
#pragma unroll
        for(int j = 0; j < kJ; ++j)
        {
            u32x4 o;
#pragma unroll
            for(int c = 0; c < 4; ++c)
                o[c] = f2bf2(y[j][2 * c], y[j][2 * c + 1]);
            out[sl.p0 + 32 * j] = o;
        }
    }
    else
    {
        // native group-32 MXFP8 of the bf16 norm output; a group is 4 lanes of a plane
        uint32_t* out = reinterpret_cast<uint32_t*>(a.out);
#pragma unroll
        for(int j = 0; j < kJ; ++j)
        {
            float yb[8];
#pragma unroll
            for(int c = 0; c < 4; ++c)
            {
                const uint32_t yp = f2bf2(y[j][2 * c], y[j][2 * c + 1]);
                yb[2 * c]         = bf2f(yp, 0);
                yb[2 * c + 1]     = bf2f(yp, 1);
            }
            float amax = 0.f;
#pragma unroll
            for(int k = 0; k < 8; ++k)
                amax = fmaxf(amax, fabsf(yb[k]));
            amax                   = fmaxf(amax, __shfl_xor(amax, 1, 32));
            amax                   = fmaxf(amax, __shfl_xor(amax, 2, 32));
            amax                   = fmaxf(amax, __uint_as_float(0x2EDBE6FFu)); // fp32(1e-10)
            const float scaled     = amax * __uint_as_float(0x3B124925u);       // fp32(1 / 448)
            const uint32_t bits    = __float_as_uint(scaled);
            const uint32_t code    = (bits >> 23) + ((bits & 0x7FFFFFu) != 0u ? 1u : 0u);
            const int e            = 254 - (int)code;
            const float reciprocal = __uint_as_float((uint32_t)(e > 0 ? e : 0) << 23);
            uint32_t q[2];
#pragma unroll
            for(int h = 0; h < 2; ++h)
            {
                float c4[4];
#pragma unroll
                for(int k = 0; k < 4; ++k)
                    c4[k] = fminf(fmaxf(yb[4 * h + k] * reciprocal, -448.0f), 448.0f);
                uint32_t pk = __builtin_amdgcn_cvt_pk_fp8_f32(c4[0], c4[1], 0, false);
                pk          = __builtin_amdgcn_cvt_pk_fp8_f32(c4[2], c4[3], pk, true);
                q[h]        = pk;
            }
            const long col = (long)r * kHidden + col0 + 256 * j;
            *reinterpret_cast<uint2*>(out + col / 4) = make_uint2(q[0], q[1]);
            if((lane & 3) == 0)
                a.scale[(long)r * (kHidden / 32) + (col0 + 256 * j) / 32] = (uint8_t)code;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// LAMPORT epilogue (rows 129-256 today): the bits of lamport_ar_rmsnorm_gfx1250 (+ the MXFP8
// steps of per_1x32_mx_quant_hip), built with fp32 denormals flushed as the custom all-reduce
// module is; its arithmetic follows the built kernel's instructions (no fused multiply-adds)
// ---------------------------------------------------------------------------------------------
// ar2s's mxfp8_quant8: per_1x32_mx_quant_hip(fp8, e8m0) of 8 bf16 values, the group being
// lanes 4g..4g+3 of a wave
__device__ __forceinline__ void mxfp8_quant8(u32x4 bf, uint32_t& lo, uint32_t& hi, uint8_t& sbyte)
{
    float v[8];
#pragma unroll
    for(int j = 0; j < 4; ++j)
    {
        v[2 * j]     = __uint_as_float(bf[j] << 16);
        v[2 * j + 1] = __uint_as_float(bf[j] & 0xffff0000u);
    }
    float amax = 0.f;
#pragma unroll
    for(int j = 0; j < 8; ++j)
        amax = fmaxf(amax, fabsf(v[j]));
    amax                  = fmaxf(amax, __shfl_xor(amax, 1));
    amax                  = fmaxf(amax, __shfl_xor(amax, 2));
    const bool degenerate = !(amax < __builtin_inff());
    amax                  = fminf(amax, 448.0f * 0x1.0p119f);
    float scale = aiter::fp_f32_to_e8m0_scale<aiter::kDefaultMxScaleRoundMode, aiter::MxDtype::FP8_E4M3>(amax);
    constexpr bool kScaleMayClip = aiter::kDefaultMxScaleRoundMode == aiter::MxScaleRoundMode::RoundDown ||
                                   aiter::kDefaultMxScaleRoundMode == aiter::MxScaleRoundMode::Even;
    if(kScaleMayClip || degenerate)
    {
        const float h      = 448.0f * scale;
        const uint32_t hb  = f2bf_sw(h);
        const uint32_t nhb = f2bf_sw(-h);
        uint32_t wd[4]     = {bf[0], bf[1], bf[2], bf[3]};
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            uint32_t e = (wd[j / 2] >> (16 * (j % 2))) & 0xffffu;
            if(v[j] > h)
                e = hb;
            if(v[j] < -h)
                e = nhb;
            wd[j / 2] = (wd[j / 2] & ~(0xffffu << (16 * (j % 2)))) | (e << (16 * (j % 2)));
        }
        bf = u32x4{wd[0], wd[1], wd[2], wd[3]};
    }
    const i32x2_t q = __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16(__builtin_bit_cast(bf16x8_t, bf), scale);
    lo              = (uint32_t)q[0];
    hi              = (uint32_t)q[1];
    sbyte           = (uint8_t)((__float_as_uint(scale) >> 23) & 0xffu);
}

template <bool FP8>
__device__ __forceinline__ void lamport_epilogue(const Args& a, const Slab& sl, int r, int w, int lane)
{
    // No contraction: the built Lamport kernel rounds every square and adds in order (packed
    // multiplies, then scalar adds), and the norm is x * (inv_rms * w) in two multiplies.
#pragma clang fp contract(off)
    // the Lamport kernel's thread holds one pack; this lane holds the packs of 3 such threads
    // (planes j), each its own wave of 32 packs: wave 3 w + j of the row
    __shared__ float part[3 * kWaves];
    float x[kJ][8];
#pragma unroll
    for(int j = 0; j < kJ; ++j)
    {
        // rank-order sum from 0, own packs canonicalized as the Lamport kernel sends them
        u32x4 mine = sl.own[j];
#pragma unroll
        for(int c = 0; c < 4; ++c)
            mine[c] ^= sentinel_mask(mine[c]);
#pragma unroll
        for(int k = 0; k < 8; ++k)
            x[j][k] = 0.f;
#pragma unroll
        for(int q = 0; q < kRanks; ++q)
        {
            const int i = (q - a.rank) & (kRanks - 1);
            u32x4 v     = mine;
            v           = i == 1 ? sl.in[0][j] : v;
            v           = i == 2 ? sl.in[1][j] : v;
            v           = i == 3 ? sl.in[2][j] : v;
#pragma unroll
            for(int k = 0; k < 8; ++k)
                x[j][k] += bf2f(v[k / 2], k);
        }
        // as the Lamport kernel: round the sum to bf16, add the residual in fp32, square in order
        u32x4 sum;
        float square_sum = 0.f;
#pragma unroll
        for(int c = 0; c < 4; ++c)
        {
            const uint32_t ab = f2bf2(x[j][2 * c], x[j][2 * c + 1]);
            x[j][2 * c]       = bf2f(ab, 0) + bf2f(sl.rs[j][c], 0);
            square_sum += x[j][2 * c] * x[j][2 * c];
            x[j][2 * c + 1] = bf2f(ab, 1) + bf2f(sl.rs[j][c], 1);
            square_sum += x[j][2 * c + 1] * x[j][2 * c + 1];
            sum[c] = f2bf2(x[j][2 * c], x[j][2 * c + 1]);
        }
        a.res_out[sl.p0 + 32 * j] = sum;
        // the 32-lane butterfly of a Lamport wave, distance 16 first
        float pw = square_sum;
        for(int off = 16; off > 0; off >>= 1)
            pw = pw + __shfl_xor(pw, off);
        if(lane == 0)
            part[3 * w + j] = pw;
    }
    lds_barrier();
    // the waves of the row in order
    float total = part[0];
    for(int g = 1; g < 3 * kWaves; ++g)
        total += part[g];
    const int hidden    = a.hidden;
    const float eps     = (float)a.eps;
    const float inv_rms = rsqrtf(total / hidden + eps);
    const int col0      = w * 768 + 8 * lane; // + 256 j
#pragma unroll
    for(int j = 0; j < kJ; ++j)
    {
        u32x4 o;
#pragma unroll
        for(int c = 0; c < 4; ++c)
        {
            const float w0 = bf2f(sl.wt[j][c], 0) + (a.gemma ? 1.f : 0.f);
            const float w1 = bf2f(sl.wt[j][c], 1) + (a.gemma ? 1.f : 0.f);
            o[c]           = f2bf2(x[j][2 * c] * (inv_rms * w0), x[j][2 * c + 1] * (inv_rms * w1));
        }
        if constexpr(!FP8)
        {
            reinterpret_cast<u32x4*>(a.out)[sl.p0 + 32 * j] = o;
        }
        else
        {
            uint32_t lo, hi;
            uint8_t sb;
            mxfp8_quant8(o, lo, hi, sb);
            const long col = (long)r * kHidden + col0 + 256 * j;
            *reinterpret_cast<uint2*>(reinterpret_cast<uint8_t*>(a.out) + col) = make_uint2(lo, hi);
            if((lane & 3) == 0)
                a.scale[(long)r * (kHidden / 32) + (col0 + 256 * j) / 32] = sb;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// the kernel
// ---------------------------------------------------------------------------------------------
// TAIL: the input rows are the deferred MoE tail (a.tail), not a.x; the Lamport-order kernels
// (flushed denormals) run the tail arithmetic with IEEE fp32 denormals, as its launch does
template <bool LAMPORT, bool FP8, bool TAIL = false>
__global__ void __launch_bounds__(kThreads) k5_ar_rms_kernel(Args a)
{
    const int wave = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if((int)blockIdx.x < a.rows)
    {
        send_slab<TAIL, LAMPORT>(a, blockIdx.x, wave, lane);
        return;
    }
    const int r = blockIdx.x - a.rows;
    Slab sl;
    receive_slab<TAIL, LAMPORT>(a, r, wave, lane, sl);
    if constexpr(LAMPORT)
        lamport_epilogue<FP8>(a, sl, r, wave, lane);
    else
        gluon_epilogue<FP8>(a, sl, r, wave, lane);
}

} // namespace k5ar
} // namespace aiter
