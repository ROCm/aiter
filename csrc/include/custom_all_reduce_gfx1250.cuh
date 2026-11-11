#pragma once
/*
 * Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Self-contained gfx1250 (MI450) custom allreduce.
 * Does NOT include aiter_hip_common.h (avoids CK dependency).
 */
#include "opus/opus.hpp"
#include "hip_float8.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Utilities copied from aiter_hip_common.h to stay CK-free
// ---------------------------------------------------------------------------
#ifndef HIP_CALL
#define HIP_CALL(call)                                                       \
    do                                                                       \
    {                                                                        \
        hipError_t err = call;                                               \
        if(err != hipSuccess) [[unlikely]]                                   \
        {                                                                    \
            std::cerr << "[AITER] " << __FILE__ << ":" << __LINE__           \
                      << " fail to call " #call " ---> [HIP error]("        \
                      << hipGetErrorString(err) << ')' << std::endl;         \
            std::abort();                                                    \
        }                                                                    \
    } while(0)
#endif

#ifndef DINLINE
#define DINLINE __device__ __forceinline__
#endif

#define DISPATCH_NGPUS_1250(ws, CALL)   \
    switch(ws) {                        \
        case 2: CALL(2); break;         \
        case 4: CALL(4); break;         \
        case 6: CALL(6); break;         \
        case 8: CALL(8); break;         \
        default:                        \
            throw std::runtime_error(   \
                "unsupported world_size " + std::to_string(ws)); \
    }

// Allgather (and the sync-only kernels) only move data, so they scale past the
// all-reduce limit; multi-node symm_mem groups reach 64 ranks.
#define DISPATCH_AG_NGPUS_1250(ws, CALL) \
    switch(ws) {                         \
        case 2:  CALL(2);  break;        \
        case 4:  CALL(4);  break;        \
        case 6:  CALL(6);  break;        \
        case 8:  CALL(8);  break;        \
        case 16: CALL(16); break;        \
        case 32: CALL(32); break;        \
        case 64: CALL(64); break;        \
        default:                         \
            throw std::runtime_error(    \
                "unsupported world_size " + std::to_string(ws)); \
    }

namespace aiter {

// ---------------------------------------------------------------------------
// Constants & data structures
// ---------------------------------------------------------------------------
constexpr int kMaxBlocks  = 512;
// kMaxNgpusAr matches DISPATCH_NGPUS_1250, kMaxNgpusAg DISPATCH_AG_NGPUS_1250.
// kMaxNgpus sizes Signal/RankData/RankSignals; the last two are passed by value
// on every launch, so widening it costs all world sizes.
constexpr int kMaxNgpusAr = 8;
constexpr int kMaxNgpusAg = 64;
constexpr int kMaxNgpus   = kMaxNgpusAg;

struct Signal
{
    alignas(128) uint32_t start[kMaxBlocks][kMaxNgpus];
    alignas(128) uint32_t end[kMaxBlocks][kMaxNgpus];
    alignas(128) uint32_t _flag[kMaxBlocks];
};

struct __align__(16) RankData
{
    const void* ptrs[kMaxNgpus];
};

struct __align__(16) RankSignals
{
    Signal* signals[kMaxNgpus];
};

// Construction-time counterpart to DISPATCH_AG_NGPUS_1250, so an unsupported
// world size fails at init instead of from a dispatch default arm on the first
// collective. init_custom_ar* are public bindings called directly downstream.
// World sizes above kMaxNgpusAr are allgather-only.
inline void check_ngpus(int world_size)
{
    bool ok = (world_size >= 2 && world_size <= kMaxNgpusAr && world_size % 2 == 0) ||
              world_size == 16 || world_size == 32 || world_size == 64;
    if(!ok)
        throw std::invalid_argument(
            "gfx1250 custom allreduce: unsupported world size " +
            std::to_string(world_size) + " (supported: 2, 4, 6, 8, 16, 32, 64)");
}

// ---------------------------------------------------------------------------
// Scalar cast helpers
// ---------------------------------------------------------------------------
template <typename inp_dtype>
DINLINE opus::fp32_t upcast_s(inp_dtype val)
{ return opus::cast<opus::fp32_t>(val); }

template <>
DINLINE opus::fp32_t upcast_s<opus::fp32_t>(opus::fp32_t val)
{ return val; }

template <typename out_dtype>
DINLINE out_dtype downcast_s(opus::fp32_t val)
{ return opus::cast<out_dtype>(val); }

template <>
DINLINE opus::fp32_t downcast_s<opus::fp32_t>(opus::fp32_t val)
{ return val; }

// ---------------------------------------------------------------------------
// LL (low-latency) small-message all-reduce
// ---------------------------------------------------------------------------
// Ported from the standalone PoC (formerly custom_all_reduce_ll_poc.*), now
// folded into the gfx1250 path. Flag-in-data, zero GPU barrier: each 16B line
// carries 8B payload + two 4B epoch flags. A rank publishes its sendbuff into
// every peer's staging scratch, then polls its own scratch for the peers' lines
// (spinning on the flag) and reduces in fp32. The scratch lives appended to the
// shared Signal meta buffer (offset kLLScratchOffset), so no extra cross-rank
// exchange is needed — peer scratch base = (char*)sg_.signals[i] + off.

constexpr int    kLLMaxRanks       = kMaxNgpusAr;
// Route to LL when bytes <= this (matches RCCL DDA_ALLREDUCE_LL_THRESHOLD).
constexpr size_t kLLArMaxBytes     = 4194304;           // 4 MiB
// Narrower rows do not amortize the extra launches of two-stage fused AR.
constexpr int kArRmsTwoStageMinHidden = 6144;
// Hard per-message payload cap (one slot). Comfortably above the routing
// threshold so all dtypes at <=128 KiB fit.
constexpr size_t kLLScratchCapBytes = 4194304;          // 4 MiB
// Per-rank staging slot capacity, in 8-byte packets.
constexpr size_t kLLPackCapacity   = kLLScratchCapBytes / 8;  // 524288

// 16-byte LL line: two (4B data, 4B flag) pairs carrying 8B of payload.
union LLPackedMsg
{
    struct
    {
        uint32_t data0;
        uint32_t flag0;
        uint32_t data1;
        uint32_t flag1;
    };
    uint4 raw;
};
static_assert(sizeof(LLPackedMsg) == 16, "LLPackedMsg must be exactly 16 bytes");

// Per-rank scratch: 2 banks * ngpus slots * 16B, i.e. 32 MiB at TP=2 and
// 128 MiB at TP=8. Sized by world size rather than kLLMaxRanks because
// ar_ll_gfx1250 indexes with its template ngpus: bankOffPkts peaks at
// (2*ngpus-1)*slot, so 2*ngpus slots is exactly what it touches.
constexpr size_t llScratchBytes(int ngpus)
{
    return (size_t)2 * ngpus * kLLPackCapacity * sizeof(LLPackedMsg);
}

// Byte offset of the LL scratch within the shared meta buffer: right after the
// Signal struct, 128-byte aligned. The meta buffer (see meta_size()) is sized
// kLLScratchOffset + llScratchBytes(world_size) + kArRmsShardBytes +
// ar2sScratchBytes(world_size) and zero-initialized, which doubles as the LL
// flag reset (flag 0 == cleared line).
constexpr size_t kLLScratchOffset =
    ((sizeof(Signal) + 127) / 128) * 128;

// Two-stage fused AR + RMSNorm stages each rank's reduced shard right after the
// LL scratch, so LL never polls a line it did not write. 16 MiB covers the
// communicator's 64 MiB fused limit at TP4.
constexpr size_t kArRmsShardBytes = 16 * 1024 * 1024;
constexpr size_t arRmsShardOffset(int ngpus)
{
    return kLLScratchOffset + llScratchBytes(ngpus);
}

// The largest message the LL two-shot all-reduce (ar_2s_ll_gfx1250) takes at
// each world size, from MI455X measurements, bf16 in a CUDA graph:
// - TP8, two trays: ahead of RCCL at 8.39 MB (39.2 vs 43.7 us); an earlier
//   two-shot was behind it at 14.68 MB (59.1 vs 51.0 us).
// - TP4, one tray: ahead of the one-shot pull up to 8.39 MB (32 vs 89 us).
//   RCCL is ahead from about 5 MiB, so plain all-reduce stops at 4 MiB by
//   default (_GFX1250_PLAIN_AR_MAX_SIZE in Python).
// 0 means the world size does not take the two-shot: TP2, and TP6, which is
// not measured. This cap also sizes the two-shot's scratch (ar2sScratchBytes).
constexpr size_t ar2sMaxBytes(int ngpus)
{
    return ngpus == 4 || ngpus == 8 ? (size_t)8 * 1024 * 1024 : 0;
}

// The smallest message the LL two-shot takes over from the one-shot at, from
// MI455X measurements, bf16:
// - TP8, two trays: the one-shot is ahead at 0.43 MB (12.5 vs 13.6 us), the
//   two-shot at 0.57 MB (12.8 vs 15.9 us).
// - TP4, one tray: the two are level at about 1 MiB.
constexpr size_t ar2sMinBytes(int ngpus) { return ngpus == 8 ? (size_t)512 * 1024 : (size_t)1024 * 1024; }

// The LL two-shot's own scratch, after the AR + RMSNorm shard. It is apart from
// the LL scratch, so a change to either kernel's layout, capacity or epochs
// leaves the other alone. Two banks, each a reduce-scatter region and an
// all-gather region of ar2sMaxBytes / 8 lines: 8 x ar2sMaxBytes a rank, i.e.
// 64 MiB at TP4 and TP8, and nothing where the two-shot does not run.
constexpr size_t ar2sRegionPk(int ngpus) { return ar2sMaxBytes(ngpus) / 8; }
constexpr size_t ar2sScratchBytes(int ngpus)
{
    return (size_t)2 * 2 * ar2sRegionPk(ngpus) * sizeof(LLPackedMsg);
}
constexpr size_t ar2sScratchOffset(int ngpus)
{
    return arRmsShardOffset(ngpus) + kArRmsShardBytes;
}

// Lines a source's slot takes in a region of the two-shot's scratch: a chunk of
// ar2sMaxBytes / ngpus in 8-byte packets, rounded to 4 KiB. A slot that does not
// start on a cache-line boundary costs ~3 us a call (MI455X, measured).
constexpr size_t ar2sSrcLines(int ngpus)
{
    return ngpus == 0 ? 0 : (ar2sMaxBytes(ngpus) / 8 / ngpus + 255) / 256 * 256;
}
static_assert(8 * ar2sSrcLines(8) <= ar2sRegionPk(8) && 4 * ar2sSrcLines(4) <= ar2sRegionPk(4),
              "every source's slot must fit the two-shot's regions");

// gfx1250 128-bit global load/store at *system* scope ("" == system). A single
// 16B transaction keeps a line's data and flags atomic w.r.t. a peer reader.
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1250__) || defined(__gfx1250_strict__)) &&                   \
    __has_builtin(__builtin_amdgcn_global_store_b128) &&                         \
    __has_builtin(__builtin_amdgcn_global_load_b128)
#define AITER_GFX1250_HAVE_B128 1
using llx_v4u      = __attribute__((__vector_size__(4 * sizeof(unsigned int)))) unsigned int;
using llx_v4u_gptr = __attribute__((address_space(1))) llx_v4u*;
#else
#define AITER_GFX1250_HAVE_B128 0
#endif

DINLINE void ll_store_b128(uint32_t* dst, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3)
{
#if AITER_GFX1250_HAVE_B128
    union
    {
        llx_v4u  v;
        uint32_t w[4];
    } u;
    u.w[0] = a0;
    u.w[1] = a1;
    u.w[2] = a2;
    u.w[3] = a3;
    __builtin_amdgcn_global_store_b128((llx_v4u_gptr)dst, u.v, "");
#else
    __builtin_nontemporal_store(a0, dst + 0);
    __builtin_nontemporal_store(a1, dst + 1);
    __builtin_nontemporal_store(a2, dst + 2);
    __builtin_nontemporal_store(a3, dst + 3);
#endif
    asm volatile("" ::: "memory");
}

DINLINE void ll_load_b128(
    const uint32_t* src, uint32_t& o0, uint32_t& o1, uint32_t& o2, uint32_t& o3)
{
    asm volatile("" ::: "memory");
#if AITER_GFX1250_HAVE_B128
    union
    {
        llx_v4u  v;
        uint32_t w[4];
    } u;
    u.v = __builtin_amdgcn_global_load_b128((llx_v4u_gptr)src, "");
    o0  = u.w[0];
    o1  = u.w[1];
    o2  = u.w[2];
    o3  = u.w[3];
#else
    o0 = __builtin_nontemporal_load(src + 0);
    o1 = __builtin_nontemporal_load(src + 1);
    o2 = __builtin_nontemporal_load(src + 2);
    o3 = __builtin_nontemporal_load(src + 3);
#endif
}

// LL all-reduce lines and roles (both kernels below).
//
// A 16-byte line carries 8 bytes of payload in words 0 and 2 and the call's
// epoch in words 1 and 3 (LLPackedMsg). It is written with one ll_store_b128
// and read with one ll_load_b128, and a reader takes it only once both flags
// carry the epoch, so a line has to land atomically only per 8-byte half.
//
// On gfx1250 a load issued after a remote store in the same wave returns only
// once that store has completed, about a fabric round trip, while a peer's line
// lands about half a round trip after it was sent. So the waves that send never
// poll, and a wave that does both polls before it sends: a poll sees a line as
// soon as it lands. The bank bases are read once, before any store, and indexed
// by k = 0..ngpus-1 (rank (rank + k) % ngpus) at compile time, so no table read
// sits between two remote stores (it would wait for the store before it). A
// poller polls every line it still needs in one batch.
//
// Every element is summed in fp32 in rank order 0..ngpus-1 with the local value
// at its own rank and rounded to T once, by both kernels, so the result does
// not depend on which kernel reduced it or on which rank.
//
// Both kernels may be called with recvbuff == sendbuff; each says why below.
//
// Graph-safe device epoch: a persistent device array block_flags[kMaxBlocks]
// that the kernel itself bumps (mirrors Signal::_flag). The grid size and block
// size follow the message size, so the block that owns a packet changes between
// calls of different sizes. Every launch therefore writes its epoch to all
// kMaxBlocks entries, including those of blocks it did not launch, so all
// blocks always start the next call on one shared epoch. Scratch is
// double-buffered (bank = epoch & 1) with a fixed per-rank slot stride; the
// per-epoch flag disambiguates stale lines, so no clearing is needed and it
// survives CUDA-graph replay.

// LL one-shot all-reduce. Block = 2 x W threads: the first W send, the other W
// poll. A sender takes U packets a round: all their loads, then their lines to
// slot[rank] of every rank's scratch, this rank's own first. The poller of
// packet i polls all ngpus slots of this rank's scratch in one batch, the own
// slot included, and sums them. It takes the local value from the own line, not
// from sendbuff: that line exists only once the sender has loaded the packet, so
// with recvbuff == sendbuff no poller can overwrite a packet before it is sent.
template <typename T, int ngpus, int U>
__global__ void __launch_bounds__(1024, 1) ar_ll_gfx1250(
    T* const* __restrict__ peer_scratch, // ngpus scratch bases (device table)
    T* __restrict__ recvbuff,
    const T* __restrict__ sendbuff,
    size_t nPk,          // number of 8-byte packets = bytes / 8
    int rank,
    uint32_t* __restrict__ block_flags) // device array[kMaxBlocks], persisted
{
    constexpr int LP = 8 / sizeof(T); // elements per 8-byte payload
    using PL         = typename opus::vector_t<T, LP>;
    constexpr size_t slot = kLLPackCapacity;

    __shared__ uint32_t s_flag;
    if(threadIdx.x == 0)
    {
        uint32_t f = block_flags[blockIdx.x] + 1u;
        if(f == 0u)
            f = 2u; // flag is never 0 (0 == cleared scratch); keep bank parity
        s_flag = f;
    }
    __syncthreads();
    const uint32_t flag = s_flag;

    LLPackedMsg* base[ngpus]; // base[k]: this call's bank of rank (rank + k) % ngpus
#pragma unroll
    for(int k = 0; k < ngpus; ++k)
        base[k] = reinterpret_cast<LLPackedMsg*>(peer_scratch[(rank + k) % ngpus]) +
                  (size_t)(flag & 1u) * ngpus * slot;

    const size_t W      = blockDim.x / 2;
    const size_t first  = (size_t)blockIdx.x * W + threadIdx.x % W;
    const size_t stride = (size_t)gridDim.x * W;

    if(threadIdx.x < W)
    {
        const uint2* in = reinterpret_cast<const uint2*>(sendbuff);
        for(size_t b = first; b < nPk; b += stride * U)
        {
            uint2 d[U];
#pragma unroll
            for(int u = 0; u < U; ++u)
                if(b + u * stride < nPk)
                    d[u] = in[b + u * stride];
#pragma unroll
            for(int k = 0; k < ngpus; ++k)
#pragma unroll
                for(int u = 0; u < U; ++u)
                    if(b + u * stride < nPk)
                        ll_store_b128(
                            reinterpret_cast<uint32_t*>(base[k] + (size_t)rank * slot + b + u * stride),
                            d[u].x, flag, d[u].y, flag);
        }
    }
    else
    {
        PL* out = reinterpret_cast<PL*>(recvbuff);
        for(size_t i = first; i < nPk; i += stride)
        {
            uint32_t w[ngpus][4];
            uint32_t pending = (1u << ngpus) - 1u;
            while(pending)
            {
#pragma unroll
                for(int r = 0; r < ngpus; ++r)
                    if(pending & (1u << r))
                        ll_load_b128(reinterpret_cast<const uint32_t*>(base[0] + (size_t)r * slot + i),
                                     w[r][0], w[r][1], w[r][2], w[r][3]);
#pragma unroll
                for(int r = 0; r < ngpus; ++r)
                    if((pending & (1u << r)) && w[r][1] == flag && w[r][3] == flag)
                        pending &= ~(1u << r);
            }
            opus::fp32_t acc[LP];
#pragma unroll
            for(int r = 0; r < ngpus; ++r)
            {
                const uint32_t pw[2] = {w[r][0], w[r][2]};
                const PL pv          = *reinterpret_cast<const PL*>(pw);
#pragma unroll
                for(int q = 0; q < LP; ++q)
                    acc[q] = r == 0 ? upcast_s(pv[q]) : acc[q] + upcast_s(pv[q]);
            }
            PL ov;
#pragma unroll
            for(int q = 0; q < LP; ++q)
                ov[q] = downcast_s<T>(acc[q]);
            out[i] = ov;
        }
    }

    for(int e = blockIdx.x + threadIdx.x * gridDim.x; e < kMaxBlocks;
        e += gridDim.x * blockDim.x)
        block_flags[e] = flag;
}

// ---------------------------------------------------------------------------
// LL two-shot all-reduce: reduce-scatter + all-gather over LL lines
// ---------------------------------------------------------------------------
// The message splits into ngpus chunks of chunk_pk 8-byte packets; rank c owns
// chunk c. The lines live in a scratch of its own (ar2sScratchOffset) with
// epochs of its own (block_flags), double-banked by epoch parity. In a bank,
// the reduce-scatter region holds rank s's lines in slot s, s * ar2sSrcLines
// lines on, and the all-gather region, ar2sRegionPk lines on, the same. Every
// thread takes packet j of every chunk, in one of ROLES wave groups a block:
//   send:   send packet j of every other chunk c to rank c's reduce-scatter
//           slot[rank]
//   reduce: poll the ngpus - 1 contributions to packet j of the own chunk,
//           reduce them with the own one in rank order, send the result to
//           every peer's all-gather slot[rank] and write it to recvbuff (its
//           polls come before its stores)
//   gather: poll the other ranks' reduced packet j into recvbuff
// ROLES = 3 runs them in three groups. ROLES = 2 runs gather in the send
// group, after its stores: a reduced packet lands only after the stores it
// depends on, about a round trip after they were sent, so waiting for them
// costs about nothing, and a block of 2 x W threads covers more packets a CU
// (allreduce_2s_ll picks ROLES by size).
// Every rank moves 2 * (ngpus - 1) / ngpus * bytes of payload (the lines double
// it) against (ngpus - 1) * bytes for the one-shot, for one more fabric
// latency: it wins from ar2sMinBytes.
// With recvbuff == sendbuff: reduce reads and writes only packet j of the own
// chunk, in that order. gather writes packet j of chunk c only once rank c has
// sent its reduction, which needs this rank's contribution, which send sent
// after it loaded the packet; send reads no other packet of recvbuff.
template <typename T, int ngpus, int ROLES>
__global__ void __launch_bounds__(1024) ar_2s_ll_gfx1250(
    T* const* __restrict__ peer_scratch, // ngpus two-shot scratch bases (device table)
    T* __restrict__ recvbuff,
    const T* __restrict__ sendbuff,
    size_t chunk_pk,     // 8-byte packets per chunk = bytes / (8 * ngpus)
    int rank,
    uint32_t* __restrict__ block_flags) // device array[kMaxBlocks], persisted
{
    static_assert(ROLES == 2 || ROLES == 3, "two or three wave groups");
    constexpr int LP = 8 / sizeof(T); // elements per 8-byte packet
    using PL         = typename opus::vector_t<T, LP>;
    constexpr size_t region = ar2sRegionPk(ngpus);
    constexpr size_t src    = ar2sSrcLines(ngpus);

    __shared__ uint32_t s_flag;
    if(threadIdx.x == 0)
    {
        uint32_t f = block_flags[blockIdx.x] + 1u;
        if(f == 0u)
            f = 2u; // flag is never 0 (0 == cleared scratch); keep bank parity
        s_flag = f;
    }
    __syncthreads();
    const uint32_t flag = s_flag;

    LLPackedMsg* rs[ngpus]; // rs[k]: this call's reduce-scatter region of rank (rank + k) % ngpus
#pragma unroll
    for(int k = 0; k < ngpus; ++k)
        rs[k] = reinterpret_cast<LLPackedMsg*>(peer_scratch[(rank + k) % ngpus]) +
                (size_t)(flag & 1u) * 2 * region;

    const size_t W      = blockDim.x / ROLES;
    const size_t role   = threadIdx.x / W; // 0 send (+ gather at ROLES 2), 1 reduce, 2 gather
    const size_t first  = (size_t)blockIdx.x * W + threadIdx.x % W;
    const size_t stride = (size_t)gridDim.x * W;
    const uint2* in     = reinterpret_cast<const uint2*>(sendbuff);
    uint2* out          = reinterpret_cast<uint2*>(recvbuff);

    for(size_t j = first; j < chunk_pk; j += stride)
    {
        if(role == 0)
        {
            uint2 d[ngpus];
#pragma unroll
            for(int k = 1; k < ngpus; ++k)
                d[k] = in[(size_t)((rank + k) % ngpus) * chunk_pk + j];
#pragma unroll
            for(int k = 1; k < ngpus; ++k)
                ll_store_b128(reinterpret_cast<uint32_t*>(rs[k] + (size_t)rank * src + j),
                              d[k].x, flag, d[k].y, flag);
        }
        if(role == 1)
        {
            const uint2 mine = in[(size_t)rank * chunk_pk + j];
            uint32_t w[ngpus][4];
            uint32_t pending = ((1u << ngpus) - 1u) & ~(1u << rank);
            while(pending)
            {
#pragma unroll
                for(int s = 0; s < ngpus; ++s)
                    if(pending & (1u << s))
                        ll_load_b128(reinterpret_cast<const uint32_t*>(rs[0] + (size_t)s * src + j),
                                     w[s][0], w[s][1], w[s][2], w[s][3]);
#pragma unroll
                for(int s = 0; s < ngpus; ++s)
                    if((pending & (1u << s)) && w[s][1] == flag && w[s][3] == flag)
                        pending &= ~(1u << s);
            }
            opus::fp32_t acc[LP];
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
            {
                const uint32_t pw[2] = {s == rank ? mine.x : w[s][0], s == rank ? mine.y : w[s][2]};
                const PL pv          = *reinterpret_cast<const PL*>(pw);
#pragma unroll
                for(int q = 0; q < LP; ++q)
                    acc[q] = s == 0 ? upcast_s(pv[q]) : acc[q] + upcast_s(pv[q]);
            }
            PL ov;
#pragma unroll
            for(int q = 0; q < LP; ++q)
                ov[q] = downcast_s<T>(acc[q]);
            const uint2 r = *reinterpret_cast<const uint2*>(&ov);
#pragma unroll
            for(int k = 1; k < ngpus; ++k)
                ll_store_b128(reinterpret_cast<uint32_t*>(rs[k] + region + (size_t)rank * src + j),
                              r.x, flag, r.y, flag);
            out[(size_t)rank * chunk_pk + j] = r;
        }
        if(role == (ROLES == 3 ? 2 : 0))
        {
            uint32_t g[ngpus][4];
            uint32_t pending = ((1u << ngpus) - 1u) & ~(1u << rank);
            while(pending)
            {
#pragma unroll
                for(int s = 0; s < ngpus; ++s)
                    if(pending & (1u << s))
                        ll_load_b128(reinterpret_cast<const uint32_t*>(rs[0] + region + (size_t)s * src + j),
                                     g[s][0], g[s][1], g[s][2], g[s][3]);
#pragma unroll
                for(int s = 0; s < ngpus; ++s)
                    if((pending & (1u << s)) && g[s][1] == flag && g[s][3] == flag)
                        pending &= ~(1u << s);
            }
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
                if(s != rank)
                    out[(size_t)s * chunk_pk + j] = make_uint2(g[s][0], g[s][2]);
        }
    }

    for(int e = blockIdx.x + threadIdx.x * gridDim.x; e < kMaxBlocks;
        e += gridDim.x * blockDim.x)
        block_flags[e] = flag;
}

// ---------------------------------------------------------------------------
// Synchronisation primitives (ROCm path only)
// ---------------------------------------------------------------------------
template <int ngpus>
DINLINE void start_sync(const RankSignals& sg, Signal* self_sg, int rank)
{
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus)
    {
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->start[blockIdx.x][rank],
                                flag,
                                __ATOMIC_RELAXED,
                                __MEMORY_SCOPE_SYSTEM);
        while(__scoped_atomic_load_n(&self_sg->start[blockIdx.x][threadIdx.x],
                                     __ATOMIC_RELAXED,
                                     __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    if(threadIdx.x == 0)
        self_sg->_flag[blockIdx.x] = flag;
}

template <int ngpus, bool final_sync = false>
DINLINE void end_sync(const RankSignals& sg, Signal* self_sg, int rank)
{
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus)
    {
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                                flag,
                                final_sync ? __ATOMIC_RELAXED : __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
        while(__scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                                     final_sync ? __ATOMIC_RELAXED : __ATOMIC_ACQUIRE,
                                     __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    if(threadIdx.x == 0)
        self_sg->_flag[blockIdx.x] = flag;
}

// ---------------------------------------------------------------------------
// gfx1250 allreduce kernel
// ---------------------------------------------------------------------------
template <typename T, int ngpus, bool is_broadcast_reg_outptr = false>
__global__ void __launch_bounds__(256, 2) ar_gfx1250_naive_unroll4(
    RankData* _input_dp,
    RankData* _output_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int unroll    = 4;
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int nthds  = blockDim.x * gridDim.x;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        // Sum in rank order so every rank gets the same bits.
        ptrs[i] = (const P*)_input_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    int aligned_size = (size / unroll) * unroll;
    for(int base = tid * unroll; base < aligned_size; base += nthds * unroll)
    {
      P inp_reg[ngpus][unroll];
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
      {
#pragma unroll
        for (int j = 0; j < unroll; ++j)
          inp_reg[i][j] = ptrs[i][base + j];
      }
      A rslt_tmp[unroll];
      P rslt_reg[unroll];
#pragma unroll
      for (int u = 0; u < unroll; ++u)
      {
#pragma unroll
        for (int j = 0; j < pack_size; ++j)
          rslt_tmp[u][j] = upcast_s(inp_reg[0][u][j]);
#pragma unroll
        for (int g = 1; g < ngpus; ++g)
        {
#pragma unroll
          for (int j = 0; j < pack_size; ++j)
            rslt_tmp[u][j] += upcast_s(inp_reg[g][u][j]);
        }
      }
#pragma unroll
      for (int u = 0; u < unroll; ++u)
      {
#pragma unroll
        for (int j = 0; j < pack_size; ++j)
          rslt_reg[u][j] = downcast_s<T>(rslt_tmp[u][j]);
        *(reinterpret_cast<P*>(result) + base + u) = rslt_reg[u];
      }
    }
    for(int idx = aligned_size + tid; idx < size; idx += nthds)
    {
      A acc;
#pragma unroll
      for (int j = 0; j < pack_size; ++j)
        acc[j] = upcast_s(ptrs[0][idx][j]);
#pragma unroll
      for (int i = 1; i < ngpus; ++i)
      {
#pragma unroll
        for (int j = 0; j < pack_size; ++j)
          acc[j] += upcast_s(ptrs[i][idx][j]);
      }
      P out_val;
#pragma unroll
      for (int j = 0; j < pack_size; ++j)
        out_val[j] = downcast_s<T>(acc[j]);
      *(reinterpret_cast<P*>(result) + idx) = out_val;
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// gfx1250 fused allreduce + residual add + RMSNorm (+ per-token FP8 quant)
// ---------------------------------------------------------------------------
// Stage 1 of the large-message path. Each rank owns one contiguous hidden
// shard, reduces that shard in fixed rank order, and writes it to its
// peer-visible staging region. Stage 2 reads the reduced shards below.
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) ar_rmsnorm_reduce_scatter_gfx1250(
    RankData* _input_dp,
    Signal* self_sg,
    int rank,
    int tokens,
    int packs)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    const int shard_packs   = packs / ngpus;
    const int shard_size    = tokens * shard_packs;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
        ptrs[i] = reinterpret_cast<const P*>(_input_dp->ptrs[i]);
    P* scratch =
        reinterpret_cast<P*>(reinterpret_cast<char*>(self_sg) + arRmsShardOffset(ngpus));

    const int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for(int out_idx = tid; out_idx < shard_size; out_idx += stride)
    {
        const int token = out_idx / shard_packs;
        const int col   = out_idx - token * shard_packs;
        const int idx   = token * packs + rank * shard_packs + col;
        A acc;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            acc[j] = upcast_s(ptrs[0][idx][j]);
#pragma unroll
        for(int i = 1; i < ngpus; ++i)
        {
#pragma unroll
            for(int j = 0; j < pack_size; ++j)
                acc[j] += upcast_s(ptrs[i][idx][j]);
        }
        P reduced;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            reduced[j] = downcast_s<T>(acc[j]);
        scratch[out_idx] = reduced;
    }
}

// Sum or max over the block; every thread gets the result. Threads past the
// row contribute 0, so the block size may exceed the row as long as it is a
// whole number of waves.
template <bool kMax, bool parallel_waves = false>
DINLINE float block_allreduce_f32(float v)
{
    __shared__ float partial[32];
    __shared__ float result;
    for(int off = warpSize / 2; off > 0; off >>= 1)
    {
        const float o = __shfl_xor(v, off);
        v             = kMax ? fmaxf(v, o) : v + o;
    }
    if(threadIdx.x % warpSize == 0)
        partial[threadIdx.x / warpSize] = v;
    __syncthreads();
    // The two-stage FP8 epilogue reduces nonnegative amax values in parallel.
    // Keep the sum's original order so normalization and split quantization
    // retain their rounding. At most 32 wave32 partials fit in the first wave.
    if constexpr(parallel_waves)
    {
        if(threadIdx.x < warpSize)
        {
            float r = threadIdx.x < blockDim.x / warpSize ? partial[threadIdx.x] : 0.f;
            for(int off = warpSize / 2; off > 0; off >>= 1)
            {
                const float o = __shfl_xor(r, off);
                r             = kMax ? fmaxf(r, o) : r + o;
            }
            if(threadIdx.x == 0)
                result = r;
        }
    }
    else if(threadIdx.x == 0)
    {
        float r = partial[0];
        for(int w = 1; w < (int)(blockDim.x / warpSize); ++w)
            r = kMax ? fmaxf(r, partial[w]) : r + partial[w];
        result = r;
    }
    __syncthreads();
    return result;
}

// One block per token, one thread per 16-byte pack of the row. Matches the
// unfused path: the all-reduce result is rounded to T before the residual add,
// and ranks are summed in rank order so every rank gets the same bits.
template <typename T, typename OutT, int ngpus, bool GEMMA_NORM, bool TWO_STAGE = false>
__global__ void __launch_bounds__(1024, 1) ar_rmsnorm_gfx1250(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    int rank,
    const T* __restrict__ residual_inp,
    T* __restrict__ residual_out,
    OutT* __restrict__ output,
    const T* __restrict__ weight,
    float* __restrict__ scale_out,
    T* __restrict__ bf16_output,
    int tokens,
    int hidden,
    float eps)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    const int packs         = hidden / pack_size;
    const bool active       = (int)threadIdx.x < packs;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        if constexpr(TWO_STAGE)
            ptrs[i] = reinterpret_cast<const P*>(
                reinterpret_cast<const char*>(sg.signals[i]) + arRmsShardOffset(ngpus));
        else
            ptrs[i] = reinterpret_cast<const P*>(_input_dp->ptrs[i]);
    }
    P w{};
    if(active)
        w = reinterpret_cast<const P*>(weight)[threadIdx.x];

    if constexpr(!TWO_STAGE)
        start_sync<ngpus>(sg, self_sg, rank);
    for(int t = blockIdx.x; t < tokens; t += gridDim.x)
    {
        const int idx = t * packs + threadIdx.x;
        float x[pack_size] = {};
        float square_sum   = 0.f;
        if(active)
        {
            if constexpr(TWO_STAGE)
            {
                const int shard_packs = packs / ngpus;
                const int owner       = threadIdx.x / shard_packs;
                const int col         = threadIdx.x - owner * shard_packs;
                const P   v           = ptrs[owner][t * shard_packs + col];
#pragma unroll
                for(int j = 0; j < pack_size; ++j)
                    x[j] = upcast_s(v[j]);
            }
            else
            {
#pragma unroll
                for(int i = 0; i < ngpus; i++)
                {
                    const P v = ptrs[i][idx];
#pragma unroll
                    for(int j = 0; j < pack_size; j++)
                        x[j] += upcast_s(v[j]);
                }
            }
            const P res = reinterpret_cast<const P*>(residual_inp)[idx];
            P sum;
#pragma unroll
            for(int j = 0; j < pack_size; j++)
            {
                x[j]   = upcast_s(downcast_s<T>(x[j])) + upcast_s(res[j]);
                sum[j] = downcast_s<T>(x[j]);
                square_sum += x[j] * x[j];
            }
            reinterpret_cast<P*>(residual_out)[idx] = sum;
        }
        const float inv_rms = rsqrtf(block_allreduce_f32<false>(square_sum) / hidden + eps);
#pragma unroll
        for(int j = 0; j < pack_size; j++)
        {
            const float wj = upcast_s(w[j]) + (GEMMA_NORM ? 1.f : 0.f);
            x[j] *= inv_rms * wj;
        }
        if constexpr(std::is_same_v<T, OutT>)
        {
            if(active)
            {
                P o;
#pragma unroll
                for(int j = 0; j < pack_size; j++)
                    o[j] = downcast_s<T>(x[j]);
                reinterpret_cast<P*>(output)[idx] = o;
            }
        }
        else
        {
            // Quantize the T-rounded output with the steps of the unfused
            // dynamic_per_token_scaled_quant on gfx1250 (scale = amax / max,
            // multiply by rcp(scale), clamp, cvt_pk), so both FP8 paths match
            // bit for bit.
            P o;
            float amax = 0.f;
#pragma unroll
            for(int j = 0; j < pack_size; j++)
            {
                o[j] = downcast_s<T>(x[j]);
                amax = fmaxf(amax, fabsf(upcast_s(o[j])));
            }
            if(active && bf16_output != nullptr)
                reinterpret_cast<P*>(bf16_output)[idx] = o;
            const float fp8_max = opus::cast<opus::fp32_t>(opus::numeric_limits<OutT>::max());
            const float scale   = block_allreduce_f32<true, TWO_STAGE>(amax) * (float)(1. / fp8_max);
            const float inv     = __builtin_amdgcn_rcpf(scale);
            if(active)
            {
                opus::vector_t<OutT, pack_size> q;
#pragma unroll
                for(int j = 0; j < pack_size; j += 2)
                {
                    const float a = __builtin_amdgcn_fmed3f(upcast_s(o[j]) * inv, -fp8_max, fp8_max);
                    const float b = __builtin_amdgcn_fmed3f(upcast_s(o[j + 1]) * inv, -fp8_max, fp8_max);
                    reinterpret_cast<uint16_t*>(&q)[j / 2] =
                        __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
                }
                reinterpret_cast<opus::vector_t<OutT, pack_size>*>(output)[idx] = q;
            }
            if(threadIdx.x == 0)
                scale_out[t] = scale;
        }
    }
    if constexpr(!TWO_STAGE)
        end_sync<ngpus, true>(sg, self_sg, rank);
}

// LL all-gather. Every rank pushes its input into each peer's LL scratch as
// 16-byte lines, then polls its own scratch for the peers' lines and writes
// each to its place in the output. A line is one ll_store_b128 and is read
// with one ll_load_b128, so a single flag guards all of it: words 0-2 carry 12
// bytes of payload and word 3 the epoch, where ar_ll_gfx1250's lines carry 8
// bytes and two flags. A rank sends 1.33 x (ngpus - 1) x its input instead of
// 2 x. It shares the other LL kernels' epoch, banks and slot stride and
// interleaves with them: a line another kernel wrote has an older epoch in
// word 3.
//
// Word w of rank r, in row y = w / row32, lands at output word
// (y * ngpus + r) * row32 + w % row32: row32 = n32 gathers along dim 0, the
// last dim's words along the last dim. A thread takes U units of 12 bytes a
// round: all their loads, then their (ngpus - 1) x U remote stores back to
// back, then its own copies (on gfx1250 a load issued after a remote store
// waits for that store's round trip), and its units' polls go out in one
// batch. The launcher picks U so that one round covers the message. ALIGNED:
// input and output 4-byte aligned; otherwise their words move byte by byte
// (same lines).
template <int ngpus, int U, bool ALIGNED>
__global__ void __launch_bounds__(512, 1) ag_ll_gfx1250(
    uint8_t* const* __restrict__ peer_scratch,
    uint8_t* __restrict__ result,
    const uint8_t* __restrict__ sendbuff,
    uint32_t n32,   // 4-byte words a rank sends
    uint32_t row32, // 4-byte words of a gathered row (n32 along dim 0)
    int rank,
    uint32_t* __restrict__ block_flags)
{
    constexpr size_t slot = kLLPackCapacity;

    __shared__ uint32_t s_flag;
    if(threadIdx.x == 0)
    {
        uint32_t f = block_flags[blockIdx.x] + 1u;
        if(f == 0u)
            f = 2u;
        s_flag = f;
    }
    __syncthreads();
    const uint32_t flag        = s_flag;
    const size_t   bankOffPkts = (size_t)(flag & 1u) * (size_t)ngpus * slot;

    const uint32_t nU     = (n32 + 2) / 3; // 12-byte units; the last may carry 4 or 8 bytes
    const uint32_t gtid   = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = gridDim.x * blockDim.x;
    const auto load = [&](uint32_t w) {
        uint32_t d = 0u;
        if(w < n32)
        {
            if constexpr(ALIGNED)
                d = reinterpret_cast<const uint32_t*>(sendbuff)[w];
            else
                __builtin_memcpy(&d, sendbuff + (size_t)w * 4, 4);
        }
        return d;
    };
    // unit i of rank r: its words 3i..3i+2 go to row w / row32 of rank r in
    // the gathered tensor, which holds every rank's rows in rank order
    const auto store_unit = [&](int r, uint32_t i, uint32_t a0, uint32_t a1, uint32_t a2) {
        const uint32_t d[3] = {a0, a1, a2};
        const uint32_t w0   = 3 * i;
        uint32_t y          = w0 / row32;
        uint32_t c          = w0 - y * row32;
#pragma unroll
        for(int q = 0; q < 3; ++q)
        {
            if(w0 + q < n32)
            {
                const size_t at = ((size_t)y * ngpus + r) * row32 + c;
                if constexpr(ALIGNED)
                    reinterpret_cast<uint32_t*>(result)[at] = d[q];
                else
                    __builtin_memcpy(result + at * 4, &d[q], 4);
            }
            if(++c == row32)
            {
                c = 0;
                ++y;
            }
        }
    };

    for(uint32_t base = gtid; base < nU; base += stride * U)
    {
        uint32_t d[U][3];
#pragma unroll
        for(int u = 0; u < U; ++u)
        {
            const uint32_t i = base + u * stride;
#pragma unroll
            for(int q = 0; q < 3; ++q)
                d[u][q] = i < nU ? load(3 * i + q) : 0u;
        }
#pragma unroll
        for(int k = 1; k < ngpus; ++k)
        {
            LLPackedMsg* dst = reinterpret_cast<LLPackedMsg*>(peer_scratch[(rank + k) % ngpus]) +
                               bankOffPkts + (size_t)rank * slot;
#pragma unroll
            for(int u = 0; u < U; ++u)
            {
                const uint32_t i = base + u * stride;
                if(i < nU)
                    ll_store_b128(reinterpret_cast<uint32_t*>(&dst[i]), d[u][0], d[u][1], d[u][2], flag);
            }
        }
#pragma unroll
        for(int u = 0; u < U; ++u)
        {
            const uint32_t i = base + u * stride;
            if(i < nU)
                store_unit(rank, i, d[u][0], d[u][1], d[u][2]);
        }
    }

    const LLPackedMsg* myBase =
        reinterpret_cast<const LLPackedMsg*>(peer_scratch[rank]) + bankOffPkts;
    for(uint32_t base = gtid; base < nU; base += stride * U)
    {
        // bit u * ngpus + k: unit u's line from rank (rank + k) % ngpus
        uint32_t pending = 0;
#pragma unroll
        for(int u = 0; u < U; ++u)
            if(base + u * stride < nU)
                pending |= (((1u << ngpus) - 1u) & ~1u) << (u * ngpus);
        while(pending)
        {
            uint32_t line[U][ngpus][4];
#pragma unroll
            for(int u = 0; u < U; ++u)
#pragma unroll
                for(int k = 1; k < ngpus; ++k)
                    if(pending & (1u << (u * ngpus + k)))
                        ll_load_b128(reinterpret_cast<const uint32_t*>(
                                         &myBase[(size_t)((rank + k) % ngpus) * slot + base + u * stride]),
                                     line[u][k][0], line[u][k][1], line[u][k][2], line[u][k][3]);
#pragma unroll
            for(int u = 0; u < U; ++u)
#pragma unroll
                for(int k = 1; k < ngpus; ++k)
                    if((pending & (1u << (u * ngpus + k))) && line[u][k][3] == flag)
                    {
                        store_unit((rank + k) % ngpus, base + u * stride, line[u][k][0], line[u][k][1],
                                   line[u][k][2]);
                        pending &= ~(1u << (u * ngpus + k));
                    }
        }
    }

    for(int e = blockIdx.x + threadIdx.x * gridDim.x; e < kMaxBlocks;
        e += gridDim.x * blockDim.x)
        block_flags[e] = flag;
}

// ---------------------------------------------------------------------------
// DCP merge: the all-to-all and LSE combine of decode context parallelism
// ---------------------------------------------------------------------------
// Under DCP each of the ngpus ranks attends all H = ngpus * HL query heads over
// its own shard of the KV cache. It holds a partial output o [B, H, 128] bf16
// and its log-sum-exp lse [B, H] fp32. Rank r owns heads r * HL .. r * HL +
// HL - 1 and needs their partials from every rank, merged by their LSE. This
// kernel does the all-to-all and the merge in one launch over the LL scratch,
// in place of a pack kernel, an all-to-all with its copy-in and a combine
// kernel.
//
// A (b, h) row of 128 bf16 travels as 32 LL lines of 8 bytes, [d0, flag, d1,
// flag], plus one line [lse, flag, 0, flag]. Rank s writes its rows for rank r
// into slot s of rank r's LL scratch: data line (b * HL + hl) * 32 + j, LSE
// line B * HL * 32 + b * HL + hl. It shares the LL kernels' epochs, banks and
// slot stride and interleaves with them: a line another call wrote has an
// older epoch.
//
// A block is 2 x W threads, in waves of 32:
// - The first W send. A wave takes U rows (b, peer, hl) a round. It loads them
//   all (a lane 8 bytes of the row, plus the row's LSE and the peer's scratch
//   base), then issues their stores. On gfx1250 a load issued after a remote
//   store waits for that store, about a fabric round trip.
// - The other W merge. A wave takes one output row (b, hl) at a time. Lane j
//   polls line j and the LSE line of every peer in one batch, reads the own
//   shard's row from o, and combines. These waves issue no remote store.
//
// The combine matches ATOM's Triton combine. A non-finite LSE counts as -inf (a
// shard with no KV for the row). m = max LSE, or 0 if all are -inf, and g =
// log(sum exp(lse - m)) + m. out = sum over shards s = 0..ngpus-1 of
// exp(lse_s - g) * o_s in fp32, rounded to bf16 once. A shard whose weight is 0
// is skipped, since its o may be NaN. owned (optional, [B] int32): a row this
// rank owns no KV slot of sends -inf.
template <int ngpus, int U>
__global__ void __launch_bounds__(1024, 1) dcp_merge_ll_gfx1250(
    uint8_t* const* __restrict__ peer_scratch,
    const opus::bf16_t* __restrict__ o,
    const float* __restrict__ lse,
    const int* __restrict__ owned,
    opus::bf16_t* __restrict__ out,
    float* __restrict__ out_lse,
    int B,
    int HL,
    int64_t o_sb,
    int64_t o_sh,
    int64_t l_sb,
    int64_t l_sh,
    int rank,
    uint32_t* __restrict__ block_flags)
{
    using PL              = typename opus::vector_t<opus::bf16_t, 4>;
    constexpr int    P    = 32; // lines a row: 128 bf16, 4 a line
    constexpr size_t slot = kLLPackCapacity;

    __shared__ uint32_t s_flag;
    if(threadIdx.x == 0)
    {
        uint32_t f = block_flags[blockIdx.x] + 1u;
        if(f == 0u)
            f = 2u; // flag is never 0 (0 == cleared scratch); keep bank parity
        s_flag = f;
    }
    __syncthreads();
    const uint32_t flag    = s_flag;
    const size_t   bankOff = (size_t)(flag & 1u) * ngpus * slot;
    const size_t   lseOff  = (size_t)B * HL * P;
    const int      W       = blockDim.x / 2;
    const int      lane    = threadIdx.x % 32;
    const int      wpb     = W / 32;
    const int      wave    = blockIdx.x * wpb + (threadIdx.x % W) / 32;
    const int      nwaves  = gridDim.x * wpb;

    auto row_lse = [&](int b, int h) {
        float v = lse[(int64_t)b * l_sb + (int64_t)h * l_sh];
        if(owned != nullptr && owned[b] == 0)
            v = -INFINITY;
        return v;
    };

    if(threadIdx.x < W)
    {
        const int rows = B * (ngpus - 1) * HL;
        for(int r0 = wave; r0 < rows; r0 += nwaves * U)
        {
            uint2        d[U];
            float        lv[U];
            LLPackedMsg* dst[U];
            int          row[U];
#pragma unroll
            for(int u = 0; u < U; ++u)
            {
                const int r = r0 + u * nwaves;
                if(r < rows)
                {
                    const int hl   = r % HL;
                    const int k    = 1 + (r / HL) % (ngpus - 1);
                    const int b    = r / (HL * (ngpus - 1));
                    const int peer = (rank + k) % ngpus;
                    const int h    = peer * HL + hl;
                    d[u]   = *reinterpret_cast<const uint2*>(o + (int64_t)b * o_sb + (int64_t)h * o_sh + 4 * lane);
                    lv[u]  = row_lse(b, h);
                    dst[u] = reinterpret_cast<LLPackedMsg*>(peer_scratch[peer]) + bankOff + (size_t)rank * slot;
                    row[u] = b * HL + hl;
                }
            }
#pragma unroll
            for(int u = 0; u < U; ++u)
            {
                if(r0 + u * nwaves < rows)
                {
                    ll_store_b128(reinterpret_cast<uint32_t*>(&dst[u][(size_t)row[u] * P + lane]),
                                  d[u].x, flag, d[u].y, flag);
                    if(lane == 0)
                        ll_store_b128(reinterpret_cast<uint32_t*>(&dst[u][lseOff + row[u]]),
                                      __float_as_uint(lv[u]), flag, 0u, flag);
                }
            }
        }
    }
    else
    {
        const LLPackedMsg* own = reinterpret_cast<const LLPackedMsg*>(peer_scratch[rank]) + bankOff;
        const int          rows = B * HL;
        for(int q = wave; q < rows; q += nwaves)
        {
            const int hl = q % HL;
            const int b  = q / HL;
            const int h  = rank * HL + hl;
            // the own shard's row, loaded before the polls so its latency overlaps them
            const uint2 mine = *reinterpret_cast<const uint2*>(o + (int64_t)b * o_sb + (int64_t)h * o_sh + 4 * lane);
            const float mlse = row_lse(b, h);
            uint32_t    w[ngpus][4];
            uint32_t    l[ngpus][4];
            uint32_t    pending = ((1u << ngpus) - 1u) & ~(1u << rank);
            while(pending)
            {
#pragma unroll
                for(int s = 0; s < ngpus; ++s)
                    if(pending & (1u << s))
                    {
                        const LLPackedMsg* src = own + (size_t)s * slot;
                        ll_load_b128(reinterpret_cast<const uint32_t*>(&src[(size_t)q * P + lane]),
                                     w[s][0], w[s][1], w[s][2], w[s][3]);
                        ll_load_b128(reinterpret_cast<const uint32_t*>(&src[lseOff + q]),
                                     l[s][0], l[s][1], l[s][2], l[s][3]);
                    }
#pragma unroll
                for(int s = 0; s < ngpus; ++s)
                    if((pending & (1u << s)) && w[s][1] == flag && w[s][3] == flag && l[s][1] == flag &&
                       l[s][3] == flag)
                        pending &= ~(1u << s);
            }
            float ls[ngpus];
            float m = -INFINITY;
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
            {
                float v = s == rank ? mlse : __uint_as_float(l[s][0]);
                if(v != v || v == INFINITY)
                    v = -INFINITY;
                ls[s] = v;
                m     = fmaxf(m, v);
            }
            if(m == -INFINITY)
                m = 0.f;
            float sum = 0.f;
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
                sum += expf(ls[s] - m);
            const float g = logf(sum) + m;
            float       acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for(int s = 0; s < ngpus; ++s)
            {
                float f = expf(ls[s] - g);
                if(f != f)
                    f = 0.f;
                if(f != 0.f)
                {
                    const uint32_t pw[2] = {s == rank ? mine.x : w[s][0], s == rank ? mine.y : w[s][2]};
                    const PL       pv    = *reinterpret_cast<const PL*>(pw);
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        acc[j] += upcast_s(pv[j]) * f;
                }
            }
            PL ov;
#pragma unroll
            for(int j = 0; j < 4; ++j)
                ov[j] = downcast_s<opus::bf16_t>(acc[j]);
            *reinterpret_cast<PL*>(out + (size_t)q * 128 + 4 * lane) = ov;
            if(out_lse != nullptr && lane == 0)
                out_lse[q] = g;
        }
    }

    for(int e = blockIdx.x + threadIdx.x * gridDim.x; e < kMaxBlocks;
        e += gridDim.x * blockDim.x)
        block_flags[e] = flag;
}

// ---------------------------------------------------------------------------
// Lamport one-shot fused allreduce + residual add + RMSNorm (2-byte dtypes)
// ---------------------------------------------------------------------------
/*
 * No barrier and no copy-in: every thread owns one 16B pack, pushes it into
 * slot[rank] of each peer's inbox, then polls its own inbox until no half-word
 * is the 0xffff sentinel (a bf16/fp16 NaN; the sender rewrites its own 0xffff
 * to 0x7fff). The epilogue then follows ar_rmsnorm_gfx1250 step by step
 * (rank-order fp32 sum, round to T, add the residual, 32-lane butterfly then a
 * serial sum over the waves of the row), so out and residual_out match it bit
 * for bit.
 *
 * Inbox (one IPC buffer per rank, memset to 0xff at init):
 *   inbox[kLamportBuffers][ngpus][slot] of 16B, slot = the most packs a call takes
 * Each wave of 32 packs has its own round counter c (rank-local), which picks
 * the buffer. Every rank runs the same calls, so the counter of a pack is the
 * same on every rank whatever the row width and block shape of each call, and
 * a pack that sits out a call keeps its counter and its inbox. A peer is at
 * most one round ahead on a pack, so buffer (c+2)%3 is re-armed here and the
 * kernel boundary orders that before my next round's sends.
 *
 * The grid covers whole rows. The sum of squares of a row is reduced
 *   - row:     one block a row (blockDim = hidden / 8), on chip.
 *   - cluster: the blocks of a row are one workgroup cluster (2 to 16 blocks);
 *              the wave partials go through rank-local memory, ordered by the
 *              cluster barrier. Spreads a row over more CUs while the rows do
 *              not fill the device.
 */
constexpr int kLamportBuffers           = 3;
constexpr int kLamportWave              = 32;   // = gfx1250 warpSize
constexpr int kLamportRowMaxThreads     = 1024; // a row is one block: hidden <= 8192
constexpr int kLamportClusterMaxThreads = 256;
constexpr int kLamportMaxClusterBlocks  = 16;
// Rows of at most this many packs stay in one block: splitting them measured
// no faster at any row count (hidden 2048, TP2/TP4).
constexpr int kLamportMinClusterRowPacks = 256;

// The cluster path needs a toolchain that knows workgroup clusters: the device
// side the gfx1250 cluster barrier and cluster memory scope, the host side the
// cluster launch attribute (neither is in ROCm 7.2). This module is also built
// for other archs and older ROCm, where the cluster kernel is compiled out and
// every call takes the block-a-row path.
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1250__) || defined(__gfx1250_strict__)) &&                   \
    __has_builtin(__builtin_amdgcn_s_cluster_barrier) && defined(__MEMORY_SCOPE_CLUSTR)
#define AITER_GFX1250_HAVE_CLUSTER 1
#else
#define AITER_GFX1250_HAVE_CLUSTER 0
#endif

template <typename V, typename = void>
struct lamport_has_cluster_dim : std::false_type
{
};
template <typename V>
struct lamport_has_cluster_dim<V, std::void_t<decltype(std::declval<V&>().clusterDim)>>
    : std::true_type
{
};
// hipLaunchAttributeValue::clusterDim comes with hipLaunchAttributeClusterDimension
constexpr bool kLamportClusterLaunch = lamport_has_cluster_dim<hipLaunchAttributeValue>::value;

// Make attr a cluster of x blocks. A template so that the cluster names are
// dependent and only looked up where the HIP headers have them.
template <typename Attr>
inline void lamport_set_cluster_dim(Attr& attr, unsigned x)
{
    if constexpr(lamport_has_cluster_dim<decltype(Attr::val)>::value)
    {
        attr.id               = decltype(Attr::id)::hipLaunchAttributeClusterDimension;
        attr.val.clusterDim.x = x;
        attr.val.clusterDim.y = 1;
        attr.val.clusterDim.z = 1;
    }
}

// slot: the most 16B packs a call takes, a multiple of kLamportWave
inline constexpr int64_t lamport_ar_rms_buffer_bytes(int ngpus, int64_t slot)
{
    return (int64_t)kLamportBuffers * ngpus * slot * 16;
}

struct __align__(16) LamportPeers
{
    void* ptrs[kMaxNgpusAr];
};

// one bit per half-word that is the 0xffff sentinel
DINLINE uint32_t lamportSentinelMask(uint32_t w)
{
    uint32_t x   = ~w;
    uint32_t tmp = ((x & 0x7fff7fff) + 0x7fff7fff) | x;
    return (tmp & 0x80008000) ^ 0x80008000;
}

// 0xffff -> 0x7fff, both NaN, so sent data is never the sentinel
DINLINE uint32_t lamportCanonicalize(uint32_t w) { return w ^ lamportSentinelMask(w); }

using lamport_u32x4 = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;

DINLINE void lamport_store_b128(void* dst, lamport_u32x4 v)
{
#if AITER_GFX1250_HAVE_B128
    __builtin_amdgcn_global_store_b128((llx_v4u_gptr)dst, v, "");
#else
    __builtin_nontemporal_store(v, reinterpret_cast<lamport_u32x4*>(dst));
#endif
}

DINLINE lamport_u32x4 lamport_load_b128(const void* src)
{
#if AITER_GFX1250_HAVE_B128
    return __builtin_amdgcn_global_load_b128((llx_v4u_gptr)src, "");
#else
    return __builtin_nontemporal_load(reinterpret_cast<const lamport_u32x4*>(src));
#endif
}

// MAX_THREADS only bounds the block; blockDim.x is the block size, a multiple
// of kLamportWave.
template <typename T, int ngpus, int MAX_THREADS, bool CLUSTER>
__global__ void __launch_bounds__(MAX_THREADS, 1) lamport_ar_rmsnorm_gfx1250(
    const T* __restrict__ input,
    const T* __restrict__ residual_inp,
    T* __restrict__ residual_out,
    T* __restrict__ output,
    const T* __restrict__ weight,
    LamportPeers peers,
    float* __restrict__ cluster_partials, // [slot / 32], CLUSTER only, rank-local
    uint32_t* __restrict__ round_counter, // [slot / 32], one a wave of packs, rank-local
    int slot,                             // inbox packs per sender
    int rank,
    int hidden,
    float eps,
    bool gemma_norm)
{
    static_assert(sizeof(T) == 2, "sentinel 0xffff is only defined for 2-byte dtypes");
    static_assert(MAX_THREADS % kLamportWave == 0, "blocks are whole waves");
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    union pack
    {
        P data;
        lamport_u32x4 i;
    };

    // the grid covers whole rows, so every thread owns a pack
    const int row_packs = hidden / pack_size;
    const int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    const int row       = tid / row_packs;
    const int col       = tid - row * row_packs;
    // blocks are whole waves, so this is uniform over the wave
    const int wave       = __builtin_amdgcn_readfirstlane(tid / kLamportWave);
    const uint32_t round = round_counter[wave];
    const uint32_t rearm = (round + 2) % kLamportBuffers;

    pack local;
    local.data = reinterpret_cast<const P*>(input)[tid];
#pragma unroll
    for(int j = 0; j < 4; ++j)
        local.i[j] = lamportCanonicalize(local.i[j]);

#pragma unroll
    for(int i = 1; i < ngpus; ++i)
    {
        P* dst = reinterpret_cast<P*>(peers.ptrs[(rank + i) % ngpus]) +
                 (round * ngpus + rank) * slot + tid;
        lamport_store_b128(dst, local.i);
    }

    // local operands of the epilogue, loaded before the wait
    const P res = reinterpret_cast<const P*>(residual_inp)[tid];
    const P w   = reinterpret_cast<const P*>(weight)[col];

    // slot[rank] is never written and stays the sentinel; it is loaded anyway
    // and masked out, a branch would make every load wait for the previous one
    const P* inbox = reinterpret_cast<const P*>(peers.ptrs[rank]) + round * ngpus * slot + tid;
    pack scratch[ngpus];
    bool pending;
    do
    {
        // without the clobber the loads are hoisted out of the loop
        asm volatile("" ::: "memory");
#pragma unroll
        for(int g = 0; g < ngpus; ++g)
            scratch[g].i = lamport_load_b128(inbox + g * slot);
        // issue all loads before any mask math
        __builtin_amdgcn_sched_barrier(0);
        uint32_t m = 0;
#pragma unroll
        for(int g = 0; g < ngpus; ++g)
        {
            uint32_t gm = 0;
#pragma unroll
            for(int j = 0; j < 4; ++j)
                gm |= lamportSentinelMask(scratch[g].i[j]);
            m |= g == rank ? 0u : gm;
        }
        pending = m != 0;
    } while(pending);

    // same steps as ar_rmsnorm_gfx1250
    float x[pack_size] = {};
#pragma unroll
    for(int g = 0; g < ngpus; ++g)
    {
        const P v = g == rank ? local.data : scratch[g].data;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            x[j] += upcast_s(v[j]);
    }
    P sum;
    float square_sum = 0.f;
#pragma unroll
    for(int j = 0; j < pack_size; ++j)
    {
        x[j]   = upcast_s(downcast_s<T>(x[j])) + upcast_s(res[j]);
        sum[j] = downcast_s<T>(x[j]);
        square_sum += x[j] * x[j];
    }
    reinterpret_cast<P*>(residual_out)[tid] = sum;

    float inv_rms;
    if constexpr(CLUSTER)
    {
#if AITER_GFX1250_HAVE_CLUSTER
        // the row is this cluster: publish a partial per wave, then every wave
        // folds the row's partials in wave order, as block_allreduce_f32 does
        float part = square_sum;
        for(int off = kLamportWave / 2; off > 0; off >>= 1)
            part = part + __shfl_xor(part, off);
        const int lane = threadIdx.x % kLamportWave;
        if(lane == 0)
            __scoped_atomic_store_n(
                cluster_partials + wave, part, __ATOMIC_RELAXED, __MEMORY_SCOPE_CLUSTR);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");
        __builtin_amdgcn_s_cluster_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "cluster");
        const int row_groups = row_packs / kLamportWave;
        const float v        = lane < row_groups
                                   ? __scoped_atomic_load_n(cluster_partials + row * row_groups + lane,
                                                            __ATOMIC_RELAXED,
                                                            __MEMORY_SCOPE_CLUSTR)
                                   : 0.f;
        float total = __shfl(v, 0);
        for(int g = 1; g < row_groups; ++g)
            total += __shfl(v, g);
        inv_rms = rsqrtf(total / hidden + eps);
#else
        // never launched: the host takes this path only with cluster launch
        __builtin_trap();
        inv_rms = 0.f;
#endif
    }
    else
    {
        inv_rms = rsqrtf(block_allreduce_f32<false>(square_sum) / hidden + eps);
    }

    P o;
#pragma unroll
    for(int j = 0; j < pack_size; ++j)
    {
        const float wj = upcast_s(w[j]) + (gemma_norm ? 1.f : 0.f);
        o[j]           = downcast_s<T>(x[j] * (inv_rms * wj));
    }
    reinterpret_cast<P*>(output)[tid] = o;

    const lamport_u32x4 sentinel{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
#pragma unroll
    for(int i = 1; i < ngpus; ++i)
    {
        P* dst = reinterpret_cast<P*>(peers.ptrs[rank]) +
                 (rearm * ngpus + (rank + i) % ngpus) * slot + tid;
        lamport_store_b128(dst, sentinel);
    }

    if(threadIdx.x % kLamportWave == 0)
        round_counter[wave] = (round + 1) % kLamportBuffers;
}

// ---------------------------------------------------------------------------
// Lamport one-shot all-reduce of input + other (2-byte dtypes)
// ---------------------------------------------------------------------------
/*
 * output = the sum over ranks of each rank's input + other. The pair is added
 * in fp32 and rounded to T first, as a T add rounds it, so the result has the
 * bits of all-reducing the materialized sum (rank-order fp32 sum, one rounding).
 *
 * Same inbox, round counters and protocol as lamport_ar_rmsnorm_gfx1250: thread
 * tid owns pack tid of the call. Waves past the input sit the call out; the
 * lanes of the last wave past the input only re-arm, so that every pack of a
 * wave that runs a round has its previous buffer re-armed, whatever the size
 * of the next call on that wave.
 */
// Scalar and pointer kernel arguments only, the inboxes one by one: the
// module preloads the kernel arguments into SGPRs up to the first aggregate,
// and a LamportPeers argument would leave the rest to s_loads in series
// before the first remote store (about 1 us of the call at decode sizes).
template <typename T, int ngpus, int BLOCK>
__global__ void __launch_bounds__(BLOCK, 1)
    lamport_ar_add_gfx1250(const T* __restrict__ input,
                           const T* __restrict__ other,
                           T* __restrict__ output,
                           uint32_t* __restrict__ round_counter, // [slot / 32], rank-local
                           int slot,                             // inbox packs per sender
                           int rank,
                           int packs, // packs of the call
                           void* p0,
                           void* p1,
                           void* p2,
                           void* p3,
                           void* p4,
                           void* p5,
                           void* p6,
                           void* p7)
{
    static_assert(sizeof(T) == 2, "sentinel 0xffff is only defined for 2-byte dtypes");
    static_assert(BLOCK % kLamportWave == 0, "blocks are whole waves");
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    union pack
    {
        P data;
        lamport_u32x4 i;
    };

    const int tid  = blockIdx.x * BLOCK + threadIdx.x;
    const int wave = __builtin_amdgcn_readfirstlane(tid / kLamportWave);
    // waves past the input sit the call out (no block-wide sync below)
    if(wave * kLamportWave >= packs)
        return;
    const uint32_t round = round_counter[wave];
    const uint32_t rearm = (round + 2) % kLamportBuffers;
    // static indices only: a runtime index into the inboxes is a scalar load
    // in series before each remote store
    void* const all[8] = {p0, p1, p2, p3, p4, p5, p6, p7};
    char* own          = static_cast<char*>(all[0]);
#pragma unroll
    for(int g = 1; g < ngpus; ++g)
        own = g == rank ? static_cast<char*>(all[g]) : own;

    if(tid < packs)
    {
        const P a = reinterpret_cast<const P*>(input)[tid];
        const P b = reinterpret_cast<const P*>(other)[tid];
        pack local;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            local.data[j] = downcast_s<T>(upcast_s(a[j]) + upcast_s(b[j]));
#pragma unroll
        for(int j = 0; j < 4; ++j)
            local.i[j] = lamportCanonicalize(local.i[j]);

        const int64_t sent = (int64_t)(round * ngpus + rank) * slot + tid;
#pragma unroll
        for(int g = 0; g < ngpus; ++g)
            if(g != rank)
                lamport_store_b128(static_cast<P*>(all[g]) + sent, local.i);

        // slot[rank] stays the sentinel; loaded and masked out, see the RMSNorm kernel
        const P* inbox = reinterpret_cast<const P*>(own) + (int64_t)round * ngpus * slot + tid;
        pack scratch[ngpus];
        bool pending;
        do
        {
            asm volatile("" ::: "memory");
#pragma unroll
            for(int g = 0; g < ngpus; ++g)
                scratch[g].i = lamport_load_b128(inbox + (int64_t)g * slot);
            __builtin_amdgcn_sched_barrier(0);
            uint32_t m = 0;
#pragma unroll
            for(int g = 0; g < ngpus; ++g)
            {
                uint32_t gm = 0;
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    gm |= lamportSentinelMask(scratch[g].i[j]);
                m |= g == rank ? 0u : gm;
            }
            pending = m != 0;
        } while(pending);

        float x[pack_size] = {};
#pragma unroll
        for(int g = 0; g < ngpus; ++g)
        {
            const P v = g == rank ? local.data : scratch[g].data;
#pragma unroll
            for(int j = 0; j < pack_size; ++j)
                x[j] += upcast_s(v[j]);
        }
        P o;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            o[j] = downcast_s<T>(x[j]);
        reinterpret_cast<P*>(output)[tid] = o;
    }

    const lamport_u32x4 sentinel{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu};
#pragma unroll
    for(int g = 0; g < ngpus; ++g)
        if(g != rank)
            lamport_store_b128(reinterpret_cast<P*>(own) + (int64_t)(rearm * ngpus + g) * slot +
                                   tid,
                               sentinel);

    if(threadIdx.x % kLamportWave == 0)
        round_counter[wave] = (round + 1) % kLamportBuffers;
}

// The Lamport inbox of a communicator: one per CustomAllreduce, any row width
// and row count up to slot packs (see lamport_ar_rmsnorm_gfx1250).
struct LamportArRms
{
    LamportPeers peers{};
    float* cluster_partials = nullptr; // [slot / 32], rank-local
    uint32_t* d_round       = nullptr; // [slot / 32], rank-local
    int slot                = 0;       // the most packs a call takes
    int rank                = 0;
    int world_size          = 0;
    int num_cus             = 0;

    ~LamportArRms()
    {
        if(d_round)
            (void)hipFree(d_round);
        if(cluster_partials)
            (void)hipFree(cluster_partials);
    }

    // Block size of a row split over a cluster, or 0 for a block a row. Split
    // only while the rows leave CUs idle: from rows >= CUs on a block a row
    // was faster (gfx1250, 256 CUs, rows 256+ at hidden 4096-8192).
    int cluster_threads(int64_t rows, int row_packs) const
    {
        if(!kLamportClusterLaunch || rows >= num_cus || row_packs <= kLamportMinClusterRowPacks)
            return 0;
        for(int bs = kLamportClusterMaxThreads; bs >= kLamportWave; bs /= 2)
        {
            if(row_packs % bs != 0)
                continue;
            const int blocks = row_packs / bs;
            return blocks <= kLamportMaxClusterBlocks ? bs : 0;
        }
        return 0;
    }

    // A stream capture drops the cluster dimension of hipLaunchKernelEx (the
    // node then runs without a cluster and the cluster barrier orders nothing),
    // so set it again on the node the launch just added.
    static void set_captured_cluster_dim(hipStream_t stream, const hipLaunchAttribute& attr)
    {
        hipStreamCaptureStatus status;
        const hipGraphNode_t* deps = nullptr;
        size_t num_deps            = 0;
        HIP_CALL(hipStreamGetCaptureInfo_v2(
            stream, &status, nullptr, nullptr, &deps, &num_deps));
        if(status != hipStreamCaptureStatusActive)
            return;
        if(num_deps != 1)
            throw std::runtime_error("lamport allreduce rmsnorm: cannot find the captured node");
        HIP_CALL(hipGraphKernelNodeSetAttribute(deps[0], attr.id, &attr.val));
    }

    template <typename T>
    void run(hipStream_t stream,
             const T* input,
             const T* residual_inp,
             T* residual_out,
             T* output,
             const T* weight,
             float eps,
             int64_t rows,
             int hidden,
             bool gemma_norm)
    {
        constexpr int pack_size = 16 / sizeof(T);
        if(hidden <= 0 || hidden % (pack_size * kLamportWave) != 0 ||
           hidden / pack_size > kLamportRowMaxThreads)
            throw std::runtime_error("lamport allreduce rmsnorm: hidden must be a multiple of " +
                                     std::to_string(pack_size * kLamportWave) + " and at most " +
                                     std::to_string(pack_size * kLamportRowMaxThreads) +
                                     ", got " + std::to_string(hidden));
        const int row_packs = hidden / pack_size;
        const int64_t packs = rows * row_packs;
        if(packs > slot)
            throw std::runtime_error("lamport allreduce rmsnorm: " + std::to_string(packs * 16) +
                                     "B exceeds the inbox cap of " +
                                     std::to_string((int64_t)slot * 16) + "B");
        if(packs == 0)
            return;
        const int threads = cluster_threads(rows, row_packs);

#define LAMPORT_RMS_ARGS                                                                    \
    input, residual_inp, residual_out, output, weight, peers, cluster_partials, d_round,    \
        slot, rank, hidden, eps, gemma_norm
#define LAUNCH_LAMPORT_RMS(NG)                                                              \
    if(threads == 0)                                                                        \
    {                                                                                       \
        lamport_ar_rmsnorm_gfx1250<T, NG, kLamportRowMaxThreads, false>                     \
            <<<(int)rows, row_packs, 0, stream>>>(LAMPORT_RMS_ARGS);                        \
    }                                                                                       \
    else if constexpr(kLamportClusterLaunch)                                                \
    {                                                                                       \
        hipLaunchAttribute attr{};                                                          \
        lamport_set_cluster_dim(attr, (unsigned)(row_packs / threads));                     \
        hipLaunchConfig_t config{};                                                         \
        config.gridDim  = dim3((unsigned)(packs / threads));                                \
        config.blockDim = dim3(threads);                                                    \
        config.stream   = stream;                                                           \
        config.attrs    = &attr;                                                            \
        config.numAttrs = 1;                                                                \
        HIP_CALL(hipLaunchKernelEx(                                                         \
            &config,                                                                        \
            lamport_ar_rmsnorm_gfx1250<T, NG, kLamportClusterMaxThreads, true>,             \
            LAMPORT_RMS_ARGS));                                                             \
        set_captured_cluster_dim(stream, attr);                                             \
    }
        DISPATCH_NGPUS_1250(world_size, LAUNCH_LAMPORT_RMS);
#undef LAUNCH_LAMPORT_RMS
#undef LAMPORT_RMS_ARGS
    }

    // Block size of a call of packs: small blocks spread a decode-size call
    // over more CUs, large ones cut the blocks of a big call (4x gfx1250, TP4,
    // bf16 [M, 5120]: 64 best at M = 1, 256 at M = 8-288, 512 at M = 409).
    static int add_threads(int64_t packs)
    {
        if(packs <= 2048)
            return 64;
        return packs <= 192 * 1024 ? 256 : 512;
    }

    // output = all-reduce of input + other, numel values (a multiple of the pack)
    template <typename T>
    void run_add(hipStream_t stream, const T* input, const T* other, T* output, int64_t numel)
    {
        constexpr int pack_size = 16 / sizeof(T);
        if(numel % pack_size != 0)
            throw std::runtime_error("lamport allreduce add: numel must be a multiple of " +
                                     std::to_string(pack_size));
        const int64_t packs = numel / pack_size;
        if(packs > slot)
            throw std::runtime_error("lamport allreduce add: " + std::to_string(packs * 16) +
                                     "B exceeds the inbox cap of " +
                                     std::to_string((int64_t)slot * 16) + "B");
        if(packs == 0)
            return;
        const int threads = add_threads(packs);
        const int blocks = (int)((packs + threads - 1) / threads);
#define LAUNCH_LAMPORT_ADD_B(NG, B)                                                          \
    lamport_ar_add_gfx1250<T, NG, B><<<blocks, B, 0, stream>>>(input,                        \
                                                               other,                        \
                                                               output,                       \
                                                               d_round,                      \
                                                               slot,                         \
                                                               rank,                         \
                                                               (int)packs,                   \
                                                               peers.ptrs[0],                \
                                                               peers.ptrs[1],                \
                                                               peers.ptrs[2],                \
                                                               peers.ptrs[3],                \
                                                               peers.ptrs[4],                \
                                                               peers.ptrs[5],                \
                                                               peers.ptrs[6],                \
                                                               peers.ptrs[7])
#define LAUNCH_LAMPORT_ADD(NG)                                                               \
    switch(threads)                                                                          \
    {                                                                                        \
    case 64: LAUNCH_LAMPORT_ADD_B(NG, 64); break;                                            \
    case 256: LAUNCH_LAMPORT_ADD_B(NG, 256); break;                                          \
    default: LAUNCH_LAMPORT_ADD_B(NG, 512); break;                                           \
    }
        DISPATCH_NGPUS_1250(world_size, LAUNCH_LAMPORT_ADD);
#undef LAUNCH_LAMPORT_ADD
#undef LAUNCH_LAMPORT_ADD_B
    }
};

// ---------------------------------------------------------------------------
// gfx1250 allgather kernel — scalar fallback (size not pack-aligned)
// ---------------------------------------------------------------------------
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) ag_gfx1250_scalar(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const T* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        ptrs[i] = (const T*)_input_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for(int idx = tid; idx < size; idx += stride)
    {
#pragma unroll
        for(int i = 0; i < ngpus; ++i)
        {
            int gpu_idx = (rank + i) % ngpus;
            result[gpu_idx * size + idx] = ptrs[gpu_idx][idx];
        }
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 2) ag_gfx1250_naive_vec(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    int index    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride  = blockDim.x * gridDim.x;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        ptrs[i] = (const P*)_input_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for (int idx = index; idx < size; idx += stride)
    {
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
      {
        int rank_idx = (rank + i) % ngpus;
        *(reinterpret_cast<P*>(result) + size * rank_idx + idx) = ptrs[rank_idx][idx];
      }
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// gfx1250 allgather kernel — vectorized unroll4
// ---------------------------------------------------------------------------
template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 2) ag_gfx1250_naive_unroll4(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int unroll    = 4;
    using P                 = typename opus::vector_t<T, pack_size>;
    int index    = blockIdx.x * blockDim.x * unroll + threadIdx.x;
    int stride  = blockDim.x * gridDim.x * unroll;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        ptrs[i] = (const P*)_input_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for (int idx = index; idx + blockDim.x * (unroll - 1) < size; idx += stride)
    {
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
      {
        int rank_idx = (rank + i) % ngpus;
#pragma unroll
        for (int j = 0; j < unroll; ++j)
        {
          *(reinterpret_cast<P*>(result) + size * rank_idx + idx + j * blockDim.x) = ptrs[rank_idx][idx + j * blockDim.x];
        }
      }
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) ag_gfx1250_lastdim(RankData* _dp,
                                                            RankSignals sg,
                                                            Signal* self_sg,
                                                            T* __restrict__ result,
                                                            int rank,
                                                            int size,
                                                            int last_dim_size)
{
    constexpr int unroll    = 4;
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    int tid                 = blockIdx.x * blockDim.x * unroll + threadIdx.x;
    int stride              = gridDim.x * blockDim.x * unroll;

    last_dim_size /= pack_size;
    const P* ptrs[ngpus];

#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        ptrs[i] = (const P*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);

    for(int idx = tid; idx < size; idx += stride)
    {
#pragma unroll
      for (int i = 0; i < ngpus; ++i)
      {
        int rank_idx = (rank + i) % ngpus;
#pragma unroll
        for (int j = 0; j < unroll; ++j)
        {
          int read_idx = idx + j * blockDim.x;
          if (read_idx >= size) break;
          int y = read_idx / last_dim_size;
          int x = read_idx % last_dim_size;
          int write_idx = (ngpus * y + rank_idx) * last_dim_size + x;
          *(reinterpret_cast<P*>(result) + write_idx) = ptrs[rank_idx][read_idx];
        }
      }
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// gfx1250 bandwidth test kernel
// ---------------------------------------------------------------------------
template <typename T, int unroll>
__global__ void  p2p_bandwidth_test_kernel(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    int index    = blockIdx.x * blockDim.x * unroll + threadIdx.x;
    int stride  = blockDim.x * gridDim.x * unroll;
    const P* ptrs[2];
#pragma unroll
    for(int i = 0; i < 2; i++)
    {
        ptrs[i] = (const P*)_input_dp->ptrs[i];
    }
    start_sync<2>(sg, self_sg, rank);
    for (int idx = index; idx < size; idx += stride)
    {
      P reg[unroll];
#pragma unroll
      for (int i = 0; i < unroll; ++i)
      {
        reg[i] = ptrs[(rank + 1) % 2][idx + i * blockDim.x];
      }
#pragma unroll
      for (int i = 0; i < unroll; ++i)
      {
        *(reinterpret_cast<P*>(result) + idx + i * blockDim.x) = reg[i];
      }
    }
    // end_sync<2, true>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// gfx1250 reduce_scatter kernels
// ---------------------------------------------------------------------------
enum class ReduceScatterSplitDim : int { kFirst = 0, kLast = 1, kMid = 2 };

// reduce_scatter, scatter on first dim — vectorized.
// cond: numel % (ngpus * pack_size) == 0
// shape: input flat numel -> output flat numel / ngpus
template <typename T, int ngpus>
__global__ void __launch_bounds__(512, 1) rs_gfx1250_split_first_dim(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    T* __restrict__ result, int rank, int range)
{
    int tid                 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride              = blockDim.x * gridDim.x;
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        // Keep the sum order independent of the reduce-scatter owner.
        ptrs[i] = (const P*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for(int idx = tid; idx < range; idx += stride)
    {
        int load_index = rank * range + idx;
        A acc;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            acc[j] = upcast_s(ptrs[0][load_index][j]);
#pragma unroll
        for(int g = 1; g < ngpus; ++g)
        {
#pragma unroll
            for(int j = 0; j < pack_size; ++j)
                acc[j] += upcast_s(ptrs[g][load_index][j]);
        }
        P out_val;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            out_val[j] = downcast_s<T>(acc[j]);
        *(reinterpret_cast<P*>(result) + idx) = out_val;
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// reduce_scatter, scatter on last dim — scalar fallback.
// cond: n % ngpus == 0
// shape: input (m, n) -> output (m, n / ngpus)
template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 1) rs_gfx1250_split_lastdim_naive(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    T* __restrict__ result, int rank, int m, int n)
{
    int size      = m * n / ngpus;
    int splited_n = n / ngpus;
    int index     = blockIdx.x * blockDim.x + threadIdx.x;
    const T* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        // Keep the sum order independent of the reduce-scatter owner.
        ptrs[i] = (const T*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for(int i = index; i < size; i += blockDim.x * gridDim.x)
    {
        int index_x    = i % splited_n;
        int index_y    = i / splited_n;
        int load_index = index_y * n + rank * splited_n + index_x;
        opus::fp32_t rslt_reg = 0.0f;
#pragma unroll
        for(int j = 0; j < ngpus; ++j)
            rslt_reg += upcast_s(ptrs[j][load_index]);
        result[i] = downcast_s<T>(rslt_reg);
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// reduce_scatter, scatter on last dim — vectorized.
// cond: n % (ngpus * pack_size) == 0
// shape: input (m, n) -> output (m, n / ngpus)
template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 1) rs_gfx1250_split_lastdim(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    T* __restrict__ result, int rank, int m, int n)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    int size        = m * n / (ngpus * pack_size);
    int splited_n   = n / (ngpus * pack_size);
    int packed_dim_n = n / pack_size;
    int index       = blockIdx.x * blockDim.x + threadIdx.x;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        // Keep the sum order independent of the reduce-scatter owner.
        ptrs[i] = (const P*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for(int i = index; i < size; i += blockDim.x * gridDim.x)
    {
        int index_x    = i % splited_n;
        int index_y    = i / splited_n;
        int load_index = index_y * packed_dim_n + rank * splited_n + index_x;
        P inp_reg[ngpus];
#pragma unroll
        for(int g = 0; g < ngpus; ++g)
            inp_reg[g] = ptrs[g][load_index];
        A acc;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            acc[j] = upcast_s(inp_reg[0][j]);
#pragma unroll
        for(int g = 1; g < ngpus; ++g)
        {
#pragma unroll
            for(int j = 0; j < pack_size; ++j)
                acc[j] += upcast_s(inp_reg[g][j]);
        }
        P out_val;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            out_val[j] = downcast_s<T>(acc[j]);
        *(reinterpret_cast<P*>(result) + i) = out_val;
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// reduce_scatter, scatter on middle dim — scalar fallback.
// cond: n % ngpus == 0
// shape: input (m, n, k) -> output (m, n / ngpus, k)
template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 1) rs_gfx1250_split_middim_naive(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    T* __restrict__ result, int rank, int m, int n, int k)
{
    int size      = m * n * k / ngpus;
    int splited_n = n / ngpus;
    int index     = blockIdx.x * blockDim.x + threadIdx.x;
    const T* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        // Keep the sum order independent of the reduce-scatter owner.
        ptrs[i] = (const T*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for(int i = index; i < size; i += blockDim.x * gridDim.x)
    {
        int index_m    = i / (splited_n * k);
        int index_n    = (i % (splited_n * k)) / k;
        int index_k    = (i % (splited_n * k)) % k;
        int load_index = index_m * (n * k) + (rank * splited_n + index_n) * k + index_k;
        opus::fp32_t rslt_reg = 0.0f;
#pragma unroll
        for(int j = 0; j < ngpus; ++j)
            rslt_reg += upcast_s(ptrs[j][load_index]);
        result[i] = downcast_s<T>(rslt_reg);
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// reduce_scatter, scatter on middle dim — vectorized along k.
// cond: n % ngpus == 0 && k % pack_size == 0
// shape: input (m, n, k) -> output (m, n / ngpus, k)
template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 1) rs_gfx1250_split_middim(
    RankData* _dp, RankSignals sg, Signal* self_sg,
    T* __restrict__ result, int rank, int m, int n, int k)
{
    constexpr int pack_size = 16 / sizeof(T);
    using P                 = typename opus::vector_t<T, pack_size>;
    using A                 = typename opus::vector_t<opus::fp32_t, pack_size>;
    int size         = m * n * k / (pack_size * ngpus);
    int splited_n    = n / ngpus;
    int packed_dim_k = k / pack_size;
    int index        = blockIdx.x * blockDim.x + threadIdx.x;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; ++i)
    {
        // Keep the sum order independent of the reduce-scatter owner.
        ptrs[i] = (const P*)_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for(int i = index; i < size; i += blockDim.x * gridDim.x)
    {
        int index_m    = i / (splited_n * packed_dim_k);
        int index_n    = (i % (splited_n * packed_dim_k)) / packed_dim_k;
        int index_k    = (i % (splited_n * packed_dim_k)) % packed_dim_k;
        int load_index = index_m * (n * packed_dim_k) + (rank * splited_n + index_n) * packed_dim_k + index_k;
        P inp_reg[ngpus];
#pragma unroll
        for(int g = 0; g < ngpus; ++g)
            inp_reg[g] = ptrs[g][load_index];
        A acc;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            acc[j] = upcast_s(inp_reg[0][j]);
#pragma unroll
        for(int g = 1; g < ngpus; ++g)
        {
#pragma unroll
            for(int j = 0; j < pack_size; ++j)
                acc[j] += upcast_s(inp_reg[g][j]);
        }
        P out_val;
#pragma unroll
        for(int j = 0; j < pack_size; ++j)
            out_val[j] = downcast_s<T>(acc[j]);
        *(reinterpret_cast<P*>(result) + i) = out_val;
    }
    end_sync<ngpus, true>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// sync latency
// ---------------------------------------------------------------------------
template <int ngpus>
__global__ void start_sync_kernel(RankSignals sg, Signal* self_sg, int rank)
{
  start_sync<ngpus>(sg, self_sg, rank);
}

template <int ngpus>
__global__ void end_sync_kernel(RankSignals sg, Signal* self_sg, int rank)
{
  end_sync<ngpus>(sg, self_sg, rank);
}

template <int ngpus>
__global__ void two_sync_kernel(RankSignals sg, Signal* self_sg, int rank)
{
  start_sync<ngpus>(sg, self_sg, rank);
  end_sync<ngpus>(sg, self_sg, rank);
}

// ---------------------------------------------------------------------------
// CustomAllreduce class (gfx1250-only, simplified)
// ---------------------------------------------------------------------------
// gfx1250: hipIpc is not available. Buffer sharing uses torch's
// cross-process CUDA tensor sharing; the C++ layer receives direct
// device pointers that torch already mapped into each process's VA.

class CustomAllreduce
{
public:
    int rank_;
    int world_size_;
    bool full_nvlink_;

    RankSignals sg_;
    std::unordered_map<void*, RankData*> input_buffer;
    std::unordered_map<void*, RankData*> output_buffers_;
    Signal* self_sg_;

    RankData *d_rank_data_base_, *d_rank_data_end_;
    std::vector<void*> graph_unreg_input_buffers_;
    std::vector<void*> graph_unreg_output_buffers_;

    // Opened hipIpc handles, kept so they can be closed at destruction. Only
    // populated on the IPC transport path (ROCm >= 7.15, where hipIpc works on
    // gfx1250). Empty on the VMM path — the destructor is then a no-op.
    using IPC_KEY = std::array<uint8_t, sizeof(hipIpcMemHandle_t)>;
    std::map<IPC_KEY, char*> ipc_handles_;

    // LL small-message fast path. Device table of per-rank scratch bases (each
    // = peer meta base + kLLScratchOffset, i.e. the region right after the peer
    // Signal in the shared meta buffer) and a persistent per-block epoch array.
    // Allocated lazily by ensure_ll_tables_() on the first LL launch.
    void**    d_ll_peers_       = nullptr;
    uint32_t* d_ll_block_flags_ = nullptr;

    // LL two-shot all-reduce, apart from the LL tables above: per-rank bases of
    // its own scratch (peer meta base + ar2sScratchOffset) and its own per-block
    // epochs. Built by ensure_2s_tables_(); null where it does not run.
    void**    d_2s_peers_       = nullptr;
    uint32_t* d_2s_block_flags_ = nullptr;

    // Lamport fused all-reduce + RMSNorm inbox (init_lamport_ar_rms), or null.
    std::unique_ptr<LamportArRms> lamport_;

    // gfx1250: hipIpc is not available. Instead, each rank's Signal buffer
    // is a torch-shared tensor whose device pointer is exchanged via the
    // distributed store.  The constructor receives the remote pointers
    // directly (torch already mapped them into this process's VA space).
    CustomAllreduce(Signal* meta,
                    void* rank_data,
                    size_t rank_data_sz,
                    const std::vector<int64_t>& all_meta_ptrs,
                    int rank,
                    bool fully_connected = true)
        : rank_(rank),
          world_size_(all_meta_ptrs.size()),
          full_nvlink_(fully_connected),
          self_sg_(meta),
          d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
          d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData))
    {
        for(int i = 0; i < world_size_; i++)
        {
            sg_.signals[i] = reinterpret_cast<Signal*>(all_meta_ptrs[i]);
        }
        // Build the LL scratch/epoch tables now, at construction, rather than
        // lazily on the first LL launch. The lazy path would run hipMalloc /
        // hipMemcpy / hipMemset the first time allreduce() routes to LL, and if
        // that first call happens inside a CUDA-graph capture (no eager warm-up
        // before capture) those runtime calls are illegal mid-capture and abort.
        // sg_.signals[] is populated above, so the peer scratch bases are ready.
        if(ll_enabled())
        {
            ensure_ll_tables_();
            ensure_2s_tables_();
        }
    }

    // IPC transport overload (ROCm >= 7.15): hipIpc works on gfx1250, so peer
    // meta buffers are shared via IPC handles instead of VMM-exported fds. This
    // mirrors the old-arch path — resolve each remote handle to a local VA via
    // hipIpcOpenMemHandle and add its offset. The kernel is unchanged; only how
    // the peer pointers are obtained differs.
    CustomAllreduce(Signal* meta,
                    void* rank_data,
                    size_t rank_data_sz,
                    const hipIpcMemHandle_t* handles,
                    const std::vector<int64_t>& offsets,
                    int rank,
                    bool fully_connected = true)
        : rank_(rank),
          world_size_(offsets.size()),
          full_nvlink_(fully_connected),
          self_sg_(meta),
          d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
          d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData))
    {
        for(int i = 0; i < world_size_; i++)
        {
            if(i != rank_)
            {
                char* handle = open_ipc_handle(&handles[i]);
                handle += offsets[i];
                sg_.signals[i] = reinterpret_cast<Signal*>(handle);
            }
            else
            {
                sg_.signals[i] = self_sg_;
            }
        }
        // Eager LL-table build (see the VMM-overload constructor above): keeps
        // hipMalloc/hipMemcpy/hipMemset out of the first LL allreduce, which may
        // land inside a CUDA-graph capture and abort. sg_.signals[] is ready.
        if(ll_enabled())
        {
            ensure_ll_tables_();
            ensure_2s_tables_();
        }
    }

    char* open_ipc_handle(const void* ipc_handle)
    {
        auto [it, new_handle] = ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
        if(new_handle)
        {
            char* ipc_ptr;
            HIP_CALL(hipIpcOpenMemHandle((void**)&ipc_ptr,
                                         *((const hipIpcMemHandle_t*)ipc_handle),
                                         hipIpcMemLazyEnablePeerAccess));
            it->second = ipc_ptr;
        }
        return it->second;
    }

    ~CustomAllreduce()
    {
        // No-op on the VMM path (ipc_handles_ empty); closes opened peer
        // handles on the IPC path.
        for(auto [_, ptr] : ipc_handles_)
        {
            HIP_CALL(hipIpcCloseMemHandle(ptr));
        }
        if(d_ll_peers_)
            (void)hipFree(d_ll_peers_);
        if(d_ll_block_flags_)
            (void)hipFree(d_ll_block_flags_);
        if(d_2s_peers_)
            (void)hipFree(d_2s_peers_);
        if(d_2s_block_flags_)
            (void)hipFree(d_2s_block_flags_);
    }

    // Whether allreduce of this size takes the LL kernel, which reads the input
    // in place and so needs no registered copy of it.
    bool use_ll(size_t bytes) const
    {
        return ll_enabled() && bytes <= kLLArMaxBytes && world_size_ <= kLLMaxRanks;
    }

    // Whether allreduce of this size takes the LL two-shot kernel: four ranks or
    // more, from where it overtakes the one-shot (ar2sMinBytes) up to its cap
    // (ar2sMaxBytes, 0 where it does not run), in whole packets per chunk.
    bool use_2s_ll(size_t bytes) const
    {
        return ll_enabled() && world_size_ <= kLLMaxRanks && bytes >= ar2sMinBytes(world_size_) &&
               bytes <= ar2sMaxBytes(world_size_) && bytes % (8 * (size_t)world_size_) == 0;
    }

    // Whether allreduce of this size reads its input in place (no copy-in).
    bool ar_reads_in_place(size_t bytes) const { return use_2s_ll(bytes) || use_ll(bytes); }

    static int cu_count()
    {
        static const int cus = []() {
            int dev = 0, n = 0;
            if(hipGetDevice(&dev) != hipSuccess ||
               hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess ||
               n <= 0)
                n = 256;
            return n;
        }();
        return cus;
    }

    // Senders (and as many pollers) a block for the LL one-shot at `units`
    // packets: 256 while one block a CU covers them, else 512. On MI455X a
    // second block a CU costs more than the wider block.
    static int ll_threads_for(size_t units) { return units <= (size_t)cu_count() * 256 ? 256 : 512; }

    // Whether the LL small-message fast path is enabled. On by default; set
    // AITER_CUSTOM_AR_DISABLE_LL=1 to force the naive kernel for all sizes.
    static bool ll_enabled()
    {
        static const bool disabled = []() {
            const char* e = std::getenv("AITER_CUSTOM_AR_DISABLE_LL");
            if(!e)
                return false;
            std::string v(e);
            return v == "1" || v == "true" || v == "yes" || v == "on" ||
                   v == "TRUE" || v == "YES" || v == "ON";
        }();
        return !disabled;
    }

    // Lazily build the per-rank LL scratch pointer table and per-block epoch.
    // sg_.signals[i] is each rank's meta base (populated by both constructors);
    // the LL scratch sits at + kLLScratchOffset. The shared meta buffer is
    // zero-initialized by the caller, which resets the LL flags (0 == cleared).
    void ensure_ll_tables_()
    {
        if(d_ll_peers_)
            return;
        std::vector<void*> peers(world_size_);
        for(int i = 0; i < world_size_; ++i)
            peers[i] = reinterpret_cast<char*>(sg_.signals[i]) + kLLScratchOffset;
        HIP_CALL(hipMalloc(&d_ll_peers_, sizeof(void*) * world_size_));
        HIP_CALL(hipMemcpy(
            d_ll_peers_, peers.data(), sizeof(void*) * world_size_, hipMemcpyHostToDevice));
        HIP_CALL(hipMalloc(&d_ll_block_flags_, sizeof(uint32_t) * kMaxBlocks));
        HIP_CALL(hipMemset(d_ll_block_flags_, 0, sizeof(uint32_t) * kMaxBlocks));
    }

    // The LL two-shot's per-rank scratch table and per-block epoch, as
    // ensure_ll_tables_() builds the LL ones; nothing at a world size it does not
    // run at. Its scratch sits at + ar2sScratchOffset in the zero-initialized
    // meta buffer.
    void ensure_2s_tables_()
    {
        if(d_2s_peers_ || world_size_ > kLLMaxRanks || ar2sMaxBytes(world_size_) == 0)
            return;
        std::vector<void*> peers(world_size_);
        for(int i = 0; i < world_size_; ++i)
            peers[i] = reinterpret_cast<char*>(sg_.signals[i]) + ar2sScratchOffset(world_size_);
        HIP_CALL(hipMalloc(&d_2s_peers_, sizeof(void*) * world_size_));
        HIP_CALL(hipMemcpy(
            d_2s_peers_, peers.data(), sizeof(void*) * world_size_, hipMemcpyHostToDevice));
        HIP_CALL(hipMalloc(&d_2s_block_flags_, sizeof(uint32_t) * kMaxBlocks));
        HIP_CALL(hipMemset(d_2s_block_flags_, 0, sizeof(uint32_t) * kMaxBlocks));
    }

    // LL one-shot all-reduce (see ar_ll_gfx1250). Reads `input` in place.
    // numel is the element count; bytes must be a multiple of 16 and <=
    // kLLScratchCapBytes (guaranteed by the routing threshold). W senders and W
    // pollers a block (ll_threads_for); past one such block a CU, a sender takes
    // 2 or 4 packets.
    template <typename T>
    void allreduce_ll(hipStream_t stream, const T* input, T* output, int numel)
    {
        ensure_ll_tables_();
        const size_t bytes = (size_t)numel * sizeof(T);
        const size_t nPk   = bytes >> 3; // 8 payload bytes per packet
        const size_t cap   = std::min<size_t>(cu_count(), kMaxBlocks);
        const int    W     = ll_threads_for(nPk);
        int U = 1;
        while(U < 4 && (nPk + (size_t)W * U - 1) / ((size_t)W * U) > cap)
            U *= 2;
        const int blocks = std::max(1, std::min<int>(kMaxBlocks,
            (int)((nPk + (size_t)W * U - 1) / ((size_t)W * U))));

        T* const* peers = reinterpret_cast<T* const*>(d_ll_peers_);
#define LAUNCH_LL_U(NG, UU) \
        ar_ll_gfx1250<T, NG, UU><<<blocks, 2 * W, 0, stream>>>( \
            peers, output, input, nPk, rank_, d_ll_block_flags_)
#define LAUNCH_LL(NG)            \
        if(U == 1)               \
        {                        \
            LAUNCH_LL_U(NG, 1);  \
        }                        \
        else if(U == 2)          \
        {                        \
            LAUNCH_LL_U(NG, 2);  \
        }                        \
        else                     \
        {                        \
            LAUNCH_LL_U(NG, 4);  \
        }
        DISPATCH_NGPUS_1250(world_size_, LAUNCH_LL);
#undef LAUNCH_LL
#undef LAUNCH_LL_U
    }

    // LL two-shot all-reduce (see ar_2s_ll_gfx1250); use_2s_ll(bytes) holds.
    // Three wave groups while one block of 3 x 128 threads a CU covers a chunk,
    // else two (2 x W threads cover more packets a CU; MI455X TP8: three groups
    // ahead by ~1 us up to 2.1 MB, two from 3.3 MB, by up to 5 us at 8.4 MB).
    // W: the fewest packets on the busiest CU, ceil(blocks / CUs) x W, for at most
    // kMaxBlocks blocks; the smaller W on a tie (more CUs issue).
    template <typename T>
    void allreduce_2s_ll(hipStream_t stream, const T* input, T* output, int numel)
    {
        ensure_2s_tables_();
        const size_t bytes    = (size_t)numel * sizeof(T);
        const size_t chunk_pk = bytes / (8 * (size_t)world_size_);
        const size_t cus      = (size_t)cu_count();
        const int    roles    = chunk_pk <= cus * 128 ? 3 : 2;
        int W = 64;
        size_t best = (size_t)-1;
        for(int w = 64; w * roles <= 1024; w += 64)
        {
            const size_t b = (chunk_pk + w - 1) / w;
            if(b > (size_t)kMaxBlocks)
                continue;
            const size_t load = (b + cus - 1) / cus * w;
            if(load < best)
            {
                best = load;
                W    = w;
            }
        }
        const int blocks = std::max(1, std::min<int>(kMaxBlocks, (int)((chunk_pk + W - 1) / W)));
        T* const* peers = reinterpret_cast<T* const*>(d_2s_peers_);
#define LAUNCH_2S_LL_R(NG, RR) \
        ar_2s_ll_gfx1250<T, NG, RR><<<blocks, RR * W, 0, stream>>>( \
            peers, output, input, chunk_pk, rank_, d_2s_block_flags_)
#define LAUNCH_2S_LL(NG)            \
        if(roles == 3)              \
        {                           \
            LAUNCH_2S_LL_R(NG, 3);  \
        }                           \
        else                        \
        {                           \
            LAUNCH_2S_LL_R(NG, 2);  \
        }
        DISPATCH_NGPUS_1250(world_size_, LAUNCH_2S_LL);
#undef LAUNCH_2S_LL
#undef LAUNCH_2S_LL_R
    }

    // -----------------------------------------------------------------------
    // Lamport fused all-reduce + RMSNorm, see lamport_ar_rmsnorm_gfx1250.
    //
    // all_ptrs[i] is rank i's buffer of lamport_ar_rms_buffer_bytes(world_size_,
    // max_bytes / 16) bytes, mapped into this process (all_ptrs[rank_] is this
    // rank's own); max_bytes is the largest input a call takes. Once per
    // communicator. The caller must barrier across ranks after this returns,
    // otherwise a fast rank's first push can be wiped by a slow rank's 0xff
    // memset.
    // -----------------------------------------------------------------------
    void init_lamport_ar_rms(const std::vector<int64_t>& all_ptrs, int64_t max_bytes)
    {
        if(lamport_)
            throw std::runtime_error("init_lamport_ar_rms: already initialized");
        if(world_size_ > kMaxNgpusAr || (int)all_ptrs.size() != world_size_)
            throw std::invalid_argument(
                "init_lamport_ar_rms: need one buffer per rank, at most 8 ranks");
        // one round counter and one partial a wave, so whole waves of packs
        if(max_bytes <= 0 || max_bytes % (16 * kLamportWave) != 0 ||
           max_bytes / 16 > std::numeric_limits<int>::max() / (kLamportBuffers * kMaxNgpusAr))
            throw std::invalid_argument("init_lamport_ar_rms: max_bytes must be a positive "
                                        "multiple of 512");
        const int64_t slot = max_bytes / 16;
        auto state         = std::make_unique<LamportArRms>();
        state->rank        = rank_;
        state->world_size  = world_size_;
        state->slot        = (int)slot;
        int device;
        HIP_CALL(hipGetDevice(&device));
        HIP_CALL(hipDeviceGetAttribute(
            &state->num_cus, hipDeviceAttributeMultiprocessorCount, device));
        for(int i = 0; i < world_size_; i++)
            state->peers.ptrs[i] = reinterpret_cast<void*>(all_ptrs[i]);
        HIP_CALL(hipMemset(
            state->peers.ptrs[rank_], 0xff, lamport_ar_rms_buffer_bytes(world_size_, slot)));
        HIP_CALL(hipMalloc((void**)&state->d_round, sizeof(uint32_t) * (slot / kLamportWave)));
        HIP_CALL(hipMemset(state->d_round, 0, sizeof(uint32_t) * (slot / kLamportWave)));
        HIP_CALL(hipMalloc((void**)&state->cluster_partials,
                           sizeof(float) * (slot / kLamportWave)));
        HIP_CALL(hipDeviceSynchronize());
        lamport_ = std::move(state);
    }

    template <typename T>
    void fused_allreduce_rmsnorm_lamport(hipStream_t stream,
                                         const T* input,
                                         const T* residual_inp,
                                         T* residual_out,
                                         T* output,
                                         const T* weight,
                                         int64_t tokens,
                                         int hidden,
                                         float eps,
                                         bool gemma_norm)
    {
        if(!lamport_)
            throw std::runtime_error("fused_allreduce_rmsnorm_lamport: call init_lamport_ar_rms "
                                     "first");
        lamport_->run<T>(stream, input, residual_inp, residual_out, output, weight, eps, tokens,
                         hidden, gemma_norm);
    }

    // Lamport all-reduce of input + other on the same inbox, see lamport_ar_add_gfx1250.
    template <typename T>
    void allreduce_add_lamport(hipStream_t stream,
                               const T* input,
                               const T* other,
                               T* output,
                               int64_t numel)
    {
        if(!lamport_)
            throw std::runtime_error("allreduce_add_lamport: call init_lamport_ar_rms first");
        lamport_->run_add<T>(stream, input, other, output, numel);
    }

    // Whether an all-gather of `bytes` a rank takes ag_ll_gfx1250: whole 8-byte
    // packets, a row's too along the last dim. The shape alone decides, so
    // every rank of the collective takes the same protocol.
    bool use_ll_ag(size_t bytes, size_t row_bytes) const
    {
        return use_ll(bytes) && bytes % 8 == 0 && row_bytes % 8 == 0;
    }

    // Whether ag_ll_gfx1250 may read this rank's input in place: clear of the
    // output it writes while still publishing the input, or a dim-0 gather into
    // the output's own slice (that slice is rewritten with the same bytes).
    bool ll_ag_reads_in_place(const void* input, const void* output, size_t bytes, size_t row_bytes) const
    {
        const auto in  = reinterpret_cast<uintptr_t>(input);
        const auto out = reinterpret_cast<uintptr_t>(output);
        return out >= in + bytes || in >= out + bytes * world_size_ ||
               (row_bytes == bytes && in == out + bytes * rank_);
    }

    // LL all-gather (see ag_ll_gfx1250); row_bytes == bytes gathers along dim 0.
    // use_ll_ag(bytes, row_bytes) holds, so both are whole 4-byte words, and a
    // message is at most kLLArMaxBytes: 349526 units, under kLLPackCapacity
    // lines a slot and under kMaxBlocks x 512 threads x 2 units. One unit a
    // thread: 256-thread blocks while kMaxBlocks of them cover the message,
    // then 512-thread blocks, then two units a thread past kMaxBlocks x 512.
    template <typename T>
    void allgather_ll(hipStream_t stream, const T* input, T* output, size_t bytes, size_t row_bytes)
    {
        ensure_ll_tables_();
        const uint32_t n32   = bytes >> 2;
        const size_t   units = ((size_t)n32 + 2) / 3;
        const int threads    = units <= (size_t)kMaxBlocks * 256 ? 256 : 512;
        const int unroll     = units <= (size_t)kMaxBlocks * threads ? 1 : 2;
        const int blocks     = std::max(1, std::min<int>(kMaxBlocks,
            (int)((units + (size_t)threads * unroll - 1) / ((size_t)threads * unroll))));

        auto peers = reinterpret_cast<uint8_t* const*>(d_ll_peers_);
        auto in    = reinterpret_cast<const uint8_t*>(input);
        auto out   = reinterpret_cast<uint8_t*>(output);
        // a rank-local choice of kernel, not of protocol: both move the same lines
        const bool aligned = (reinterpret_cast<uintptr_t>(input) & 3) == 0 &&
                             (reinterpret_cast<uintptr_t>(output) & 3) == 0;
#define LAUNCH_AG_LL12_U(NG, U)                                                                    \
        if(aligned)                                                                                \
            ag_ll_gfx1250<NG, U, true><<<blocks, threads, 0, stream>>>(                          \
                peers, out, in, n32, (uint32_t)(row_bytes >> 2), rank_, d_ll_block_flags_);        \
        else                                                                                       \
            ag_ll_gfx1250<NG, U, false><<<blocks, threads, 0, stream>>>(                         \
                peers, out, in, n32, (uint32_t)(row_bytes >> 2), rank_, d_ll_block_flags_)
#define LAUNCH_AG_LL12(NG)            \
        if(unroll == 1)               \
        {                             \
            LAUNCH_AG_LL12_U(NG, 1);  \
        }                             \
        else                          \
        {                             \
            LAUNCH_AG_LL12_U(NG, 2);  \
        }
        DISPATCH_NGPUS_1250(world_size_, LAUNCH_AG_LL12);
#undef LAUNCH_AG_LL12
#undef LAUNCH_AG_LL12_U
    }

    // Rows of 128 bf16 a source that dcp_merge fits in an LL slot (32 data lines
    // and one LSE line a row), or 0 where it does not run: LL off, or more than
    // kLLMaxRanks ranks. dcp_merge takes B x H / world_size rows at most this.
    int64_t dcp_merge_max_rows() const
    {
        return ll_enabled() && world_size_ <= kLLMaxRanks ? (int64_t)(kLLPackCapacity / 33) : 0;
    }

    // DCP merge (see dcp_merge_ll_gfx1250): o [B, H, 128] bf16 and lse [B, H]
    // fp32 in, out [B, H / world_size, 128] bf16 and out_lse (optional) out.
    // W = 256 threads a role. Blocks: one per 8 sender rows, up to one a CU; a
    // sender wave then takes 1, 2, 4 or 8 rows a round, so that one round covers
    // the message when it can.
    void dcp_merge(hipStream_t stream,
                   const void* o,
                   const float* lse,
                   const int* owned,
                   void* out,
                   float* out_lse,
                   int B,
                   int H,
                   int64_t o_sb,
                   int64_t o_sh,
                   int64_t l_sb,
                   int64_t l_sh)
    {
        ensure_ll_tables_();
        static const int cus = []() {
            int dev = 0, n = 0;
            if(hipGetDevice(&dev) != hipSuccess ||
               hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || n <= 0)
                n = 256;
            return n;
        }();
        constexpr int W   = 256;
        constexpr int wpb = W / 32;
        const int     HL  = H / world_size_;
        const int     rows   = B * (world_size_ - 1) * HL;
        const int     blocks = std::max(1, std::min({cus, kMaxBlocks, (rows + wpb - 1) / wpb}));
        const int     per    = (rows + blocks * wpb - 1) / (blocks * wpb);
        const int     U      = per <= 1 ? 1 : per <= 2 ? 2 : per <= 4 ? 4 : 8;

        auto peers = reinterpret_cast<uint8_t* const*>(d_ll_peers_);
        auto in    = reinterpret_cast<const opus::bf16_t*>(o);
        auto res   = reinterpret_cast<opus::bf16_t*>(out);
#define LAUNCH_DCP_U(NG, UU)                                                                      \
        dcp_merge_ll_gfx1250<NG, UU><<<blocks, 2 * W, 0, stream>>>(                              \
            peers, in, lse, owned, res, out_lse, B, HL, o_sb, o_sh, l_sb, l_sh, rank_, d_ll_block_flags_)
#define LAUNCH_DCP(NG)            \
        if(U == 1)                \
        {                         \
            LAUNCH_DCP_U(NG, 1);  \
        }                         \
        else if(U == 2)           \
        {                         \
            LAUNCH_DCP_U(NG, 2);  \
        }                         \
        else if(U == 4)           \
        {                         \
            LAUNCH_DCP_U(NG, 4);  \
        }                         \
        else                      \
        {                         \
            LAUNCH_DCP_U(NG, 8);  \
        }
        DISPATCH_NGPUS_1250(world_size_, LAUNCH_DCP);
#undef LAUNCH_DCP
#undef LAUNCH_DCP_U
    }

    // gfx1250: return raw device pointers (no hipIpc handles).
    std::vector<int64_t> get_graph_buffer_ptrs()
    {
        auto num_input_buffers  = graph_unreg_input_buffers_.size();
        auto num_output_buffers = graph_unreg_output_buffers_.size();
        auto num_buffers        = num_input_buffers + num_output_buffers;
        std::vector<int64_t> ptrs(num_buffers);
        for(size_t i = 0; i < num_input_buffers; i++)
            ptrs[i] = (int64_t)graph_unreg_input_buffers_[i];
        for(size_t i = 0; i < num_output_buffers; i++)
            ptrs[num_input_buffers + i] = (int64_t)graph_unreg_output_buffers_[i];
        return ptrs;
    }

    // IPC transport: the same buffers as hipIpc handles plus offsets. A handle
    // covers the whole allocation that contains the pointer, so the offset
    // into that allocation travels alongside.
    std::pair<std::vector<uint8_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta()
    {
        auto ptrs      = get_graph_buffer_ptrs();
        auto handle_sz = sizeof(hipIpcMemHandle_t);
        std::vector<uint8_t> handles(handle_sz * ptrs.size(), 0);
        std::vector<int64_t> offsets(ptrs.size());
        for(size_t i = 0; i < ptrs.size(); i++)
        {
            void* base_ptr;
            if(hipPointerGetAttribute(&base_ptr,
                                      HIP_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                      (hipDeviceptr_t)ptrs[i]) != hipSuccess)
                throw std::runtime_error("failed to get the allocation of a graph buffer");
            HIP_CALL(hipIpcGetMemHandle((hipIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
            offsets[i] = ptrs[i] - (int64_t)base_ptr;
        }
        return {handles, offsets};
    }

    void check_rank_data_capacity(size_t num = 1)
    {
        if(d_rank_data_base_ + num > d_rank_data_end_)
            throw std::runtime_error("Rank data buffer is overflowed by " +
                                     std::to_string(d_rank_data_base_ + num - d_rank_data_end_));
    }

    // gfx1250: receive direct device pointers instead of IPC handles.
    void register_input_buffer(const std::vector<int64_t>& all_ptrs, void* self)
    {
        check_rank_data_capacity();
        RankData data;
        for(int i = 0; i < world_size_; i++)
            data.ptrs[i] = (i != rank_) ? (void*)all_ptrs[i] : self;
        auto d_data = d_rank_data_base_++;
        HIP_CALL(hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
        input_buffer[self] = d_data;
    }

    void register_output_buffer(const std::vector<int64_t>& all_ptrs, void* self)
    {
        check_rank_data_capacity();
        RankData data;
        for(int i = 0; i < world_size_; i++)
            data.ptrs[i] = (i != rank_) ? (void*)all_ptrs[i] : self;
        auto d_data = d_rank_data_base_++;
        HIP_CALL(hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
        output_buffers_[self] = d_data;
    }

    // IPC transport overloads (ROCm >= 7.15): resolve peer handles to VAs.
    void register_input_buffer(const hipIpcMemHandle_t* handles,
                               const int64_t* offsets,
                               void* self)
    {
        check_rank_data_capacity();
        RankData data;
        for(int i = 0; i < world_size_; i++)
        {
            if(i != rank_)
            {
                char* handle = open_ipc_handle((void*)&handles[i]);
                handle += offsets[i];
                data.ptrs[i] = handle;
            }
            else
            {
                data.ptrs[i] = self;
            }
        }
        auto d_data = d_rank_data_base_++;
        HIP_CALL(hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
        input_buffer[self] = d_data;
    }

    void register_output_buffer(const hipIpcMemHandle_t* handles,
                                const int64_t* offsets,
                                void* self)
    {
        check_rank_data_capacity();
        RankData data;
        for(int i = 0; i < world_size_; i++)
        {
            if(i != rank_)
            {
                char* handle = open_ipc_handle((void*)&handles[i]);
                handle += offsets[i];
                data.ptrs[i] = handle;
            }
            else
            {
                data.ptrs[i] = self;
            }
        }
        auto d_data = d_rank_data_base_++;
        HIP_CALL(hipMemcpy(d_data, &data, sizeof(RankData), hipMemcpyHostToDevice));
        output_buffers_[self] = d_data;
    }

    RankData* get_buffer_RD(hipStream_t stream, void* input)
    {
        auto it = input_buffer.find(input);
        if(it != input_buffer.end())
            return it->second;
        hipStreamCaptureStatus status;
        HIP_CALL(hipStreamIsCapturing(stream, &status));
        if(status == hipStreamCaptureStatusActive)
        {
            auto ptrs = d_rank_data_base_ + graph_unreg_input_buffers_.size();
            graph_unreg_input_buffers_.push_back(input);
            return ptrs;
        }
        throw std::runtime_error("buffer address " +
                                 std::to_string(reinterpret_cast<uint64_t>(input)) +
                                 " is not registered!");
    }

    // gfx1250: receive direct device pointers per rank per buffer.
    // ptrs_per_rank[rank_j] points to a flat array of int64_t device pointers,
    // one per buffer (inputs first, then outputs), in the same order as
    // graph_unreg_input_buffers_ + graph_unreg_output_buffers_.
    void register_graph_buffers(const int64_t* const* ptrs_per_rank)
    {
        auto num_input_buffers  = graph_unreg_input_buffers_.size();
        auto num_output_buffers = graph_unreg_output_buffers_.size();
        auto total_buffers      = num_input_buffers + num_output_buffers;
        check_rank_data_capacity(total_buffers);
        std::vector<RankData> rank_data(total_buffers);
        for(size_t i = 0; i < num_input_buffers; i++)
        {
            auto self_ptr = graph_unreg_input_buffers_[i];
            auto& rd      = rank_data[i];
            for(int j = 0; j < world_size_; j++)
                rd.ptrs[j] = (j != rank_) ? (void*)ptrs_per_rank[j][i] : self_ptr;
        }
        for(size_t i = 0; i < num_output_buffers; i++)
        {
            auto self_ptr = graph_unreg_output_buffers_[i];
            auto& rd      = rank_data[num_input_buffers + i];
            for(int j = 0; j < world_size_; j++)
                rd.ptrs[j] = (j != rank_) ? (void*)ptrs_per_rank[j][num_input_buffers + i]
                                          : self_ptr;
        }
        HIP_CALL(hipMemcpy(d_rank_data_base_,
                           rank_data.data(),
                           sizeof(RankData) * total_buffers,
                           hipMemcpyHostToDevice));
        d_rank_data_base_ += total_buffers;
        graph_unreg_input_buffers_.clear();
        graph_unreg_output_buffers_.clear();
    }

    // IPC transport overload: handles_per_rank[j] and offsets_per_rank[j] are
    // rank j's get_graph_buffer_ipc_meta output. Each peer handle is opened
    // once (open_ipc_handle caches it) and resolved to a pointer here.
    void register_graph_buffers(const void* const* handles_per_rank,
                                const int64_t* const* offsets_per_rank)
    {
        auto total_buffers =
            graph_unreg_input_buffers_.size() + graph_unreg_output_buffers_.size();
        check_rank_data_capacity(total_buffers);
        std::vector<std::vector<int64_t>> ptrs(world_size_,
                                               std::vector<int64_t>(total_buffers));
        std::vector<const int64_t*> ptrs_per_rank(world_size_);
        for(int j = 0; j < world_size_; j++)
        {
            for(size_t i = 0; j != rank_ && i < total_buffers; i++)
            {
                auto* handle = (const hipIpcMemHandle_t*)handles_per_rank[j] + i;
                ptrs[j][i]   = (int64_t)(open_ipc_handle(handle) + offsets_per_rank[j][i]);
            }
            ptrs_per_rank[j] = ptrs[j].data();
        }
        register_graph_buffers(ptrs_per_rank.data());
    }

    // The allgather kernels index the output with int. allgather_scalar does it
    // per element; vec, naive and lastdim do it per 16-byte pack. Each launcher
    // takes the element count as int64_t and checks its own index range before
    // narrowing, so the vectorized kernels take any output of up to INT_MAX
    // packs (16 * INT_MAX bytes, about 32 GiB) whatever the dtype.
    // _resolve_custom_sizes in custom_all_reduce.py caps
    // AITER_CUSTOM_AG_MAX_SIZE at the same bound; keep the two in sync.
    template <typename T>
    int allgather_packs(int64_t numel, const char* who) const
    {
        constexpr int64_t d = 16 / sizeof(T);
        if(numel % d != 0)
            throw std::runtime_error(std::string(who) +
                                     " requires input length to be multiple of " +
                                     std::to_string(d));
        const int64_t packs = numel / d;
        if(packs * world_size_ > std::numeric_limits<int>::max())
            throw std::runtime_error(
                std::string(who) + ": output of " + std::to_string(packs * world_size_) +
                " 16-byte packs (numel " + std::to_string(numel) + " x world_size " +
                std::to_string(world_size_) + ") exceeds the int32 index range; "
                "route this size through should_custom_ag / RCCL instead");
        return static_cast<int>(packs);
    }

    template <typename T>
    void allgather_scalar(hipStream_t stream,
                          T* input,
                          T* output,
                          int64_t numel)
    {
        // Indexes the output per element, not per pack.
        if(numel * world_size_ > std::numeric_limits<int>::max())
            throw std::runtime_error(
                "gfx1250 allgather_scalar: output element count " +
                std::to_string(numel * world_size_) + " (numel " + std::to_string(numel) +
                " x world_size " + std::to_string(world_size_) +
                ") exceeds the int32 index range; route this size through "
                "should_custom_ag / RCCL instead");
        const int size       = static_cast<int>(numel);
        RankData* input_ptrs = get_buffer_RD(stream, input);

        constexpr int threads = 512;
        int blocks = std::min(kMaxBlocks,
                              (size + threads - 1) / threads);
#define LAUNCH_AG_SCALAR(NG) \
        ag_gfx1250_scalar<T, NG><<<blocks, threads, 0, stream>>>( \
            input_ptrs, sg_, self_sg_, output, rank_, size)
        DISPATCH_AG_NGPUS_1250(world_size_, LAUNCH_AG_SCALAR);
#undef LAUNCH_AG_SCALAR
    }

    template <typename T>
    void allgather_vec(hipStream_t stream,
                       T* input,
                       T* output,
                       int64_t numel)
    {
        const int size       = allgather_packs<T>(numel, "allgather_vec");
        RankData* input_ptrs = get_buffer_RD(stream, input);

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads - 1) / threads);
#define LAUNCH_AG_VEC(NG) \
        ag_gfx1250_naive_vec<T, NG><<<blocks, threads, 0, stream>>>( \
            input_ptrs, sg_, self_sg_, output, rank_, size)
        DISPATCH_AG_NGPUS_1250(world_size_, LAUNCH_AG_VEC);
#undef LAUNCH_AG_VEC
    }

    template <typename T>
    void allgather_naive(hipStream_t stream,
                         T* input,
                         T* output,
                         int64_t numel)
    {
        const int size       = allgather_packs<T>(numel, "allgather");
        RankData* input_ptrs = get_buffer_RD(stream, input);

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * 4 - 1) / (threads * 4));
#define LAUNCH_AG_NAIVE(NG) \
        ag_gfx1250_naive_unroll4<T, NG><<<blocks, threads, 0, stream>>>( \
            input_ptrs, sg_, self_sg_, output, rank_, size)
        DISPATCH_AG_NGPUS_1250(world_size_, LAUNCH_AG_NAIVE);
#undef LAUNCH_AG_NAIVE
    }

    template <typename T>
    void allgather_lastdim(hipStream_t stream,
                           T* input,
                           T* output,
                           int64_t numel,
                           int64_t last_dim_size)
    {
        auto d = 16 / sizeof(T);
        if(last_dim_size % d != 0)
            throw std::runtime_error(
                "allgather_lastdim requires input length and last_dim_size "
                "to be multiples of " + std::to_string(d));
        // The kernel takes the row length in elements and divides it by the
        // pack size itself.
        if(last_dim_size > std::numeric_limits<int>::max())
            throw std::runtime_error("allgather_lastdim: last_dim_size " +
                                     std::to_string(last_dim_size) +
                                     " exceeds the int32 range");
        const int size       = allgather_packs<T>(numel, "allgather_lastdim");
        RankData* input_ptrs = get_buffer_RD(stream, input);

        constexpr int threads = 512;
        constexpr int unroll  = 4;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * unroll - 1) / (threads * unroll));
#define LAUNCH_AG_LAST(NG) \
        ag_gfx1250_lastdim<T, NG><<<blocks, threads, 0, stream>>>( \
            input_ptrs, sg_, self_sg_, output, rank_, size, static_cast<int>(last_dim_size))
        DISPATCH_AG_NGPUS_1250(world_size_, LAUNCH_AG_LAST);
#undef LAUNCH_AG_LAST
    }

    template <typename T, int unroll>
    void p2p_bw_test(hipStream_t stream,
                     T* input,
                     T* output,
                     int size,
                     int threads,
                     int blocks)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "p2p_bw_test requires input length to be multiple of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        size /= d;

        p2p_bandwidth_test_kernel<T, unroll><<<blocks, threads, 0, stream>>>(
            input_ptrs, sg_, self_sg_, output, rank_, size);
    }

    template <typename T>
    void allreduce(hipStream_t stream,
                   T* input,
                   T* output,
                   int size,
                   bool use_new = true)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "custom allreduce requires input length to be multiple of " + std::to_string(d));

        // LL small-message fast lane: bytes <= kLLArMaxBytes route to the
        // flag-in-data kernel, which reads input locally and pushes to the
        // shared peer scratch — no registered peer-input table (_input_dp) is
        // needed, so branch before get_buffer_RD.
        const size_t bytes = (size_t)size * sizeof(T);
        if(use_2s_ll(bytes))
        {
            allreduce_2s_ll<T>(stream, input, output, size);
            return;
        }
        if(use_ll(bytes))
        {
            allreduce_ll<T>(stream, input, output, size);
            return;
        }

        // The kernel reads the peers' inputs and writes only this rank's output,
        // so a captured all-reduce registers its input alone.
        RankData* input_ptrs = get_buffer_RD(stream, input);

        size /= d;

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * 4 - 1) / (threads * 4));
#define LAUNCH_AR(NG) \
        ar_gfx1250_naive_unroll4<T, NG><<<blocks, threads, 0, stream>>>( \
            input_ptrs, nullptr, sg_, self_sg_, output, rank_, size)
        DISPATCH_NGPUS_1250(world_size_, LAUNCH_AR);
#undef LAUNCH_AR
    }

    // Fused allreduce + residual add + RMSNorm over [tokens, hidden]. OutT is T,
    // or fp8 for per-token quantized output with one scale per token.
    template <typename T, typename OutT>
    void fused_allreduce_rmsnorm(hipStream_t stream,
                                 T* input,
                                 const T* residual_inp,
                                 T* residual_out,
                                 OutT* output,
                                 const T* weight,
                                 float* scale_out,
                                 T* bf16_output,
                                 int tokens,
                                 int hidden,
                                 float eps,
                                 bool gemma_norm,
                                 bool use_1stage)
    {
        constexpr int pack_size = 16 / sizeof(T);
        if(hidden % pack_size != 0 || hidden / pack_size > 1024)
            throw std::runtime_error("gfx1250 fused allreduce rmsnorm: unsupported hidden size " +
                                     std::to_string(hidden));
        if(tokens == 0)
            return;
        RankData* input_ptrs = get_buffer_RD(stream, input);
        const int threads    = (hidden / pack_size + 63) / 64 * 64;
        const int blocks     = std::min(tokens, kMaxBlocks);
        const size_t bytes   = (size_t)tokens * hidden * sizeof(T);
        // At TP4+, two-stage halves or better the peer traffic of direct-read.
        // Its three single-block barriers do not amortize at TP2 or narrow rows.
        // Each rank's reduced shard must fit its staging region.
        const bool two_stage =
            !use_1stage && world_size_ >= 4 && hidden >= kArRmsTwoStageMinHidden &&
            hidden / pack_size % world_size_ == 0 &&
            bytes / world_size_ <= kArRmsShardBytes && bytes > kLLArMaxBytes;
#define LAUNCH_AR_RMS(NG)                                                                   \
        if(two_stage)                                                                       \
        {                                                                                   \
            start_sync_kernel<NG><<<1, 64, 0, stream>>>(sg_, self_sg_, rank_);              \
            ar_rmsnorm_reduce_scatter_gfx1250<T, NG><<<blocks, 512, 0, stream>>>(           \
                input_ptrs, self_sg_, rank_, tokens, hidden / pack_size);                   \
            end_sync_kernel<NG><<<1, 64, 0, stream>>>(sg_, self_sg_, rank_);                \
            if(gemma_norm)                                                                  \
                ar_rmsnorm_gfx1250<T, OutT, NG, true, true>                                 \
                    <<<blocks, threads, 0, stream>>>(                                       \
                        nullptr, sg_, self_sg_, rank_, residual_inp, residual_out, output,   \
                        weight, scale_out, bf16_output, tokens, hidden, eps);                \
            else                                                                            \
                ar_rmsnorm_gfx1250<T, OutT, NG, false, true>                                \
                    <<<blocks, threads, 0, stream>>>(                                       \
                        nullptr, sg_, self_sg_, rank_, residual_inp, residual_out, output,   \
                        weight, scale_out, bf16_output, tokens, hidden, eps);                \
            end_sync_kernel<NG><<<1, 64, 0, stream>>>(sg_, self_sg_, rank_);                \
        }                                                                                   \
        else if(gemma_norm)                                                                 \
            ar_rmsnorm_gfx1250<T, OutT, NG, true><<<blocks, threads, 0, stream>>>(          \
                input_ptrs, sg_, self_sg_, rank_, residual_inp, residual_out, output,       \
                weight, scale_out, bf16_output, tokens, hidden, eps);                       \
        else                                                                                \
            ar_rmsnorm_gfx1250<T, OutT, NG, false><<<blocks, threads, 0, stream>>>(         \
                input_ptrs, sg_, self_sg_, rank_, residual_inp, residual_out, output,       \
                weight, scale_out, bf16_output, tokens, hidden, eps)
        DISPATCH_NGPUS_1250(world_size_, LAUNCH_AR_RMS);
#undef LAUNCH_AR_RMS
    }

    template <typename T>
    void dispatchReduceScatter(hipStream_t stream, T* input, T* output,
                               int m, int n, int k,
                               ReduceScatterSplitDim split_dim)
    {
        RankData* ptrs          = get_buffer_RD(stream, input);
        constexpr int pack_size = 16 / sizeof(T);
        constexpr int kGridCap  = kMaxBlocks;

        switch(split_dim)
        {
        case ReduceScatterSplitDim::kFirst: {
            int range = k / (world_size_ * pack_size);
            dim3 block(512);
            dim3 grid(std::min(kGridCap, (range + 511) / 512));
#define LAUNCH_RS_FIRST(NG) \
            rs_gfx1250_split_first_dim<T, NG> \
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, range)
            DISPATCH_NGPUS_1250(world_size_, LAUNCH_RS_FIRST);
#undef LAUNCH_RS_FIRST
            break;
        }
        case ReduceScatterSplitDim::kLast: {
            bool vec  = (k % (world_size_ * pack_size) == 0);
            int size  = vec ? (n * k) / (world_size_ * pack_size)
                            : (n * k) / world_size_;
            dim3 block(256);
            dim3 grid(std::min(kGridCap, (size + 255) / 256));
#define LAUNCH_LAST_1250(NG)                                                    \
    do {                                                                        \
        if(vec)                                                                 \
            rs_gfx1250_split_lastdim<T, NG>                                     \
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output,       \
                                             rank_, n, k);                      \
        else                                                                    \
            rs_gfx1250_split_lastdim_naive<T, NG>                               \
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output,       \
                                             rank_, n, k);                      \
    } while(0)
            DISPATCH_NGPUS_1250(world_size_, LAUNCH_LAST_1250);
#undef LAUNCH_LAST_1250
            break;
        }
        case ReduceScatterSplitDim::kMid: {
            bool vec  = (k % pack_size == 0);
            int size  = vec ? (m * n * k) / (world_size_ * pack_size)
                            : (m * n * k) / world_size_;
            dim3 block(256);
            dim3 grid(std::min(kGridCap, (size + 255) / 256));
#define LAUNCH_MID_1250(NG)                                                     \
    do {                                                                        \
        if(vec)                                                                 \
            rs_gfx1250_split_middim<T, NG>                                      \
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output,       \
                                             rank_, m, n, k);                   \
        else                                                                    \
            rs_gfx1250_split_middim_naive<T, NG>                                \
                <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output,       \
                                             rank_, m, n, k);                   \
    } while(0)
            DISPATCH_NGPUS_1250(world_size_, LAUNCH_MID_1250);
#undef LAUNCH_MID_1250
            break;
        }
        default: printf("reduce_scatter split_dim error!\n");
        }
    }
};

} // namespace aiter
