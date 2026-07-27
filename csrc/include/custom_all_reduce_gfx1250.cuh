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
#include <map>
#include <string>
#include <unordered_map>
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

namespace aiter {

// ---------------------------------------------------------------------------
// Constants & data structures
// ---------------------------------------------------------------------------
constexpr int kMaxBlocks = 512;

struct Signal
{
    alignas(128) uint32_t start[kMaxBlocks][8];
    alignas(128) uint32_t end[kMaxBlocks][8];
    alignas(128) uint32_t _flag[kMaxBlocks];
};

struct __align__(16) RankData
{
    const void* ptrs[8];
};

struct __align__(16) RankSignals
{
    Signal* signals[8];
};

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

// gfx1250 AR supports world_size <= 4; size scratch for the max.
constexpr int    kLLMaxRanks       = 4;
// Route to LL when bytes <= this (matches RCCL DDA_ALLREDUCE_LL_THRESHOLD).
constexpr size_t kLLArMaxBytes     = 131072;            // 128 KiB
// Hard per-message payload cap (one slot). Comfortably above the routing
// threshold so all dtypes at <=128 KiB fit.
constexpr size_t kLLScratchCapBytes = 262144;           // 256 KiB
// Per-rank staging slot capacity, in 8-byte packets.
constexpr size_t kLLPackCapacity   = kLLScratchCapBytes / 8;  // 32768

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

// Per-rank scratch footprint: 2 banks * kLLMaxRanks slots * slotStride * 16B.
// 4 MiB at the 256 KiB / 4-rank defaults. Uniform across ranks so the
// double-buffered slot layout is identical everywhere.
constexpr size_t llScratchBytes()
{
    return (size_t)2 * kLLMaxRanks * kLLPackCapacity * sizeof(LLPackedMsg);
}

// Byte offset of the LL scratch within the shared meta buffer: right after the
// Signal struct, 128-byte aligned. The meta buffer (see meta_size()) is sized
// kLLScratchOffset + llScratchBytes() and zero-initialized, which doubles as the
// LL flag reset (flag 0 == cleared line).
constexpr size_t kLLScratchOffset =
    ((sizeof(Signal) + 127) / 128) * 128;

// gfx1250 128-bit global load/store at *system* scope ("" == system). A single
// 16B transaction keeps a line's data and flags atomic w.r.t. a peer reader.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx1250__) &&                   \
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

// LL flat all-reduce. 1D grid over 8-byte packets.
//
// Phase 1 (publish): rank writes its full sendbuff into every peer's scratch at
// slot[rank], as LL lines carrying the epoch flag.
// Phase 2 (reduce): rank polls its own scratch slots for the other ranks
// (waiting on the flag), sums them with its own sendbuff (fp32 accumulation),
// and writes recvbuff. Self is read directly from sendbuff.
//
// Graph-safe device epoch: each block owns a persistent device flag
// block_flags[blockIdx.x] that the kernel itself bumps (mirrors Signal::_flag).
// A packet is always published+consumed by the same (blockIdx, threadIdx) on
// every rank, so only the *same block across ranks* must agree on the flag; each
// block runs once per launch, keeping per-block flags in lockstep. Scratch is
// double-buffered (bank = epoch & 1); the per-epoch flag disambiguates stale
// lines, so no clearing is needed and it survives CUDA-graph replay.
template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 2) ar_ll_gfx1250(
    T* const* __restrict__ peer_scratch, // ngpus scratch bases (device table)
    T* __restrict__ recvbuff,
    const T* __restrict__ sendbuff,
    size_t nPk,          // number of 8-byte packets = bytes / 8
    int rank,
    uint32_t* __restrict__ block_flags) // device array[gridDim.x], persisted
{
    constexpr int LP = 8 / sizeof(T); // elements per 8-byte payload
    using PL         = typename opus::vector_t<T, LP>;
    using AL         = typename opus::vector_t<opus::fp32_t, LP>;
    constexpr size_t slot = kLLPackCapacity;

    __shared__ uint32_t s_flag;
    if(threadIdx.x == 0)
    {
        uint32_t f = block_flags[blockIdx.x] + 1u;
        if(f == 0u)
            f = 1u; // flag is never 0 (0 == cleared scratch)
        s_flag = f;
    }
    __syncthreads();
    const uint32_t flag        = s_flag;
    const size_t   bankOffPkts = (size_t)(flag & 1u) * (size_t)ngpus * slot;

    const size_t gtid   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    const uint32_t* in = reinterpret_cast<const uint32_t*>(sendbuff);

    // Phase 1: publish my payload into every peer's slot[rank].
    for(size_t pk = gtid; pk < nPk; pk += stride)
    {
        const uint32_t d0 = in[2 * pk];
        const uint32_t d1 = in[2 * pk + 1];
#pragma unroll
        for(int r = 1; r < ngpus; ++r)
        {
            int          peer = (rank + r) % ngpus;
            LLPackedMsg* dst  = reinterpret_cast<LLPackedMsg*>(peer_scratch[peer]) +
                               bankOffPkts + (size_t)rank * slot;
            ll_store_b128(reinterpret_cast<uint32_t*>(&dst[pk]), d0, flag, d1, flag);
        }
    }

    // Phase 2: poll my slots for the other ranks, reduce with my own data.
    LLPackedMsg* myBase =
        reinterpret_cast<LLPackedMsg*>(peer_scratch[rank]) + bankOffPkts;
    for(size_t pk = gtid; pk < nPk; pk += stride)
    {
        PL selfv = *reinterpret_cast<const PL*>(&in[2 * pk]);
        AL acc;
#pragma unroll
        for(int j = 0; j < LP; ++j)
            acc[j] = upcast_s(selfv[j]);

        for(int r = 1; r < ngpus; ++r)
        {
            int                   peer = (rank + r) % ngpus;
            volatile LLPackedMsg* src  = myBase + (size_t)peer * slot;
            uint32_t d0, f0, d1, f1;
            do
            {
                ll_load_b128(
                    reinterpret_cast<const uint32_t*>(const_cast<LLPackedMsg*>(&src[pk])),
                    d0, f0, d1, f1);
            } while(f0 != flag || f1 != flag);

            const uint32_t w[2] = {d0, d1};
            PL             pv   = *reinterpret_cast<const PL*>(w);
#pragma unroll
            for(int j = 0; j < LP; ++j)
                acc[j] += upcast_s(pv[j]);
        }

        PL ov;
#pragma unroll
        for(int j = 0; j < LP; ++j)
            ov[j] = downcast_s<T>(acc[j]);
        reinterpret_cast<PL*>(recvbuff)[pk] = ov;
    }

    if(threadIdx.x == 0)
        block_flags[blockIdx.x] = flag;
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
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const P*)_input_dp->ptrs[target];
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


template <typename T, int ngpus>
__global__ void __launch_bounds__(256, 2) ag_gfx1250_warpsplit_unroll4(
    RankData* _input_dp,
    RankSignals sg,
    Signal* self_sg,
    T* __restrict__ result,
    int rank,
    int size)
{
    constexpr int pack_size = 16 / sizeof(T);
    constexpr int unroll    = 4;
    constexpr int tnum_gpu = 256 / ngpus;
    using P                 = typename opus::vector_t<T, pack_size>;
    int warp_id = threadIdx.x / tnum_gpu;
    int lane_id = threadIdx.x % tnum_gpu;
    int index    = blockIdx.x * tnum_gpu * unroll + lane_id;
    int stride  = blockDim.x * tnum_gpu * unroll;
    const P* ptrs[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        ptrs[i] = (const P*)_input_dp->ptrs[i];
    }
    start_sync<ngpus>(sg, self_sg, rank);
    for (int idx = index; idx + tnum_gpu * (unroll - 1) < size; idx += stride)
    {
#pragma unroll
      for (int i = 0; i < unroll; ++i)
      {
        P* rslt_addr = reinterpret_cast<P*>(result) + warp_id * size + idx + tnum_gpu * i;
        *rslt_addr = ptrs[warp_id][idx + i * tnum_gpu];
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
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const P*)_dp->ptrs[target];
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
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const T*)_dp->ptrs[target];
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
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const P*)_dp->ptrs[target];
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
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const T*)_dp->ptrs[target];
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
        int target = (rank + i) % ngpus;
        ptrs[i]    = (const P*)_dp->ptrs[target];
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
__global__ void start_sync_latency(RankSignals sg, Signal* self_sg, int rank)
{
  start_sync<ngpus>(sg, self_sg, rank);
}

template <int ngpus>
__global__ void end_sync_latency(RankSignals sg, Signal* self_sg, int rank)
{
  end_sync<ngpus>(sg, self_sg, rank);
}

template <int ngpus>
__global__ void two_sync_latency(RankSignals sg, Signal* self_sg, int rank)
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
            ensure_ll_tables_();
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
            ensure_ll_tables_();
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
    }

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

    // LL small-message all-reduce. Reads `input` locally, pushes to peer scratch,
    // reduces into `output`. numel is the element count; bytes must be a multiple
    // of 16 and <= kLLScratchCapBytes (guaranteed by the routing threshold).
    template <typename T>
    void allreduce_ll(hipStream_t stream, const T* input, T* output, int numel)
    {
        ensure_ll_tables_();
        const size_t bytes = (size_t)numel * sizeof(T);
        const size_t nPk   = bytes >> 3; // 8 payload bytes per packet

        constexpr int threads = 256;
        int blocks = std::min<int>(kMaxBlocks, (int)((nPk + threads - 1) / threads));
        if(blocks < 1)
            blocks = 1;

        T* const* peers = reinterpret_cast<T* const*>(d_ll_peers_);
        if(world_size_ == 2)
            ar_ll_gfx1250<T, 2><<<blocks, threads, 0, stream>>>(
                peers, output, input, nPk, rank_, d_ll_block_flags_);
        else
            ar_ll_gfx1250<T, 4><<<blocks, threads, 0, stream>>>(
                peers, output, input, nPk, rank_, d_ll_block_flags_);
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

    RankData* get_output_buffer_RD(hipStream_t stream, void* output)
    {
        auto it = output_buffers_.find(output);
        if(it != output_buffers_.end())
            return it->second;
        hipStreamCaptureStatus status;
        HIP_CALL(hipStreamIsCapturing(stream, &status));
        if(status == hipStreamCaptureStatusActive)
        {
            auto ptrs = d_rank_data_base_ + graph_unreg_input_buffers_.size() +
                        graph_unreg_output_buffers_.size();
            graph_unreg_output_buffers_.push_back(output);
            return ptrs;
        }
        throw std::runtime_error("output buffer address " +
                                 std::to_string(reinterpret_cast<uint64_t>(output)) +
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
            output_buffers_[self_ptr] = d_rank_data_base_ + num_input_buffers + i;
        }
        HIP_CALL(hipMemcpy(d_rank_data_base_,
                           rank_data.data(),
                           sizeof(RankData) * total_buffers,
                           hipMemcpyHostToDevice));
        d_rank_data_base_ += total_buffers;
        graph_unreg_input_buffers_.clear();
        graph_unreg_output_buffers_.clear();
    }

    template <typename T>
    void allgather_scalar(hipStream_t stream,
                          T* input,
                          T* output,
                          int size)
    {
        RankData* input_ptrs = get_buffer_RD(stream, input);

        constexpr int threads = 512;
        int blocks = std::min(kMaxBlocks,
                              (size + threads - 1) / threads);
        if(world_size_ == 2)
            ag_gfx1250_scalar<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
        else
            ag_gfx1250_scalar<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
    }

    template <typename T>
    void allgather_vec(hipStream_t stream,
                       T* input,
                       T* output,
                       int size)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "allgather_vec requires input length to be multiple of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        size /= d;

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads - 1) / threads);
        if(world_size_ == 2)
            ag_gfx1250_naive_vec<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
        else
            ag_gfx1250_naive_vec<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
    }

    template <typename T>
    void allgather_naive(hipStream_t stream,
                         T* input,
                         T* output,
                         int size)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "allgather requires input length to be multiple of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        size /= d;

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * 4 - 1) / (threads * 4));
        if(world_size_ == 2)
            ag_gfx1250_naive_unroll4<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
        else
            ag_gfx1250_naive_unroll4<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
    }

    template <typename T>
    void allgather_warpsplit(hipStream_t stream,
                             T* input,
                             T* output,
                             int size)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "allgather requires input length to be multiple of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        size /= d;

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * 4 - 1) / (threads * 4));
        if(world_size_ == 2)
            ag_gfx1250_warpsplit_unroll4<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
        else
            ag_gfx1250_warpsplit_unroll4<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size);
    }

    template <typename T>
    void allgather_lastdim(hipStream_t stream,
                           T* input,
                           T* output,
                           int size,
                           int last_dim_size)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0 || last_dim_size % d != 0)
            throw std::runtime_error(
                "allgather_lastdim requires input length and last_dim_size "
                "to be multiples of " + std::to_string(d));

        RankData* input_ptrs = get_buffer_RD(stream, input);
        size /= d;

        constexpr int threads = 512;
        constexpr int unroll  = 4;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * unroll - 1) / (threads * unroll));
        if(world_size_ == 2)
            ag_gfx1250_lastdim<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size, last_dim_size);
        else
            ag_gfx1250_lastdim<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, sg_, self_sg_, output, rank_, size, last_dim_size);
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
                   bool use_new                 = true,
                   bool is_broadcast_reg_outptr = false)
    {
        auto d = 16 / sizeof(T);
        if(size % d != 0)
            throw std::runtime_error(
                "custom allreduce requires input length to be multiple of " + std::to_string(d));

        // LL small-message fast lane: bytes <= kLLArMaxBytes (128 KiB) route to
        // the flag-in-data kernel, which reads input locally and pushes to the
        // shared peer scratch — no registered peer-input table (_input_dp) is
        // needed, so branch before get_buffer_RD.
        const size_t bytes = (size_t)size * sizeof(T);
        if(ll_enabled() && bytes <= kLLArMaxBytes &&
           (world_size_ == 2 || world_size_ == 4))
        {
            allreduce_ll<T>(stream, input, output, size);
            return;
        }

        RankData* input_ptrs = get_buffer_RD(stream, input);
        RankData* output_ptrs = nullptr;
        if(is_broadcast_reg_outptr)
            output_ptrs = get_output_buffer_RD(stream, output);

        size /= d;

        if(world_size_ > 4)
            throw std::runtime_error(
                "gfx1250 custom allreduce only supports world_size <= 4, got " +
                std::to_string(world_size_));

        constexpr int threads = 256;
        int blocks = std::min(kMaxBlocks,
                              (size + threads * 4 - 1) / (threads * 4));
        if(world_size_ == 2)
        {
            ar_gfx1250_naive_unroll4<T, 2><<<blocks, threads, 0, stream>>>(
                input_ptrs, output_ptrs, sg_, self_sg_, output, rank_, size);
        }
        else
        {
            ar_gfx1250_naive_unroll4<T, 4><<<blocks, threads, 0, stream>>>(
                input_ptrs, output_ptrs, sg_, self_sg_, output, rank_, size);
        }
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
            if(world_size_ == 2)
                rs_gfx1250_split_first_dim<T, 2>
                    <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, range);
            else
                rs_gfx1250_split_first_dim<T, 4>
                    <<<grid, block, 0, stream>>>(ptrs, sg_, self_sg_, output, rank_, range);
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
            if(world_size_ == 2) { LAUNCH_LAST_1250(2); }
            else                 { LAUNCH_LAST_1250(4); }
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
            if(world_size_ == 2) { LAUNCH_MID_1250(2); }
            else                 { LAUNCH_MID_1250(4); }
#undef LAUNCH_MID_1250
            break;
        }
        default: printf("reduce_scatter split_dim error!\n");
        }
    }
};

} // namespace aiter
