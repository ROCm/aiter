// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// IFOE cross-node custom all-reduce kernels for gfx1250.
//
// This reuses aiter's 2-stage custom all-reduce structure (reduce-scatter +
// allgather, guarded by start_sync / end_sync per-block barriers).  The only
// difference from the intra-node IPC path is that peer buffers are shared with
// HIP *fabric* handles (see custom_all_reduce_ifoe.cu), which are node
// independent -- so the identical kernel runs cross-node over the IFOE fabric.
//
// Two data paths are provided:
//   * allreduce2_opt  -- fp32, MLP-unrolled (U float4 in flight per thread).
//   * allreduce2_bf16 -- bf16 on the wire (half the fabric bytes, fp32
//                        accumulate); lossy.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace aiter {
namespace ifoe {

constexpr int kMaxBlocks = 304; // >= CU count for full occupancy

struct Signal
{
    alignas(128) uint32_t start[kMaxBlocks][8];
    alignas(128) uint32_t end[kMaxBlocks][8];
    alignas(128) uint32_t _flag[kMaxBlocks];
};
struct __align__(16) RankData { const void* ptrs[8]; };
struct __align__(16) RankSignals { Signal* signals[8]; };

#define IFOE_DINLINE __device__ __forceinline__

// Per-block cross-rank barrier: publish own arrival to every peer, spin on all
// peers' arrival for this block.  System scope so the atomics traverse the
// fabric; the peer data ordering is provided by the surrounding kernel launch
// boundaries (inputs / bf16 casts are produced by prior kernels).
template <int ngpus>
IFOE_DINLINE void start_sync(const RankSignals& sg, Signal* self_sg, int rank)
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

template <int ngpus>
IFOE_DINLINE void end_sync(const RankSignals& sg, Signal* self_sg, int rank)
{
    __syncthreads();
    uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
    if(threadIdx.x < ngpus)
    {
        __scoped_atomic_store_n(&sg.signals[threadIdx.x]->end[blockIdx.x][rank],
                                flag,
                                __ATOMIC_RELEASE,
                                __MEMORY_SCOPE_SYSTEM);
        while(__scoped_atomic_load_n(&self_sg->end[blockIdx.x][threadIdx.x],
                                     __ATOMIC_ACQUIRE,
                                     __MEMORY_SCOPE_DEVICE) < flag)
            ;
    }
    __syncthreads();
    if(threadIdx.x == 0)
        self_sg->_flag[blockIdx.x] = flag;
}

template <typename P>
IFOE_DINLINE P* tmp_buf(Signal* s)
{
    return (P*)(s + 1); // reduced-shard scratch lives right after the Signal
}

// ---- fp32 MLP-optimized path -------------------------------------------------
// size is the float4 count of the (whole) tensor.
template <int ngpus, int U>
__global__ void __launch_bounds__(512, 1)
    allreduce2_opt(RankData* in_dp, RankSignals sg, Signal* self_sg, float* result, int rank, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    int part = size / ngpus, start = rank * part, end = (rank == ngpus - 1) ? size : start + part,
        largest = part + size % ngpus;
    const float4* ptrs[ngpus];
    float4* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int t   = (rank + i) % ngpus;
        ptrs[i] = (const float4*)in_dp->ptrs[t];
        tmps[i] = tmp_buf<float4>(sg.signals[t]);
    }
    start_sync<ngpus>(sg, self_sg, rank);
    // reduce-scatter: U float4 in flight per thread across the whole peer set
    for(int i0 = start + tid; i0 < end; i0 += stride * U)
    {
        float4 acc[U];
#pragma unroll
        for(int u = 0; u < U; u++)
        {
            int idx = i0 + u * stride;
            acc[u]  = idx < end ? ptrs[0][idx] : float4{0, 0, 0, 0};
        }
#pragma unroll
        for(int p = 1; p < ngpus; p++)
        {
#pragma unroll
            for(int u = 0; u < U; u++)
            {
                int idx = i0 + u * stride;
                if(idx < end)
                {
                    float4 v = ptrs[p][idx];
                    acc[u].x += v.x;
                    acc[u].y += v.y;
                    acc[u].z += v.z;
                    acc[u].w += v.w;
                }
            }
        }
#pragma unroll
        for(int u = 0; u < U; u++)
        {
            int idx = i0 + u * stride;
            if(idx < end)
                tmps[0][idx - start] = acc[u];
        }
    }
    end_sync<ngpus>(sg, self_sg, rank);
    // allgather: each peer's reduced shard copied contiguously
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int gr = (rank + i) % ngpus, cnt = (gr == ngpus - 1) ? largest : part;
        float4* dst       = (float4*)result + (size_t)gr * part;
        const float4* src = tmps[i];
        for(int i0 = tid; i0 < cnt; i0 += stride * U)
        {
#pragma unroll
            for(int u = 0; u < U; u++)
            {
                int idx = i0 + u * stride;
                if(idx < cnt)
                    dst[idx] = src[idx];
            }
        }
    }
}

// ---- bf16 on-wire path -------------------------------------------------------
// One "octet" = 8 fp32 elems = 2 float4 = a bf16x8 (16B) on the wire.
using bf16 = __hip_bfloat16;
struct __align__(16) bf16x8 { bf16 v[8]; };

IFOE_DINLINE bf16x8 cvt_f2b(float4 a, float4 b)
{
    bf16x8 o;
    o.v[0] = __float2bfloat16(a.x);
    o.v[1] = __float2bfloat16(a.y);
    o.v[2] = __float2bfloat16(a.z);
    o.v[3] = __float2bfloat16(a.w);
    o.v[4] = __float2bfloat16(b.x);
    o.v[5] = __float2bfloat16(b.y);
    o.v[6] = __float2bfloat16(b.z);
    o.v[7] = __float2bfloat16(b.w);
    return o;
}
IFOE_DINLINE void cvt_b2f(bf16x8 o, float4& a, float4& b)
{
    a.x = __bfloat162float(o.v[0]);
    a.y = __bfloat162float(o.v[1]);
    a.z = __bfloat162float(o.v[2]);
    a.w = __bfloat162float(o.v[3]);
    b.x = __bfloat162float(o.v[4]);
    b.y = __bfloat162float(o.v[5]);
    b.z = __bfloat162float(o.v[6]);
    b.w = __bfloat162float(o.v[7]);
}

// Cast own fp32 input -> own bf16 wire buffer.  Separate kernel so its writes
// are grid-globally visible (kernel boundary) before the reduce reads them
// cross-block.  size8 = element count / 8.
__global__ void cast_bf16(const float4* own_in, bf16x8* my_bf, int size8)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for(int o = tid; o < size8; o += stride)
        my_bf[o] = cvt_f2b(own_in[2 * o], own_in[2 * o + 1]);
}

// size8 = element count / 8 (octets).  in_bf_dp: peer bf16 buffers (already cast).
template <int ngpus, int U>
__global__ void __launch_bounds__(512, 1) allreduce2_bf16(
    RankData* in_bf_dp, RankSignals sg, Signal* self_sg, float4* result, int rank, int size8)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    int part = size8 / ngpus, start = rank * part, end = (rank == ngpus - 1) ? size8 : start + part,
        largest = part + size8 % ngpus;
    const bf16x8* ptrs[ngpus];
    bf16x8* tmps[ngpus];
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int t   = (rank + i) % ngpus;
        ptrs[i] = (const bf16x8*)in_bf_dp->ptrs[t];
        tmps[i] = tmp_buf<bf16x8>(sg.signals[t]);
    }
    start_sync<ngpus>(sg, self_sg, rank);
    // reduce-scatter: bf16 reads, fp32 accumulate, U-way MLP
    for(int o0 = start + tid; o0 < end; o0 += stride * U)
    {
        float4 accA[U], accB[U];
#pragma unroll
        for(int u = 0; u < U; u++)
        {
            int o = o0 + u * stride;
            if(o < end)
                cvt_b2f(ptrs[0][o], accA[u], accB[u]);
        }
#pragma unroll
        for(int p = 1; p < ngpus; p++)
        {
#pragma unroll
            for(int u = 0; u < U; u++)
            {
                int o = o0 + u * stride;
                if(o < end)
                {
                    float4 va, vb;
                    cvt_b2f(ptrs[p][o], va, vb);
                    accA[u].x += va.x;
                    accA[u].y += va.y;
                    accA[u].z += va.z;
                    accA[u].w += va.w;
                    accB[u].x += vb.x;
                    accB[u].y += vb.y;
                    accB[u].z += vb.z;
                    accB[u].w += vb.w;
                }
            }
        }
#pragma unroll
        for(int u = 0; u < U; u++)
        {
            int o = o0 + u * stride;
            if(o < end)
                tmps[0][o - start] = cvt_f2b(accA[u], accB[u]);
        }
    }
    end_sync<ngpus>(sg, self_sg, rank);
    // allgather: read peers' bf16 reduced shards, cast to fp32 result
#pragma unroll
    for(int i = 0; i < ngpus; i++)
    {
        int gr = (rank + i) % ngpus, cnt = (gr == ngpus - 1) ? largest : part;
        for(int o = tid; o < cnt; o += stride)
        {
            float4 a, b;
            cvt_b2f(tmps[i][o], a, b);
            result[2 * ((size_t)gr * part + o)]     = a;
            result[2 * ((size_t)gr * part + o) + 1] = b;
        }
    }
}

} // namespace ifoe
} // namespace aiter
