// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "moe_nvfp4_common.h"

// ---------------------------------------------------------------- fused prep
// One launch replaces moe_sorting + activation cast: block 0 sorts the routing (one
// workgroup), blocks 1.. zero the buffer of the stage2 atomics (out_vecs 16 B vectors: bf16
// output, the f32 workspace of f16 output; 0 for Cfg::reduce) and cast X (bf16) to f16
// (x_f16 null: no cast, for f16 X or _inbf16 stage-1 kernels). Sorted layout matches the gemm
// kernels: per expert, ceil(count / BM) blocks of BM entries (token | slot << 24, padding
// token = T).
// Cast blocks: all SORT_THREADS lanes, one 16 B vector per lane (zero, 8 X elements); the
// f16 X store is non-temporal (gives stage1 2-4% of layer time back at M384-768).
constexpr int SORT_THREADS     = 1024;
constexpr int SORT_PER_THREAD  = 4; // T * topk <= 4096
constexpr int SORT_MAX_EXPERTS = 512;
constexpr int SORT_MAX_BLOCKS  = SORT_THREADS * SORT_PER_THREAD / 16 + SORT_MAX_EXPERTS;
static_assert(SORT_MAX_BLOCKS < (1 << 15), "packed block counts of the sort scan");

struct PrepParams
{
    const uint16_t* x;
    uint8_t* x_f16;
    uint8_t* out;
    const int* ids;
    const float* w;
    int* sorted_ids;
    float* sorted_w;
    int* sorted_eids;
    int* num_valid;
    int T, H, topk, E, BM, nblk, out_vecs;
};

__global__ __launch_bounds__(SORT_THREADS) void nvfp4_moe_sort_cast(PrepParams p)
{
    const int tid = threadIdx.x;
    if(blockIdx.x > 0)
    {
        const int vec = (blockIdx.x - 1) * SORT_THREADS + tid;
        if(vec < p.out_vecs)
            ((v4i*)p.out)[vec] = v4i{0, 0, 0, 0};
        if(p.x_f16 && vec < p.T * p.H / 8)
            __builtin_nontemporal_store(cast_bf16x8_f16(((const v4i*)p.x)[vec]),
                                        (v4i*)p.x_f16 + vec);
        return;
    }

    __shared__ int count[SORT_MAX_EXPERTS];
    __shared__ int first_block[SORT_MAX_EXPERTS + 1]; // first block of each expert
    __shared__ int block_expert[SORT_MAX_BLOCKS];     // expert of each block
    const int n_rows = p.T * p.topk;
    const int BM     = p.BM;
    // all global loads first: one memory latency instead of SORT_PER_THREAD dependent ones
    int expert[SORT_PER_THREAD], rank[SORT_PER_THREAD];
    float weight[SORT_PER_THREAD];
#pragma unroll
    for(int i = 0; i < SORT_PER_THREAD; i++)
    {
        const int idx = tid + SORT_THREADS * i;
        expert[i]     = idx < n_rows ? p.ids[idx] : 0;
        weight[i]     = idx < n_rows ? p.w[idx] : 0.0f;
    }
    for(int e = tid; e < p.E; e += SORT_THREADS)
        count[e] = 0;
    __syncthreads();
#pragma unroll
    for(int i = 0; i < SORT_PER_THREAD; i++)
        rank[i] = tid + SORT_THREADS * i < n_rows ? atomicAdd(&count[expert[i]], 1) : 0;
    __syncthreads();
    if(tid < 64)
    {
        // Experts with more than one block go first (stable), so the long multi-block experts
        // (e.g. a fused shared expert) do not form the grid tail. One scan of both block
        // counts: multi-block experts' blocks in bits 16.., single-block experts' in bits
        // 0..15 (each total at most SORT_MAX_BLOCKS < 2^16).
        const int experts_per_lane = (p.E + 63) / 64;
        const int expert_begin     = tid * experts_per_lane;
        auto n_blocks              = [&](int e) { return (count[e] + BM - 1) / BM; };
        int blocks                 = 0;
        for(int k = 0; k < experts_per_lane; k++)
            if(const int e = expert_begin + k; e < p.E)
                blocks += n_blocks(e) > 1 ? n_blocks(e) << 16 : n_blocks(e);
        int scan = blocks;
#pragma unroll
        for(int d = 1; d < 64; d <<= 1)
        {
            const int lane_scan = __shfl_up(scan, d);
            scan += tid >= d ? lane_scan : 0;
        }
        const int total       = __shfl(scan, 63);
        const int total_multi = total >> 16;
        int next_multi        = (scan - blocks) >> 16;
        int next_single       = total_multi + ((scan - blocks) & 0xffff);
        for(int k = 0; k < experts_per_lane; k++)
            if(const int e = expert_begin + k; e < p.E)
            {
                int& next      = n_blocks(e) > 1 ? next_multi : next_single;
                first_block[e] = next;
                next += n_blocks(e);
            }
        if(tid == 63)
        {
            first_block[p.E] = total_multi + (total & 0xffff);
            p.num_valid[0]   = first_block[p.E] * BM;
        }
    }
    __syncthreads();
    for(int e = tid; e < p.E; e += SORT_THREADS)
        for(int b = first_block[e]; b < first_block[e] + (count[e] + BM - 1) / BM; b++)
            block_expert[b] = e;
    __syncthreads();
    const int n_valid_blocks = first_block[p.E];
    for(int b = tid; b < p.nblk; b += SORT_THREADS)
        p.sorted_eids[b] = b < n_valid_blocks ? block_expert[b] : 0;
    for(int pos = tid; pos < n_valid_blocks * BM; pos += SORT_THREADS)
    {
        const int e = block_expert[pos / BM];
        if(pos - first_block[e] * BM >= count[e])
        {
            p.sorted_ids[pos] = p.T;
            p.sorted_w[pos]   = 0.0f;
        }
    }
#pragma unroll
    for(int i = 0; i < SORT_PER_THREAD; i++)
    {
        const int idx = tid + SORT_THREADS * i;
        if(idx < n_rows)
        {
            const int pos     = first_block[expert[i]] * BM + rank[i];
            p.sorted_ids[pos] = ((idx % p.topk) << 24) | (idx / p.topk);
            p.sorted_w[pos]   = weight[i];
        }
    }
}

extern "C" int moe_nvfp4_kernels_sort(const PrepParams* hp, hipStream_t stream)
{
    const PrepParams p = *hp;
    if(p.T * p.topk > SORT_THREADS * SORT_PER_THREAD || p.E > SORT_MAX_EXPERTS)
        return -2;
    // one 16 B vector (8 X elements) per cast lane
    const int cast_vecs     = p.x_f16 ? p.T * p.H / 8 : 0;
    const int n_vecs        = p.out_vecs > cast_vecs ? p.out_vecs : cast_vecs;
    const int n_cast_blocks = (n_vecs + SORT_THREADS - 1) / SORT_THREADS;
    hipLaunchKernelGGL(
        nvfp4_moe_sort_cast, dim3(1 + n_cast_blocks), dim3(SORT_THREADS), 0, stream, p);
    return (int)hipGetLastError();
}
