// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// K/V preprocessing for the gfx1201 SageAttention H3 core (backend="asm_h3"). It differs from
// gfx1201_sage_prepare in the contract the H3 core expects: Q stays raw BF16 (the core quantizes
// it internally), K keeps its original row count and is mean-centered per (batch, head, channel)
// before INT8 quantization with one scale per 64 rows, and V is padded to 16 rows, not 32.
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <hip/hip_runtime.h>

#include "aiter_stream.h"
#include "gfx1201_sage_prepare_h3.h"

#include <cstdint>
#include <stdexcept>

namespace {

constexpr int HEAD_DIM      = 128;
constexpr int WARP          = 32; // HIP runs gfx1201 at wave32
constexpr int V_TILE_D      = 32;
constexpr int V_TILE_S      = 128;
constexpr int V_SEQ_LANES   = 8;
constexpr int V_THREADS     = 256;
constexpr int V_PAD_ROWS    = 16;
constexpr int K_GROUP_ROWS  = 64; // one INT8 scale per 64 key rows
constexpr int K_PACK_ELEMS  = 8;
constexpr int K_PACKS       = 2;
constexpr int K_GROUPS      = 4;
constexpr int K_THREADS     = 512;

__device__ __forceinline__ float value_to_float(const __hip_bfloat16 value)
{
    return __bfloat162float(value);
}

__device__ __forceinline__ int8_t float_to_int8_nearby_gfx12(const float x)
{
    const float clipped = fminf(127.0f, fmaxf(-128.0f, nearbyintf(x)));
    return static_cast<int8_t>(clipped);
}

// Block-wide maximum, valid in every thread. Max is associative and exact, so the result does not
// depend on the reduction order. The caller must separate two calls with a barrier: the next call
// overwrites partial[] that this one reads.
template <int BlockThreads>
__device__ __forceinline__ float block_reduce_max(float v)
{
    static_assert(BlockThreads % WARP == 0, "block must be a whole number of waves");
    constexpr int WAVES = BlockThreads / WARP;
    __shared__ float partial[WAVES];
#pragma unroll
    for(int offset = WARP / 2; offset > 0; offset >>= 1)
        v = fmaxf(v, __shfl_xor(v, offset, WARP));
    if(threadIdx.x % WARP == 0)
        partial[threadIdx.x / WARP] = v;
    __syncthreads();
    float result = partial[0];
#pragma unroll
    for(int wave = 1; wave < WAVES; ++wave)
        result = fmaxf(result, partial[wave]);
    return result;
}

// Grid: (HEAD_DIM / V_TILE_D, heads, batch). One block owns V_TILE_D channels of one (batch, head):
// it reduces the K mean and the V per-channel amax over all rows, then writes V transposed as FP8.
__global__ __launch_bounds__(V_THREADS) void prepare_kmean_value_fp8_kernel(
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ value,
    __hip_bfloat16* __restrict__ key_mean,
    uint8_t* __restrict__ value_out,
    float* __restrict__ value_scale,
    const int64_t seq_len,
    const int64_t heads,
    const int64_t padded_rows)
{
    // One uint4 per thread covers 8 bf16, so D_THREADS threads span the V_TILE_D-wide tile and the
    // rest of the block becomes extra token lanes.
    constexpr int STAT_VEC   = 8;
    constexpr int D_THREADS  = V_TILE_D / STAT_VEC;
    constexpr int STAT_LANES = (V_TILE_D * V_SEQ_LANES) / D_THREADS;
    __shared__ float partial_sum[STAT_LANES * V_TILE_D];
    __shared__ float partial_amax[STAT_LANES * V_TILE_D];
    __shared__ float scale_tile[V_TILE_D];
    __shared__ uint8_t tile[V_TILE_S][V_TILE_D];

    const int tid        = threadIdx.x;
    const int64_t d_base = (int64_t)blockIdx.x * V_TILE_D;
    const int64_t h      = blockIdx.y;
    const int64_t b      = blockIdx.z;
    const int64_t d_vec  = d_base + (int64_t)(tid % D_THREADS) * STAT_VEC;
    const int slane      = tid / D_THREADS;

    float vsum[STAT_VEC];
    float vamax[STAT_VEC];
#pragma unroll
    for(int i = 0; i < STAT_VEC; ++i)
    {
        vsum[i]  = 0.0f;
        vamax[i] = 0.0f;
    }
    __hip_bfloat16 kbuf[STAT_VEC];
    __hip_bfloat16 vbuf[STAT_VEC];
    for(int64_t token = slane; token < seq_len; token += STAT_LANES)
    {
        const int64_t offset = ((b * seq_len + token) * heads + h) * HEAD_DIM + d_vec;
        *reinterpret_cast<uint4*>(kbuf) = *reinterpret_cast<const uint4*>(key + offset);
        *reinterpret_cast<uint4*>(vbuf) = *reinterpret_cast<const uint4*>(value + offset);
#pragma unroll
        for(int i = 0; i < STAT_VEC; ++i)
        {
            vsum[i] += value_to_float(kbuf[i]);
            vamax[i] = fmaxf(vamax[i], fabsf(value_to_float(vbuf[i])));
        }
    }
#pragma unroll
    for(int i = 0; i < STAT_VEC; ++i)
    {
        const int slot     = slane * V_TILE_D + (tid % D_THREADS) * STAT_VEC + i;
        partial_sum[slot]  = vsum[i];
        partial_amax[slot] = vamax[i];
    }
    __syncthreads();
    if(tid < V_TILE_D)
    {
        float sum  = 0.0f;
        float amax = 0.0f;
#pragma unroll
        for(int lane = 0; lane < STAT_LANES; ++lane)
        {
            sum += partial_sum[lane * V_TILE_D + tid];
            amax = fmaxf(amax, partial_amax[lane * V_TILE_D + tid]);
        }
        const int64_t stat_offset = (b * heads + h) * HEAD_DIM + d_base + tid;
        key_mean[stat_offset]     = __float2bfloat16(sum / (float)seq_len);
        // 2.25 is the V quantization headroom the H3 core's FP8 PV step is built around.
        const float scale         = amax / 2.25f;
        scale_tile[tid]           = scale;
        value_scale[stat_offset]  = scale;
    }
    __syncthreads();

    for(int64_t s_base = 0; s_base < seq_len; s_base += V_TILE_S)
    {
        constexpr int PACKS_PER_ROW = V_TILE_D / STAT_VEC;
        for(int pack = tid; pack < V_TILE_S * PACKS_PER_ROW; pack += V_THREADS)
        {
            const int load_s    = pack / PACKS_PER_ROW;
            const int load_d    = (pack - load_s * PACKS_PER_ROW) * STAT_VEC;
            const int64_t token = s_base + load_s;
            if(token < seq_len)
            {
                const int64_t offset =
                    ((b * seq_len + token) * heads + h) * HEAD_DIM + d_base + load_d;
                *reinterpret_cast<uint4*>(vbuf) = *reinterpret_cast<const uint4*>(value + offset);
#pragma unroll
                for(int elem = 0; elem < STAT_VEC; ++elem)
                {
                    const float scale = scale_tile[load_d + elem];
                    const float quant = scale == 0.0f ? 0.0f : value_to_float(vbuf[elem]) / scale;
                    tile[load_s][load_d + elem] =
                        __hip_cvt_float_to_fp8(quant, __HIP_SATFINITE, __HIP_E4M3);
                }
            }
        }
        __syncthreads();
        for(int linear = tid; linear < V_TILE_S * V_TILE_D; linear += V_THREADS)
        {
            const int store_d   = linear / V_TILE_S;
            const int store_s   = linear - store_d * V_TILE_S;
            const int64_t token = s_base + store_s;
            if(token < seq_len)
                value_out[((b * heads + h) * HEAD_DIM + d_base + store_d) * padded_rows + token] =
                    tile[store_s][store_d];
        }
        __syncthreads();
    }

    // The core reads whole 16-row V tiles, so the rows past seq_len must be zero, not stale memory.
    const int64_t tail = padded_rows - seq_len;
    for(int64_t item = tid; item < tail * V_TILE_D; item += V_THREADS)
    {
        const int64_t local_d = item / tail;
        const int64_t token   = seq_len + item - local_d * tail;
        value_out[((b * heads + h) * HEAD_DIM + d_base + local_d) * padded_rows + token] = 0;
    }
}

// Grid: (ceil(groups / K_GROUPS), heads, batch). Each block sweeps K_GROUPS groups of
// K_GROUP_ROWS key rows, so the launch, the mean load and the address setup are amortised.
__global__ __launch_bounds__(K_THREADS) void quant_k_sub_mean_kernel(
    const __hip_bfloat16* __restrict__ key,
    const __hip_bfloat16* __restrict__ mean,
    int8_t* __restrict__ key_out,
    float* __restrict__ key_scale,
    const int64_t seq_len,
    const int64_t heads)
{
    constexpr int THREADS_PER_TOKEN = HEAD_DIM / K_PACK_ELEMS;
    constexpr int ITER_STRIDE       = K_GROUP_ROWS / K_PACKS;

    const int tid          = threadIdx.x;
    const int h            = blockIdx.y;
    const int b            = blockIdx.z;
    const int local_token  = tid / THREADS_PER_TOKEN;
    const int d            = (tid % THREADS_PER_TOKEN) * K_PACK_ELEMS;
    const int64_t mean_off = ((int64_t)b * heads + h) * HEAD_DIM + d;
    const int num_groups   = (int)((seq_len + K_GROUP_ROWS - 1) / K_GROUP_ROWS);

    __hip_bfloat16 mean_val[K_PACK_ELEMS];
    float mean_float[K_PACK_ELEMS];
    *reinterpret_cast<uint4*>(mean_val) = *reinterpret_cast<const uint4*>(mean + mean_off);
#pragma unroll
    for(int i = 0; i < K_PACK_ELEMS; ++i)
        mean_float[i] = value_to_float(mean_val[i]);

    __hip_bfloat16 x_val[K_PACKS][K_PACK_ELEMS];
    float centered[K_PACKS][K_PACK_ELEMS];
    for(int g = 0; g < K_GROUPS; ++g)
    {
        const int group = blockIdx.x * K_GROUPS + g;
        if(group >= num_groups)
            break;
        const int64_t token_base = (int64_t)group * K_GROUP_ROWS + local_token;

        float local_amax = 1e-7f; // keeps the scale finite for an all-zero group
#pragma unroll
        for(int pack = 0; pack < K_PACKS; ++pack)
        {
            const int64_t s = token_base + (int64_t)pack * ITER_STRIDE;
            if(s < seq_len)
            {
                const int64_t off = (((int64_t)b * seq_len + s) * heads + h) * HEAD_DIM + d;
                *reinterpret_cast<uint4*>(x_val[pack]) =
                    *reinterpret_cast<const uint4*>(key + off);
#pragma unroll
                for(int i = 0; i < K_PACK_ELEMS; ++i)
                {
                    centered[pack][i] = value_to_float(x_val[pack][i]) - mean_float[i];
                    local_amax        = fmaxf(local_amax, fabsf(centered[pack][i]));
                }
            }
            else
            {
#pragma unroll
                for(int i = 0; i < K_PACK_ELEMS; ++i)
                    centered[pack][i] = 0.0f;
            }
        }

        const float group_amax = block_reduce_max<K_THREADS>(local_amax);
        if(tid == 0)
            key_scale[((int64_t)b * heads + h) * num_groups + group] = group_amax / 127.0f;
        const float inv_scale = 127.0f / group_amax;

#pragma unroll
        for(int pack = 0; pack < K_PACKS; ++pack)
        {
            const int64_t s = token_base + (int64_t)pack * ITER_STRIDE;
            if(s < seq_len)
            {
                const int64_t off = (((int64_t)b * seq_len + s) * heads + h) * HEAD_DIM + d;
                char4 lo, hi;
                lo.x = float_to_int8_nearby_gfx12(centered[pack][0] * inv_scale);
                lo.y = float_to_int8_nearby_gfx12(centered[pack][1] * inv_scale);
                lo.z = float_to_int8_nearby_gfx12(centered[pack][2] * inv_scale);
                lo.w = float_to_int8_nearby_gfx12(centered[pack][3] * inv_scale);
                hi.x = float_to_int8_nearby_gfx12(centered[pack][4] * inv_scale);
                hi.y = float_to_int8_nearby_gfx12(centered[pack][5] * inv_scale);
                hi.z = float_to_int8_nearby_gfx12(centered[pack][6] * inv_scale);
                hi.w = float_to_int8_nearby_gfx12(centered[pack][7] * inv_scale);
                *reinterpret_cast<char4*>(key_out + off)     = lo;
                *reinterpret_cast<char4*>(key_out + off + 4) = hi;
            }
        }
        // block_reduce_max of the next group overwrites the partial[] this one read.
        __syncthreads();
    }
}

} // namespace

void gfx1201_sage_prepare_h3_hip(const aiter_tensor_t& key,
                                 const aiter_tensor_t& value,
                                 aiter_tensor_t& key_mean,
                                 aiter_tensor_t& key_out,
                                 aiter_tensor_t& key_scale,
                                 aiter_tensor_t& value_out,
                                 aiter_tensor_t& value_scale)
{
    const int batch       = (int)key.size(0);
    const int seq_len     = (int)key.size(1);
    const int heads       = (int)key.size(2);
    const int head_dim    = (int)key.size(3);
    const int padded_rows = (int)value_out.size(3);
    if(batch <= 0 || seq_len <= 0 || heads <= 0 || head_dim != HEAD_DIM)
        throw std::invalid_argument("gfx1201_sage_prepare_h3: expected key [B, S, H, 128]");
    if(padded_rows % V_PAD_ROWS != 0 || padded_rows < seq_len ||
       padded_rows - seq_len >= V_PAD_ROWS)
        throw std::invalid_argument(
            "gfx1201_sage_prepare_h3: value_out must pad the sequence to a multiple of 16");
    const int num_groups = (seq_len + K_GROUP_ROWS - 1) / K_GROUP_ROWS;
    if((int)key_scale.size(2) != num_groups)
        throw std::invalid_argument("gfx1201_sage_prepare_h3: key_scale must be [B, H, S / 64]");

    const hipStream_t stream = aiter::getCurrentHIPStream();
    hipLaunchKernelGGL(prepare_kmean_value_fp8_kernel,
                       dim3(HEAD_DIM / V_TILE_D, heads, batch),
                       dim3(V_THREADS),
                       0,
                       stream,
                       (const __hip_bfloat16*)key.ptr,
                       (const __hip_bfloat16*)value.ptr,
                       (__hip_bfloat16*)key_mean.ptr,
                       (uint8_t*)value_out.ptr,
                       (float*)value_scale.ptr,
                       (int64_t)seq_len,
                       (int64_t)heads,
                       (int64_t)padded_rows);
    hipLaunchKernelGGL(quant_k_sub_mean_kernel,
                       dim3((num_groups + K_GROUPS - 1) / K_GROUPS, heads, batch),
                       dim3(K_THREADS),
                       0,
                       stream,
                       (const __hip_bfloat16*)key.ptr,
                       (const __hip_bfloat16*)key_mean.ptr,
                       (int8_t*)key_out.ptr,
                       (float*)key_scale.ptr,
                       (int64_t)seq_len,
                       (int64_t)heads);
}
