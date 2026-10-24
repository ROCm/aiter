// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1201 SageAttention input preparation, D = 128:
//   Q/K -> INT8 with one FP32 scale per (batch, head, 32-row block); Q is pre-multiplied by
//   sm_scale. V   -> FP8 e4m3 transposed to [B, H, D, S_pad] with one FP32 scale per (batch, head,
//   channel).
// Optionally fuses QK RMSNorm + rotate-half RoPE on the first rope_dim channels in front of the
// Q/K quantization; the arithmetic follows eager PyTorch so the result is bitwise reproducible.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#include <stdexcept>

#include "aiter_stream.h"
#include "gfx1201_sage_prepare.h"

namespace {

constexpr int DIM           = 128;
constexpr int BLOCK_ROWS    = 32;
constexpr int THREADS       = 256;
constexpr int V_ROWS        = 128;
constexpr int ABSMAX_CHUNKS = 512;

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits)
{ return __uint_as_float(((uint32_t)bits) << 16); }

__device__ __forceinline__ float round_bf16(float value)
{ return __bfloat162float(__float2bfloat16(value)); }

__device__ __forceinline__ void load16(const uint16_t* source, bool valid, float out[16])
{
    uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
    if(valid)
    {
        a = *reinterpret_cast<const uint4*>(source);
        b = *reinterpret_cast<const uint4*>(source + 8);
    }
    const uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        out[2 * i]     = bf16_bits_to_float(words[i] & 0xffff);
        out[2 * i + 1] = bf16_bits_to_float(words[i] >> 16);
    }
}

__device__ __forceinline__ float block_max(float value, float* shared)
{
#pragma unroll
    for(int mask = 16; mask > 0; mask >>= 1)
        value = fmaxf(value, __shfl_xor(value, mask, 32));
    const int wave = threadIdx.x / 32;
    if((threadIdx.x & 31) == 0)
        shared[wave] = value;
    __syncthreads();
    value = shared[0];
#pragma unroll
    for(int i = 1; i < THREADS / 32; ++i)
        value = fmaxf(value, shared[i]);
    return value;
}

// Per-channel |V| max over the sequence. |bf16| bit patterns are monotone, NaN maps to 0.
__global__ __launch_bounds__(THREADS) void v_absmax_kernel(const uint16_t* __restrict__ value,
                                                           uint32_t* __restrict__ maximum,
                                                           int rows,
                                                           int columns,
                                                           int rows_per_block)
{
    const int column = (blockIdx.y * THREADS + threadIdx.x) * 8;
    if(column >= columns)
        return;
    const int batch = blockIdx.z;
    value += (size_t)batch * rows * columns;
    maximum += (size_t)batch * columns;
    const int begin = blockIdx.x * rows_per_block;
    const int end   = min(rows, begin + rows_per_block);
    auto key        = [](uint32_t h) {
        h &= 0x7fffu;
        return h > 0x7f80u ? 0u : h;
    };
    uint32_t best[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int row          = begin;
    for(; row + 4 <= end; row += 4)
    {
        uint4 v[4];
#pragma unroll
        for(int u = 0; u < 4; ++u)
            v[u] = *reinterpret_cast<const uint4*>(value + (size_t)(row + u) * columns + column);
#pragma unroll
        for(int u = 0; u < 4; ++u)
        {
            const uint32_t w[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                best[2 * i]     = max(best[2 * i], key(w[i]));
                best[2 * i + 1] = max(best[2 * i + 1], key(w[i] >> 16));
            }
        }
    }
    for(; row < end; ++row)
    {
        const uint4 v = *reinterpret_cast<const uint4*>(value + (size_t)row * columns + column);
        const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            best[2 * i]     = max(best[2 * i], key(w[i]));
            best[2 * i + 1] = max(best[2 * i + 1], key(w[i] >> 16));
        }
    }
#pragma unroll
    for(int i = 0; i < 8; ++i)
        atomicMax(maximum + column + i, best[i] << 16);
}

// One 32-row x 128-channel tile of Q (blockIdx.y == 0) or K (blockIdx.y == 1) per block.
template <bool NORM_ROPE>
__global__
__launch_bounds__(THREADS) void qk_quant_kernel(const uint16_t* __restrict__ query,
                                                const uint16_t* __restrict__ key,
                                                const float* __restrict__ cosine,
                                                const float* __restrict__ sine,
                                                const uint16_t* __restrict__ query_weight,
                                                const uint16_t* __restrict__ key_weight,
                                                int8_t* __restrict__ query_out,
                                                int8_t* __restrict__ key_out,
                                                float* __restrict__ query_scale,
                                                float* __restrict__ key_scale,
                                                int rows,
                                                int padded_rows,
                                                int heads,
                                                int rope_dim,
                                                float eps,
                                                float sm_scale)
{
    __shared__ float shared[THREADS / 32];
    const int head         = blockIdx.x % heads;
    const int block        = blockIdx.x / heads;
    const int blocks       = padded_rows / BLOCK_ROWS;
    const int batch        = blockIdx.z;
    const int tid          = threadIdx.x;
    const bool is_query    = blockIdx.y == 0;
    const uint16_t* source = (is_query ? query : key) + (size_t)batch * rows * heads * DIM;
    const int group        = tid % 8;
    const int row          = block * BLOCK_ROWS + tid / 8;
    const bool valid       = row < rows;
    const size_t base      = ((size_t)row * heads + head) * DIM;
    float x[16];
    load16(source + base + group * 16, valid, x);
    if constexpr(NORM_ROPE)
    {
        const int rope_groups   = rope_dim / 16;
        const int half_groups   = rope_groups / 2;
        const bool rotates      = group < rope_groups;
        const int partner_group = group < half_groups ? group + half_groups : group - half_groups;
        float partner[16];
        load16(source + base + partner_group * 16, valid && rotates, partner);
        // ATen vectorized RMSNorm order: per-4 fma chains, then a 16-lane xor tree.
        float sums[4];
#pragma unroll
        for(int j = 0; j < 4; ++j)
        {
            float acc = 0.f;
#pragma unroll
            for(int i = 0; i < 4; ++i)
                acc = __fmaf_rn(x[4 * j + i], x[4 * j + i], acc);
            sums[j] = acc;
        }
#pragma unroll
        for(int mask = 4; mask > 0; mask >>= 1)
#pragma unroll
            for(int j = 0; j < 4; ++j)
                sums[j] = __fadd_rn(sums[j], __shfl_xor(sums[j], mask, 32));
        const float total = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
        const float rstd  = rsqrtf(__fadd_rn(__fdiv_rn(total, (float)DIM), eps));
        const uint16_t* weight = is_query ? query_weight : key_weight;
#pragma unroll
        for(int i = 0; i < 16; ++i)
        {
            x[i] = round_bf16(
                __fmul_rn(__fmul_rn(rstd, x[i]), bf16_bits_to_float(weight[group * 16 + i])));
            partner[i] = round_bf16(__fmul_rn(__fmul_rn(rstd, partner[i]),
                                              bf16_bits_to_float(weight[partner_group * 16 + i])));
        }
        if(rotates)
        {
            const size_t rope_base = (size_t)row * rope_dim + group * 16;
#pragma unroll
            for(int i = 0; i < 16; ++i)
            {
                const float c       = valid ? cosine[rope_base + i] : 0.f;
                const float s       = valid ? sine[rope_base + i] : 0.f;
                const float rotated = group < half_groups ? -partner[i] : partner[i];
                x[i] = round_bf16(__fadd_rn(__fmul_rn(x[i], c), __fmul_rn(rotated, s)));
            }
        }
    }
    float local = 0.f;
#pragma unroll
    for(int i = 0; i < 16; ++i)
    {
        if(is_query)
            x[i] = __fmul_rn(x[i], sm_scale);
        local = fmaxf(local, fabsf(x[i]));
    }
    const float scale = __fdiv_rn(block_max(local, shared), 127.0f);
    if(tid == 0)
        (is_query ? query_scale : key_scale)[((size_t)batch * heads + head) * blocks + block] =
            scale;
    uint32_t packed[4];
#pragma unroll
    for(int word = 0; word < 4; ++word)
    {
        uint32_t bits = 0;
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            float q        = __fdiv_rn(x[word * 4 + i], scale);
            q              = __fadd_rn(q, q >= 0.f ? 0.5f : -0.5f);
            const int8_t v = valid && scale > 0.f ? (int8_t)__float2int_rz(q) : (int8_t)0;
            bits |= ((uint32_t)(uint8_t)v) << (8 * i);
        }
        packed[word] = bits;
    }
    int8_t* out = (is_query ? query_out : key_out) + (size_t)batch * padded_rows * heads * DIM;
    *reinterpret_cast<uint4*>(out + base + group * 16) =
        make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

// 128-row tiles so each (head, channel) output row gets one contiguous 128-byte FP8 segment.
__global__ __launch_bounds__(THREADS) void v_quant_kernel(const uint16_t* __restrict__ value,
                                                          const uint32_t* __restrict__ v_maximum,
                                                          uint8_t* __restrict__ value_out,
                                                          float* __restrict__ value_scale,
                                                          int rows,
                                                          int padded_rows,
                                                          int heads)
{
    __shared__ uint16_t tile[V_ROWS][DIM + 2];
    const int head  = blockIdx.x % heads;
    const int block = blockIdx.x / heads;
    const int batch = blockIdx.z;
    const int tid   = threadIdx.x;
    value += (size_t)batch * rows * heads * DIM;
    v_maximum += (size_t)batch * heads * DIM;
    value_scale += (size_t)batch * heads * DIM;
    value_out += (size_t)batch * heads * DIM * padded_rows;
    const int load_col = (tid % 8) * 16;
#pragma unroll
    for(int pass = 0; pass < V_ROWS / 32; ++pass)
    {
        const int local_row = pass * 32 + tid / 8;
        const int load_row  = block * V_ROWS + local_row;
        uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
        if(load_row < rows)
        {
            const uint16_t* source = value + ((size_t)load_row * heads + head) * DIM + load_col;
            a                      = *reinterpret_cast<const uint4*>(source);
            b                      = *reinterpret_cast<const uint4*>(source + 8);
        }
        const uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for(int i = 0; i < 8; ++i)
        {
            tile[local_row][load_col + 2 * i]     = (uint16_t)(words[i] & 0xffff);
            tile[local_row][load_col + 2 * i + 1] = (uint16_t)(words[i] >> 16);
        }
    }
    __syncthreads();
    const int first = (tid % 8) * 16;
#pragma unroll
    for(int pass = 0; pass < DIM / 32; ++pass)
    {
        const int dim     = pass * 32 + tid / 8;
        const float scale = __fmul_rn(__uint_as_float(v_maximum[head * DIM + dim]), 1.0f / 448.0f);
        if(block == 0 && tid % 8 == 0)
            value_scale[head * DIM + dim] = scale;
        if(block * V_ROWS + first >= padded_rows)
            continue;
        uint32_t packed[4];
#pragma unroll
        for(int word = 0; word < 4; ++word)
        {
            float x[4];
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                const int local = first + word * 4 + i;
                x[i]            = block * V_ROWS + local < rows && scale > 0.f
                                      ? __fdiv_rn(bf16_bits_to_float(tile[local][dim]), scale)
                                      : 0.f;
            }
            int bits     = 0;
            bits         = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], bits, false);
            bits         = __builtin_amdgcn_cvt_pk_fp8_f32(x[2], x[3], bits, true);
            packed[word] = (uint32_t)bits;
        }
        *reinterpret_cast<uint4*>(value_out + ((size_t)head * DIM + dim) * padded_rows +
                                  block * V_ROWS + first) =
            make_uint4(packed[0], packed[1], packed[2], packed[3]);
    }
}

} // namespace

void gfx1201_sage_prepare_hip(const aiter_tensor_t& query,
                              const aiter_tensor_t& key,
                              const aiter_tensor_t& value,
                              std::optional<aiter_tensor_t> query_weight,
                              std::optional<aiter_tensor_t> key_weight,
                              std::optional<aiter_tensor_t> cosine,
                              std::optional<aiter_tensor_t> sine,
                              aiter_tensor_t& maximum,
                              aiter_tensor_t& query_out,
                              aiter_tensor_t& key_out,
                              aiter_tensor_t& value_out,
                              aiter_tensor_t& query_scale,
                              aiter_tensor_t& key_scale,
                              aiter_tensor_t& value_scale,
                              int64_t rope_dim,
                              double eps,
                              double sm_scale)
{
    const int batch = (int)query.size(0), rows = (int)query.size(1), heads = (int)query.size(2);
    const int padded_rows = (int)query_out.size(1);
    if(batch <= 0 || rows <= 0 || heads <= 0 || padded_rows % BLOCK_ROWS != 0 || padded_rows < rows)
        throw std::invalid_argument("gfx1201_sage_prepare: invalid shape");
    const bool norm_rope = query_weight.has_value();
    if(norm_rope != key_weight.has_value() || norm_rope != cosine.has_value() ||
       norm_rope != sine.has_value())
        throw std::invalid_argument(
            "gfx1201_sage_prepare: norm weights and cosine/sine must be given together");
    if(norm_rope && (rope_dim <= 0 || rope_dim > DIM || rope_dim % 32 != 0))
        throw std::invalid_argument("gfx1201_sage_prepare: rope_dim must be 32, 64, 96 or 128");
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int columns        = heads * DIM;
    hipMemsetAsync(maximum.ptr, 0, (size_t)batch * columns * sizeof(uint32_t), stream);
    const int rows_per_block = (rows + ABSMAX_CHUNKS - 1) / ABSMAX_CHUNKS;
    const dim3 absmax_grid(
        (rows + rows_per_block - 1) / rows_per_block, (columns / 8 + THREADS - 1) / THREADS, batch);
    hipLaunchKernelGGL(v_absmax_kernel,
                       absmax_grid,
                       dim3(THREADS),
                       0,
                       stream,
                       (const uint16_t*)value.ptr,
                       (uint32_t*)maximum.ptr,
                       rows,
                       columns,
                       rows_per_block);
    hipLaunchKernelGGL(v_quant_kernel,
                       dim3((padded_rows + V_ROWS - 1) / V_ROWS * heads, 1, batch),
                       dim3(THREADS),
                       0,
                       stream,
                       (const uint16_t*)value.ptr,
                       (const uint32_t*)maximum.ptr,
                       (uint8_t*)value_out.ptr,
                       (float*)value_scale.ptr,
                       rows,
                       padded_rows,
                       heads);
    const dim3 qk_grid(padded_rows / BLOCK_ROWS * heads, 2, batch);
    const void* qw = norm_rope ? query_weight->ptr : nullptr;
    const void* kw = norm_rope ? key_weight->ptr : nullptr;
    const void* cs = norm_rope ? cosine->ptr : nullptr;
    const void* sn = norm_rope ? sine->ptr : nullptr;
#define GFX1201_QK_ARGS                                                                          \
    (const uint16_t*)query.ptr, (const uint16_t*)key.ptr, (const float*)cs, (const float*)sn,    \
        (const uint16_t*)qw, (const uint16_t*)kw, (int8_t*)query_out.ptr, (int8_t*)key_out.ptr,  \
        (float*)query_scale.ptr, (float*)key_scale.ptr, rows, padded_rows, heads, (int)rope_dim, \
        (float)eps, (float)sm_scale
    if(norm_rope)
        hipLaunchKernelGGL(
            qk_quant_kernel<true>, qk_grid, dim3(THREADS), 0, stream, GFX1201_QK_ARGS);
    else
        hipLaunchKernelGGL(
            qk_quant_kernel<false>, qk_grid, dim3(THREADS), 0, stream, GFX1201_QK_ARGS);
#undef GFX1201_QK_ARGS
}
