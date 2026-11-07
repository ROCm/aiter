// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// QK RMSNorm + RoPE + Sage INT8 Q/K + FP8 V prepare; the head count is a template parameter
// (7/14/28/56: TP2 shards and Ulysses SP2/4/8 head shards of MiniMax-H3).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <stdexcept>

#include "gfx1201_norm_rope_prepare.h"

static constexpr int DIM = 128;
static constexpr int BLOCK_ROWS = 32;
static constexpr int ROPE_DIM = 96;
static constexpr int THREADS = 256;
static constexpr float NORM_EPS = 1e-5f;

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
  return __uint_as_float(((uint32_t)bits) << 16);
}

__device__ __forceinline__ float round_bf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

__device__ __forceinline__ void load16(const uint16_t* source, bool valid, float out[16]) {
  uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
  if (valid) {
    a = *reinterpret_cast<const uint4*>(source);
    b = *reinterpret_cast<const uint4*>(source + 8);
  }
  const uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[2 * i] = bf16_bits_to_float(words[i] & 0xffff);
    out[2 * i + 1] = bf16_bits_to_float(words[i] >> 16);
  }
}

__device__ __forceinline__ float block_max(float value, float* shared) {
  #pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) value = fmaxf(value, __shfl_xor(value, mask, 32));
  const int wave = threadIdx.x / 32;
  if ((threadIdx.x & 31) == 0) shared[wave] = value;
  __syncthreads();
  value = shared[0];
  #pragma unroll
  for (int i = 1; i < THREADS / 32; ++i) value = fmaxf(value, shared[i]);
  return value;
}

template <int HEADS>
__global__ __launch_bounds__(HEADS * DIM / 8)  // one uint4 (8 bf16) per thread
void v_absmax_kernel(const uint16_t* __restrict__ value, uint32_t* __restrict__ maximum, int rows, int rows_per_block) {
  constexpr int COLUMNS = HEADS * DIM;
  const int column = threadIdx.x * 8;
  const int begin = blockIdx.x * rows_per_block;
  const int end = min(rows, begin + rows_per_block);
  // |bf16| bits are monotone; NaN keys map to 0 because fmaxf ignores NaN operands.
  auto key = [](uint32_t h) { h &= 0x7fffu; return h > 0x7f80u ? 0u : h; };
  uint32_t best[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int row = begin;
  for (; row + 4 <= end; row += 4) {
    uint4 v[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) v[u] = *reinterpret_cast<const uint4*>(value + (size_t)(row + u) * COLUMNS + column);
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
      const uint32_t w[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        best[2 * i] = max(best[2 * i], key(w[i]));
        best[2 * i + 1] = max(best[2 * i + 1], key(w[i] >> 16));
      }
    }
  }
  for (; row < end; ++row) {
    const uint4 v = *reinterpret_cast<const uint4*>(value + (size_t)row * COLUMNS + column);
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      best[2 * i] = max(best[2 * i], key(w[i]));
      best[2 * i + 1] = max(best[2 * i + 1], key(w[i] >> 16));
    }
  }
  #pragma unroll
  for (int i = 0; i < 8; ++i) atomicMax(maximum + column + i, best[i] << 16);
}

template <int HEADS>
__global__ __launch_bounds__(THREADS)
void rope_sage_prepare_kernel(
    const uint16_t* __restrict__ query, const uint16_t* __restrict__ key,
    const float* __restrict__ cosine, const float* __restrict__ sine,
    const uint16_t* __restrict__ query_weight, const uint16_t* __restrict__ key_weight,
    int8_t* __restrict__ query_out, int8_t* __restrict__ key_out,
    float* __restrict__ query_scale, float* __restrict__ key_scale,
    int rows, int padded_rows, float sm_scale) {
  __shared__ float shared[THREADS / 32];
  const int head = blockIdx.x % HEADS;
  const int block = blockIdx.x / HEADS;
  const int blocks = padded_rows / BLOCK_ROWS;
  const int tid = threadIdx.x;
  const bool is_query = blockIdx.y == 0;
  const uint16_t* source = is_query ? query : key;
  const int group = tid % 8;
  const int row = block * BLOCK_ROWS + tid / 8;
  const bool valid = row < rows;
  const size_t base = ((size_t)row * HEADS + head) * DIM;
  const int src_row = row;
  const size_t src_base = ((size_t)src_row * HEADS + head) * DIM;
  float x[16], partner[16];
  const int partner_group = group < 3 ? group + 3 : group - 3;
  load16(source + src_base + group * 16, valid, x);
  load16(source + src_base + partner_group * 16, valid && group < 6, partner);
  // Reproduces ATen vectorized RMSNorm: per-4 fma chains, then shfl_down tree 16/8/4/2/1.
  float sums[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    float acc = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc = __fmaf_rn(x[4 * j + i], x[4 * j + i], acc);
    sums[j] = acc;
  }
  #pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1)
    #pragma unroll
    for (int j = 0; j < 4; ++j) sums[j] = __fadd_rn(sums[j], __shfl_xor(sums[j], mask, 32));
  const float total = __fadd_rn(__fadd_rn(sums[0], sums[2]), __fadd_rn(sums[1], sums[3]));
  const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(total, 128.0f), NORM_EPS));
  const uint16_t* weight = is_query ? query_weight : key_weight;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    x[i] = round_bf16(__fmul_rn(__fmul_rn(rstd, x[i]), bf16_bits_to_float(weight[group * 16 + i])));
    partner[i] = round_bf16(__fmul_rn(__fmul_rn(rstd, partner[i]), bf16_bits_to_float(weight[partner_group * 16 + i])));
  }
  if (group < 6) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      const float c = valid ? cosine[(size_t)src_row * ROPE_DIM + group * 16 + i] : 0.f;
      const float s = valid ? sine[(size_t)src_row * ROPE_DIM + group * 16 + i] : 0.f;
      const float rotated = group < 3 ? -partner[i] : partner[i];
      x[i] = __fadd_rn(__fmul_rn(x[i], c), __fmul_rn(rotated, s));
    }
  }
  float local = 0.f;
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    x[i] = round_bf16(x[i]);
    if (is_query) x[i] = __fmul_rn(x[i], sm_scale);
    local = fmaxf(local, fabsf(x[i]));
  }
  const float scale = __fdiv_rn(block_max(local, shared), 127.0f);
  if (tid == 0) (is_query ? query_scale : key_scale)[head * blocks + block] = scale;
  uint32_t packed[4];
  #pragma unroll
  for (int word = 0; word < 4; ++word) {
    uint32_t bits = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float q = __fdiv_rn(x[word * 4 + i], scale);
      q = __fadd_rn(q, q >= 0.f ? 0.5f : -0.5f);
      const int8_t v = valid ? (int8_t)__float2int_rz(q) : (int8_t)0;
      bits |= ((uint32_t)(uint8_t)v) << (8 * i);
    }
    packed[word] = bits;
  }
  *reinterpret_cast<uint4*>((is_query ? query_out : key_out) + base + group * 16) =
      make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

// V pass: 128-row tiles so each (head, dim) output row gets one contiguous 128-byte FP8 segment
// (the 32-row tile in rope_sage_prepare_kernel emitted scattered 32-byte writes). Same per-element math.
static constexpr int V_ROWS = 128;

template <int HEADS>
__global__ __launch_bounds__(THREADS)
void v_quant_kernel(const uint16_t* __restrict__ value, const uint32_t* __restrict__ v_maximum,
                    uint8_t* __restrict__ value_out, float* __restrict__ value_scale, int rows, int padded_rows) {
  __shared__ uint16_t tile[V_ROWS][DIM + 2];
  const int head = blockIdx.x % HEADS;
  const int block = blockIdx.x / HEADS;
  const int tid = threadIdx.x;
  const int load_col = (tid % 8) * 16;
  #pragma unroll
  for (int pass = 0; pass < V_ROWS / 32; ++pass) {
    const int local_row = pass * 32 + tid / 8;
    const int load_row = block * V_ROWS + local_row;
    uint4 a = make_uint4(0, 0, 0, 0), b = make_uint4(0, 0, 0, 0);
    if (load_row < rows) {
      const uint16_t* source = value + ((size_t)load_row * HEADS + head) * DIM + load_col;
      a = *reinterpret_cast<const uint4*>(source);
      b = *reinterpret_cast<const uint4*>(source + 8);
    }
    const uint32_t words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      tile[local_row][load_col + 2 * i] = (uint16_t)(words[i] & 0xffff);
      tile[local_row][load_col + 2 * i + 1] = (uint16_t)(words[i] >> 16);
    }
  }
  __syncthreads();
  const int first = (tid % 8) * 16;
  #pragma unroll
  for (int pass = 0; pass < DIM / 32; ++pass) {
    const int dim = pass * 32 + tid / 8;
    const float scale = __fmul_rn(__uint_as_float(v_maximum[head * DIM + dim]), 1.0f / 448.0f);
    if (block == 0 && tid % 8 == 0) value_scale[head * DIM + dim] = scale;
    if (block * V_ROWS + first >= padded_rows) continue;
    uint32_t packed[4];
    #pragma unroll
    for (int word = 0; word < 4; ++word) {
      float x[4];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int local = first + word * 4 + i;
        x[i] = block * V_ROWS + local < rows ? __fdiv_rn(bf16_bits_to_float(tile[local][dim]), scale) : 0.f;
      }
      int bits = 0;
      bits = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], bits, false);
      bits = __builtin_amdgcn_cvt_pk_fp8_f32(x[2], x[3], bits, true);
      packed[word] = (uint32_t)bits;
    }
    *reinterpret_cast<uint4*>(value_out + ((size_t)head * DIM + dim) * padded_rows + block * V_ROWS + first) =
        make_uint4(packed[0], packed[1], packed[2], packed[3]);
  }
}

template <int HEADS>
static void launch(hipStream_t s, const uint16_t* query, const uint16_t* key, const uint16_t* value,
                   const float* cosine, const float* sine, uint32_t* maximum,
                   const uint16_t* query_weight, const uint16_t* key_weight, int8_t* query_out, int8_t* key_out,
                   uint8_t* value_out, float* query_scale, float* key_scale, float* value_scale,
                   int rows, int padded_rows, float sm_scale, int parts) {
  if (parts > 2) {
    hipMemsetAsync(maximum, 0, HEADS * DIM * sizeof(uint32_t), s);
    const int chunks = 512;
    const int rows_per_block = (rows + chunks - 1) / chunks;
    hipLaunchKernelGGL(v_absmax_kernel<HEADS>, dim3((unsigned)((rows + rows_per_block - 1) / rows_per_block)),
                       dim3(HEADS * DIM / 8), 0, s, value, maximum, rows, rows_per_block);
    hipLaunchKernelGGL(v_quant_kernel<HEADS>, dim3((unsigned)((padded_rows + V_ROWS - 1) / V_ROWS * HEADS)), dim3(THREADS), 0, s,
                       value, (const uint32_t*)maximum, value_out, value_scale, rows, padded_rows);
  }
  hipLaunchKernelGGL(rope_sage_prepare_kernel<HEADS>, dim3((unsigned)(padded_rows / BLOCK_ROWS * HEADS), (unsigned)(parts > 2 ? 2 : parts)),
                     dim3(THREADS), 0, s, query, key, cosine, sine, query_weight, key_weight, query_out, key_out,
                     query_scale, key_scale, rows, padded_rows, sm_scale);
}

void launch_gfx1201_norm_rope_prepare(int64_t query, int64_t key, int64_t value, int64_t cosine, int64_t sine, int64_t maximum,
                    int64_t query_weight, int64_t key_weight, int64_t query_out, int64_t key_out, int64_t value_out,
                    int64_t query_scale, int64_t key_scale, int64_t value_scale,
                    int64_t rows, int64_t padded_rows, double sm_scale, int64_t stream, int64_t parts, int64_t heads) {
#define GFX1201_PREPARE_CASE(H)                                                                                   \
  case H:                                                                                                        \
    launch<H>((hipStream_t)stream, (const uint16_t*)query, (const uint16_t*)key, (const uint16_t*)value,         \
              (const float*)cosine, (const float*)sine, (uint32_t*)maximum, (const uint16_t*)query_weight,       \
              (const uint16_t*)key_weight, (int8_t*)query_out, (int8_t*)key_out, (uint8_t*)value_out,            \
              (float*)query_scale, (float*)key_scale, (float*)value_scale, (int)rows, (int)padded_rows,          \
              (float)sm_scale, (int)parts);                                                                      \
    return;
  switch (heads) {
    GFX1201_PREPARE_CASE(7)
    GFX1201_PREPARE_CASE(14)
    GFX1201_PREPARE_CASE(28)
    GFX1201_PREPARE_CASE(56)
  }
#undef GFX1201_PREPARE_CASE
  throw std::invalid_argument("gfx1201_norm_rope_prepare supports 7/14/28/56 heads");
}
