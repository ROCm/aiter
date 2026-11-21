// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Fused RMSNorm(weight) + adaLN modulation, BF16 rows of 5376, eager PyTorch rounding.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "gfx1201_h3_ops.h"

namespace {

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {  // round-to-nearest-even, NaN-preserving
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// out = bf16(bf16(bf16(x*rstd*w) * bf16(1+scale[g])) + shift[g]), g = indices[row]
template <int THREADS, int CHUNKS, typename Index>
__global__ void __launch_bounds__(THREADS) rms_modulate_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const uint16_t* __restrict__ scale,
    const uint16_t* __restrict__ shift, const Index* __restrict__ indices, uint16_t* __restrict__ out,
    long scale_stride, long shift_stride, float eps) {
  constexpr int D = THREADS * CHUNKS * 8;
  __shared__ float red[THREADS / 32];
  const long row = blockIdx.x;
  const uint16_t* xr = x + row * D;
  float v[CHUNKS][8];
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int col = (c * THREADS + threadIdx.x) * 8;
    uint4 raw = *reinterpret_cast<const uint4*>(xr + col);
    const uint16_t* h = reinterpret_cast<const uint16_t*>(&raw);
#pragma unroll
    for (int i = 0; i < 8; ++i) { v[c][i] = bf2f(h[i]); ss = fmaf(v[c][i], v[c][i], ss); }
  }
  ss = wave_sum(ss);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  if (lane == 0) red[wave] = ss;
  __syncthreads();
  float t = lane < THREADS / 32 ? red[lane] : 0.f;
  t = wave_sum(t);
  const float rstd = rsqrtf(t / static_cast<float>(D) + eps);
  const long g = static_cast<long>(indices[row]);
  const uint16_t* sr = scale + g * scale_stride;
  const uint16_t* hr = shift + g * shift_stride;
#pragma unroll
  for (int c = 0; c < CHUNKS; ++c) {
    const int col = (c * THREADS + threadIdx.x) * 8;
    uint4 wr = *reinterpret_cast<const uint4*>(w + col);
    uint4 scr = *reinterpret_cast<const uint4*>(sr + col);
    uint4 shr = *reinterpret_cast<const uint4*>(hr + col);
    const uint16_t* wh = reinterpret_cast<const uint16_t*>(&wr);
    const uint16_t* sch = reinterpret_cast<const uint16_t*>(&scr);
    const uint16_t* shh = reinterpret_cast<const uint16_t*>(&shr);
    uint4 oraw;
    uint16_t* oh = reinterpret_cast<uint16_t*>(&oraw);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float normed = bf2f(f2bf(bf2f(wh[i]) * (v[c][i] * rstd)));
      const float factor = bf2f(f2bf(1.0f + bf2f(sch[i])));
      const float product = bf2f(f2bf(normed * factor));
      oh[i] = f2bf(product + bf2f(shh[i]));
    }
    *reinterpret_cast<uint4*>(out + row * D + col) = oraw;
  }
}

}  // namespace

void launch_gfx1201_rms_modulate(int64_t x, int64_t weight, int64_t scale, int64_t shift, int64_t indices,
                                 bool indices_int64, int64_t out, int64_t rows, int64_t scale_stride,
                                 int64_t shift_stride, double eps, int64_t stream) {
  if (rows == 0) return;
  auto p = [](int64_t v) { return reinterpret_cast<const uint16_t*>(v); };
  hipStream_t s = reinterpret_cast<hipStream_t>(stream);
  if (indices_int64)
    hipLaunchKernelGGL((rms_modulate_kernel<224, 3, int64_t>), dim3(rows), dim3(224), 0, s, p(x), p(weight), p(scale),
                       p(shift), reinterpret_cast<const int64_t*>(indices), reinterpret_cast<uint16_t*>(out),
                       scale_stride, shift_stride, static_cast<float>(eps));
  else
    hipLaunchKernelGGL((rms_modulate_kernel<224, 3, int32_t>), dim3(rows), dim3(224), 0, s, p(x), p(weight), p(scale),
                       p(shift), reinterpret_cast<const int32_t*>(indices), reinterpret_cast<uint16_t*>(out),
                       scale_stride, shift_stride, static_cast<float>(eps));
}
