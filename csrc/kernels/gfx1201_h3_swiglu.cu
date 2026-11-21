// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SwiGLU on a fused [rows, 2F] BF16 up-projection: out = value * silu(gate), eager PyTorch rounding.
#include <hip/hip_runtime.h>

#include <algorithm>

#include "gfx1201_h3_ops.h"

namespace {

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {  // round-to-nearest-even, NaN-preserving (matches c10::BFloat16)
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

// out[r, c] = bf16(value * bf16(silu(gate))), value = in[r, c], gate = in[r, F + c]: eager `v * F.silu(g)` rounding
__global__ void __launch_bounds__(256) swiglu_kernel(const uint16_t* __restrict__ in, uint16_t* __restrict__ out,
                                                     long rows, long features) {
  const long vecs = features / 8;
  const long total = rows * vecs;
  for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<long>(gridDim.x) * blockDim.x) {
    const long r = i / vecs, c = (i - r * vecs) * 8;
    const uint4 v = *reinterpret_cast<const uint4*>(in + r * 2 * features + c);
    const uint4 g = *reinterpret_cast<const uint4*>(in + r * 2 * features + features + c);
    const uint16_t* vh = reinterpret_cast<const uint16_t*>(&v);
    const uint16_t* gh = reinterpret_cast<const uint16_t*>(&g);
    uint4 o;
    uint16_t* oh = reinterpret_cast<uint16_t*>(&o);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const float x = bf2f(gh[k]);
      const float s = bf2f(f2bf(x / (1.0f + ::expf(-x))));  // at::native silu: x / (1 + exp(-x)) in float
      oh[k] = f2bf(bf2f(vh[k]) * s);
    }
    *reinterpret_cast<uint4*>(out + r * features + c) = o;
  }
}

}  // namespace

void launch_gfx1201_swiglu(int64_t in, int64_t out, int64_t rows, int64_t features, int64_t stream) {
  const long total = rows * (features / 8);
  if (total == 0) return;
  const int blocks = static_cast<int>(std::min<long>((total + 255) / 256, 65535L * 8));
  hipLaunchKernelGGL(swiglu_kernel, dim3(blocks), dim3(256), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(in), reinterpret_cast<uint16_t*>(out), rows, features);
}
