// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// Gated residual: out = bf16(res + bf16(gate[idx[row]] * proj)), FP32 math, NaN -> 0x7fff like Triton.
#include <hip/hip_runtime.h>

#include <algorithm>

#include "gfx1201_h3_ops.h"

namespace {

__device__ __forceinline__ float bf2f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {  // round-to-nearest-even; NaN -> 0x7fff like Triton's AMD cvt
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fffu;
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

// out[r, c] = bf16(res[r, c] + bf16(gate[idx[r], c] * proj[r, c])), all math in FP32 (radeon_coresw_ops._gated_residual)
template <typename Index, int VEC>
__global__ void __launch_bounds__(256) gated_residual_kernel(const uint16_t* __restrict__ res, const uint16_t* __restrict__ proj,
                                                             const uint16_t* __restrict__ gate, const Index* __restrict__ idx,
                                                             uint16_t* __restrict__ out, long rows, long dim, long gate_stride) {
  const long vecs = dim / VEC, total = rows * vecs;
  for (long i = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; i < total; i += static_cast<long>(gridDim.x) * blockDim.x) {
    const long r = i / vecs, c = (i - r * vecs) * VEC, o = r * dim + c;
    const uint16_t* g = gate + static_cast<long>(idx[r]) * gate_stride + c;
    if constexpr (VEC == 8) {
      const uint4 a = *reinterpret_cast<const uint4*>(res + o), b = *reinterpret_cast<const uint4*>(proj + o),
                  w = *reinterpret_cast<const uint4*>(g);
      const uint16_t *ah = reinterpret_cast<const uint16_t*>(&a), *bh = reinterpret_cast<const uint16_t*>(&b),
                     *wh = reinterpret_cast<const uint16_t*>(&w);
      uint4 y;
      uint16_t* yh = reinterpret_cast<uint16_t*>(&y);
#pragma unroll
      for (int k = 0; k < 8; ++k) yh[k] = f2bf(bf2f(ah[k]) + bf2f(f2bf(bf2f(wh[k]) * bf2f(bh[k]))));
      *reinterpret_cast<uint4*>(out + o) = y;
    } else {
      out[o] = f2bf(bf2f(res[o]) + bf2f(f2bf(bf2f(g[0]) * bf2f(proj[o]))));
    }
  }
}

}  // namespace

namespace {

template <typename Index>
void launch(const uint16_t* residual, const uint16_t* projected, const uint16_t* gate, const Index* indices,
            uint16_t* out, long rows, long dim, long gate_stride, bool vectorized, hipStream_t stream) {
  const long total = rows * (vectorized ? dim / 8 : dim);
  if (total == 0) return;
  const int blocks = static_cast<int>(std::min<long>((total + 255) / 256, 65535L * 8));
  if (vectorized)
    hipLaunchKernelGGL((gated_residual_kernel<Index, 8>), dim3(blocks), dim3(256), 0, stream, residual, projected, gate,
                       indices, out, rows, dim, gate_stride);
  else
    hipLaunchKernelGGL((gated_residual_kernel<Index, 1>), dim3(blocks), dim3(256), 0, stream, residual, projected, gate,
                       indices, out, rows, dim, gate_stride);
}

}  // namespace

void launch_gfx1201_gated_residual(int64_t residual, int64_t projected, int64_t gate, int64_t indices,
                                   bool indices_int64, int64_t out, int64_t rows, int64_t dim, int64_t gate_stride,
                                   bool vectorized, int64_t stream) {
  auto p = [](int64_t v) { return reinterpret_cast<const uint16_t*>(v); };
  hipStream_t s = reinterpret_cast<hipStream_t>(stream);
  if (indices_int64)
    launch(p(residual), p(projected), p(gate), reinterpret_cast<const int64_t*>(indices),
           reinterpret_cast<uint16_t*>(out), rows, dim, gate_stride, vectorized, s);
  else
    launch(p(residual), p(projected), p(gate), reinterpret_cast<const int32_t*>(indices),
           reinterpret_cast<uint16_t*>(out), rows, dim, gate_stride, vectorized, s);
}
