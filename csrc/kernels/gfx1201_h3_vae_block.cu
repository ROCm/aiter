// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// MiniMax-H3 video VAE ViT decoder block pieces: residual + RMSNorm (FP32 stream, FP16 out), per-head QK
// RMSNorm + RoPE (FP16), SwiGLU (FP16).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "gfx1201_h3_ops.h"

namespace {

constexpr int kThreads = 256;
constexpr int kMaxChunks = 4;  // D <= 256 * 8 * 4

__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

__device__ __forceinline__ float block_sum(float v, float* red) {
  v = wave_sum(v);
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  if (lane == 0) red[wave] = v;
  __syncthreads();
  float t = lane < (kThreads / 32) ? red[lane] : 0.f;
  t = wave_sum(t);
  return t;
}

// h[row] (+)= float(a[row]) * scale ; out = half(h * rsqrt(mean(h^2)+eps) * w)
__global__ void __launch_bounds__(kThreads) residual_rms_kernel(
    float* __restrict__ h, const __half* __restrict__ a, const float* __restrict__ scale,
    const float* __restrict__ w, __half* __restrict__ out, int D, float eps) {
  __shared__ float red[kThreads / 32];
  const long row = blockIdx.x;
  float* hr = h + row * D;
  const int chunks = D / (kThreads * 8);
  float v[kMaxChunks][8];
  float ss = 0.f;
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    if (c >= chunks) break;
    const int col = (c * kThreads + threadIdx.x) * 8;
    float4 x0 = *reinterpret_cast<const float4*>(hr + col);
    float4 x1 = *reinterpret_cast<const float4*>(hr + col + 4);
    v[c][0] = x0.x; v[c][1] = x0.y; v[c][2] = x0.z; v[c][3] = x0.w;
    v[c][4] = x1.x; v[c][5] = x1.y; v[c][6] = x1.z; v[c][7] = x1.w;
    if (a != nullptr) {
      uint4 araw = *reinterpret_cast<const uint4*>(a + row * D + col);
      const __half* ah = reinterpret_cast<const __half*>(&araw);
      float4 s0 = *reinterpret_cast<const float4*>(scale + col);
      float4 s1 = *reinterpret_cast<const float4*>(scale + col + 4);
      const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
      for (int i = 0; i < 8; ++i) v[c][i] = __fadd_rn(v[c][i], __fmul_rn(__half2float(ah[i]), sv[i]));
      *reinterpret_cast<float4*>(hr + col) = make_float4(v[c][0], v[c][1], v[c][2], v[c][3]);
      *reinterpret_cast<float4*>(hr + col + 4) = make_float4(v[c][4], v[c][5], v[c][6], v[c][7]);
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) ss = fmaf(v[c][i], v[c][i], ss);
  }
  if (out == nullptr) return;
  ss = block_sum(ss, red);
  const float rstd = rsqrtf(ss / static_cast<float>(D) + eps);
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    if (c >= chunks) break;
    const int col = (c * kThreads + threadIdx.x) * 8;
    float4 w0 = *reinterpret_cast<const float4*>(w + col);
    float4 w1 = *reinterpret_cast<const float4*>(w + col + 4);
    const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
    uint4 oraw;
    __half* oh = reinterpret_cast<__half*>(&oraw);
#pragma unroll
    for (int i = 0; i < 8; ++i) oh[i] = __float2half_rn(__fmul_rn(__fmul_rn(v[c][i], rstd), wv[i]));
    *reinterpret_cast<uint4*>(out + row * D + col) = oraw;
  }
}

// In-place per-(token, head) RMSNorm(64, no affine) -> fp16, then fp16 rotary on the first ROT dims.
template <int HD, int ROT>
__global__ void __launch_bounds__(kThreads) qk_norm_rope_kernel(
    __half* __restrict__ q, __half* __restrict__ k, const __half* __restrict__ cos16,
    const __half* __restrict__ sin16, long rows, int heads, long q_stride, long k_stride, float eps) {
  const long t = static_cast<long>(blockIdx.x) * kThreads + threadIdx.x;
  if (t >= 2 * rows * heads) return;
  const bool is_k = t >= rows * heads;
  const long r = is_k ? t - rows * heads : t;
  const long tok = r / heads;
  const int head = static_cast<int>(r - tok * heads);
  __half* p = (is_k ? k + tok * k_stride : q + tok * q_stride) + head * HD;
  __half y[HD];
  float ss = 0.f;
  float x[HD];
#pragma unroll
  for (int i = 0; i < HD; i += 8) {
    uint4 raw = *reinterpret_cast<const uint4*>(p + i);
    const __half* hv = reinterpret_cast<const __half*>(&raw);
#pragma unroll
    for (int j = 0; j < 8; ++j) { x[i + j] = __half2float(hv[j]); ss = fmaf(x[i + j], x[i + j], ss); }
  }
  const float rstd = rsqrtf(ss / static_cast<float>(HD) + eps);
#pragma unroll
  for (int i = 0; i < HD; ++i) y[i] = __float2half_rn(__fmul_rn(x[i], rstd));
  const __half* cr = cos16 + tok * ROT;
  const __half* sr = sin16 + tok * ROT;
  __half o[HD];
#pragma unroll
  for (int i = 0; i < ROT; ++i) {
    const __half rot = i < ROT / 2 ? __hneg(y[i + ROT / 2]) : y[i - ROT / 2];
    const __half a = __float2half_rn(__fmul_rn(__half2float(y[i]), __half2float(cr[i])));
    const __half b = __float2half_rn(__fmul_rn(__half2float(rot), __half2float(sr[i])));
    o[i] = __float2half_rn(__fadd_rn(__half2float(a), __half2float(b)));
  }
#pragma unroll
  for (int i = ROT; i < HD; ++i) o[i] = y[i];
#pragma unroll
  for (int i = 0; i < HD; i += 8) *reinterpret_cast<uint4*>(p + i) = *reinterpret_cast<const uint4*>(o + i);
}

// out[r, c] = x * half(silu(gate)), x = in[r, c], gate = in[r, F + c]
__global__ void __launch_bounds__(kThreads) swiglu_kernel(
    const __half* __restrict__ in, __half* __restrict__ out, long rows, int F, long in_stride) {
  const long t = static_cast<long>(blockIdx.x) * kThreads + threadIdx.x;
  const int vec_per_row = F / 8;
  if (t >= rows * vec_per_row) return;
  const long r = t / vec_per_row;
  const int c = static_cast<int>(t - r * vec_per_row) * 8;
  uint4 xr = *reinterpret_cast<const uint4*>(in + r * in_stride + c);
  uint4 gr = *reinterpret_cast<const uint4*>(in + r * in_stride + F + c);
  const __half* xh = reinterpret_cast<const __half*>(&xr);
  const __half* gh = reinterpret_cast<const __half*>(&gr);
  uint4 oraw;
  __half* oh = reinterpret_cast<__half*>(&oraw);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const float g = __half2float(gh[j]);
    const __half s = __float2half_rn(g / (1.0f + expf(-g)));
    oh[j] = __float2half_rn(__fmul_rn(__half2float(xh[j]), __half2float(s)));
  }
  *reinterpret_cast<uint4*>(out + r * static_cast<long>(F) + c) = oraw;
}

}  // namespace

void launch_gfx1201_vae_residual_rms(int64_t hidden, int64_t addend, int64_t scale, int64_t weight, int64_t out,
                                     int64_t rows, int64_t dim, double eps, int64_t stream) {
  if (rows == 0) return;
  hipLaunchKernelGGL(residual_rms_kernel, dim3(rows), dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<float*>(hidden), reinterpret_cast<const __half*>(addend),
                     reinterpret_cast<const float*>(scale), reinterpret_cast<const float*>(weight),
                     reinterpret_cast<__half*>(out), static_cast<int>(dim), static_cast<float>(eps));
}

void launch_gfx1201_vae_qk_norm_rope(int64_t query, int64_t key, int64_t cos16, int64_t sin16, int64_t rows,
                                     int64_t heads, int64_t query_stride, int64_t key_stride, double eps,
                                     int64_t stream) {
  const long total = 2 * rows * heads;
  if (total == 0) return;
  hipLaunchKernelGGL((qk_norm_rope_kernel<64, 48>), dim3((total + kThreads - 1) / kThreads), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<__half*>(query),
                     reinterpret_cast<__half*>(key), reinterpret_cast<const __half*>(cos16),
                     reinterpret_cast<const __half*>(sin16), rows, static_cast<int>(heads), query_stride, key_stride,
                     static_cast<float>(eps));
}

void launch_gfx1201_vae_swiglu(int64_t in, int64_t out, int64_t rows, int64_t features, int64_t in_stride,
                               int64_t stream) {
  const long total = rows * (features / 8);
  if (total == 0) return;
  hipLaunchKernelGGL(swiglu_kernel, dim3((total + kThreads - 1) / kThreads), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const __half*>(in),
                     reinterpret_cast<__half*>(out), rows, static_cast<int>(features), in_stride);
}
