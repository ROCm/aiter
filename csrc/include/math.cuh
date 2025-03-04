/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
// #ifdef __HIPCC_RTC__  // Probably redundant
// #undef __HIPCC_RTC__
// #endif
// #ifndef __HIP_PLATFORM_AMD__
// #define __HIP_PLATFORM_AMD__
// #endif
// #ifdef __HIP_PLATFORM_NVIDIA__  // Redundant
// #undef __HIP_PLATFORM_NVIDIA__
// #endif
// #ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
// #define HIP_ENABLE_WARP_SYNC_BUILTINS 1
// #endif
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#elif defined(__CUDACC__) || defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__)) || defined(__CUDACC_RTC__)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif

#include <cstdint>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
__device__ float compute_ex2_approx_ftz(float x) {
  // Define a small threshold for FTZ
  // Alternative __FLT_DENORM_MIN__
  const float ftz_threshold = 1e-37f;

  // Calculate the base-2 exponential
  float y = exp2f(x);

  // Apply FTZ behavior
  if (fabsf(y) < ftz_threshold) {
    y = 0.0f;
  }
  return y;
}
#endif

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
  float y;
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  y = compute_ex2_approx_ftz(x);
#else
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
#endif
  return y;
}

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
__device__ float compute_log2_approx_ftz(float x) {
  // Flush subnormals to zero
  if (fabsf(x) < __FLT_DENORM_MIN__) {
    x = 0.0f;
  }
  // Hardware-accelerated log2 approximation
  return __log2f(x);
}
#endif

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
  float y;
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  y = compute_log2_approx_ftz(x);
#else
  asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
#endif
  return y;
}

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
__device__ half2 compute_ex2_approx_f16x2(uint32_t x) {
  // Unpack the two 16-bit half-precision floats from the input
  // Extract lower 16 bits
  __half x0 = __ushort_as_half(x & 0xFFFF);
  // Extract upper 16 bits
  __half x1 = __ushort_as_half((x >> 16) & 0xFFFF);

  // Compute exp2 (approximation) for each half
  // CUDA intrinsic for approximate exp2
  __half y0 = __float2half(exp2f(__half2float(x0)));
  __half y1 = __float2half(exp2f(__half2float(x1)));

  return __halves2half2(y0, y1);
}
#endif

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  return compute_ex2_approx_f16x2(x_u32);
#else
  asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
  return uint32_as_half2(y_u32);
#endif
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  return __float2half(exp2f(__half2float(x)));
#else
  asm volatile("ex2.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
#endif
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  // FIXME:
  // 1. FTZ may globally be controlled by hardware level configurations.
  // 2. The "round-to-nearest-even" mode is indicated by this intrinsic.
  y = __frcp_rn(x);
#else
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
#endif
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  // FIXME: May miss the synchronization across threads in the warp.
  return __shfl_xor(x, lane_mask);
#else
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
#endif
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  // FIXME: May miss the synchronization across threads in the warp.
  return __shfl_xor(x, lane_mask);
#else
  return __shfl_xor_sync(0xffffffff, x, lane_mask);
#endif
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  return __frsqrt_rn(x);
#else
  float y;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#endif
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  // FIXME:
  // In terms of precision vs. performance, a custom "tanhf" may be needed.
  return tanhf(x);
#else
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#endif
}

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
__device__ uint32_t tanh_approx_uint32(uint32_t x) {
  // Convert uint32_t to float
  float x_f32 = __uint_as_float(x);
  // FIXME:
  // In terms of precision vs. performance, a custom "tanhf" may be needed.
  float res = tanhf(x_f32);
  return __float_as_uint(res);
}
#endif

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  y_u32 = tanh_approx_uint32(x_u32);
#else
  asm volatile("tanh.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
#endif
  return uint32_as_half2(y_u32);
}

#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
__device__ __half tanh_approx_half(__half x) {
  // Approximation: tanh(x) ~ x * (1 - 0.5 * x^2)
  // FIXME:
  // In terms of precision vs. performance, a custom "tanhf" may be needed.
  float x_f32 = __half2float(x);
  float y_f32 = tanhf(x_f32);
  return __float2half(y_f32);
}
#endif

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
#if defined(__HIPCC__) || (defined(__clang__) && defined(__HIP__)) || defined(__HIPCC_RTC__)
  return tanh_approx_half(x);
#else
  ushort y_u16;
  asm volatile("tanh.approx.f16 %0, %1;" : "=h"(y_u16) : "h"(__half_as_ushort(x)));
  return __ushort_as_half(y_u16);
#endif
}

}  // namespace math
}  // namespace flashinfer
