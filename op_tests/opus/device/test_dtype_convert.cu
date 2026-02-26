// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_dtype_convert.cu
 * @brief Unit test kernels for OPUS data type conversion functions.
 *
 * All tests are FP32 -> low-precision -> FP32 round-trips via opus::cast<>.
 * Each kernel reads fp32 input, converts down, converts back, writes fp32 output.
 * The host (Python) compares output with a reference computed in PyTorch.
 *
 * Coverage matrix:
 *
 *   Conversion     | Width   | Cast path                          | Arch
 *   ---------------+---------+------------------------------------+-----------------
 *   FP32 <-> BF16  | scalar  | cast<bf16_t>(fp32_t)               | all (RNE)
 *   FP32 <-> BF16  | x4 vec  | cast<bf16_t>(fp32x4_t)             | all (RNE)
 *   FP32 <-> FP16  | scalar  | cast<fp16_t>(fp32_t)               | all
 *   FP32 <-> FP16  | x4 vec  | cast<fp16_t>(fp32x4_t)             | all
 *   FP32 <-> FP8   | scalar  | cast<fp8_t>(fp32_t)                | gfx942 + gfx950
 *   FP32 <-> FP8   | x2 pk   | cast<fp8_t>(fp32x2_t)              | gfx942 + gfx950
 *   FP32 <-> FP8   | x4 pk   | cast<fp8_t>(fp32x4_t)              | gfx942 + gfx950
 *   FP32 <-> FP8   | x8 fold | cast<fp8_t>(fp32x8_t)  auto-fold   | gfx942 + gfx950
 *   FP32 <-> FP4   | x2 pk   | cast<fp4_t>(fp32x2_t)              | gfx950
 *   FP32 <-> FP4   | x4 pk   | cast<fp4_t>(fp32x4_t)              | gfx950
 *   FP32 <-> FP4   | x8 pk   | cast<fp4_t>(fp32x8_t)              | gfx950
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include "opus/opus.hpp"
#include "test_dtype_convert.h"

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// FP32 <-> BF16 round-trip (round-to-nearest-even on all architectures)
//
// FP32 -> BF16 rounding behaviour differs by GPU architecture:
//   gfx942: opus::cast<bf16_t>(val) defaults to truncation (rm=2).
//           Pass 0_I as 2nd argument to force round-to-nearest-even (RNE).
//   gfx950: opus::cast<bf16_t>(val) uses hardware RNE by default,
//           no 2nd argument needed.
// Both paths produce the same numerical result (IEEE 754 RNE).
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_bf16_kernel(const float* __restrict__ in,
                                               float* __restrict__ out,
                                               int n)
{
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (gid >= n) return;

    using opus::operator""_I;
    opus::fp32_t val = in[gid];
#if defined(__gfx942__) || defined(__gfx9_4_generic__)
    // gfx942: explicit RNE via 2nd parameter (0_I)
    opus::bf16_t tmp = opus::cast<opus::bf16_t>(val, 0_I);
#else
    // gfx950+: hardware default is already RNE
    opus::bf16_t tmp = opus::cast<opus::bf16_t>(val);
#endif
    out[gid] = opus::cast<opus::fp32_t>(tmp);
}

extern "C" void run_dtype_convert_fp32_bf16(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_bf16_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP32 <-> FP16 round-trip
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp16_kernel(const float* __restrict__ in,
                                               float* __restrict__ out,
                                               int n)
{
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (gid >= n) return;

    opus::fp32_t val = in[gid];
    opus::fp16_t tmp = opus::cast<opus::fp16_t>(val);
    out[gid] = opus::cast<opus::fp32_t>(tmp);
}

extern "C" void run_dtype_convert_fp32_fp16(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp16_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP32 <-> FP8 (e4m3) round-trip  --  packed x4 conversion
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp8_kernel(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              int n)
{
    // Each thread processes 4 elements via packed fp32x4 <-> fp8x4 conversion.
    int gid = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;
    if (gid >= n) return;

    opus::fp32x4_t v_in;
    v_in[0] = in[gid + 0];
    v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2];
    v_in[3] = in[gid + 3];

    // FP32 -> FP8 (packed x4)
    auto v_fp8 = opus::cast<opus::fp8_t>(v_in);
    // FP8 -> FP32 (packed x4)
    auto v_out = opus::cast<opus::fp32_t>(v_fp8);

    out[gid + 0] = v_out[0];
    out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2];
    out[gid + 3] = v_out[3];
}

extern "C" void run_dtype_convert_fp32_fp8(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMS_PER_THREAD = 4;
    int threads_needed = n / ELEMS_PER_THREAD;
    int blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp8_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP32 <-> FP4 (e2m1) round-trip  --  packed x8 conversion (gfx950 only)
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp4_kernel(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              int n)
{
    // Each thread processes 8 elements via packed fp32x8 <-> fp4x8 conversion.
    int gid = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 8;
    if (gid >= n) return;

    opus::fp32x8_t v_in;
    v_in[0] = in[gid + 0];
    v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2];
    v_in[3] = in[gid + 3];
    v_in[4] = in[gid + 4];
    v_in[5] = in[gid + 5];
    v_in[6] = in[gid + 6];
    v_in[7] = in[gid + 7];

    // FP32 -> FP4 (packed x8)
    auto v_fp4 = opus::cast<opus::fp4_t>(v_in);
    // FP4 -> FP32 (packed x8)
    auto v_out = opus::cast<opus::fp32_t>(v_fp4);

    out[gid + 0] = v_out[0];
    out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2];
    out[gid + 3] = v_out[3];
    out[gid + 4] = v_out[4];
    out[gid + 5] = v_out[5];
    out[gid + 6] = v_out[6];
    out[gid + 7] = v_out[7];
}

extern "C" void run_dtype_convert_fp32_fp4(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMS_PER_THREAD = 8;
    int threads_needed = n / ELEMS_PER_THREAD;
    int blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp4_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP32 <-> FP8 (e4m3) scalar round-trip
//   fp32 -> fp8 via __builtin_amdgcn_cvt_pk_fp8_f32 (lo half only)
//   fp8 -> fp32 via __builtin_amdgcn_cvt_f32_fp8
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp8_scalar_kernel(const float* __restrict__ in,
                                                     float* __restrict__ out,
                                                     int n)
{
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (gid >= n) return;

    opus::fp32_t val = in[gid];
    opus::fp8_t tmp = opus::cast<opus::fp8_t>(val);
    out[gid] = opus::cast<opus::fp32_t>(tmp);
}

extern "C" void run_dtype_convert_fp32_fp8_scalar(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp8_scalar_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Vectorized BF16: FP32x4 -> BF16x4 -> FP32x4 via generic vectorized cast()
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_bf16_vec4_kernel(const float* __restrict__ in,
                                                    float* __restrict__ out,
                                                    int n)
{
    int gid = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;
    if (gid >= n) return;

    using opus::operator""_I;
    opus::fp32x4_t v_in;
    v_in[0] = in[gid + 0];
    v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2];
    v_in[3] = in[gid + 3];

#if defined(__gfx942__) || defined(__gfx9_4_generic__)
    auto v_bf16 = opus::cast<opus::bf16_t>(v_in, 0_I);
#else
    auto v_bf16 = opus::cast<opus::bf16_t>(v_in);
#endif
    auto v_out = opus::cast<opus::fp32_t>(v_bf16);

    out[gid + 0] = v_out[0];
    out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2];
    out[gid + 3] = v_out[3];
}

extern "C" void run_dtype_convert_fp32_bf16_vec4(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int threads_needed = n / 4;
    int blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_bf16_vec4_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// Vectorized FP16: FP32x4 -> FP16x4 -> FP32x4 via generic vectorized cast()
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp16_vec4_kernel(const float* __restrict__ in,
                                                    float* __restrict__ out,
                                                    int n)
{
    int gid = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;
    if (gid >= n) return;

    opus::fp32x4_t v_in;
    v_in[0] = in[gid + 0];
    v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2];
    v_in[3] = in[gid + 3];

    auto v_fp16 = opus::cast<opus::fp16_t>(v_in);
    auto v_out  = opus::cast<opus::fp32_t>(v_fp16);

    out[gid + 0] = v_out[0];
    out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2];
    out[gid + 3] = v_out[3];
}

extern "C" void run_dtype_convert_fp32_fp16_vec4(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int threads_needed = n / 4;
    int blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp16_vec4_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP8 packed x2: FP32x2 -> FP8x2 -> FP32x2
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp8_x2_kernel(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int n)
{
    int gid = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 2;
    if (gid >= n) return;

    opus::fp32x2_t v_in;
    v_in[0] = in[gid + 0];
    v_in[1] = in[gid + 1];

    auto v_fp8 = opus::cast<opus::fp8_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp8);

    out[gid + 0] = v_out[0];
    out[gid + 1] = v_out[1];
}

extern "C" void run_dtype_convert_fp32_fp8_x2(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int threads_needed = n / 2;
    int blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp8_x2_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP8 auto-fold x8: FP32x8 -> (auto-fold 2x FP8x4) -> FP32x8
// Tests the generic vectorized cast() entry point for fp8 with size%4==0
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp8_vec8_kernel(const float* __restrict__ in,
                                                   float* __restrict__ out,
                                                   int n)
{
    int gid = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 8;
    if (gid >= n) return;

    opus::fp32x8_t v_in;
    for (int i = 0; i < 8; ++i) v_in[i] = in[gid + i];

    // cast<fp8_t>(fp32x8_t) auto-folds internally, returns flat fp8x8_t
    auto v_fp8 = opus::cast<opus::fp8_t>(v_in);
    // cast<fp32_t>(fp8x8_t) auto-folds internally, returns flat fp32x8_t
    auto v_out = opus::cast<opus::fp32_t>(v_fp8);

    for (int i = 0; i < 8; ++i) out[gid + i] = v_out[i];
}

extern "C" void run_dtype_convert_fp32_fp8_vec8(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int threads_needed = n / 8;
    int blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp8_vec8_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP4 packed x2: FP32x2 -> FP4(x2) -> FP32x2 (gfx950 only)
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp4_x2_kernel(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int n)
{
    int gid = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 2;
    if (gid >= n) return;

    opus::fp32x2_t v_in;
    v_in[0] = in[gid + 0];
    v_in[1] = in[gid + 1];

    auto v_fp4 = opus::cast<opus::fp4_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp4);

    out[gid + 0] = v_out[0];
    out[gid + 1] = v_out[1];
}

extern "C" void run_dtype_convert_fp32_fp4_x2(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int threads_needed = n / 2;
    int blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp4_x2_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP4 packed x4: FP32x4 -> FP4(x4) -> FP32x4 (gfx950 only)
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp4_x4_kernel(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int n)
{
    int gid = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * 4;
    if (gid >= n) return;

    opus::fp32x4_t v_in;
    v_in[0] = in[gid + 0];
    v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2];
    v_in[3] = in[gid + 3];

    auto v_fp4 = opus::cast<opus::fp4_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp4);

    out[gid + 0] = v_out[0];
    out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2];
    out[gid + 3] = v_out[3];
}

extern "C" void run_dtype_convert_fp32_fp4_x4(const void* d_in, void* d_out, int n)
{
    constexpr int BLOCK_SIZE = 256;
    int threads_needed = n / 4;
    int blocks = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(
        (dtype_convert_fp32_fp4_x4_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
