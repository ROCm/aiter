// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_dtype_convert.cu
 * @brief Unit test kernels for OPUS data type conversion functions.
 *
 * Tests round-trip conversions using opus::cast<> API:
 *   1. FP32 -> BF16 -> FP32  (opus::cast<bf16_t>, opus::cast<fp32_t>)
 *   2. FP32 -> FP16 -> FP32  (opus::cast<fp16_t>, opus::cast<fp32_t>)
 *   3. FP32 -> FP8  -> FP32  (opus::cast<fp8_t>(fp32x4), opus::cast<fp32_t>(fp8x4))
 *   4. FP32 -> FP4  -> FP32  (opus::cast<fp4_t>(fp32x8), opus::cast<fp32_t>(fp4x8))
 *                              gfx950 only -- uses packed x8 conversion
 *
 * Each kernel reads fp32 input, converts down, converts back, writes fp32 output.
 * The host compares output with a reference computed in PyTorch.
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
// FP32 <-> BF16 round-trip
// ---------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_bf16_kernel(const float* __restrict__ in,
                                               float* __restrict__ out,
                                               int n)
{
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (gid >= n) return;

    opus::fp32_t val = in[gid];
    opus::bf16_t tmp = opus::cast<opus::bf16_t>(val);
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
