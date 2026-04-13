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

#include "opus/opus.hpp"
#ifndef __HIP_DEVICE_COMPILE__
// #include <hip/hip_runtime.h>   // replaced by hip_host_minimal.h for faster builds
#include "hip_host_minimal.h"
#endif

// ═══════════════════════════════════════════════════════════════════════════
// Kernel definitions (visible to both device and host passes)
// ═══════════════════════════════════════════════════════════════════════════

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_bf16_kernel(const float* __restrict__ in,
                                               float* __restrict__ out,
                                               int n)
{
    int gid = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x();
    if (gid >= n) return;

    using opus::operator""_I;
    opus::fp32_t val = in[gid];
#if defined(__gfx942__) || defined(__gfx9_4_generic__)
    opus::bf16_t tmp = opus::cast<opus::bf16_t>(val, 0_I);
#else
    opus::bf16_t tmp = opus::cast<opus::bf16_t>(val);
#endif
    out[gid] = opus::cast<opus::fp32_t>(tmp);
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp16_kernel(const float* __restrict__ in,
                                               float* __restrict__ out,
                                               int n)
{
    int gid = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x();
    if (gid >= n) return;

    opus::fp32_t val = in[gid];
    opus::fp16_t tmp = opus::cast<opus::fp16_t>(val);
    out[gid] = opus::cast<opus::fp32_t>(tmp);
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp8_kernel(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              int n)
{
    int gid = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * 4;
    if (gid >= n) return;

    opus::fp32x4_t v_in;
    v_in[0] = in[gid + 0]; v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2]; v_in[3] = in[gid + 3];

    auto v_fp8 = opus::cast<opus::fp8_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp8);

    out[gid + 0] = v_out[0]; out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2]; out[gid + 3] = v_out[3];
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp4_kernel(const float* __restrict__ in,
                                              float* __restrict__ out,
                                              int n)
{
    int gid = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * 8;
    if (gid >= n) return;

    opus::fp32x8_t v_in;
    for (int i = 0; i < 8; ++i) v_in[i] = in[gid + i];

    auto v_fp4 = opus::cast<opus::fp4_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp4);

    for (int i = 0; i < 8; ++i) out[gid + i] = v_out[i];
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp8_scalar_kernel(const float* __restrict__ in,
                                                     float* __restrict__ out,
                                                     int n)
{
    int gid = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x();
    if (gid >= n) return;

    opus::fp32_t val = in[gid];
    opus::fp8_t tmp = opus::cast<opus::fp8_t>(val);
    out[gid] = opus::cast<opus::fp32_t>(tmp);
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_bf16_vec4_kernel(const float* __restrict__ in,
                                                    float* __restrict__ out,
                                                    int n)
{
    int gid = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * 4;
    if (gid >= n) return;

    using opus::operator""_I;
    opus::fp32x4_t v_in;
    v_in[0] = in[gid + 0]; v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2]; v_in[3] = in[gid + 3];

#if defined(__gfx942__) || defined(__gfx9_4_generic__)
    auto v_bf16 = opus::cast<opus::bf16_t>(v_in, 0_I);
#else
    auto v_bf16 = opus::cast<opus::bf16_t>(v_in);
#endif
    auto v_out = opus::cast<opus::fp32_t>(v_bf16);

    out[gid + 0] = v_out[0]; out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2]; out[gid + 3] = v_out[3];
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp16_vec4_kernel(const float* __restrict__ in,
                                                    float* __restrict__ out,
                                                    int n)
{
    int gid = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * 4;
    if (gid >= n) return;

    opus::fp32x4_t v_in;
    v_in[0] = in[gid + 0]; v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2]; v_in[3] = in[gid + 3];

    auto v_fp16 = opus::cast<opus::fp16_t>(v_in);
    auto v_out  = opus::cast<opus::fp32_t>(v_fp16);

    out[gid + 0] = v_out[0]; out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2]; out[gid + 3] = v_out[3];
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp8_x2_kernel(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int n)
{
    int gid = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * 2;
    if (gid >= n) return;

    opus::fp32x2_t v_in;
    v_in[0] = in[gid + 0]; v_in[1] = in[gid + 1];

    auto v_fp8 = opus::cast<opus::fp8_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp8);

    out[gid + 0] = v_out[0]; out[gid + 1] = v_out[1];
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp8_vec8_kernel(const float* __restrict__ in,
                                                   float* __restrict__ out,
                                                   int n)
{
    int gid = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * 8;
    if (gid >= n) return;

    opus::fp32x8_t v_in;
    for (int i = 0; i < 8; ++i) v_in[i] = in[gid + i];

    auto v_fp8 = opus::cast<opus::fp8_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp8);

    for (int i = 0; i < 8; ++i) out[gid + i] = v_out[i];
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp4_x2_kernel(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int n)
{
    int gid = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * 2;
    if (gid >= n) return;

    opus::fp32x2_t v_in;
    v_in[0] = in[gid + 0]; v_in[1] = in[gid + 1];

    auto v_fp4 = opus::cast<opus::fp4_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp4);

    out[gid + 0] = v_out[0]; out[gid + 1] = v_out[1];
}

template<int BLOCK_SIZE>
__global__ void dtype_convert_fp32_fp4_x4_kernel(const float* __restrict__ in,
                                                 float* __restrict__ out,
                                                 int n)
{
    int gid = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * 4;
    if (gid >= n) return;

    opus::fp32x4_t v_in;
    v_in[0] = in[gid + 0]; v_in[1] = in[gid + 1];
    v_in[2] = in[gid + 2]; v_in[3] = in[gid + 3];

    auto v_fp4 = opus::cast<opus::fp4_t>(v_in);
    auto v_out = opus::cast<opus::fp32_t>(v_fp4);

    out[gid + 0] = v_out[0]; out[gid + 1] = v_out[1];
    out[gid + 2] = v_out[2]; out[gid + 3] = v_out[3];
}

// ═══════════════════════════════════════════════════════════════════════════
// Explicit template instantiations (device pass needs these since it can't
// see the hipLaunchKernelGGL calls in the host-only block below)
// ═══════════════════════════════════════════════════════════════════════════
template __global__ void dtype_convert_fp32_bf16_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp16_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp8_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp4_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp8_scalar_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_bf16_vec4_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp16_vec4_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp8_x2_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp8_vec8_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp4_x2_kernel<256>(const float*, float*, int);
template __global__ void dtype_convert_fp32_fp4_x4_kernel<256>(const float*, float*, int);

// ═══════════════════════════════════════════════════════════════════════════
// Host launch functions (host pass only)
// ═══════════════════════════════════════════════════════════════════════════
#ifndef __HIP_DEVICE_COMPILE__
#include <cstdio>

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

extern "C" void run_dtype_convert_fp32_bf16(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_bf16_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp16(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp16_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp8(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n / 4 + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp8_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp4(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n / 8 + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp4_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp8_scalar(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp8_scalar_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_bf16_vec4(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n / 4 + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_bf16_vec4_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp16_vec4(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n / 4 + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp16_vec4_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp8_x2(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n / 2 + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp8_x2_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp8_vec8(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n / 8 + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp8_vec8_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp4_x2(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n / 2 + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp4_x2_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_dtype_convert_fp32_fp4_x4(const void* d_in, void* d_out, int n)
{
    constexpr int BS = 256;
    int blocks = (n / 4 + BS - 1) / BS;
    hipLaunchKernelGGL((dtype_convert_fp32_fp4_x4_kernel<BS>),
        dim3(blocks), dim3(BS), 0, 0,
        static_cast<const float*>(d_in), static_cast<float*>(d_out), n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

#endif // !__HIP_DEVICE_COMPILE__
