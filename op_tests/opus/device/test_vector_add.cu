// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_vector_add.cu
 * @brief Element-wise vector addition kernel using OPUS gmem helpers.
 * Reference: https://github.com/carlushuang/gcnasm/blob/master/vector_add/vector_add.cpp
 *
 * Demonstrates OPUS make_gmem load / store in a grid-stride loop.
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass: opus.hpp + full kernel body, no hip_runtime.h ──────────────
#include "opus/opus.hpp"

template<int BLOCK_SIZE, int VECTOR_SIZE>
__global__ void vector_add_kernel(const float* a, const float* b, float* result, int n)
{
    auto g_a = opus::make_gmem(a);
    auto g_b = opus::make_gmem(b);
    auto g_r = opus::make_gmem(result);

    int idx = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x();
    int stride = __builtin_amdgcn_grid_size_x();

    for (int i = idx * VECTOR_SIZE; i < n; i += stride * VECTOR_SIZE) {
        auto va = g_a.load<VECTOR_SIZE>(i);
        auto vb = g_b.load<VECTOR_SIZE>(i);

        decltype(va) vr;
        for (int j = 0; j < VECTOR_SIZE; j++) {
            vr[j] = va[j] + vb[j];
        }

        g_r.store<VECTOR_SIZE>(vr, i);
    }
}

template __global__ void vector_add_kernel<256, 4>(const float*, const float*, float*, int);

#else
// ── Host pass: hip_runtime.h for launch API, empty kernel stubs ─────────────
// #include <hip/hip_runtime.h>   // replaced by hip_host_minimal.h for faster builds
#include "hip_host_minimal.h"
#include <cstdio>

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

template<int BLOCK_SIZE, int VECTOR_SIZE>
__global__ void vector_add_kernel(const float* a, const float* b, float* result, int n) {}

extern "C" void run_vector_add(
    const void* d_a,
    const void* d_b,
    void* d_result,
    int n)
{
    const auto* a = static_cast<const float*>(d_a);
    const auto* b = static_cast<const float*>(d_b);
    auto* r = static_cast<float*>(d_result);

    constexpr int BLOCK_SIZE = 256;
    constexpr int VECTOR_SIZE = 4;
    int blocks = n / (BLOCK_SIZE * VECTOR_SIZE);

    hipLaunchKernelGGL(
        (vector_add_kernel<BLOCK_SIZE, VECTOR_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        a, b, r, n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif
