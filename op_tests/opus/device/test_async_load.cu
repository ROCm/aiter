// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_async_load.cu
 * @brief Unit test kernel for opus gmem::async_load (global -> LDS async copy).
 *
 * Demonstrates the async_load path:
 *   1. Each thread issues async_load to copy its portion of global memory into LDS.
 *   2. s_waitcnt_vmcnt(0) waits for all async loads to complete.
 *   3. Data is read back from LDS and written to an output buffer in global memory.
 *
 * The host compares output with input to verify correctness.
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"

template<int BLOCK_SIZE>
__global__ void async_load_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst,
                                  int n)
{
    __shared__ float smem_buf[BLOCK_SIZE];

    int tid = __builtin_amdgcn_workitem_id_x();
    int gid = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + tid;

    if (gid >= n) return;

    auto g_src = opus::make_gmem(src, static_cast<unsigned int>(n * sizeof(float)));
    g_src.async_load<1>(smem_buf + tid, gid);
    opus::s_waitcnt_vmcnt(opus::number<0>{});
    __builtin_amdgcn_s_barrier();

    dst[gid] = smem_buf[tid];
}

template __global__ void async_load_kernel<256>(const float*, float*, int);

#else
// ── Host pass ───────────────────────────────────────────────────────────────
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

template<int BLOCK_SIZE>
__global__ void async_load_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst,
                                  int n) {}

extern "C" void run_async_load(
    const void* d_src,
    void* d_dst,
    int n)
{
    const auto* src = static_cast<const float*>(d_src);
    auto* dst = static_cast<float*>(d_dst);

    constexpr int BLOCK_SIZE = 256;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    hipLaunchKernelGGL(
        (async_load_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        src, dst, n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif
