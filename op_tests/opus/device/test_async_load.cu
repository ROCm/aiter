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

#include <hip/hip_runtime.h>
#include <cstdio>
#include "opus/opus.hpp"
#include "test_async_load.h"

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

// Each thread loads one float via async_load from global memory into LDS,
// then reads it back from LDS and writes to the output buffer.
// This pattern mirrors the production usage in quant_kernels.cu.
template<int BLOCK_SIZE>
__global__ void async_load_kernel(const float* __restrict__ src,
                                  float* __restrict__ dst,
                                  int n)
{
    __shared__ float smem_buf[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    if (gid >= n) return;

    // Create gmem accessor for src with an explicit size (bytes) for the block.
    auto g_src = opus::make_gmem(src, static_cast<uint32_t>(n * sizeof(float)));

    // Phase 1: async_load one float from global memory into LDS.
    // Each thread in the wavefront has a unique LDS destination (smem_buf + tid)
    // and a unique global offset (gid).
    g_src.async_load<1>(smem_buf + tid, gid);

    // Phase 2: Wait for all async loads in this wavefront to complete.
    opus::s_waitcnt_vmcnt(opus::number<0>{});
    __syncthreads();

    // Phase 3: Read from LDS and write to output global memory.
    dst[gid] = smem_buf[tid];
}

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
