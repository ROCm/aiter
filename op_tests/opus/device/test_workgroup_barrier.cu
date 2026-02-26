// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_workgroup_barrier.cu
 * @brief Device tests for opus::workgroup_barrier.
 *
 * Test 1 (cumulative): N workgroups synchronize via wait_lt + inc.
 * Test 2 (stream-K reduce): N+1 workgroups cooperate — N producers reduce chunks,
 *         1 consumer waits for all producers then sums partial results.
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include "opus/opus.hpp"
#include "test_workgroup_barrier.h"

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Test 1: cumulative barrier (wait_lt + inc)
// ---------------------------------------------------------------------------
__global__ void cumulative_barrier_kernel(uint32_t* sem, int* accumulator, int n_workgroups)
{
    opus::workgroup_barrier wb{sem};
    int i = blockIdx.x;
    if (i >= n_workgroups) return;

    wb.wait_lt(static_cast<uint32_t>(i));
    if (threadIdx.x == 0)
        atomicAdd(accumulator, i + 1);
    wb.inc();
}

extern "C" void run_workgroup_barrier_cumulative(void* d_accumulator, int n_workgroups)
{
    uint32_t* d_sem = nullptr;
    HIP_CALL(hipMalloc(&d_sem, sizeof(uint32_t)));
    HIP_CALL(hipMemset(d_sem, 0, sizeof(uint32_t)));
    HIP_CALL(hipMemset(d_accumulator, 0, sizeof(int)));

    hipLaunchKernelGGL(
        cumulative_barrier_kernel,
        dim3(n_workgroups), dim3(64), 0, 0,
        d_sem, static_cast<int*>(d_accumulator), n_workgroups);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipFree(d_sem));
}

// ---------------------------------------------------------------------------
// Test 2: stream-K style reduce (wait_eq + inc)
// ---------------------------------------------------------------------------
constexpr int BLOCK_SIZE = 256;

__global__ void streamk_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ workspace,
    float* __restrict__ result,
    uint32_t* sem,
    int n_chunks)
{
    int bid = blockIdx.x;

    if (bid < n_chunks) {
        // Producer: reduce 256 elements starting at input[bid * 256]
        const float* chunk = input + bid * BLOCK_SIZE;
        float val = chunk[threadIdx.x];

        // Tree reduction in shared memory
        __shared__ float smem[BLOCK_SIZE];
        smem[threadIdx.x] = val;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            workspace[bid] = smem[0];

        // Ensure workspace write is globally visible before signaling the consumer.
        __threadfence();

        opus::workgroup_barrier wb{sem};
        wb.inc();
    }
    else {
        // Consumer: wait for all N producers, then sum workspace[0..N-1]
        opus::workgroup_barrier wb{sem};
        wb.wait_eq(static_cast<uint32_t>(n_chunks));

        // Use threads cooperatively to load workspace into smem, then reduce
        __shared__ float smem[BLOCK_SIZE];
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < n_chunks; i += BLOCK_SIZE)
            local_sum += workspace[i];
        smem[threadIdx.x] = local_sum;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            *result = smem[0];
    }
}

extern "C" void run_workgroup_barrier_streamk_reduce(
    const void* d_input,
    void* d_workspace,
    void* d_result,
    int n_chunks)
{
    uint32_t* d_sem = nullptr;
    HIP_CALL(hipMalloc(&d_sem, sizeof(uint32_t)));
    HIP_CALL(hipMemset(d_sem, 0, sizeof(uint32_t)));

    hipLaunchKernelGGL(
        streamk_reduce_kernel,
        dim3(n_chunks + 1), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_input),
        static_cast<float*>(d_workspace),
        static_cast<float*>(d_result),
        d_sem,
        n_chunks);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipFree(d_sem));
}
