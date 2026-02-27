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

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"

constexpr int BLOCK_SIZE = 256;

__global__ void cumulative_barrier_kernel(unsigned int* sem, int* accumulator, int n_workgroups)
{
    opus::workgroup_barrier wb{sem};
    int i = __builtin_amdgcn_workgroup_id_x();
    if (i >= n_workgroups) return;

    wb.wait_lt(static_cast<unsigned int>(i));
    if (__builtin_amdgcn_workitem_id_x() == 0)
        __atomic_fetch_add(accumulator, i + 1, __ATOMIC_RELAXED);
    wb.inc();
}

__global__ void streamk_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ workspace,
    float* __restrict__ result,
    unsigned int* sem,
    int n_chunks)
{
    int bid = __builtin_amdgcn_workgroup_id_x();
    int tid = __builtin_amdgcn_workitem_id_x();

    if (bid < n_chunks) {
        const float* chunk = input + bid * BLOCK_SIZE;
        float val = chunk[tid];

        __shared__ float smem[BLOCK_SIZE];
        smem[tid] = val;
        __builtin_amdgcn_s_barrier();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                smem[tid] += smem[tid + stride];
            __builtin_amdgcn_s_barrier();
        }

        if (tid == 0)
            workspace[bid] = smem[0];

        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");

        opus::workgroup_barrier wb{sem};
        wb.inc();
    }
    else {
        opus::workgroup_barrier wb{sem};
        wb.wait_eq(static_cast<unsigned int>(n_chunks));

        __shared__ float smem[BLOCK_SIZE];
        float local_sum = 0.0f;
        for (int i = tid; i < n_chunks; i += BLOCK_SIZE)
            local_sum += workspace[i];
        smem[tid] = local_sum;
        __builtin_amdgcn_s_barrier();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
            if (tid < stride)
                smem[tid] += smem[tid + stride];
            __builtin_amdgcn_s_barrier();
        }

        if (tid == 0)
            *result = smem[0];
    }
}

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

constexpr int BLOCK_SIZE = 256;

__global__ void cumulative_barrier_kernel(unsigned int* sem, int* accumulator, int n_workgroups) {}

__global__ void streamk_reduce_kernel(
    const float* __restrict__ input,
    float* __restrict__ workspace,
    float* __restrict__ result,
    unsigned int* sem,
    int n_chunks) {}

extern "C" void run_workgroup_barrier_cumulative(void* d_accumulator, int n_workgroups)
{
    unsigned int* d_sem = nullptr;
    HIP_CALL(hipMalloc(&d_sem, sizeof(unsigned int)));
    HIP_CALL(hipMemset(d_sem, 0, sizeof(unsigned int)));
    HIP_CALL(hipMemset(d_accumulator, 0, sizeof(int)));

    hipLaunchKernelGGL(
        cumulative_barrier_kernel,
        dim3(n_workgroups), dim3(64), 0, 0,
        d_sem, static_cast<int*>(d_accumulator), n_workgroups);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipFree(d_sem));
}

extern "C" void run_workgroup_barrier_streamk_reduce(
    const void* d_input,
    void* d_workspace,
    void* d_result,
    int n_chunks)
{
    unsigned int* d_sem = nullptr;
    HIP_CALL(hipMalloc(&d_sem, sizeof(unsigned int)));
    HIP_CALL(hipMemset(d_sem, 0, sizeof(unsigned int)));

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
#endif
