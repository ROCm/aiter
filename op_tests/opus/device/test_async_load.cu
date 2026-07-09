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

// Sentinel pre-filled into LDS for the i_os test; chosen far outside the
// randn() input range so it can never collide with a real copied value.
#define ASYNC_LOAD_IOS_SENTINEL (-123456.0f)

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"

OPUS_D void async_load_waitall()
{
#if defined(__gfx1250__)
    opus::s_wait_loadcnt(opus::number<0>{});
    opus::s_wait_asynccnt(opus::number<0>{});
#else
    opus::s_waitcnt_vmcnt(opus::number<0>{});
#endif
}

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
    async_load_waitall();
    __builtin_amdgcn_s_barrier();

    dst[gid] = smem_buf[tid];
}

template __global__ void async_load_kernel<256>(const float*, float*, int);

// Validates the compile-time immediate offset i_os of gmem::async_load.
// Per the SP3 LDS-DMA spec i_os shifts BOTH ends of the copy: it is added to
// the source byte address AND to the LDS destination address. This kernel is
// arranged so a missing shift on either end is observable:
//   thread tid issues  async_load(smem_buf + tid, /*v_os*/tid, 0, number<IOS_BYTES>)
//     -> reads  src[tid + IOS_ELEMS]      (source shifted by i_os)
//     -> writes smem_buf[tid + IOS_ELEMS] (LDS dest shifted by i_os)
// LDS is pre-seeded with a sentinel and the whole region is copied out, so:
//   dst[0 .. IOS_ELEMS)                         == sentinel  (proves dest shift)
//   dst[IOS_ELEMS .. BLOCK_SIZE + IOS_ELEMS)    == src[j]    (proves source shift)
template<int BLOCK_SIZE, int IOS_BYTES>
__global__ void async_load_ioffset_kernel(const float* __restrict__ src,
                                          float* __restrict__ dst)
{
    constexpr int IOS_ELEMS = IOS_BYTES / sizeof(float);
    constexpr int LDS_SIZE  = BLOCK_SIZE + IOS_ELEMS;
    __shared__ float smem_buf[LDS_SIZE];

    int tid = __builtin_amdgcn_workitem_id_x();

    for (int i = tid; i < LDS_SIZE; i += BLOCK_SIZE) smem_buf[i] = ASYNC_LOAD_IOS_SENTINEL;
    __builtin_amdgcn_s_barrier();

    auto g_src = opus::make_gmem(src, static_cast<unsigned int>(LDS_SIZE * sizeof(float)));
    g_src.async_load<1>(smem_buf + tid, tid, 0, opus::number<IOS_BYTES>{});
    async_load_waitall();
    __builtin_amdgcn_s_barrier();

    for (int i = tid; i < LDS_SIZE; i += BLOCK_SIZE) dst[i] = smem_buf[i];
}

template __global__ void async_load_ioffset_kernel<256, 32>(const float*, float*);
template __global__ void async_load_ioffset_kernel<256, 4092>(const float*, float*);

#else
// ── Host pass ───────────────────────────────────────────────────────────────
// #include <hip/hip_runtime.h>   // replaced by hip_minimal.h for faster builds
#include "opus/hip_minimal.hpp"
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

template<int BLOCK_SIZE, int IOS_BYTES>
__global__ void async_load_ioffset_kernel(const float* __restrict__ src,
                                          float* __restrict__ dst) {}

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

// Single-block launcher for the i_os test. Reads src[0 .. BLOCK_SIZE + IOS_ELEMS)
// and writes dst[0 .. BLOCK_SIZE + IOS_ELEMS). ios_bytes selects the variant.
extern "C" void run_async_load_ioffset(
    const void* d_src,
    void* d_dst,
    int ios_bytes)
{
    const auto* src = static_cast<const float*>(d_src);
    auto* dst = static_cast<float*>(d_dst);

    constexpr int BLOCK_SIZE = 256;
    if (ios_bytes == 32) {
        hipLaunchKernelGGL((async_load_ioffset_kernel<BLOCK_SIZE, 32>),
                           dim3(1), dim3(BLOCK_SIZE), 0, 0, src, dst);
    } else if (ios_bytes == 4092) {
        hipLaunchKernelGGL((async_load_ioffset_kernel<BLOCK_SIZE, 4092>),
                           dim3(1), dim3(BLOCK_SIZE), 0, 0, src, dst);
    } else {
        fprintf(stderr, "run_async_load_ioffset: unsupported ios_bytes=%d\n", ios_bytes);
        return;
    }
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif
