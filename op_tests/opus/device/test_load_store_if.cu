// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_load_store_if.cu
 * @brief Device tests for PR #2020: predicated load/store and free function API.
 *
 * Tests:
 *   1) predicated_copy      — opus::load_if + opus::store_if free functions
 *                              (gmem::load_if / store_if with layout + predicate)
 *   2) free_func_vector_add — opus::load + opus::store free functions
 *                              (also exercises is_gmem_v / is_mem_v type traits)
 *   3) predicated_async_load — opus::async_load_if free function
 *                              (gmem::async_load_if with layout + predicate)
 *
 * All kernels run on all GPU architectures (gfx942, gfx950, ...).
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"

template<int BLOCK_SIZE, int ELEMS>
__global__ void predicated_copy_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       int n)
{
    using namespace opus;
    int base = (__builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x()) * ELEMS;

    auto g_src = make_gmem(src);
    auto g_dst = make_gmem(dst);

    constexpr auto shape = opus::make_tuple(number<ELEMS>{});
    auto u = make_layout_packed(shape, opus::make_tuple(_)) + base;

    auto pred = [&](auto id) -> bool { return (base + id.value) < n; };

    auto data = load_if(g_src, pred, u);
    store_if(g_dst, pred, data, u);
}

template<int BLOCK_SIZE, int VEC>
__global__ void free_func_add_kernel(const float* a, const float* b,
                                     float* result, int n)
{
    auto g_a = opus::make_gmem(a);
    auto g_b = opus::make_gmem(b);
    auto g_r = opus::make_gmem(result);

    int idx = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x();
    int stride = __builtin_amdgcn_grid_size_x();

    for (int i = idx * VEC; i < n; i += stride * VEC) {
        auto va = opus::load<VEC>(g_a, i);
        auto vb = opus::load<VEC>(g_b, i);

        decltype(va) vr;
        for (int j = 0; j < VEC; j++) {
            vr[j] = va[j] + vb[j];
        }

        opus::store<VEC>(g_r, vr, i);
    }
}

template<int BLOCK_SIZE>
__global__ void predicated_async_load_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              int n, int n_padded)
{
    using namespace opus;
    __shared__ float smem_buf[BLOCK_SIZE];

    int tid = __builtin_amdgcn_workitem_id_x();
    int gid = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + tid;

    if (gid >= n_padded) return;

    auto g_src = make_gmem(src, static_cast<unsigned int>(n * sizeof(float)));

    auto u_gmem = make_layout(seq<1>{}) + gid;
    auto u_smem = make_layout(seq<1>{}) + tid;

    auto pred = [&](auto) -> bool { return gid < n; };

    async_load_if(g_src, pred, smem_buf, u_gmem, u_smem);

    s_waitcnt_vmcnt(number<0>{});
    __builtin_amdgcn_s_barrier();

    dst[gid] = smem_buf[tid];
}

template __global__ void predicated_copy_kernel<256, 4>(const float*, float*, int);
template __global__ void free_func_add_kernel<256, 4>(const float*, const float*, float*, int);
template __global__ void predicated_async_load_kernel<256>(const float*, float*, int, int);

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

template<int BLOCK_SIZE, int ELEMS>
__global__ void predicated_copy_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       int n) {}

template<int BLOCK_SIZE, int VEC>
__global__ void free_func_add_kernel(const float* a, const float* b,
                                     float* result, int n) {}

template<int BLOCK_SIZE>
__global__ void predicated_async_load_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              int n, int n_padded) {}

extern "C" void run_predicated_copy(const void* d_src, void* d_dst, int n)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int ELEMS = 4;
    int total_threads = (n + ELEMS - 1) / ELEMS;
    int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    hipLaunchKernelGGL(
        (predicated_copy_kernel<BLOCK_SIZE, ELEMS>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_src),
        static_cast<float*>(d_dst),
        n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_free_func_add(const void* d_a, const void* d_b,
                                   void* d_result, int n)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int VEC = 4;
    int blocks = n / (BLOCK_SIZE * VEC);

    hipLaunchKernelGGL(
        (free_func_add_kernel<BLOCK_SIZE, VEC>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_a),
        static_cast<const float*>(d_b),
        static_cast<float*>(d_result),
        n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_predicated_async_load(const void* d_src, void* d_dst,
                                           int n, int n_padded)
{
    constexpr int BLOCK_SIZE = 256;
    int blocks = n_padded / BLOCK_SIZE;

    hipLaunchKernelGGL(
        (predicated_async_load_kernel<BLOCK_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        static_cast<const float*>(d_src),
        static_cast<float*>(d_dst),
        n, n_padded);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif
