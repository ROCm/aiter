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

#include <hip/hip_runtime.h>
#include <cstdio>
#include "opus/opus.hpp"
#include "test_load_store_if.h"

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

// ==========================================================================
// Kernel 1: predicated copy — gmem load_if / store_if via free functions
// ==========================================================================
// Each thread processes ELEMS elements using layout-based load_if/store_if.
// The predicate checks bounds, so the last block safely handles partial tiles.
template<int BLOCK_SIZE, int ELEMS>
__global__ void predicated_copy_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst,
                                       int n)
{
    using namespace opus;
    int base = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * ELEMS;

    auto g_src = make_gmem(src);
    auto g_dst = make_gmem(dst);

    // Layout: maps issue coords {0..ELEMS-1} to global offsets {base..base+ELEMS-1}
    // Tests layout_linear::operator+ from PR #2020.
    // The coord (make_tuple(_)) marks the dimension as issue-space so that
    // layout_to_issue_space returns (ELEMS,) instead of the default (1,).
    // Use opus::make_tuple(number<ELEMS>{}) rather than seq<ELEMS>{} so the shape
    // contains number<> types that vectorize_issue_space can decompose.
    constexpr auto shape = opus::make_tuple(number<ELEMS>{});
    auto u = make_layout_packed(shape, opus::make_tuple(_)) + base;

    // Predicate: only access if within bounds
    auto pred = [&](auto id) -> bool { return (base + id.value) < n; };

    // Free function wrappers (tests opus::load_if / opus::store_if and is_gmem_v)
    auto data = load_if(g_src, pred, u);
    store_if(g_dst, pred, data, u);
}

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

// ==========================================================================
// Kernel 2: vector add via free function API
// ==========================================================================
// Same as test_vector_add but uses opus::load / opus::store free functions
// instead of member functions, exercising is_mem_v / is_gmem_v type traits.
template<int BLOCK_SIZE, int VEC>
__global__ void free_func_add_kernel(const float* a, const float* b,
                                     float* result, int n)
{
    // NOTE: deliberately NOT using "using namespace opus" here to test
    // that the free functions work with explicit opus:: qualification.
    auto g_a = opus::make_gmem(a);
    auto g_b = opus::make_gmem(b);
    auto g_r = opus::make_gmem(result);

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;

    for (int i = idx * VEC; i < n; i += stride * VEC) {
        auto va = opus::load<VEC>(g_a, i);   // free function wrapper
        auto vb = opus::load<VEC>(g_b, i);   // free function wrapper

        decltype(va) vr;
        for (int j = 0; j < VEC; j++) {
            vr[j] = va[j] + vb[j];
        }

        opus::store<VEC>(g_r, vr, i);        // free function wrapper
    }
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

// ==========================================================================
// Kernel 3: predicated async load via free function API
// ==========================================================================
// Copies src -> LDS (async) -> dst with a bounds-checking predicate.
// Out-of-bounds smem slots are zero-filled by async_load_if.
template<int BLOCK_SIZE>
__global__ void predicated_async_load_kernel(const float* __restrict__ src,
                                              float* __restrict__ dst,
                                              int n, int n_padded)
{
    using namespace opus;
    __shared__ float smem_buf[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * BLOCK_SIZE + tid;

    if (gid >= n_padded) return;

    auto g_src = make_gmem(src, static_cast<uint32_t>(n * sizeof(float)));

    // Layouts: single element per thread
    auto u_gmem = make_layout(seq<1>{}) + gid;  // global offset
    auto u_smem = make_layout(seq<1>{}) + tid;   // smem offset

    // Predicate: only load if within actual data bounds
    auto pred = [&](auto) -> bool { return gid < n; };

    // Free function wrapper (tests opus::async_load_if and is_gmem_v)
    async_load_if(g_src, pred, smem_buf, u_gmem, u_smem);

    s_waitcnt_vmcnt(number<0>{});
    __syncthreads();

    // Write back from LDS to global output
    dst[gid] = smem_buf[tid];
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
