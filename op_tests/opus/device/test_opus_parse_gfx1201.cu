// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_opus_parse_gfx1201.cu
 * @brief Verify opus.hpp parses + opus::make_gmem load/store work on gfx1201
 *        (Navi 48 / RX 9070 XT, RDNA4).
 *
 * Two issues this test covers, both addressed in this commit:
 *
 *  1) Parse-time: without the forward declarations of mfma_adaptor /
 *     wmma_adaptor at the top of the opus namespace, opus.hpp fails to
 *     compile for gfx1201 device code because make_tiled_mma()'s default
 *     template argument names types that are gated behind __GFX9__ /
 *     __gfx1250__ blocks.
 *
 *  2) Runtime: even after the header parses, opus::make_gmem<>.store<N>()
 *     and .load<N>() silently produced wrong results on gfx1201 because
 *     buffer_default_config() returned the 0xffffffff fallback (the
 *     __gfx11__ / __gfx12__ checks on the prior line are typos — clang
 *     only predefines the uppercase __GFX11__ / __GFX12__). The invalid
 *     buffer rsrc made all buffer_load_b32 lanes return 0 and all
 *     buffer_store_b32 lanes drop on the floor. Fix: add explicit
 *     __gfx1201__ / __gfx1200__ branches with the correct 0x31004000
 *     config word that gfx1250 already uses.
 *
 * The kernel below exercises the exact opus API that sample_kernels.cu /
 * topk_softmax_kernels_group.cu / etc. depend on:
 *
 *     auto g = opus::make_gmem(ptr);
 *     auto v = g.load<VEC>(i);    // buffer_load via cached rsrc
 *     ... opus::cast<float>(v[j]) ...
 *     g.store<VEC>(vr, i);        // buffer_store via cached rsrc
 *
 * If this test produces correct results on gfx1201, both fixes hold.
 * Kernel body is gated by __gfx1201__ so other archs see an empty no-op
 * pass — gfx1250 / gfx9x behavior is unchanged.
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass: opus.hpp + kernel body, no hip_runtime.h ──────────────────
#include "opus/opus.hpp"

#if defined(__gfx1201__)
// Element-wise add via opus make_gmem load / store + per-lane opus::cast.
// Mirrors the load → cast<float> → store pattern in sample_kernels.cu.
template<int BLOCK_SIZE, int VECTOR_SIZE>
__global__ void opus_parse_gfx1201_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ result,
    int n)
{
    auto g_a = opus::make_gmem(a);
    auto g_b = opus::make_gmem(b);
    auto g_r = opus::make_gmem(result);

    int idx    = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE + __builtin_amdgcn_workitem_id_x();
    int stride = __builtin_amdgcn_grid_size_x();

    for (int i = idx * VECTOR_SIZE; i < n; i += stride * VECTOR_SIZE) {
        auto va = g_a.load<VECTOR_SIZE>(i);
        auto vb = g_b.load<VECTOR_SIZE>(i);

        decltype(va) vr;
        for (int j = 0; j < VECTOR_SIZE; ++j) {
            // opus::cast<float>(float) is a compile-time pass-through.
            // Including it exercises the same template path sample_kernels.cu
            // instantiates for its per-lane DTYPE_I → float conversion.
            vr[j] = opus::cast<float>(va[j]) + opus::cast<float>(vb[j]);
        }
        g_r.store<VECTOR_SIZE>(vr, i);
    }
}

template __global__ void opus_parse_gfx1201_kernel<256, 4>(const float*, const float*, float*, int);
#endif // __gfx1201__

#else
// ── Host pass: launcher + empty kernel stub ────────────────────────────────
#include "opus/hip_minimal.hpp"
#include <cstdio>

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

template<int BLOCK_SIZE, int VECTOR_SIZE>
__global__ void opus_parse_gfx1201_kernel(const float*, const float*, float*, int) {}

extern "C" void run_opus_parse_gfx1201(
    const void* d_a,
    const void* d_b,
    void*       d_result,
    int n)
{
    const auto* a = static_cast<const float*>(d_a);
    const auto* b = static_cast<const float*>(d_b);
    auto*       r = static_cast<float*>(d_result);

    constexpr int BLOCK_SIZE  = 256;
    constexpr int VECTOR_SIZE = 4;
    int blocks = (n + (BLOCK_SIZE * VECTOR_SIZE) - 1) / (BLOCK_SIZE * VECTOR_SIZE);

    hipLaunchKernelGGL(
        (opus_parse_gfx1201_kernel<BLOCK_SIZE, VECTOR_SIZE>),
        dim3(blocks), dim3(BLOCK_SIZE), 0, 0,
        a, b, r, n);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif
