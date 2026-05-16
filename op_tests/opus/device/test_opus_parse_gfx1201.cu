// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_opus_parse_gfx1201.cu
 * @brief Verify opus.hpp parses cleanly on gfx1201 (Navi 48 / RX 9070 XT, RDNA4).
 *
 * Without the gfx1201 forward-declaration fix in opus.hpp, kernels that
 * include opus.hpp on gfx1201 fail to compile with:
 *
 *     csrc/include/opus/opus.hpp:3065:24: error: unknown type name 'mfma_adaptor'
 *
 * because make_tiled_mma()'s default template argument references
 * mfma_adaptor (defined only under __GFX9__) and wmma_adaptor (defined only
 * under __gfx1250__) — neither of which is active in gfx1201 device code.
 *
 * This test exercises the opus utilities sample_kernels.cu actually uses
 * (opus::vector_t for vectorized lane storage, opus::cast for type
 * conversion). They are pure compile-time / C++ template machinery — no HIP
 * intrinsics — so they work on any arch the opus.hpp header parses for.
 *
 * What this test does NOT exercise:
 *   - opus::make_gmem .load<N>/.store<N> — these route through buffer-load /
 *     buffer-store intrinsics that are not available on gfx1201 today; the
 *     kernel uses plain pointer arithmetic for memory I/O instead.
 *   - mfma_adaptor / wmma_adaptor instantiation — neither is defined for
 *     gfx1201; that is the subject of opus.hpp Phase 2 (full WMMA support).
 *
 * If this test builds and produces correct results on gfx1201, the
 * forward declarations in opus.hpp are sufficient to keep gfx1201 device
 * code compiling. Behavior on gfx1250 / gfx9x is unchanged — the kernel
 * body is gated by __gfx1201__ so other archs see an empty no-op pass.
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass: opus.hpp + kernel body, no hip_runtime.h ──────────────────
#include "opus/opus.hpp"

#if defined(__gfx1201__)
// Element-wise add via opus::vector_t lanes + opus::cast<float>; the
// load/store goes through plain pointer arithmetic because opus's
// buffer-intrinsic backed make_gmem store path is not available on
// gfx1201 today. Sample_kernels.cu uses the same vector_t + cast pattern
// for its per-lane FP8/BF16 → FP32 conversion.
template<int BLOCK_SIZE, int VECTOR_SIZE>
__global__ void opus_parse_gfx1201_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ result,
    int n)
{
    int idx    = __builtin_amdgcn_workgroup_id_x() * BLOCK_SIZE
               + __builtin_amdgcn_workitem_id_x();
    int stride = __builtin_amdgcn_grid_size_x();

    for (int base = idx * VECTOR_SIZE; base < n; base += stride * VECTOR_SIZE) {
        opus::vector_t<float, VECTOR_SIZE> va, vb, vr;
        for (int j = 0; j < VECTOR_SIZE; ++j) {
            va[j] = a[base + j];
            vb[j] = b[base + j];
        }
        for (int j = 0; j < VECTOR_SIZE; ++j) {
            // opus::cast<float>(float) is a compile-time pass-through.
            // Including it in the test exercises the same template path
            // sample_kernels.cu instantiates for its DTYPE_I → float lanes.
            vr[j] = opus::cast<float>(va[j]) + opus::cast<float>(vb[j]);
        }
        for (int j = 0; j < VECTOR_SIZE; ++j) {
            result[base + j] = vr[j];
        }
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
