// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_wmma_gfx1201.cu
 * @brief Direct WMMA tests on gfx1201 (Navi 48, RDNA4) via the opus::wmma<>
 *        struct dispatch (which targets __builtin_amdgcn_wmma_*_w32_gfx12).
 *
 * Covers the wave32 16x16x16 WMMA variants gfx1201 supports:
 *
 *   - f32 <- f16 / f16
 *   - f32 <- bf16 / bf16
 *   - f16 <- f16 / f16
 *   - bf16 <- bf16 / bf16
 *   - f32 <- fp8 / fp8
 *   - f32 <- fp8 / bf8
 *   - f32 <- bf8 / fp8
 *   - f32 <- bf8 / bf8
 *
 * Lane / register layout used here (per the gfx12 WMMA fragment diagrams in
 * AMD's internal Navi 4 layout reference and the community-verified guide
 * https://github.com/JohnTDI-cpu/rdna4-wmma-guide):
 *
 *     lane i in [0, 31], j in [0, 7]:
 *         A_frag[i][j] = A[i % 16,         (i/16)*8 + j]   // ROW-distributed
 *         B_frag[i][j] = B[(i/16)*8 + j,  i % 16]         // COLUMN-distributed
 *         C_frag[i][j] = C[(i/16)*8 + j,  i % 16]         // COLUMN-distributed
 *
 * That is:
 *   - A: lanes 0..15 cover rows 0..15 with K=0..7 within each lane; lanes
 *     16..31 cover the same rows with K=8..15.
 *   - B and C: lanes 0..15 cover columns 0..15 with K (for B) or M-rows
 *     0..7 (for C) within each lane; lanes 16..31 cover the same columns
 *     with K=8..15 (B) or M-rows 8..15 (C).
 *
 * This asymmetry (A row-distributed, B and C column-distributed) is the
 * native gfx12 wmma_128b fragment encoding and matches CK's wmma_gemm.hpp.
 *
 * The tests deliberately go through opus::wmma<>::operator() to exercise the
 * dispatch table (DISPATCH_WMMA_GFX12_F32_ / DISPATCH_WMMA_GFX12_8BIT_), not
 * the high-level make_tiled_mma / partition_layout_* path — the latter
 * relies on opus::wmma_adaptor whose lane encoding is row-distributed and
 * matches gfx1250 but NOT gfx12. A dedicated gfx12 wmma_adaptor would be
 * needed for the tiled API to work on gfx1201 (see TODO in opus.hpp).
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
#if defined(__gfx1201__) || defined(__gfx1200__)

// Generic 1-wave, 1-tile WMMA driver. Each lane loads its column-distributed
// A and B fragment from global memory, calls opus::wmma<>::operator(), and
// stores the C fragment back.
template<typename DIN_A, typename DIN_B, typename DOUT>
__global__ void wmma_gfx12_kernel(
    const DIN_A* __restrict__ ptr_a,
    const DIN_B* __restrict__ ptr_b,
    DOUT*        __restrict__ ptr_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int ELEM_A = WM * WK / 32;   // 8
    constexpr int ELEM_B = WN * WK / 32;   // 8
    constexpr int ELEM_C = WM * WN / 32;   // 8

    using vtype_a = opus::vector_t<DIN_A, ELEM_A>;
    using vtype_b = opus::vector_t<DIN_B, ELEM_B>;
    using vtype_c = opus::vector_t<DOUT,  ELEM_C>;

    int lane     = static_cast<int>(__builtin_amdgcn_workitem_id_x() % 32);
    int col      = lane % 16;            // for B / C (column-distributed)
    int row_base = (lane / 16) * 8;      // for B (K block) / C (M row block)
    int a_row    = lane % 16;            // A is row-distributed: lane selects row
    int a_k_base = (lane / 16) * 8;      // and lane group selects an 8-wide K block

    vtype_a v_a{};
    vtype_b v_b{};
    vtype_c v_c{};

    #pragma unroll
    for (int j = 0; j < ELEM_A; ++j) v_a[j] = ptr_a[a_row * stride_a + (a_k_base + j)];
    #pragma unroll
    for (int j = 0; j < ELEM_B; ++j) v_b[j] = ptr_b[(row_base + j) * stride_b + col];

    // Call through the opus::wmma<> dispatch — this is the path users of the
    // library hit when they call opus::wmma<...>{}(a, b, c).
    opus::wmma<DIN_A, DIN_B, DOUT, WM, WN, WK> mma;
    v_c = mma(v_a, v_b, v_c);

    #pragma unroll
    for (int j = 0; j < ELEM_C; ++j) ptr_c[(row_base + j) * stride_c + col] = v_c[j];
}

template __global__ void wmma_gfx12_kernel<opus::fp16_t, opus::fp16_t, opus::fp32_t>(const opus::fp16_t*, const opus::fp16_t*, opus::fp32_t*, int, int, int);
template __global__ void wmma_gfx12_kernel<opus::bf16_t, opus::bf16_t, opus::fp32_t>(const opus::bf16_t*, const opus::bf16_t*, opus::fp32_t*, int, int, int);
template __global__ void wmma_gfx12_kernel<opus::fp16_t, opus::fp16_t, opus::fp16_t>(const opus::fp16_t*, const opus::fp16_t*, opus::fp16_t*, int, int, int);
template __global__ void wmma_gfx12_kernel<opus::bf16_t, opus::bf16_t, opus::bf16_t>(const opus::bf16_t*, const opus::bf16_t*, opus::bf16_t*, int, int, int);
template __global__ void wmma_gfx12_kernel<opus::fp8_t , opus::fp8_t , opus::fp32_t>(const opus::fp8_t* , const opus::fp8_t* , opus::fp32_t*, int, int, int);
template __global__ void wmma_gfx12_kernel<opus::fp8_t , opus::bf8_t , opus::fp32_t>(const opus::fp8_t* , const opus::bf8_t* , opus::fp32_t*, int, int, int);
template __global__ void wmma_gfx12_kernel<opus::bf8_t , opus::fp8_t , opus::fp32_t>(const opus::bf8_t* , const opus::fp8_t* , opus::fp32_t*, int, int, int);
template __global__ void wmma_gfx12_kernel<opus::bf8_t , opus::bf8_t , opus::fp32_t>(const opus::bf8_t* , const opus::bf8_t* , opus::fp32_t*, int, int, int);

#endif // __gfx1201__ / __gfx1200__

#else
// ── Host pass: empty kernel stubs + extern "C" launchers ────────────────────
#include "opus/opus.hpp"
#include "opus/hip_minimal.hpp"
#include <cstdio>

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

template<typename DIN_A, typename DIN_B, typename DOUT>
__global__ void wmma_gfx12_kernel(const DIN_A*, const DIN_B*, DOUT*, int, int, int) {}

#define LAUNCHER_(NAME, DA, DB, DC) \
extern "C" void run_wmma_gfx1201_ ## NAME ( \
    const void* d_a, const void* d_b, void* d_c, \
    int stride_a, int stride_b, int stride_c) \
{ \
    hipLaunchKernelGGL((wmma_gfx12_kernel<opus::DA, opus::DB, opus::DC>), \
                       dim3(1, 1), 32, 0, 0, \
                       static_cast<const opus::DA*>(d_a), \
                       static_cast<const opus::DB*>(d_b), \
                       static_cast<opus::DC*>(d_c), \
                       stride_a, stride_b, stride_c); \
    HIP_CALL(hipGetLastError()); \
    HIP_CALL(hipDeviceSynchronize()); \
}

LAUNCHER_(f32_f16,      fp16_t, fp16_t, fp32_t)
LAUNCHER_(f32_bf16,     bf16_t, bf16_t, fp32_t)
LAUNCHER_(f16_f16,      fp16_t, fp16_t, fp16_t)
LAUNCHER_(bf16_bf16,    bf16_t, bf16_t, bf16_t)
LAUNCHER_(f32_fp8_fp8,  fp8_t,  fp8_t,  fp32_t)
LAUNCHER_(f32_fp8_bf8,  fp8_t,  bf8_t,  fp32_t)
LAUNCHER_(f32_bf8_fp8,  bf8_t,  fp8_t,  fp32_t)
LAUNCHER_(f32_bf8_bf8,  bf8_t,  bf8_t,  fp32_t)

#undef LAUNCHER_
#endif // __HIP_DEVICE_COMPILE__
