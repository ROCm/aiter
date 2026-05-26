// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
#if defined(__gfx1201__) || defined(__gfx1200__)

template<typename DIN_A, typename DIN_B, typename DOUT>
__global__ void wmma_gfx12_w64_kernel(
    const DIN_A* __restrict__ ptr_a,
    const DIN_B* __restrict__ ptr_b,
    DOUT*        __restrict__ ptr_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    constexpr int WM = 16, WN = 16, WK = 16;
    constexpr int WARP = 64;
    constexpr int ELEM = WM * WK / WARP;  // 4

    using vtype_a = opus::vector_t<DIN_A, ELEM>;
    using vtype_b = opus::vector_t<DIN_B, ELEM>;
    using vtype_c = opus::vector_t<DOUT, ELEM>;

    int lane = static_cast<int>(opus::lane_id());
    int group = lane / 16;     // 0,1,2,3
    int sublane = lane % 16;   // column for B/C, row for A

    // A: row-distributed. A[row=sublane][K=(group*4+j)]
    // Same group order as my original assumption — confirmed by AMD matrix calculator.
    int a_row = sublane;
    int a_k_base = group * 4;

    // B: column-distributed. B[K=(group*4+j)][col=sublane]
    int b_col = sublane;
    int b_k_base = group * 4;

    // D/C: column-distributed with INTERLEAVED group order {0,2,1,3}
    // Group 0 (lanes 0-15)  → rows 0-3
    // Group 1 (lanes 16-31) → rows 8-11  (NOT 4-7!)
    // Group 2 (lanes 32-47) → rows 4-7
    // Group 3 (lanes 48-63) → rows 12-15
    constexpr int c_row_base_lut[4] = {0, 8, 4, 12};
    int c_col = sublane;
    int c_row_base = c_row_base_lut[group];

    vtype_a v_a{};
    vtype_b v_b{};
    vtype_c v_c{};

    #pragma unroll
    for (int j = 0; j < ELEM; ++j) v_a[j] = ptr_a[a_row * stride_a + (a_k_base + j)];
    #pragma unroll
    for (int j = 0; j < ELEM; ++j) v_b[j] = ptr_b[(b_k_base + j) * stride_b + b_col];

    opus::wmma<DIN_A, DIN_B, DOUT, WM, WN, WK, WARP> mma;
    v_c = mma(v_a, v_b, v_c);

    #pragma unroll
    for (int j = 0; j < ELEM; ++j) ptr_c[(c_row_base + j) * stride_c + c_col] = v_c[j];
}

template __global__ void wmma_gfx12_w64_kernel<opus::fp16_t, opus::fp16_t, opus::fp32_t>(const opus::fp16_t*, const opus::fp16_t*, opus::fp32_t*, int, int, int);
template __global__ void wmma_gfx12_w64_kernel<opus::bf16_t, opus::bf16_t, opus::fp32_t>(const opus::bf16_t*, const opus::bf16_t*, opus::fp32_t*, int, int, int);
template __global__ void wmma_gfx12_w64_kernel<opus::fp16_t, opus::fp16_t, opus::fp16_t>(const opus::fp16_t*, const opus::fp16_t*, opus::fp16_t*, int, int, int);
template __global__ void wmma_gfx12_w64_kernel<opus::bf16_t, opus::bf16_t, opus::bf16_t>(const opus::bf16_t*, const opus::bf16_t*, opus::bf16_t*, int, int, int);

#endif
#else
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
__global__ void wmma_gfx12_w64_kernel(const DIN_A*, const DIN_B*, DOUT*, int, int, int) {}

#define LAUNCHER_(NAME, DA, DB, DC) \
extern "C" void run_wmma_gfx1201_w64_ ## NAME ( \
    const void* d_a, const void* d_b, void* d_c, \
    int stride_a, int stride_b, int stride_c) \
{ \
    hipLaunchKernelGGL((wmma_gfx12_w64_kernel<opus::DA, opus::DB, opus::DC>), \
                       dim3(1, 1), 64, 0, 0, \
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

#undef LAUNCHER_
#endif
