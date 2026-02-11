// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_opus_mfma.cu
 * @brief OPUS MFMA kernel and host launch (no main).
 * Uses matrix_core_kernel_block_v2 style from
 * https://github.com/carlushuang/gcnasm/blob/master/matrix_core_opus/matrix_core.cc
 * Single block 32x32x8 with mfma_adaptor_swap_ab: C = A @ B^T (A 32x8, B 32x8, C 32x32).
 * swap_ab internally swaps A/B in the MFMA and transposes the C layout,
 * so the net result in row-major memory is C = A @ B^T (gemm_rcr).
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include "opus/opus.hpp"
#include "test_opus_mfma.h"

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

// This kernel requires gfx942 (MI300) MFMA instructions.
// The __gfx942__ macro is defined by hipcc during device compilation for that target.
#if defined(__gfx942__) || defined(__gfx9_4_generic__) || !defined(__HIP_DEVICE_COMPILE__)

// Single-block 32x32x8 kernel matching matrix_core_kernel_block_v2 (E_M=E_N=E_K=1, T_M=T_N=T_K=1, W_M=32, W_N=32, W_K=8).
__global__ void mfma_kernel_32x32x8_f16(
    const opus::fp16_t* __restrict__ ptr_a,
    const opus::fp16_t* __restrict__ ptr_b,
    opus::fp16_t* __restrict__ ptr_c,
    int k,
    int stride_a,
    int stride_b,
    int stride_c)
{
    using opus::operator""_I;
    constexpr int BLOCK_M = 32;
    constexpr int BLOCK_N = 32;
    constexpr int BLOCK_K = 8;
    constexpr int T_M = 1, T_N = 1, T_K = 1;
    constexpr int W_M = 32, W_N = 32, W_K = 8;
    constexpr int E_M = BLOCK_M / (W_M * T_M);
    constexpr int E_N = BLOCK_N / (W_N * T_N);
    constexpr int E_K = BLOCK_K / (W_K * T_K);

    using d_a = opus::fp16_t;
    using d_b = opus::fp16_t;
    using d_c = opus::fp32_t;

    int lane_id = static_cast<int>(threadIdx.x % opus::get_warp_size());
    int wave_id = static_cast<int>(threadIdx.x / opus::get_warp_size());
    int g_im = blockIdx.x * BLOCK_M;
    int g_in = blockIdx.y * BLOCK_N;

    auto mma = opus::make_tiled_mma<d_a, d_b, d_c>(
        opus::seq<E_M, E_N, E_K>{},
        opus::seq<T_M, T_N, T_K>{},
        opus::seq<W_M, W_N, W_K>{},
        opus::mfma_adaptor_swap_ab{});

    auto u_a = opus::partition_layout_a<4>(
        mma, opus::make_tuple(stride_a, 1_I),
        opus::make_tuple(wave_id / 2, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a));
    auto u_b = opus::partition_layout_b<4>(
        mma, opus::make_tuple(stride_b, 1_I),
        opus::make_tuple(wave_id % 2, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b));
    auto u_c = opus::partition_layout_c(
        mma, opus::make_tuple(stride_c, 1_I),
        opus::make_tuple(wave_id / 2, lane_id % mma.grpn_c, wave_id % 2, lane_id / mma.grpn_c));

    auto g_a = opus::make_gmem(ptr_a + g_im * stride_a);
    auto g_b = opus::make_gmem(ptr_b + g_in * stride_b);
    auto g_c = opus::make_gmem(ptr_c + g_im * stride_c + g_in);

    int loops = (k + BLOCK_K - 1) / BLOCK_K;
    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);

    for (int i = 0; i < loops; i++) {
        auto v_a = g_a.load<4>(u_a);
        u_a += BLOCK_K;
        auto v_b = g_b.load<4>(u_b);
        u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    auto v_c_f16 = opus::cast<opus::fp16_t>(v_c);
    g_c.store<4>(v_c_f16, u_c);
}

#endif // gfx942 guard

extern "C" void run_mfma_32x32x8_f16(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::fp16_t*>(d_a);
    const auto* b = static_cast<const opus::fp16_t*>(d_b);
    auto* c = static_cast<opus::fp16_t*>(d_c);
    const int M = 32, N = 32, K = 8;
    hipLaunchKernelGGL(mfma_kernel_32x32x8_f16, dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
