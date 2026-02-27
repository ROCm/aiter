// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_mfma_f32.cu
 * @brief OPUS MFMA f32 variants: 32x32x2, 16x16x4 (gfx942 + gfx950).
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
#if defined(__gfx942__) || defined(__gfx9_4_generic__) || defined(__gfx950__)

template<typename DIN, typename DOUT, int WM, int WN, int WK>
__global__ void mfma_kernel_generic(
    const DIN* __restrict__ ptr_a,
    const DIN* __restrict__ ptr_b,
    DOUT* __restrict__ ptr_c,
    int k, int stride_a, int stride_b, int stride_c)
{
    using opus::operator""_I;
    constexpr int BLOCK_M = WM, BLOCK_N = WN, BLOCK_K = WK;
    constexpr int T_M = 1, T_N = 1, T_K = 1;
    constexpr int E_M = BLOCK_M / (WM * T_M);
    constexpr int E_N = BLOCK_N / (WN * T_N);
    constexpr int E_K = BLOCK_K / (WK * T_K);
    constexpr int ELEM_A = WM * WK / 64;
    constexpr int ELEM_B = WN * WK / 64;

    using d_a = DIN; using d_b = DIN; using d_c = opus::fp32_t;

    int lane_id = static_cast<int>(__builtin_amdgcn_workitem_id_x() % opus::get_warp_size());
    int wave_id = static_cast<int>(__builtin_amdgcn_workitem_id_x() / opus::get_warp_size());
    int g_im = __builtin_amdgcn_workgroup_id_x() * BLOCK_M;
    int g_in = __builtin_amdgcn_workgroup_id_y() * BLOCK_N;

    auto mma = opus::make_tiled_mma<d_a, d_b, d_c>(
        opus::seq<E_M, E_N, E_K>{}, opus::seq<T_M, T_N, T_K>{},
        opus::seq<WM, WN, WK>{}, opus::mfma_adaptor_swap_ab{});

    auto u_a = opus::partition_layout_a<ELEM_A>(
        mma, opus::make_tuple(stride_a, 1_I),
        opus::make_tuple(wave_id / 2, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a));
    auto u_b = opus::partition_layout_b<ELEM_B>(
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
        auto v_a = g_a.template load<ELEM_A>(u_a); u_a += BLOCK_K;
        auto v_b = g_b.template load<ELEM_B>(u_b); u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    if constexpr (std::is_same_v<DOUT, d_c>) {
        g_c.template store<4>(v_c, u_c);
    } else if constexpr (std::is_same_v<DIN, opus::bf16_t>) {
        auto v_c_out = opus::cast<DOUT>(v_c, 0_I);
        g_c.template store<4>(v_c_out, u_c);
    } else {
        auto v_c_out = opus::cast<DOUT>(v_c);
        g_c.template store<4>(v_c_out, u_c);
    }
}

#endif // gfx942 / gfx950 guard

template __global__ void mfma_kernel_generic<opus::fp32_t, opus::fp32_t, 32, 32, 2>(
    const opus::fp32_t*, const opus::fp32_t*, opus::fp32_t*, int, int, int, int);
template __global__ void mfma_kernel_generic<opus::fp32_t, opus::fp32_t, 16, 16, 4>(
    const opus::fp32_t*, const opus::fp32_t*, opus::fp32_t*, int, int, int, int);

#else
// ── Host pass ───────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
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

template<typename DIN, typename DOUT, int WM, int WN, int WK>
__global__ void mfma_kernel_generic(
    const DIN* __restrict__ ptr_a, const DIN* __restrict__ ptr_b,
    DOUT* __restrict__ ptr_c,
    int k, int stride_a, int stride_b, int stride_c) {}

extern "C" void run_mfma_32x32x2_f32(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c)
{
    const auto* a = static_cast<const opus::fp32_t*>(d_a);
    const auto* b = static_cast<const opus::fp32_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp32_t, opus::fp32_t, 32, 32, 2>),
                       dim3(1, 1), 64, 0, 0, a, b, c, 2, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_16x16x4_f32(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c)
{
    const auto* a = static_cast<const opus::fp32_t*>(d_a);
    const auto* b = static_cast<const opus::fp32_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp32_t, opus::fp32_t, 16, 16, 4>),
                       dim3(1, 1), 64, 0, 0, a, b, c, 4, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif // __HIP_DEVICE_COMPILE__
