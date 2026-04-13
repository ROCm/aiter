// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_wmma_f32.cu
 * @brief OPUS WMMA additional variants: f32 16x16x4, f16->f16, bf16->bf16 (gfx1250 only, wave32).
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
#if defined(__gfx1250__)

template<typename DIN, typename DOUT, int WM, int WN, int WK>
__global__ void wmma_kernel_v2(
    const DIN* __restrict__ ptr_a,
    const DIN* __restrict__ ptr_b,
    DOUT* __restrict__ ptr_c,
    int k,
    int stride_a,
    int stride_b,
    int stride_c)
{
    using opus::operator""_I;
    constexpr int BLOCK_M = WM;
    constexpr int BLOCK_N = WN;
    constexpr int BLOCK_K = WK;
    constexpr int T_M = 1, T_N = 1, T_K = 1;
    constexpr int E_M = BLOCK_M / (WM * T_M);
    constexpr int E_N = BLOCK_N / (WN * T_N);
    constexpr int E_K = BLOCK_K / (WK * T_K);

    constexpr int ELEM_A = WM * WK / 32;
    constexpr int PACK_A = (16 / static_cast<int>(sizeof(DIN)) < ELEM_A) ? 16 / static_cast<int>(sizeof(DIN)) : ELEM_A;
    constexpr int PACK_B = PACK_A;
    constexpr int ELEM_C = WM * WN / 32;
    constexpr int PACK_C = (16 / static_cast<int>(sizeof(DOUT)) < ELEM_C) ? 16 / static_cast<int>(sizeof(DOUT)) : ELEM_C;

    using d_a = DIN;
    using d_b = DIN;
    using d_c = DOUT;

    int lane_id = static_cast<int>(__builtin_amdgcn_workitem_id_x() % opus::get_warp_size());
    int wave_id = static_cast<int>(__builtin_amdgcn_workitem_id_x() / opus::get_warp_size());
    int g_im = __builtin_amdgcn_workgroup_id_x() * BLOCK_M;
    int g_in = __builtin_amdgcn_workgroup_id_y() * BLOCK_N;

    auto mma = opus::make_tiled_mma(
        opus::make_wmma<d_a, d_b, d_c>(opus::seq<WM, WN, WK>{}, opus::wmma_adaptor_swap_ab{}),
        opus::seq<E_M, E_N, E_K>{},
        opus::seq<T_M, T_N, T_K>{});

    auto u_a = opus::partition_layout_a<PACK_A>(
        mma, opus::make_tuple(stride_a, 1_I),
        opus::make_tuple(wave_id / 2, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a));
    auto u_b = opus::partition_layout_b<PACK_B>(
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
        auto v_a = g_a.template load<PACK_A>(u_a);
        u_a += BLOCK_K;
        auto v_b = g_b.template load<PACK_B>(u_b);
        u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    g_c.template store<PACK_C>(v_c, u_c);
}

// f16->f16 16x16x32
template __global__ void wmma_kernel_v2<opus::fp16_t, opus::fp16_t, 16, 16, 32>(
    const opus::fp16_t*, const opus::fp16_t*, opus::fp16_t*, int, int, int, int);
// bf16->bf16 16x16x32
template __global__ void wmma_kernel_v2<opus::bf16_t, opus::bf16_t, 16, 16, 32>(
    const opus::bf16_t*, const opus::bf16_t*, opus::bf16_t*, int, int, int, int);
// f32 16x16x4
template __global__ void wmma_kernel_v2<opus::fp32_t, opus::fp32_t, 16, 16, 4>(
    const opus::fp32_t*, const opus::fp32_t*, opus::fp32_t*, int, int, int, int);

#endif // gfx1250 guard

#else
// ── Host pass ───────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
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
__global__ void wmma_kernel_v2(
    const DIN* __restrict__ ptr_a, const DIN* __restrict__ ptr_b,
    DOUT* __restrict__ ptr_c,
    int k, int stride_a, int stride_b, int stride_c) {}

extern "C" void run_wmma_16x16x32_f16_f16(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c)
{
    const auto* a = static_cast<const opus::fp16_t*>(d_a);
    const auto* b = static_cast<const opus::fp16_t*>(d_b);
    auto* c = static_cast<opus::fp16_t*>(d_c);
    hipLaunchKernelGGL((wmma_kernel_v2<opus::fp16_t, opus::fp16_t, 16, 16, 32>),
                       dim3(1, 1), 32, 0, 0, a, b, c, 32, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_wmma_16x16x32_bf16_bf16(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c)
{
    const auto* a = static_cast<const opus::bf16_t*>(d_a);
    const auto* b = static_cast<const opus::bf16_t*>(d_b);
    auto* c = static_cast<opus::bf16_t*>(d_c);
    hipLaunchKernelGGL((wmma_kernel_v2<opus::bf16_t, opus::bf16_t, 16, 16, 32>),
                       dim3(1, 1), 32, 0, 0, a, b, c, 32, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_wmma_16x16x4_f32(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c)
{
    const auto* a = static_cast<const opus::fp32_t*>(d_a);
    const auto* b = static_cast<const opus::fp32_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    hipLaunchKernelGGL((wmma_kernel_v2<opus::fp32_t, opus::fp32_t, 16, 16, 4>),
                       dim3(1, 1), 32, 0, 0, a, b, c, 4, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif // __HIP_DEVICE_COMPILE__
