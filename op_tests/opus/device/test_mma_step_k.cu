// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_mma_step_k.cu
 * @brief Minimal tiled_mma_adaptor::step_k() test for bf16 32x32x128.
 *
 * The kernel builds a single 32x32x128 logical tile from a 32x32x16 wave MMA
 * with EXPAND_K=8, then accumulates only through step_k() to exercise the
 * public tiled_mma_adaptor API directly.
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"

__global__ void mma_step_k_bf16_kernel(
    const opus::bf16_t* __restrict__ ptr_a,
    const opus::bf16_t* __restrict__ ptr_b,
    opus::bf16_t* __restrict__ ptr_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
#if defined(__gfx942__) || defined(__gfx9_4_generic__) || defined(__gfx950__)
    using opus::operator""_I;
    constexpr int WM = 32;
    constexpr int WN = 32;
    constexpr int WK = 16;
    constexpr int E_M = 1;
    constexpr int E_N = 1;
    constexpr int E_K = 8;
    constexpr int T_M = 1;
    constexpr int T_N = 1;
    constexpr int T_K = 1;
    constexpr int BLOCK_M = WM * E_M * T_M;
    constexpr int BLOCK_N = WN * E_N * T_N;
    constexpr int ELEM_A = WM * WK / 64;
    constexpr int ELEM_B = WN * WK / 64;

    using d_a = opus::bf16_t;
    using d_b = opus::bf16_t;
    using d_c = opus::fp32_t;

    int lane_id = static_cast<int>(__builtin_amdgcn_workitem_id_x() % opus::get_warp_size());
    int wave_id = static_cast<int>(__builtin_amdgcn_workitem_id_x() / opus::get_warp_size());
    int g_im = static_cast<int>(__builtin_amdgcn_workgroup_id_x()) * BLOCK_M;
    int g_in = static_cast<int>(__builtin_amdgcn_workgroup_id_y()) * BLOCK_N;

    auto mma = opus::make_tiled_mma<d_a, d_b, d_c>(
        opus::seq<E_M, E_N, E_K>{},
        opus::seq<T_M, T_N, T_K>{},
        opus::seq<WM, WN, WK>{},
        opus::mfma_adaptor_swap_ab{});

    auto u_a = opus::partition_layout_a<ELEM_A>(
        mma, opus::make_tuple(stride_a, 1_I),
        opus::make_tuple(0_I, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a));
    auto u_b = opus::partition_layout_b<ELEM_B>(
        mma, opus::make_tuple(stride_b, 1_I),
        opus::make_tuple(0_I, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b));
    auto u_c = opus::partition_layout_c(
        mma, opus::make_tuple(stride_c, 1_I),
        opus::make_tuple(0_I, lane_id % mma.grpn_c, 0_I, lane_id / mma.grpn_c));

    auto g_a = opus::make_gmem(ptr_a + g_im * stride_a);
    auto g_b = opus::make_gmem(ptr_b + g_in * stride_b);
    auto g_c = opus::make_gmem(ptr_c + g_im * stride_c + g_in);

    auto v_a = g_a.template load<ELEM_A>(u_a);
    auto v_b = g_b.template load<ELEM_B>(u_b);

    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);
    opus::static_for<E_K>([&](auto step) { v_c = mma.step_k(step, v_a, v_b, v_c); });

    auto v_c_out = opus::cast<d_a>(v_c, 0_I);
    g_c.template store<4>(v_c_out, u_c);
#else
    (void)ptr_a;
    (void)ptr_b;
    (void)ptr_c;
    (void)stride_a;
    (void)stride_b;
    (void)stride_c;
#endif
}

#else
// ── Host pass ───────────────────────────────────────────────────────────────
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

__global__ void mma_step_k_bf16_kernel(
    const opus::bf16_t* __restrict__,
    const opus::bf16_t* __restrict__,
    opus::bf16_t* __restrict__,
    int,
    int,
    int)
{}

extern "C" void run_mma_step_k_bf16(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c)
{
    constexpr int threads = 64;
    const auto* a = static_cast<const opus::bf16_t*>(d_a);
    const auto* b = static_cast<const opus::bf16_t*>(d_b);
    auto* c = static_cast<opus::bf16_t*>(d_c);
    hipLaunchKernelGGL(
        (mma_step_k_bf16_kernel),
        dim3(1, 1), dim3(threads), 0, 0,
        a, b, c, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif // __HIP_DEVICE_COMPILE__
