// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
// Tiled WMMA test for gfx1201/gfx1200: make_tiled_mma + partition_layout + gmem load/store.
// Computes C = A @ B^T (swap_ab) with 16x16x16 wave32 WMMA, same pattern as test_wmma_f32.cu (gfx1250).

#ifdef __HIP_DEVICE_COMPILE__
#include "opus/opus.hpp"
#if defined(__gfx1201__) || defined(__gfx1200__)

template<typename DIN, typename DOUT, int WM, int WN, int WK>
__global__ void wmma_gfx12_tiled_kernel(
    const DIN* __restrict__ ptr_a,
    const DIN* __restrict__ ptr_b,
    DOUT* __restrict__ ptr_c,
    int k, int stride_a, int stride_b, int stride_c)
{
    using opus::operator""_I;
    constexpr int T_M = 1, T_N = 1, T_K = 1;
    constexpr int E_M = 1, E_N = 1, E_K = 1;
    constexpr int ELEM_A = WM * WK / 32;
    constexpr int PACK_A = (16 / static_cast<int>(sizeof(DIN)) < ELEM_A) ? 16 / static_cast<int>(sizeof(DIN)) : ELEM_A;
    constexpr int PACK_B = PACK_A;
    constexpr int ELEM_C = WM * WN / 32;
    constexpr int PACK_C = (16 / static_cast<int>(sizeof(DOUT)) < ELEM_C) ? 16 / static_cast<int>(sizeof(DOUT)) : ELEM_C;
    using d_a = DIN; using d_b = DIN; using d_c = DOUT;

    int lane_id = static_cast<int>(opus::lane_id());
    int g_im = __builtin_amdgcn_workgroup_id_x() * WM;
    int g_in = __builtin_amdgcn_workgroup_id_y() * WN;

    auto mma = opus::make_tiled_mma(
        opus::make_wmma<d_a, d_b, d_c>(opus::seq<WM, WN, WK>{}, opus::wmma_adaptor_swap_ab{}),
        opus::seq<E_M, E_N, E_K>{}, opus::seq<T_M, T_N, T_K>{});

    auto u_a = opus::partition_layout_a<PACK_A>(mma, opus::make_tuple(stride_a, 1_I),
        opus::make_tuple(0_I, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a));
    auto u_b = opus::partition_layout_b<PACK_B>(mma, opus::make_tuple(stride_b, 1_I),
        opus::make_tuple(0_I, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b));
    auto u_c = opus::partition_layout_c(mma, opus::make_tuple(stride_c, 1_I),
        opus::make_tuple(0_I, lane_id % mma.grpn_c, 0_I, lane_id / mma.grpn_c));

    auto g_a = opus::make_gmem(ptr_a + g_im * stride_a);
    auto g_b = opus::make_gmem(ptr_b + g_in * stride_b);
    auto g_c = opus::make_gmem(ptr_c + g_im * stride_c + g_in);

    int loops = (k + WK - 1) / WK;
    typename decltype(mma)::vtype_c v_c;
    opus::clear(v_c);
    for (int i = 0; i < loops; i++) {
        auto v_a = g_a.template load<PACK_A>(u_a);
        u_a += WK;
        auto v_b = g_b.template load<PACK_B>(u_b);
        u_b += WK;
        v_c = mma(v_a, v_b, v_c);
    }
    g_c.template store<PACK_C>(v_c, u_c);
}

template __global__ void wmma_gfx12_tiled_kernel<opus::fp16_t, opus::fp32_t, 16, 16, 16>(const opus::fp16_t*, const opus::fp16_t*, opus::fp32_t*, int, int, int, int);
template __global__ void wmma_gfx12_tiled_kernel<opus::bf16_t, opus::fp32_t, 16, 16, 16>(const opus::bf16_t*, const opus::bf16_t*, opus::fp32_t*, int, int, int, int);

#endif
#else
#include "opus/opus.hpp"
#include "opus/hip_minimal.hpp"
#include <cstdio>
#define HIP_CALL(call) do { hipError_t err = (call); if (err != hipSuccess) { fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); return; } } while(0)

template<typename DIN, typename DOUT, int WM, int WN, int WK>
__global__ void wmma_gfx12_tiled_kernel(const DIN*, const DIN*, DOUT*, int, int, int, int) {}

#define LAUNCHER_(NAME, DIN, DOUT, WM, WN, WK) \
extern "C" void run_wmma_gfx1201_tiled_ ## NAME ( \
    const void* d_a, const void* d_b, void* d_c, int stride_a, int stride_b, int stride_c) { \
    hipLaunchKernelGGL((wmma_gfx12_tiled_kernel<opus::DIN, opus::DOUT, WM, WN, WK>), \
        dim3(1, 1), 32, 0, 0, \
        static_cast<const opus::DIN*>(d_a), static_cast<const opus::DIN*>(d_b), \
        static_cast<opus::DOUT*>(d_c), WK, stride_a, stride_b, stride_c); \
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize()); }

LAUNCHER_(f32_f16,  fp16_t, fp32_t, 16, 16, 16)
LAUNCHER_(f32_bf16, bf16_t, fp32_t, 16, 16, 16)
#undef LAUNCHER_
#endif
