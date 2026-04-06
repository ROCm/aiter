// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_tr_load_f16.cu
 * @brief Single-wave (64-thread) gfx950 test: async_load 16×32 row-major fp16 to LDS,
 *        smem::tr_load via layouts, store via partition_layout_b as MFMA B (32×16).
 *
 * Device body is compiled only for __gfx950__ (ds_read_tr*). Other targets get a
 * no-op kernel; Python gates verification on gfx950 (`test_tr_load_f16`).
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"

using opus::operator""_I;

template<int kNPerWarp = 32, int kKPerWarp = 16, int Vec = 8>
__device__ inline auto make_layout_g_s(int lane_id, int stride) {
    constexpr int lanes_n = kNPerWarp / Vec;
    constexpr int lanes_k = opus::get_warp_size() / lanes_n;

    constexpr auto shape = opus::make_tuple(
        opus::number<kKPerWarp / lanes_k>{},
        opus::number<lanes_k>{},
        opus::number<lanes_n>{},
        opus::number<Vec>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<Vec>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(stride, 1_I)),
        opus::unfold_p_coord(dim, opus::make_tuple(lane_id / lanes_n, lane_id % lanes_n)));
}

template<int kNPerWarp = 32, int kKPerWarp = 16, int Vec = 4>
__device__ inline auto make_layout_r(int lane_id, int stride) {
    constexpr int lane_per_grp = 16;
    constexpr int lane_lo = 4;
    constexpr int lane_hi = lane_per_grp / lane_lo;

    constexpr int num_grps = opus::get_warp_size() / lane_per_grp;
    constexpr int grp_n = kNPerWarp / (lane_lo * Vec);
    constexpr int grp_k = num_grps / grp_n;

    constexpr auto shape = opus::make_tuple(
        opus::number<grp_k>{},
        opus::number<kKPerWarp / (lane_hi * grp_k)>{},
        opus::number<lane_hi>{},
        opus::number<grp_n>{},
        opus::number<lane_lo>{},
        opus::number<Vec>{});

    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    int grp_id = lane_id / lane_per_grp;
    int lane_in_grp = lane_id % lane_per_grp;

    return opus::make_layout<Vec>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::make_tuple(stride, 1_I)),
        opus::unfold_p_coord(dim, opus::make_tuple(
            grp_id / grp_n,
            lane_in_grp / lane_lo,
            grp_id % grp_n,
            lane_in_grp % lane_lo)));
}

template<int ROWS, int COLS>
__global__ void tr_load_f16_kernel(const opus::fp16_t* __restrict__ in_row_major,
                                   opus::fp16_t* __restrict__ out_b_layout)
{
#if defined(__gfx950__)
    using namespace opus;
    static_assert(get_warp_size() == 64, "layout assumes wave64");

    __shared__ alignas(16) fp16_t smem_tile[ROWS * COLS];

    const int lane_id = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    constexpr int stride = COLS;

    auto g_in = make_gmem(in_row_major);
    auto g_out = make_gmem(out_b_layout);

    auto u_g = make_layout_g_s(lane_id, stride);
    auto u_s = make_layout_g_s(lane_id, stride);

    g_in.async_load<8>(reinterpret_cast<void*>(smem_tile), u_g, u_s);
    s_waitcnt_vmcnt(number<0>{});
    __builtin_amdgcn_s_barrier();

    auto smem_m = make_smem(smem_tile);
    auto u_r = make_layout_r(lane_id, stride);
    auto r = tr_load<4>(smem_m, u_r);

    auto mma = make_tiled_mma<fp16_t, fp16_t, fp32_t>(
        seq<1_I, 1_I, 1_I>{},
        seq<1_I, 1_I, 1_I>{},
        seq<32_I, 32_I, 16_I>{},
        mfma_adaptor{});

    auto u_b = partition_layout_b<8>(
        mma,
        make_tuple(16_I, 1_I),
        make_tuple(0_I, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b));

    g_out.template store<8>(__builtin_bit_cast(fp16x8_t, r), u_b);
#else
    (void)in_row_major;
    (void)out_b_layout;
#endif
}

template __global__ void tr_load_f16_kernel<16, 32>(const opus::fp16_t*, opus::fp16_t*);

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

template<int ROWS, int COLS>
__global__ void tr_load_f16_kernel(const opus::fp16_t* __restrict__ in_row_major,
                                   opus::fp16_t* __restrict__ out_b_layout) {}

extern "C" void run_tr_load_f16(const void* d_in, void* d_out)
{
    constexpr int ROWS = 16;
    constexpr int COLS = 32;
    constexpr int threads = 64;
    const auto* in = static_cast<const opus::fp16_t*>(d_in);
    auto* out = static_cast<opus::fp16_t*>(d_out);
    hipLaunchKernelGGL(
        (tr_load_f16_kernel<ROWS, COLS>),
        dim3(1), dim3(threads), 0, 0,
        in, out);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
#endif
