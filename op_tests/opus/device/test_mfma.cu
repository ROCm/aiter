// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_mfma.cu
 * @brief Templatized OPUS MFMA kernel and host launchers (no main).
 * Uses matrix_core_kernel_block_v2 style from
 * https://github.com/carlushuang/gcnasm/blob/master/matrix_core_opus/matrix_core.cc
 *
 * Supports 14 variants:
 *   1)  32x32x2  FP32  — gfx942 + gfx950 (native)
 *   2)  16x16x4  FP32  — gfx942 + gfx950 (native)
 *   3)  32x32x8  FP16  — gfx942 only
 *   4)  32x32x8  BF16  — gfx942 only
 *   5)  16x16x16 FP16  — gfx942 only
 *   6)  16x16x16 BF16  — gfx942 only
 *   7)  32x32x16 FP16  — gfx942 (step_k) + gfx950 (native)
 *   8)  32x32x16 BF16  — gfx942 (step_k) + gfx950 (native)
 *   9)  16x16x32 FP16  — gfx942 (step_k) + gfx950 (native)
 *   10) 16x16x32 BF16  — gfx942 (step_k) + gfx950 (native)
 *   11) 32x32x16 FP8   — gfx942 + gfx950 (native, fp32 output)
 *   12) 32x32x16 BF8   — gfx942 + gfx950 (native, fp32 output)
 *   13) 16x16x32 FP8   — gfx942 + gfx950 (native, fp32 output)
 *   14) 16x16x32 BF8   — gfx942 + gfx950 (native, fp32 output)
 *
 * swap_ab internally swaps A/B in the MFMA and transposes the C layout,
 * so the net result in row-major memory is C = A @ B^T (gemm_rcr).
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include "opus/opus.hpp"
#include "test_mfma.h"

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

// This kernel requires gfx942 (MI300) or gfx950 (MI350) MFMA instructions.
#if defined(__gfx942__) || defined(__gfx9_4_generic__) || defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)

/**
 * @brief Generic single-block MFMA kernel.
 * @tparam DIN      Data type for A/B inputs  (fp16_t, bf16_t, fp8_t, bf8_t)
 * @tparam DOUT     Data type for C output    (same as DIN, or fp32_t for fp8/bf8)
 * @tparam WM       Wave tile M dimension (32 or 16)
 * @tparam WN       Wave tile N dimension (32 or 16)
 * @tparam WK       Wave tile K dimension (8, 16, or 32)
 *
 * Single block: BLOCK_M=WM, BLOCK_N=WN, BLOCK_K=WK, T_M=T_N=T_K=1, E_M=E_N=E_K=1.
 * C = A @ B^T   (A: [M,K], B: [N,K], C: [M,N])
 */
template<typename DIN, typename DOUT, int WM, int WN, int WK>
__global__ void mfma_kernel_generic(
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

    // elem_a/elem_b per thread: WM*WK/warp_size
    constexpr int ELEM_A = WM * WK / 64;
    constexpr int ELEM_B = WN * WK / 64;

    using d_a = DIN;
    using d_b = DIN;
    using d_c = opus::fp32_t;

    int lane_id = static_cast<int>(threadIdx.x % opus::get_warp_size());
    int wave_id = static_cast<int>(threadIdx.x / opus::get_warp_size());
    int g_im = blockIdx.x * BLOCK_M;
    int g_in = blockIdx.y * BLOCK_N;

    auto mma = opus::make_tiled_mma<d_a, d_b, d_c>(
        opus::seq<E_M, E_N, E_K>{},
        opus::seq<T_M, T_N, T_K>{},
        opus::seq<WM, WN, WK>{},
        opus::mfma_adaptor_swap_ab{});

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
        auto v_a = g_a.template load<ELEM_A>(u_a);
        u_a += BLOCK_K;
        auto v_b = g_b.template load<ELEM_B>(u_b);
        u_b += BLOCK_K;
        v_c = mma(v_a, v_b, v_c);
    }

    // Store result.
    // - fp8/bf8 → fp32 output: store accumulator directly (no cast)
    // - bf16:  cast with RNE rounding (0_I = round-to-nearest-even)
    // - fp16:  cast (default truncation is fine, matches PyTorch)
    if constexpr (std::is_same_v<DOUT, d_c>) {
        // DOUT == fp32_t: store fp32 accumulator directly (fp8/bf8 path)
        g_c.template store<4>(v_c, u_c);
    } else if constexpr (std::is_same_v<DIN, opus::bf16_t>) {
        auto v_c_out = opus::cast<DOUT>(v_c, 0_I);   // RNE for bf16
        g_c.template store<4>(v_c_out, u_c);
    } else {
        auto v_c_out = opus::cast<DOUT>(v_c);
        g_c.template store<4>(v_c_out, u_c);
    }
}

#endif // gfx942 / gfx950 guard

// ---------------------------------------------------------------------------
// Host launch functions
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// FP32 variants: native 32x32x2 and 16x16x4 MFMA, fp32 input/output
// ---------------------------------------------------------------------------

extern "C" void run_mfma_32x32x2_f32(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::fp32_t*>(d_a);
    const auto* b = static_cast<const opus::fp32_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    const int K = 2;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp32_t, opus::fp32_t, 32, 32, 2>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_16x16x4_f32(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::fp32_t*>(d_a);
    const auto* b = static_cast<const opus::fp32_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    const int K = 4;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp32_t, opus::fp32_t, 16, 16, 4>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

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
    const int K = 8;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp16_t, opus::fp16_t, 32, 32, 8>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_32x32x8_bf16(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::bf16_t*>(d_a);
    const auto* b = static_cast<const opus::bf16_t*>(d_b);
    auto* c = static_cast<opus::bf16_t*>(d_c);
    const int K = 8;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::bf16_t, opus::bf16_t, 32, 32, 8>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_16x16x16_f16(
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
    const int K = 16;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp16_t, opus::fp16_t, 16, 16, 16>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_16x16x16_bf16(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::bf16_t*>(d_a);
    const auto* b = static_cast<const opus::bf16_t*>(d_b);
    auto* c = static_cast<opus::bf16_t*>(d_c);
    const int K = 16;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::bf16_t, opus::bf16_t, 16, 16, 16>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// 32x32x16 variants: use base 32x32x8 instruction, loop K=16 in 2 iterations.
// On gfx942 this is equivalent to STEP_K; on gfx950 the base instruction also exists.

extern "C" void run_mfma_32x32x16_f16(
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
    const int K = 16;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp16_t, opus::fp16_t, 32, 32, 8>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_32x32x16_bf16(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::bf16_t*>(d_a);
    const auto* b = static_cast<const opus::bf16_t*>(d_b);
    auto* c = static_cast<opus::bf16_t*>(d_c);
    const int K = 16;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::bf16_t, opus::bf16_t, 32, 32, 8>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// 16x16x32 variants: use base 16x16x16 instruction, loop K=32 in 2 iterations.

extern "C" void run_mfma_16x16x32_f16(
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
    const int K = 32;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp16_t, opus::fp16_t, 16, 16, 16>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_16x16x32_bf16(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::bf16_t*>(d_a);
    const auto* b = static_cast<const opus::bf16_t*>(d_b);
    auto* c = static_cast<opus::bf16_t*>(d_c);
    const int K = 32;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::bf16_t, opus::bf16_t, 16, 16, 16>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// FP8 / BF8 variants: native 32x32x16 and 16x16x32 MFMA, fp32 output
// ---------------------------------------------------------------------------

extern "C" void run_mfma_32x32x16_fp8(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::fp8_t*>(d_a);
    const auto* b = static_cast<const opus::fp8_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    const int K = 16;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp8_t, opus::fp32_t, 32, 32, 16>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_32x32x16_bf8(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::bf8_t*>(d_a);
    const auto* b = static_cast<const opus::bf8_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    const int K = 16;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::bf8_t, opus::fp32_t, 32, 32, 16>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_16x16x32_fp8(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::fp8_t*>(d_a);
    const auto* b = static_cast<const opus::fp8_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    const int K = 32;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::fp8_t, opus::fp32_t, 16, 16, 32>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_16x16x32_bf8(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c)
{
    const auto* a = static_cast<const opus::bf8_t*>(d_a);
    const auto* b = static_cast<const opus::bf8_t*>(d_b);
    auto* c = static_cast<opus::fp32_t*>(d_c);
    const int K = 32;
    hipLaunchKernelGGL((mfma_kernel_generic<opus::bf8_t, opus::fp32_t, 16, 16, 32>),
                       dim3(1, 1), 64, 0, 0, a, b, c, K, stride_a, stride_b, stride_c);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
