// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_wmma_scale.cu
 * @brief WMMA scaled instruction tests for gfx1250.
 *
 * Tests __builtin_amdgcn_wmma_scale[16]_f32_{16x16x128_f8f6f4, 32x16x128_f4}
 * via the opus::wmma struct (BX32 and BX16 scaled overloads).
 *
 * Part A: Raw warp-level tests (direct builtin call through opus::wmma)
 *   1) wmma_scale_16x16x128_fp8     — BX32, fp8 via f8f6f4 (fmt=0)
 *   2) wmma_scale16_16x16x128_fp8   — BX16, fp8
 *   3) wmma_scale_16x16x128_fp4     — BX32, fp4 via f8f6f4 (fmt=4)
 *   4) wmma_scale16_16x16x128_fp4   — BX16, fp4
 *   5) wmma_scale_32x16x128_fp4     — BX32, dedicated f4
 *   6) wmma_scale16_32x16x128_fp4   — BX16, dedicated f4
 *
 * Part B: Tiled MMA test via make_tiled_mma + wmma_adaptor_swap_ab
 *   7) tiled_wmma_scale_16x16x128_fp8 — uses opus layout infrastructure
 *
 * Lane mapping (confirmed by probing):
 *   16x16x128 f8f6f4 (fp8): A: lane%16→m, lane/16→k_half (64 fp8/lane)
 *                            B: lane%16→n, lane/16→k_half
 *                            C: lane%16→n, lane/16→m_grp, c[i]→m%8
 *   16x16x128 f8f6f4 (fp4): same but 32 bytes per lane (64 fp4), rest zero-padded
 *   32x16x128 f4:            A: lane%16→m_base, lane/16→k_half
 *                              bytes 0-31 = m_base K-half, bytes 32-63 = m_base+16 K-half
 *                            B: lane%16→n, lane/16→k_half (32 bytes = 64 fp4)
 *                            C: lane%16→n, lane/16→m_grp2
 *                              c[0-7]→m_grp2*8+i, c[8-15]→m_grp2*8+16+i
 *   Scale: per-lane E8M0 exponent. scale_sel=0 → lanes 0-15, =1 → lanes 16-31.
 *          BX32 no-scale = 0x7F7F7F7F, BX16 no-scale = 0x7F7F7F7F7F7F7F7FL
 */

#ifdef __HIP_DEVICE_COMPILE__
// ── Device pass ─────────────────────────────────────────────────────────────
#include "opus/opus.hpp"
#if defined(__gfx1250__)

// ═══ Part A: Raw warp-level kernels ═════════════════════════════════════════

// --- 16x16x128 fp8 (f8f6f4 fmt=0) ---
template<bool BX16>
__global__ void wmma_scale_16x16x128_fp8_kernel(
    const opus::fp8_t* __restrict__ A,   // [16][128]
    const opus::fp8_t* __restrict__ B,   // [16][128] (NxK)
    opus::fp32_t* __restrict__ C,        // [16][16]
    int scale_a_bx32, int scale_b_bx32,
    long scale_a_bx16, long scale_b_bx16)
{
    using namespace opus;
    int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    int m = lane % 16, k_half = lane / 16, n = lane % 16;

    auto mma = wmma<fp8_t, fp8_t, fp32_t, 16, 16, 128>{};
    fp8x64_t a_reg, b_reg;
    #pragma unroll
    for (int i = 0; i < 64; i++) a_reg[i] = A[m * 128 + k_half * 64 + i];
    #pragma unroll
    for (int i = 0; i < 64; i++) b_reg[i] = B[n * 128 + k_half * 64 + i];

    fp32x8_t c_reg{0};
    if constexpr (BX16)
        c_reg = mma(a_reg, b_reg, c_reg, scale_a_bx16, scale_b_bx16);
    else
        c_reg = mma(a_reg, b_reg, c_reg, scale_a_bx32, scale_b_bx32);

    int m_grp = lane / 16; n = lane % 16;
    #pragma unroll
    for (int i = 0; i < 8; i++) C[(m_grp * 8 + i) * 16 + n] = c_reg[i];
}

template __global__ void wmma_scale_16x16x128_fp8_kernel<false>(
    const opus::fp8_t*, const opus::fp8_t*, opus::fp32_t*, int, int, long, long);
template __global__ void wmma_scale_16x16x128_fp8_kernel<true>(
    const opus::fp8_t*, const opus::fp8_t*, opus::fp32_t*, int, int, long, long);

// --- 16x16x128 fp4 (f8f6f4 fmt=4) ---
template<bool BX16>
__global__ void wmma_scale_16x16x128_fp4_kernel(
    const unsigned char* __restrict__ A,  // [16][64] packed fp4x2
    const unsigned char* __restrict__ B,  // [16][64]
    opus::fp32_t* __restrict__ C,         // [16][16]
    int scale_a_bx32, int scale_b_bx32,
    long scale_a_bx16, long scale_b_bx16)
{
    using namespace opus;
    int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    int m = lane % 16, k_half = lane / 16, n = lane % 16;

    auto mma = wmma<fp4_t, fp4_t, fp32_t, 16, 16, 128>{};
    // fp4: 64 fp4 per k-half = 32 bytes. Load into first 32 bytes of vtype_a (64 elements).
    typename decltype(mma)::vtype_a a_reg{};
    typename decltype(mma)::vtype_b b_reg{};
    auto* a_bytes = reinterpret_cast<unsigned char*>(&a_reg);
    auto* b_bytes = reinterpret_cast<unsigned char*>(&b_reg);
    #pragma unroll
    for (int i = 0; i < 32; i++) a_bytes[i] = A[m * 64 + k_half * 32 + i];
    #pragma unroll
    for (int i = 0; i < 32; i++) b_bytes[i] = B[n * 64 + k_half * 32 + i];

    fp32x8_t c_reg{0};
    if constexpr (BX16)
        c_reg = mma(a_reg, b_reg, c_reg, scale_a_bx16, scale_b_bx16);
    else
        c_reg = mma(a_reg, b_reg, c_reg, scale_a_bx32, scale_b_bx32);

    int m_grp = lane / 16; n = lane % 16;
    #pragma unroll
    for (int i = 0; i < 8; i++) C[(m_grp * 8 + i) * 16 + n] = c_reg[i];
}

template __global__ void wmma_scale_16x16x128_fp4_kernel<false>(
    const unsigned char*, const unsigned char*, opus::fp32_t*, int, int, long, long);
template __global__ void wmma_scale_16x16x128_fp4_kernel<true>(
    const unsigned char*, const unsigned char*, opus::fp32_t*, int, int, long, long);

// --- 32x16x128 fp4 (dedicated f4) ---
template<bool BX16>
__global__ void wmma_scale_32x16x128_fp4_kernel(
    const unsigned char* __restrict__ A,  // [32][64] packed fp4x2
    const unsigned char* __restrict__ B,  // [16][64]
    opus::fp32_t* __restrict__ C,         // [32][16]
    int scale_a_bx32, int scale_b_bx32,
    long scale_a_bx16, long scale_b_bx16)
{
    using namespace opus;
    int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    int lane16 = lane % 16, k_half = lane / 16;

    auto mma = wmma<fp4_t, fp4_t, fp32_t, 32, 16, 128>{};
    // vtype_a = i32x16 (64 bytes), vtype_b = i32x8 (32 bytes)
    // A: bytes 0-31 = m_base K-half, bytes 32-63 = m_base+16 K-half
    typename decltype(mma)::vtype_a a_reg{};
    typename decltype(mma)::vtype_b b_reg{};
    auto* a_bytes = reinterpret_cast<unsigned char*>(&a_reg);
    auto* b_bytes = reinterpret_cast<unsigned char*>(&b_reg);
    #pragma unroll
    for (int i = 0; i < 32; i++) a_bytes[i]      = A[lane16 * 64 + k_half * 32 + i];
    #pragma unroll
    for (int i = 0; i < 32; i++) a_bytes[32 + i]  = A[(lane16 + 16) * 64 + k_half * 32 + i];
    #pragma unroll
    for (int i = 0; i < 32; i++) b_bytes[i]       = B[lane16 * 64 + k_half * 32 + i];

    fp32x16_t c_reg{0};
    if constexpr (BX16)
        c_reg = mma(a_reg, b_reg, c_reg, scale_a_bx16, scale_b_bx16);
    else
        c_reg = mma(a_reg, b_reg, c_reg, scale_a_bx32, scale_b_bx32);

    int m_grp2 = lane / 16, n = lane % 16;
    #pragma unroll
    for (int i = 0; i < 8; i++)  C[(m_grp2 * 8 + i) * 16 + n]      = c_reg[i];
    #pragma unroll
    for (int i = 0; i < 8; i++)  C[(m_grp2 * 8 + 16 + i) * 16 + n] = c_reg[8 + i];
}

template __global__ void wmma_scale_32x16x128_fp4_kernel<false>(
    const unsigned char*, const unsigned char*, opus::fp32_t*, int, int, long, long);
template __global__ void wmma_scale_32x16x128_fp4_kernel<true>(
    const unsigned char*, const unsigned char*, opus::fp32_t*, int, int, long, long);

// --- 16x16x128 fp8 BX32 with per-lane scale ---
__global__ void wmma_scale_16x16x128_fp8_perlane_kernel(
    const opus::fp8_t* __restrict__ A,
    const opus::fp8_t* __restrict__ B,
    opus::fp32_t* __restrict__ C,
    const int* __restrict__ scale_a_buf,   // [32] per-lane packed BX32
    const int* __restrict__ scale_b_buf)   // [32] per-lane packed BX32
{
    using namespace opus;
    int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    int m = lane % 16, k_half = lane / 16, n = lane % 16;

    auto mma = wmma<fp8_t, fp8_t, fp32_t, 16, 16, 128>{};
    fp8x64_t a_reg, b_reg;
    #pragma unroll
    for (int i = 0; i < 64; i++) a_reg[i] = A[m * 128 + k_half * 64 + i];
    #pragma unroll
    for (int i = 0; i < 64; i++) b_reg[i] = B[n * 128 + k_half * 64 + i];

    int sa = scale_a_buf[lane];
    int sb = scale_b_buf[lane];

    fp32x8_t c_reg{0};
    c_reg = mma(a_reg, b_reg, c_reg, sa, sb);

    int m_grp = lane / 16; n = lane % 16;
    #pragma unroll
    for (int i = 0; i < 8; i++) C[(m_grp * 8 + i) * 16 + n] = c_reg[i];
}

// ═══ Part B: Tiled MMA kernels ══════════════════════════════════════════════
// T_M, T_N: number of waves tiling M and N dimensions.
// Total threads per block = T_M * T_N * warp_size.
// Block covers (WM * T_M) x (WN * T_N) output elements.

template<typename DIN, int WM, int WN, int WK, int T_M, int T_N>
__global__ void tiled_wmma_scale_kernel(
    const DIN* __restrict__ ptr_a,
    const DIN* __restrict__ ptr_b,
    opus::fp32_t* __restrict__ ptr_c,
    int k,
    int stride_a,
    int stride_b,
    int stride_c,
    int scale_a, int scale_b)
{
    using opus::operator""_I;
    constexpr int BLOCK_M = WM * T_M;
    constexpr int BLOCK_N = WN * T_N;
    constexpr int BLOCK_K = WK;
    constexpr int T_K = 1;
    constexpr int E_M = 1, E_N = 1, E_K = 1;

    using d_a = DIN;
    using d_b = DIN;
    using d_c = opus::fp32_t;

    constexpr int ELEM_A = WM * WK / 32;
    constexpr int PACK_A = (16 / static_cast<int>(sizeof(DIN)) < ELEM_A) ? 16 / static_cast<int>(sizeof(DIN)) : ELEM_A;
    constexpr int ELEM_B = WN * WK / 32;
    constexpr int PACK_B = (16 / static_cast<int>(sizeof(DIN)) < ELEM_B) ? 16 / static_cast<int>(sizeof(DIN)) : ELEM_B;
    constexpr int ELEM_C = WM * WN / 32;
    constexpr int PACK_C = (16 / static_cast<int>(sizeof(d_c)) < ELEM_C) ? 16 / static_cast<int>(sizeof(d_c)) : ELEM_C;

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
        opus::make_tuple(wave_id / T_N, lane_id % mma.grpm_a, 0_I, lane_id / mma.grpm_a));
    auto u_b = opus::partition_layout_b<PACK_B>(
        mma, opus::make_tuple(stride_b, 1_I),
        opus::make_tuple(wave_id % T_N, lane_id % mma.grpn_b, 0_I, lane_id / mma.grpn_b));
    auto u_c = opus::partition_layout_c(
        mma, opus::make_tuple(stride_c, 1_I),
        opus::make_tuple(wave_id / T_N, lane_id % mma.grpn_c, wave_id % T_N, lane_id / mma.grpn_c));

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
        v_c = mma(v_a, v_b, v_c, scale_a, scale_b);
    }

    g_c.template store<PACK_C>(v_c, u_c);
}

// fp8 tiled: T_M=1, T_N=1 (1 wave)
template __global__ void tiled_wmma_scale_kernel<opus::fp8_t, 16, 16, 128, 1, 1>(
    const opus::fp8_t*, const opus::fp8_t*, opus::fp32_t*, int, int, int, int, int, int);
// fp8 tiled: T_M=2, T_N=2 (4 waves, 32x32 block)
template __global__ void tiled_wmma_scale_kernel<opus::fp8_t, 16, 16, 128, 2, 2>(
    const opus::fp8_t*, const opus::fp8_t*, opus::fp32_t*, int, int, int, int, int, int);
// fp8 tiled: T_M=4, T_N=1 (4 waves, 64x16 block)
template __global__ void tiled_wmma_scale_kernel<opus::fp8_t, 16, 16, 128, 4, 1>(
    const opus::fp8_t*, const opus::fp8_t*, opus::fp32_t*, int, int, int, int, int, int);

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

// Forward declarations (empty bodies for host pass)
template<bool BX16>
__global__ void wmma_scale_16x16x128_fp8_kernel(
    const opus::fp8_t*, const opus::fp8_t*, opus::fp32_t*, int, int, long, long) {}
template<bool BX16>
__global__ void wmma_scale_16x16x128_fp4_kernel(
    const unsigned char*, const unsigned char*, opus::fp32_t*, int, int, long, long) {}
template<bool BX16>
__global__ void wmma_scale_32x16x128_fp4_kernel(
    const unsigned char*, const unsigned char*, opus::fp32_t*, int, int, long, long) {}
__global__ void wmma_scale_16x16x128_fp8_perlane_kernel(
    const opus::fp8_t*, const opus::fp8_t*, opus::fp32_t*, const int*, const int*) {}
template<typename DIN, int WM, int WN, int WK, int T_M, int T_N>
__global__ void tiled_wmma_scale_kernel(
    const DIN*, const DIN*, opus::fp32_t*, int, int, int, int, int, int) {}

// ═══ Host launchers ═════════════════════════════════════════════════════════

// --- 16x16x128 fp8 BX32 ---
extern "C" void run_wmma_scale_16x16x128_fp8_bx32(
    const void* d_a, const void* d_b, void* d_c, int scale_a, int scale_b)
{
    hipLaunchKernelGGL((wmma_scale_16x16x128_fp8_kernel<false>),
                       dim3(1), 32, 0, 0,
                       static_cast<const opus::fp8_t*>(d_a),
                       static_cast<const opus::fp8_t*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       scale_a, scale_b, 0L, 0L);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- 16x16x128 fp8 BX16 ---
extern "C" void run_wmma_scale16_16x16x128_fp8_bx16(
    const void* d_a, const void* d_b, void* d_c, long scale_a, long scale_b)
{
    hipLaunchKernelGGL((wmma_scale_16x16x128_fp8_kernel<true>),
                       dim3(1), 32, 0, 0,
                       static_cast<const opus::fp8_t*>(d_a),
                       static_cast<const opus::fp8_t*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       0, 0, scale_a, scale_b);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- 16x16x128 fp4 BX32 ---
extern "C" void run_wmma_scale_16x16x128_fp4_bx32(
    const void* d_a, const void* d_b, void* d_c, int scale_a, int scale_b)
{
    hipLaunchKernelGGL((wmma_scale_16x16x128_fp4_kernel<false>),
                       dim3(1), 32, 0, 0,
                       static_cast<const unsigned char*>(d_a),
                       static_cast<const unsigned char*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       scale_a, scale_b, 0L, 0L);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- 16x16x128 fp4 BX16 ---
extern "C" void run_wmma_scale16_16x16x128_fp4_bx16(
    const void* d_a, const void* d_b, void* d_c, long scale_a, long scale_b)
{
    hipLaunchKernelGGL((wmma_scale_16x16x128_fp4_kernel<true>),
                       dim3(1), 32, 0, 0,
                       static_cast<const unsigned char*>(d_a),
                       static_cast<const unsigned char*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       0, 0, scale_a, scale_b);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- 32x16x128 fp4 BX32 ---
extern "C" void run_wmma_scale_32x16x128_fp4_bx32(
    const void* d_a, const void* d_b, void* d_c, int scale_a, int scale_b)
{
    hipLaunchKernelGGL((wmma_scale_32x16x128_fp4_kernel<false>),
                       dim3(1), 32, 0, 0,
                       static_cast<const unsigned char*>(d_a),
                       static_cast<const unsigned char*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       scale_a, scale_b, 0L, 0L);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- 32x16x128 fp4 BX16 ---
extern "C" void run_wmma_scale16_32x16x128_fp4_bx16(
    const void* d_a, const void* d_b, void* d_c, long scale_a, long scale_b)
{
    hipLaunchKernelGGL((wmma_scale_32x16x128_fp4_kernel<true>),
                       dim3(1), 32, 0, 0,
                       static_cast<const unsigned char*>(d_a),
                       static_cast<const unsigned char*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       0, 0, scale_a, scale_b);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- 16x16x128 fp8 BX32 per-lane scale ---
extern "C" void run_wmma_scale_16x16x128_fp8_bx32_perlane(
    const void* d_a, const void* d_b, void* d_c,
    const void* d_scale_a, const void* d_scale_b)
{
    hipLaunchKernelGGL(wmma_scale_16x16x128_fp8_perlane_kernel,
                       dim3(1), 32, 0, 0,
                       static_cast<const opus::fp8_t*>(d_a),
                       static_cast<const opus::fp8_t*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       static_cast<const int*>(d_scale_a),
                       static_cast<const int*>(d_scale_b));
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- Tiled WMMA scale fp8: T_M=1, T_N=1 (1 wave, 16x16 block) ---
extern "C" void run_tiled_wmma_scale_16x16x128_fp8(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c,
    int scale_a, int scale_b)
{
    hipLaunchKernelGGL((tiled_wmma_scale_kernel<opus::fp8_t, 16, 16, 128, 1, 1>),
                       dim3(1, 1), 32, 0, 0,
                       static_cast<const opus::fp8_t*>(d_a),
                       static_cast<const opus::fp8_t*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       128, stride_a, stride_b, stride_c,
                       scale_a, scale_b);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- Tiled WMMA scale fp8: T_M=2, T_N=2 (4 waves, 32x32 block) ---
extern "C" void run_tiled_wmma_scale_16x16x128_fp8_2x2(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c,
    int scale_a, int scale_b)
{
    hipLaunchKernelGGL((tiled_wmma_scale_kernel<opus::fp8_t, 16, 16, 128, 2, 2>),
                       dim3(1, 1), 128, 0, 0,
                       static_cast<const opus::fp8_t*>(d_a),
                       static_cast<const opus::fp8_t*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       128, stride_a, stride_b, stride_c,
                       scale_a, scale_b);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

// --- Tiled WMMA scale fp8: T_M=4, T_N=1 (4 waves, 64x16 block) ---
extern "C" void run_tiled_wmma_scale_16x16x128_fp8_4x1(
    const void* d_a, const void* d_b, void* d_c,
    int stride_a, int stride_b, int stride_c,
    int scale_a, int scale_b)
{
    hipLaunchKernelGGL((tiled_wmma_scale_kernel<opus::fp8_t, 16, 16, 128, 4, 1>),
                       dim3(1, 1), 128, 0, 0,
                       static_cast<const opus::fp8_t*>(d_a),
                       static_cast<const opus::fp8_t*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       128, stride_a, stride_b, stride_c,
                       scale_a, scale_b);
    HIP_CALL(hipGetLastError()); HIP_CALL(hipDeviceSynchronize());
}

#endif // __HIP_DEVICE_COMPILE__
