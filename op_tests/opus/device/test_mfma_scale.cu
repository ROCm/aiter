// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_mfma_scale.cu
 * @brief Scaled MFMA (f8f6f4) kernel tests for gfx950.
 *
 * Tests __builtin_amdgcn_mfma_scale_f32_{32x32x64,16x16x128}_f8f6f4
 * via the opus::mfma struct (scaled overload), with fp8*fp8 and fp4*fp4 inputs.
 *
 * Variants:
 *   1) 32x32x64  FP8*FP8  — gfx950 only
 *   2) 16x16x128 FP8*FP8  — gfx950 only
 *   3) 32x32x64  FP4*FP4  — gfx950 only
 *   4) 16x16x128 FP4*FP4  — gfx950 only
 *
 * Data layout follows the CDNA4 Matrix Core specification:
 *   A is [M,K], B is [K,N], C is [M,N]; all row-major.
 *   C = A @ B  (standard matmul, NOT swap_ab).
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include "opus/opus.hpp"
#include "test_mfma_scale.h"

#define HIP_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %d at %s:%d\n", (int)err, __FILE__, __LINE__); \
        return; \
    } \
} while(0)

// Helper: extract one fp4 nibble (0=low, 1=high) from a packed fp4x2 byte
__device__ inline unsigned char fp4_extract(unsigned char packed, int idx) {
    return (idx == 0) ? (packed & 0xFu) : (packed >> 4);
}

// Helper: pack two fp4 nibbles into one fp4x2 byte
__device__ inline unsigned char fp4_pack(unsigned char lo, unsigned char hi) {
    return (lo & 0xFu) | ((hi & 0xFu) << 4);
}

// ==========================================================================
// FP8 * FP8 scaled MFMA kernel
// ==========================================================================
template<int M, int N, int K>
__global__ void mfma_scale_kernel_fp8(
    const opus::fp8_t* __restrict__ ptr_a,   // A[M][K] row-major
    const opus::fp8_t* __restrict__ ptr_b,   // B[K][N] row-major
    opus::fp32_t* __restrict__ ptr_c,        // C[M][N] row-major
    int scale_a, int scale_b)
{
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
    using namespace opus;
    int lane = static_cast<int>(threadIdx.x);

    if constexpr (M == 32 && N == 32 && K == 64) {
        int lane32 = lane % 32;
        int group  = lane / 32;

        // Load A[32][64]: 2 stripes of 16 fp8 each, stride 32 between stripes
        fp8x32_t a_reg;
        const fp8_t* ldg_a = ptr_a + lane32 * K + group * 16;
        #pragma unroll
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 16; j++)
                a_reg[i * 16 + j] = ldg_a[i * 32 + j];

        // Load B[64][32]: 2 stripes of 16 fp8 each
        fp8x32_t b_reg;
        const fp8_t* ldg_b = ptr_b + lane32 + N * 16 * group;
        #pragma unroll
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 16; j++)
                b_reg[i * 16 + j] = ldg_b[N * j + i * N * 32];

        // Scaled MFMA via opus::mfma (scaled overload)
        auto mma = mfma<fp8_t, fp8_t, fp32_t, 32, 32, 64>{};
        fp32x16_t c_reg{0};
        c_reg = mma(__builtin_bit_cast(i32x8_t, a_reg),
                    __builtin_bit_cast(i32x8_t, b_reg),
                    c_reg, scale_a, scale_b);

        // Store C[32][32]: same layout as standard 32x32 MFMA
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ptr_c[(group * 4 + i * 8 + 0) * N + lane32] = c_reg[i * 4 + 0];
            ptr_c[(group * 4 + i * 8 + 1) * N + lane32] = c_reg[i * 4 + 1];
            ptr_c[(group * 4 + i * 8 + 2) * N + lane32] = c_reg[i * 4 + 2];
            ptr_c[(group * 4 + i * 8 + 3) * N + lane32] = c_reg[i * 4 + 3];
        }
    }
    else if constexpr (M == 16 && N == 16 && K == 128) {
        int lane16 = lane % 16;
        int group4 = lane / 16;

        // Load A[16][128]: 4 stripes of 8 fp8 each, stride 32 between stripes
        fp8x32_t a_reg;
        const fp8_t* ldg_a = ptr_a + lane16 * K + group4 * 8;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                a_reg[i * 8 + j] = ldg_a[i * 32 + j];

        // Load B[128][16]: 4 stripes of 8 fp8 each
        fp8x32_t b_reg;
        const fp8_t* ldg_b = ptr_b + lane16 + N * 8 * group4;
        #pragma unroll
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                b_reg[i * 8 + j] = ldg_b[N * j + i * N * 32];

        auto mma = mfma<fp8_t, fp8_t, fp32_t, 16, 16, 128>{};
        fp32x4_t c_reg{0};
        c_reg = mma(__builtin_bit_cast(i32x8_t, a_reg),
                    __builtin_bit_cast(i32x8_t, b_reg),
                    c_reg, scale_a, scale_b);

        // Store C[16][16]: same layout as standard 16x16 MFMA
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ptr_c[(group4 * 4 + i) * N + lane16] = c_reg[i];
        }
    }
#endif // gfx950 guard
}

// ==========================================================================
// FP4 * FP4 scaled MFMA kernel
// ==========================================================================
template<int M, int N, int K>
__global__ void mfma_scale_kernel_fp4(
    const unsigned char* __restrict__ ptr_a,  // A[M][K] packed fp4x2, row-major
    const unsigned char* __restrict__ ptr_b,  // B[K][N] packed fp4x2, row-major
    opus::fp32_t* __restrict__ ptr_c,         // C[M][N] row-major
    int scale_a, int scale_b)
{
#if defined(__gfx950__) || !defined(__HIP_DEVICE_COMPILE__)
    using namespace opus;
    int lane = static_cast<int>(threadIdx.x);
    constexpr int A_BYTES_PER_ROW = K / 2;  // 2 fp4 values packed per byte
    constexpr int B_BYTES_PER_ROW = N / 2;

    if constexpr (M == 32 && N == 32 && K == 64) {
        int lane32 = lane % 32;
        int group  = lane / 32;

        // Load A: 16 contiguous bytes (32 fp4 values) -> 128 bits of 256-bit register
        union { i32x8_t v; unsigned char b[32]; } a_buf;
        #pragma unroll
        for (int i = 0; i < 8; i++) a_buf.v[i] = 0;
        const unsigned char* ldg_a = ptr_a + lane32 * A_BYTES_PER_ROW + group * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++) a_buf.b[i] = ldg_a[i];

        // Load B: extract fp4 nibbles and repack for register layout
        union { i32x8_t v; unsigned char b[32]; } b_buf;
        #pragma unroll
        for (int i = 0; i < 8; i++) b_buf.v[i] = 0;
        const unsigned char* ldg_b = ptr_b + lane32 / 2 + B_BYTES_PER_ROW * 32 * group;
        int b_nibble = lane32 % 2;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte0 = ldg_b[B_BYTES_PER_ROW * 2 * i];
            unsigned char byte1 = ldg_b[B_BYTES_PER_ROW * (2 * i + 1)];
            b_buf.b[i] = fp4_pack(fp4_extract(byte0, b_nibble),
                                   fp4_extract(byte1, b_nibble));
        }

        auto mma = mfma<fp4_t, fp4_t, fp32_t, 32, 32, 64>{};
        fp32x16_t c_reg{0};
        c_reg = mma(a_buf.v, b_buf.v, c_reg, scale_a, scale_b);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ptr_c[(group * 4 + i * 8 + 0) * N + lane32] = c_reg[i * 4 + 0];
            ptr_c[(group * 4 + i * 8 + 1) * N + lane32] = c_reg[i * 4 + 1];
            ptr_c[(group * 4 + i * 8 + 2) * N + lane32] = c_reg[i * 4 + 2];
            ptr_c[(group * 4 + i * 8 + 3) * N + lane32] = c_reg[i * 4 + 3];
        }
    }
    else if constexpr (M == 16 && N == 16 && K == 128) {
        int lane16 = lane % 16;
        int group4 = lane / 16;

        // Load A: 16 contiguous bytes (32 fp4 values)
        union { i32x8_t v; unsigned char b[32]; } a_buf;
        #pragma unroll
        for (int i = 0; i < 8; i++) a_buf.v[i] = 0;
        const unsigned char* ldg_a = ptr_a + lane16 * A_BYTES_PER_ROW + group4 * 16;
        #pragma unroll
        for (int i = 0; i < 16; i++) a_buf.b[i] = ldg_a[i];

        // Load B: extract fp4 nibbles
        union { i32x8_t v; unsigned char b[32]; } b_buf;
        #pragma unroll
        for (int i = 0; i < 8; i++) b_buf.v[i] = 0;
        const unsigned char* ldg_b = ptr_b + lane16 / 2 + B_BYTES_PER_ROW * 32 * group4;
        int b_nibble = lane16 % 2;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            unsigned char byte0 = ldg_b[B_BYTES_PER_ROW * 2 * i];
            unsigned char byte1 = ldg_b[B_BYTES_PER_ROW * (2 * i + 1)];
            b_buf.b[i] = fp4_pack(fp4_extract(byte0, b_nibble),
                                   fp4_extract(byte1, b_nibble));
        }

        auto mma = mfma<fp4_t, fp4_t, fp32_t, 16, 16, 128>{};
        fp32x4_t c_reg{0};
        c_reg = mma(a_buf.v, b_buf.v, c_reg, scale_a, scale_b);

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            ptr_c[(group4 * 4 + i) * N + lane16] = c_reg[i];
        }
    }
#endif // gfx950 guard
}

// ---------------------------------------------------------------------------
// Host launch functions
// ---------------------------------------------------------------------------

extern "C" void run_mfma_scale_32x32x64_fp8(
    const void* d_a, const void* d_b, void* d_c, int scale_a, int scale_b)
{
    hipLaunchKernelGGL((mfma_scale_kernel_fp8<32, 32, 64>),
                       dim3(1), 64, 0, 0,
                       static_cast<const opus::fp8_t*>(d_a),
                       static_cast<const opus::fp8_t*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       scale_a, scale_b);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_scale_16x16x128_fp8(
    const void* d_a, const void* d_b, void* d_c, int scale_a, int scale_b)
{
    hipLaunchKernelGGL((mfma_scale_kernel_fp8<16, 16, 128>),
                       dim3(1), 64, 0, 0,
                       static_cast<const opus::fp8_t*>(d_a),
                       static_cast<const opus::fp8_t*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       scale_a, scale_b);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_scale_32x32x64_fp4(
    const void* d_a, const void* d_b, void* d_c, int scale_a, int scale_b)
{
    hipLaunchKernelGGL((mfma_scale_kernel_fp4<32, 32, 64>),
                       dim3(1), 64, 0, 0,
                       static_cast<const unsigned char*>(d_a),
                       static_cast<const unsigned char*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       scale_a, scale_b);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}

extern "C" void run_mfma_scale_16x16x128_fp4(
    const void* d_a, const void* d_b, void* d_c, int scale_a, int scale_b)
{
    hipLaunchKernelGGL((mfma_scale_kernel_fp4<16, 16, 128>),
                       dim3(1), 64, 0, 0,
                       static_cast<const unsigned char*>(d_a),
                       static_cast<const unsigned char*>(d_b),
                       static_cast<opus::fp32_t*>(d_c),
                       scale_a, scale_b);
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());
}
