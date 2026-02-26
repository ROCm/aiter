// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef OP_TESTS_OPUS_DEVICE_TEST_MFMA_H
#define OP_TESTS_OPUS_DEVICE_TEST_MFMA_H

#ifdef __cplusplus
extern "C" {
#endif

// All functions: d_a, d_b, d_c are device pointers; strides in elements.
// C = A @ B^T  (swap_ab adaptor).

// --- f32 MFMA: gfx942 + gfx950 ---
void run_mfma_32x32x2_f32(const void* d_a, const void* d_b, void* d_c,
                           int stride_a, int stride_b, int stride_c);
void run_mfma_16x16x4_f32(const void* d_a, const void* d_b, void* d_c,
                           int stride_a, int stride_b, int stride_c);

// --- gfx942 only ---
void run_mfma_32x32x8_f16(const void* d_a, const void* d_b, void* d_c,
                           int stride_a, int stride_b, int stride_c);
void run_mfma_32x32x8_bf16(const void* d_a, const void* d_b, void* d_c,
                            int stride_a, int stride_b, int stride_c);
void run_mfma_16x16x16_f16(const void* d_a, const void* d_b, void* d_c,
                            int stride_a, int stride_b, int stride_c);
void run_mfma_16x16x16_bf16(const void* d_a, const void* d_b, void* d_c,
                             int stride_a, int stride_b, int stride_c);

// --- gfx942 (step_k) + gfx950 (native) ---
void run_mfma_32x32x16_f16(const void* d_a, const void* d_b, void* d_c,
                            int stride_a, int stride_b, int stride_c);
void run_mfma_32x32x16_bf16(const void* d_a, const void* d_b, void* d_c,
                             int stride_a, int stride_b, int stride_c);
void run_mfma_16x16x32_f16(const void* d_a, const void* d_b, void* d_c,
                            int stride_a, int stride_b, int stride_c);
void run_mfma_16x16x32_bf16(const void* d_a, const void* d_b, void* d_c,
                             int stride_a, int stride_b, int stride_c);

// --- fp8/bf8: gfx942 + gfx950 (native), fp32 output ---
void run_mfma_32x32x16_fp8(const void* d_a, const void* d_b, void* d_c,
                            int stride_a, int stride_b, int stride_c);
void run_mfma_32x32x16_bf8(const void* d_a, const void* d_b, void* d_c,
                            int stride_a, int stride_b, int stride_c);
void run_mfma_16x16x32_fp8(const void* d_a, const void* d_b, void* d_c,
                            int stride_a, int stride_b, int stride_c);
void run_mfma_16x16x32_bf8(const void* d_a, const void* d_b, void* d_c,
                            int stride_a, int stride_b, int stride_c);

#ifdef __cplusplus
}
#endif

#endif
