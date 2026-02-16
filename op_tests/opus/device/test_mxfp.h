// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef OP_TESTS_OPUS_DEVICE_TEST_MXFP_H
#define OP_TESTS_OPUS_DEVICE_TEST_MXFP_H

#ifdef __cplusplus
extern "C" {
#endif

// MXFP (gfx950 only): C = A @ B with E8M0 block exponent scaling.
// A is [M,K], B is [K,N], C is [M,N]; all row-major.
// scale_a, scale_b: E8M0 exponent (127 = no scaling).

// --- MXFP8 (FP8 * FP8) ---
void run_mxfp8_32x32x64(const void* d_a, const void* d_b, void* d_c,
                         int scale_a, int scale_b);
void run_mxfp8_16x16x128(const void* d_a, const void* d_b, void* d_c,
                          int scale_a, int scale_b);

// --- MXFP4 (FP4 * FP4, packed fp4x2 bytes) ---
void run_mxfp4_32x32x64(const void* d_a, const void* d_b, void* d_c,
                         int scale_a, int scale_b);
void run_mxfp4_16x16x128(const void* d_a, const void* d_b, void* d_c,
                          int scale_a, int scale_b);

#ifdef __cplusplus
}
#endif

#endif
