// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef OP_TESTS_OPUS_TEST_OPUS_MFMA_H
#define OP_TESTS_OPUS_TEST_OPUS_MFMA_H

#ifdef __cplusplus
extern "C" {
#endif

// Run 32x32x8 fp16 MFMA: C = A @ B^T (A 32x8, B 32x8, C 32x32).
// d_a, d_b, d_c are device pointers to fp16 data; strides in elements.
void run_mfma_32x32x8_f16(
    const void* d_a,
    const void* d_b,
    void* d_c,
    int stride_a,
    int stride_b,
    int stride_c);

#ifdef __cplusplus
}
#endif

#endif
