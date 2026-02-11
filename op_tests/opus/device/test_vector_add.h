// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef OP_TESTS_OPUS_DEVICE_TEST_VECTOR_ADD_H
#define OP_TESTS_OPUS_DEVICE_TEST_VECTOR_ADD_H

#ifdef __cplusplus
extern "C" {
#endif

// Element-wise vector addition using OPUS gmem:
//   result[i] = a[i] + b[i],  i = 0 .. n-1
// d_a, d_b, d_result are device pointers to float data.
void run_vector_add(
    const void* d_a,
    const void* d_b,
    void* d_result,
    int n);

#ifdef __cplusplus
}
#endif

#endif
