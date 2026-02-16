// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifndef OP_TESTS_OPUS_DEVICE_TEST_DTYPE_CONVERT_H
#define OP_TESTS_OPUS_DEVICE_TEST_DTYPE_CONVERT_H

#ifdef __cplusplus
extern "C" {
#endif

// FP32 <-> BF16 round-trip: convert fp32 input to bf16, then back to fp32.
// d_in: device pointer to float input, d_out: device pointer to float output.
// n must be a multiple of 256.
void run_dtype_convert_fp32_bf16(const void* d_in, void* d_out, int n);

// FP32 <-> FP16 round-trip: convert fp32 input to fp16, then back to fp32.
// d_in: device pointer to float input, d_out: device pointer to float output.
// n must be a multiple of 256.
void run_dtype_convert_fp32_fp16(const void* d_in, void* d_out, int n);

// FP32 <-> FP8 (e4m3) round-trip: convert fp32 input to fp8, then back to fp32.
// d_in: device pointer to float input, d_out: device pointer to float output.
// n must be a multiple of 256 * 4 (uses packed x4 conversions).
void run_dtype_convert_fp32_fp8(const void* d_in, void* d_out, int n);

// FP32 <-> FP4 (e2m1) round-trip: convert fp32 input to fp4, then back to fp32.
// d_in: device pointer to float input, d_out: device pointer to float output.
// n must be a multiple of 256 * 8 (uses packed x8 conversions). gfx950 only.
void run_dtype_convert_fp32_fp4(const void* d_in, void* d_out, int n);

#ifdef __cplusplus
}
#endif

#endif
