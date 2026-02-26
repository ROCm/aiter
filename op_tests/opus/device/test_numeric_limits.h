// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#ifndef OP_TESTS_OPUS_DEVICE_TEST_NUMERIC_LIMITS_H
#define OP_TESTS_OPUS_DEVICE_TEST_NUMERIC_LIMITS_H

#ifdef __cplusplus
extern "C" {
#endif

// Writes 55 uint32 values: 11 types * 5 fields (min,max,lowest,quiet_nan,infinity).
// Layout: [0..4]=fp32, [5..9]=fp16, [10..14]=bf16, [15..19]=fp8, [20..24]=bf8,
//         [25..29]=i32, [30..34]=u32, [35..39]=i16, [40..44]=u16, [45..49]=i8, [50..54]=u8
void run_numeric_limits(void* d_out);

#ifdef __cplusplus
}
#endif
#endif
