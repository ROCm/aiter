// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#ifndef OP_TESTS_OPUS_DEVICE_TEST_MDIV_H
#define OP_TESTS_OPUS_DEVICE_TEST_MDIV_H
#ifdef __cplusplus
extern "C" {
#endif

void run_mdiv(const void* d_dividends, void* d_out_q, void* d_out_r,
              int divisor, int n);

#ifdef __cplusplus
}
#endif
#endif
