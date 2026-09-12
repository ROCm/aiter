#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#ifdef __cplusplus
extern "C" {
#endif

int getPaddedM(int M, int N, int K, int gl /*granularity level*/);

#ifdef __cplusplus
}
#endif
