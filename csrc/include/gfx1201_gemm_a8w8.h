// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

// Y[M, N] bf16 = (XQ[M, K] fp8 @ WQ^T) * x_scale * w_scale on gfx1201.
// WQ is fp8 [N, K] preshuffled by shuffle_weight_gfx1201; x_scale fp32 [M] or [1],
// w_scale fp32 [N] or [1]. Requires N % 256 == 0 and K % 128 == 0.
void gfx1201_gemm_a8w8_bpreshuffle_hip(const aiter_tensor_t& XQ,
                                       const aiter_tensor_t& WQ,
                                       const aiter_tensor_t& x_scale,
                                       const aiter_tensor_t& w_scale,
                                       aiter_tensor_t& Y);
