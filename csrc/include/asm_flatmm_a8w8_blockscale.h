#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor flatmm_a8w8_blockscale_asm(
    torch::Tensor &A,       // [M, K]
    torch::Tensor &B,       // [N, K] -> [N/128, K*128]
    torch::Tensor &A_scale, // [K/128, M]
    torch::Tensor &B_scale, // [K/128, N/128]
    torch::Tensor &out      // Out:[M, N] fp16
);
