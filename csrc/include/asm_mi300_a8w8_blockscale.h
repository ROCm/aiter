#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor a8w8_blockscale_bpreshuffle_asm(
    torch::Tensor &A,      // [M, K]
    torch::Tensor &B,      // [N, K] -> [N/128, K*128]
    torch::Tensor &a_scale, // [M, K/128]
    torch::Tensor &b_scale, // [N/128, K/128]
    torch::Tensor &out,      // Out:[M, N] bf16
    torch::Tensor &bias     // [1, N]      fp32
);
