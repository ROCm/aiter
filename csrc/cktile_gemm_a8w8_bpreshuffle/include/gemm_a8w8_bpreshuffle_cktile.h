#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor gemm_a8w8_bpreshuffle_cktile(
    torch::Tensor &XQ,      // [M, K]
    torch::Tensor &WQ,      // [N, K] -> [N/128, K*128]
    torch::Tensor &x_scale, // [K/128, M]
    torch::Tensor &w_scale, // [K/128, N/128]
    torch::Tensor &out      // Out:[M, N] fp16
);
torch::Tensor gemm_a8w8_bpreshuffle_cktile_tune(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &out,
    int kernelId,
    int splitK);
