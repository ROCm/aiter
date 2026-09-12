#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
torch::Tensor gemm_a8w8(
    // void gemm_a8w8(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int splitK);

torch::Tensor gemm_a8w8_tune(
    // void gemm_a8w8(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    int kernelId,
    int splitK);
