#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
torch::Tensor batched_gemm_bf16(torch::Tensor& XQ,
                                torch::Tensor& WQ,
                                torch::Tensor& Y,
                                std::optional<torch::Tensor> bias,
                                int splitK);

torch::Tensor batched_gemm_bf16_tune(
    torch::Tensor& XQ, torch::Tensor& WQ, torch::Tensor& Y, int kernelId, int splitK);
