#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor
gemm_a8w8_blockscale_bpreshuffle_asm(torch::Tensor& A,       // A:[M, K] fp8
                                     torch::Tensor& B,       // B:[N, K] fp8 -> [N/128, K*128]
                                     torch::Tensor& out,     // Out:[M, N] bf16
                                     torch::Tensor& A_scale, // A_scale:[M, K/128] fp32
                                     torch::Tensor& B_scale, // B_scale:[N/128, K/128] fp32
                                     std::optional<torch::Tensor> bias, // bias:[1, N] fp32
                                     std::optional<int> splitK,
                                     std::optional<std::string> kernelName,
                                     std::optional<bool> bpreshuffle);
