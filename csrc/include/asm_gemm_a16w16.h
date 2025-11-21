#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor gemm_a16w16_asm(torch::Tensor& A,   // A:[M, K] bf16
                              torch::Tensor& B,   // B:[N, K] bf16
                              torch::Tensor& out, // Out:[M, N] f32
                              std::optional<torch::Tensor> bias,
                              std::optional<int> splitK,
                              std::optional<std::string> kernelName,
                              bool bpreshuffle = false);
