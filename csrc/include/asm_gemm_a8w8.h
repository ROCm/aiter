#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor gemm_a8w8_asm(torch::Tensor& A,       // A:[M, K] i8
                            torch::Tensor& B,       //  B:[N, K] i8 -> shuffle layout(32,16)
                            torch::Tensor& A_scale, // A_scale:[M, 1] f32
                            torch::Tensor& B_scale, // B_scale:[1, N] f32
                            torch::Tensor& out,     // Out:[M, N] bf16
                            std::string& kernelName,
                            torch::Tensor& bias, // bias:[1, N] f32
                            std::optional<bool> bpreshuffle = true,
                            std::optional<int> splitK       = std::nullopt);