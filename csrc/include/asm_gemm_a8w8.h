#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor gemm_a8w8_asm(torch::Tensor &A,       // A:[M, K] i8
                            torch::Tensor &B,       //  B:[N, K] i8 -> shuffle layout(32,16)
                            torch::Tensor &A_scale, // A_scale:[M, 1] f32
                            torch::Tensor &B_scale, // B_scale:[1, N] f32
                            torch::Tensor &out,     // Out:[M, N] bf16
                            torch::Tensor &bias,    // bias:[1, N] f32
                            std::optional<int> sub_m = 128,
                            std::optional<int> sub_n = 128,
                            std::optional<int> pad_a = 0,
                            std::optional<int> pad_b = 0,
                            std::optional<int> pad_c = 0,
                            std::optional<int> splitK = 0);