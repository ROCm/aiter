// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/torch.h>

void static_scaled_fp8_quant(torch::Tensor &out,          // [..., d]
                             torch::Tensor const &input,  // [..., d]
                             torch::Tensor const &scale); // [1]

void dynamic_scaled_fp8_quant(torch::Tensor &out,         // [..., d]
                              torch::Tensor const &input, // [..., d]
                              torch::Tensor &scale);      // [1]

void dynamic_per_token_scaled_fp8_quant(
    torch::Tensor &out,         // [..., d]
    torch::Tensor const &input, // [..., d]
    torch::Tensor &scales, std::optional<at::Tensor> const &scale_ub);
