// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/torch.h>

namespace aiter {

void static_per_tensor_quant(torch::Tensor &out,          // [..., d]
                             torch::Tensor const &input,  // [..., d]
                             torch::Tensor const &scale); // [1]

void dynamic_per_tensor_quant(torch::Tensor &out,         // [..., d]
                              torch::Tensor const &input, // [..., d]
                              torch::Tensor &scale);      // [1]

void dynamic_per_token_scaled_quant(
    torch::Tensor &out,         // [..., d]
    torch::Tensor const &input, // [..., d]
    torch::Tensor &scales, std::optional<at::Tensor> const &scale_ub);

} // namespace aiter
