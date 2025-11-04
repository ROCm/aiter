// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

namespace aiter {

void static_per_tensor_quant(torch::Tensor& out,          // [..., d]
                             torch::Tensor const& input,  // [..., d]
                             torch::Tensor const& scale); // [1]

void dynamic_per_tensor_quant(torch::Tensor& out,         // [..., d]
                              torch::Tensor const& input, // [..., d]
                              torch::Tensor& scale);      // [1]

void dynamic_per_token_scaled_quant(torch::Tensor& out,         // [..., d]
                                    torch::Tensor const& input, // [..., d]
                                    torch::Tensor& scales,
                                    std::optional<torch::Tensor> scale_ub = std::nullopt,
                                    bool shuffle_scale                    = false,
                                    std::optional<torch::Tensor> num_rows = std::nullopt,
                                    int num_rows_factor                   = 1);

void dynamic_per_group_scaled_quant_fp4(torch::Tensor& out,         // [..., d]
                                        torch::Tensor const& input, // [..., d]
                                        torch::Tensor& scales,
                                        int group_size                            = 32,
                                        bool shuffle_scale                        = true,
                                        std::optional<at::Tensor> const& num_rows = std::nullopt,
                                        int num_rows_factor                       = 1);

void smooth_per_token_scaled_quant(
    torch::Tensor& out,         // [..., d]
    torch::Tensor const& input, // [..., d]
    torch::Tensor& scales,
    torch::Tensor const& smooth_scale,
    std::optional<torch::Tensor> const& smooth_scale_map = std::nullopt,
    bool shuffle_scale                                   = false,
    std::optional<torch::Tensor> const& num_rows         = std::nullopt,
    int num_rows_factor                                  = 1);

void partial_transpose(torch::Tensor& out,         // [rows, d]
                       torch::Tensor const& input, // [rows, d]
                       torch::Tensor const& num_rows);

} // namespace aiter
