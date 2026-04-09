// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <optional>
#include <torch/extension.h>

namespace aiter {

void static_per_tensor_quant(const aiter_tensor_t& out,          // [..., d]
                             const aiter_tensor_t& input,  // [..., d]
                             const aiter_tensor_t& scale); // [1]

void dynamic_per_tensor_quant(const aiter_tensor_t& out,         // [..., d]
                              const aiter_tensor_t& input, // [..., d]
                              const aiter_tensor_t& scale);      // [1]

void dynamic_per_token_scaled_quant(const aiter_tensor_t& out,         // [..., d]
                                    const aiter_tensor_t& input, // [..., d]
                                    const aiter_tensor_t& scales,
                                    std::optional<aiter_tensor_t> scale_ub = std::nullopt,
                                    bool shuffle_scale                    = false,
                                    std::optional<aiter_tensor_t> num_rows = std::nullopt,
                                    int num_rows_factor                   = 1);

void dynamic_per_group_scaled_quant_fp4(const aiter_tensor_t& out,         // [..., d]
                                        const aiter_tensor_t& input, // [..., d]
                                        const aiter_tensor_t& scales,
                                        int group_size                            = 32,
                                        bool shuffle_scale                        = true,
                                        std::optional<aiter_tensor_t> const& num_rows = std::nullopt,
                                        int num_rows_factor                       = 1);

void smooth_per_token_scaled_quant(
    const aiter_tensor_t& out,         // [..., d]
    const aiter_tensor_t& input, // [..., d]
    const aiter_tensor_t& scales,
    const aiter_tensor_t& smooth_scale,
    std::optional<aiter_tensor_t> const& smooth_scale_map      = std::nullopt,
    bool shuffle_scale                                        = false,
    std::optional<aiter_tensor_t> const& num_rows              = std::nullopt,
    int num_rows_factor                                       = 1,
    std::optional<aiter_tensor_t> const& smooth_scale_map_hash = std::nullopt,
    bool enable_ps                                            = true);

void partial_transpose(const aiter_tensor_t& out,         // [rows, d]
                       const aiter_tensor_t& input, // [rows, d]
                       const aiter_tensor_t& num_rows);

void moe_smooth_per_token_scaled_quant_v1(
    const aiter_tensor_t& out,         // [..., d]
    const aiter_tensor_t& input, // [..., d]
    const aiter_tensor_t& scales,
    const aiter_tensor_t& smooth_scale,
    const aiter_tensor_t& smooth_scale_map,
    bool shuffle_scale                                        = false,
    std::optional<aiter_tensor_t> const& smooth_scale_map_hash = std::nullopt,
    bool transpose_out                                        = false);

void moe_smooth_per_token_scaled_quant_v2(const aiter_tensor_t& out,         // [..., d]
                                          const aiter_tensor_t& input, // [..., d]
                                          const aiter_tensor_t& scales,
                                          const aiter_tensor_t& smooth_scale,
                                          const aiter_tensor_t& sorted_token_ids,
                                          const aiter_tensor_t& sorted_expert_ids,
                                          const aiter_tensor_t& num_valid_ids,
                                          int block_m,
                                          bool shuffle_scale = false,
                                          bool transpose_out = false);

void fused_dynamic_mxfp4_quant_moe_sort_hip(torch::Tensor& out,         // [token_num * topk, d / 2]
                                            torch::Tensor& scales,      // swizzled e8m0 bytes
                                            torch::Tensor const& input, // [token_num * topk, d]
                                            torch::Tensor const& sorted_ids,
                                            torch::Tensor const& num_valid_ids,
                                            int token_num,
                                            int block_m,
                                            int group_size = 32);

void mxfp4_moe_sort_hip(torch::Tensor& out_scale,
                         torch::Tensor const& scale,
                         torch::Tensor const& sorted_ids,
                         torch::Tensor const& num_valid_ids,
                         int token_num,
                         int cols);
} // namespace aiter
