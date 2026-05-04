// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <optional>
#include <torch/extension.h>

namespace aiter {

void fused_qk_rmsnorm_group_quant(std::optional<torch::Tensor> q_out_quantized,
                                  std::optional<torch::Tensor> q_out_scale,
                                  std::optional<torch::Tensor> q,
                                  std::optional<torch::Tensor> q_weight,
                                  std::optional<double> q_epsilon,
                                  std::optional<torch::Tensor> q_out_unquantized,
                                  std::optional<torch::Tensor> k_out,
                                  std::optional<torch::Tensor> q_res_out,
                                  std::optional<torch::Tensor> k,
                                  std::optional<torch::Tensor> k_weight,
                                  std::optional<double> k_epsilon,
                                  std::optional<torch::Tensor> q_residual,
                                  int64_t group_size,
                                  bool transpose_scale,
                                  bool gemma_norm);

} // namespace aiter
