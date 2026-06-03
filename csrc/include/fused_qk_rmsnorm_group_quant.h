// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

void fused_qk_rmsnorm_group_quant(std::optional<aiter_tensor_t> q_out_quantized,
                                  std::optional<aiter_tensor_t> q_out_scale,
                                  std::optional<aiter_tensor_t> q,
                                  std::optional<aiter_tensor_t> q_weight,
                                  std::optional<double> q_epsilon,
                                  std::optional<aiter_tensor_t> q_out_unquantized,
                                  std::optional<aiter_tensor_t> k_out,
                                  std::optional<aiter_tensor_t> q_res_out,
                                  std::optional<aiter_tensor_t> k,
                                  std::optional<aiter_tensor_t> k_weight,
                                  std::optional<double> k_epsilon,
                                  std::optional<aiter_tensor_t> q_residual,
                                  int64_t group_size,
                                  bool transpose_scale,
                                  bool gemma_norm);

void fused_qk_rmsnorm_per_token_quant(aiter_tensor_t& q_out_quantized,
                                      aiter_tensor_t& q_out_scale,
                                      aiter_tensor_t& q,
                                      aiter_tensor_t& q_weight,
                                      double q_epsilon,
                                      std::optional<aiter_tensor_t> q_out_unquantized,
                                      std::optional<aiter_tensor_t> k_out,
                                      std::optional<aiter_tensor_t> q_res_out,
                                      std::optional<aiter_tensor_t> k,
                                      std::optional<aiter_tensor_t> k_weight,
                                      std::optional<double> k_epsilon,
                                      std::optional<aiter_tensor_t> q_residual,
                                      bool gemma_norm);

} // namespace aiter
