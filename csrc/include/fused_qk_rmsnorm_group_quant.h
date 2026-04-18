// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <optional>

namespace aiter {

void fused_qk_rmsnorm_group_quant(const aiter_tensor_t& q_out_quantized,
                                  const aiter_tensor_t& q_out_scale,
                                  const aiter_tensor_t& q,
                                  const aiter_tensor_t& q_weight,
                                  double q_epsilon,
                                  std::optional<aiter_tensor_t> q_out_unquantized,
                                  std::optional<aiter_tensor_t> k_out,
                                  std::optional<aiter_tensor_t> q_res_out,
                                  std::optional<aiter_tensor_t> k,
                                  std::optional<aiter_tensor_t> k_weight,
                                  std::optional<double> k_epsilon,
                                  std::optional<aiter_tensor_t> q_residual,
                                  int64_t group_size,
                                  bool transpose_scale);

} // namespace aiter
