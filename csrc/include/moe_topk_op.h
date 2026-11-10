#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <optional>
#include <string>

namespace aiter {

void topk_gating(aiter_tensor_t& topk_weights,
                 aiter_tensor_t& topk_indices,
                 aiter_tensor_t& gating_output,
                 aiter_tensor_t& correction_bias,
                 bool need_renorm,
                 float routed_scaling_factor = 1.0,
                 const std::string& score_func = "sqrtsoftplus",
                 int num_shared_experts = 0,
                 std::optional<aiter_tensor_t> num_valid_rows = std::nullopt);

} // namespace aiter
