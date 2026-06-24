#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

namespace aiter {

void topk_softplus(aiter_tensor_t& topk_weights,
                    aiter_tensor_t& topk_indices,
                    aiter_tensor_t& gating_output,
                    aiter_tensor_t& correction_bias,
                    bool need_renorm,
                    float routed_scaling_factor = 1.0,
                    const std::string& score_func = "sqrtsoftplus");

} // namespace aiter
