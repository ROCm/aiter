#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"
#include <optional>

namespace aiter {

AiterTensor fused_split_gdr_update(
    const aiter_tensor_t& mixed_qkv,
    const aiter_tensor_t& A_log,
    const aiter_tensor_t& a,
    const aiter_tensor_t& dt_bias,
    const aiter_tensor_t& b_gate,
    const aiter_tensor_t& initial_state_source,
    const aiter_tensor_t& initial_state_indices,
    int key_dim,
    int value_dim,
    int num_heads_qk,
    int num_heads_v,
    int head_dim,
    float softplus_beta,
    float softplus_threshold,
    float scale,
    bool use_qk_l2norm_in_kernel,
    std::optional<aiter_tensor_t> output);

} // namespace aiter
