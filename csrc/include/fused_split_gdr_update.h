#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"

namespace aiter {

void fused_split_gdr_update(
    aiter_tensor_t& mixed_qkv,
    aiter_tensor_t& A_log,
    aiter_tensor_t& a,
    aiter_tensor_t& dt_bias,
    aiter_tensor_t& b_gate,
    aiter_tensor_t& initial_state_source,
    aiter_tensor_t& initial_state_indices,
    int key_dim,
    int value_dim,
    int num_heads_qk,
    int num_heads_v,
    int head_dim,
    float softplus_beta,
    float softplus_threshold,
    float scale,
    bool use_qk_l2norm_in_kernel,
    aiter_tensor_t& output);

} // namespace aiter
