#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {

torch::Tensor fused_split_gdr_update(
    torch::Tensor mixed_qkv,
    torch::Tensor A_log,
    torch::Tensor a,
    torch::Tensor dt_bias,
    torch::Tensor b_gate,
    torch::Tensor initial_state_source,
    torch::Tensor initial_state_indices,
    int key_dim,
    int value_dim,
    int num_heads_qk,
    int num_heads_v,
    int head_dim,
    float softplus_beta,
    float softplus_threshold,
    float scale,
    bool use_qk_l2norm_in_kernel,
    c10::optional<torch::Tensor> output);

} // namespace aiter
