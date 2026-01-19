#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {

void causal_conv1d_fn(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias_,
    const c10::optional<torch::Tensor> &seq_idx_,
    const c10::optional<torch::Tensor> &initial_states_,
    torch::Tensor &out,
    c10::optional<torch::Tensor> &final_states_out_,
    bool silu_activation);

void causal_conv1d_update(
    torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& out,
    bool use_silu,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& conv_state_indices,
    int pad_slot_id);

} // namespace aiter

