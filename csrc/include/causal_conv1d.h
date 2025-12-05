#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {

void causal_conv1d_fwd(
    const torch::Tensor &x,
    const torch::Tensor &weight,
    const c10::optional<torch::Tensor> &bias_,
    const c10::optional<torch::Tensor> &seq_idx_,
    const c10::optional<torch::Tensor> &initial_states_,
    torch::Tensor &out,
    c10::optional<torch::Tensor> &final_states_out_,
    bool silu_activation);

void causal_conv1d_update(
    torch::Tensor& x,                          // [batch, dim, seqlen] - new input (typically seqlen=1)
    torch::Tensor& conv_state,                 // [batch, dim, state_len] - state buffer (updated in-place)
    const torch::Tensor& weight,               // [dim, width]
    const torch::Tensor& bias,                 // [dim] or empty
    torch::Tensor& out,                        // [batch, dim, seqlen] - output
    bool use_silu,
    const torch::Tensor& cache_seqlens,        // [batch] - optional, for circular buffer
    const torch::Tensor& conv_state_indices);  // [batch] - optional, for continuous batching

} // namespace aiter

