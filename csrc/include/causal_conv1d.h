#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {

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
