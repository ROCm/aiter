#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {

void causal_conv1d_fwd(
    torch::Tensor& out,           // [batch, dim, seqlen]
    const torch::Tensor& x,       // [batch, dim, seqlen]
    const torch::Tensor& weight,  // [dim, width]
    const torch::Tensor& bias,    // [dim] or empty
    bool use_silu);

} // namespace aiter

