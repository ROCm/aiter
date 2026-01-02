#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);
void scaled_silu_and_mul(torch::Tensor &out, torch::Tensor &input, torch::Tensor &scale);
void gelu_and_mul(torch::Tensor &out, torch::Tensor &input);
void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input);

// Fused activation+mul+quant kernels
void fused_silu_mul_per_token_quant(torch::Tensor &out, torch::Tensor &scales, torch::Tensor &input);
void fused_gelu_mul_per_token_quant(torch::Tensor &out, torch::Tensor &scales, torch::Tensor &input);
void fused_gelu_tanh_mul_per_token_quant(torch::Tensor &out, torch::Tensor &scales, torch::Tensor &input);

} // namespace aiter
