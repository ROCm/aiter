#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {

void silu_and_mul(torch::Tensor &out, torch::Tensor &input);
void scaled_silu_and_mul(torch::Tensor &out, torch::Tensor &input, torch::Tensor &scale);
void gelu_and_mul(torch::Tensor &out, torch::Tensor &input);
void gelu_tanh_and_mul(torch::Tensor &out, torch::Tensor &input);

} // namespace aiter
