#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor aiter_add(torch::Tensor &input, torch::Tensor &other);
torch::Tensor aiter_mul(torch::Tensor &input, torch::Tensor &other);
torch::Tensor aiter_sub(torch::Tensor &input, torch::Tensor &other);
torch::Tensor aiter_div(torch::Tensor &input, torch::Tensor &other);

torch::Tensor aiter_add_(torch::Tensor &input, torch::Tensor &other);
torch::Tensor aiter_mul_(torch::Tensor &input, torch::Tensor &other);
torch::Tensor aiter_sub_(torch::Tensor &input, torch::Tensor &other);
torch::Tensor aiter_div_(torch::Tensor &input, torch::Tensor &other);

torch::Tensor aiter_sigmoid(torch::Tensor &input);
torch::Tensor aiter_tanh(torch::Tensor &input);
