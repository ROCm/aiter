#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void aiter_add(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output);
void aiter_mul(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output);
void aiter_sub(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output);
void aiter_div(torch::Tensor &input, torch::Tensor &other, torch::Tensor &output);

void aiter_add_(torch::Tensor &input, torch::Tensor &other);
void aiter_mul_(torch::Tensor &input, torch::Tensor &other);
void aiter_sub_(torch::Tensor &input, torch::Tensor &other);
void aiter_div_(torch::Tensor &input, torch::Tensor &other);

void aiter_sigmoid(torch::Tensor &input, torch::Tensor &output);
void aiter_tanh(torch::Tensor &input, torch::Tensor &output);