#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor aiter_sigmoid(torch::Tensor &input);
torch::Tensor aiter_tanh(torch::Tensor &input);
