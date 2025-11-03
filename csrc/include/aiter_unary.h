#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor aiter_sigmoid(torch::Tensor &input);
torch::Tensor aiter_tanh(torch::Tensor &input);
