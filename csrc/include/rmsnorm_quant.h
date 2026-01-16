// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

namespace aiter {

void add_rmsnorm_quant(torch::Tensor& out,
                       torch::Tensor& input,
                       torch::Tensor& residual_in,
                       torch::Tensor& residual_out,
                       torch::Tensor& scale,
                       torch::Tensor& weight,
                       double epsilon,
                       int group_size     = 0,
                       bool shuffle_scale = false);

void add_rmsnorm(torch::Tensor& out,
                 torch::Tensor& input,
                 torch::Tensor& residual_in,
                 torch::Tensor& residual_out,
                 torch::Tensor& weight,
                 double epsilon);

} // namespace aiter
