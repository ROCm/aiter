// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

void gemma_rmsnorm(torch::Tensor& output,
                   const torch::Tensor& input,
                   const torch::Tensor& weight,
                   double eps = 1e-6);

void gemma_fused_add_rmsnorm(torch::Tensor& input,
                             torch::Tensor& residual,
                             const torch::Tensor& weight,
                             double eps = 1e-6);
