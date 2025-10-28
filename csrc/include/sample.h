// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/torch.h>

namespace aiter {

void greedy_sample(torch::Tensor& out, torch::Tensor& input);

void random_sample_outer_exponential(torch::Tensor& out,
                                     torch::Tensor& input,
                                     torch::Tensor& exponentials,
                                     torch::Tensor& temperatures,
                                     float eps = 1e-10);

void random_sample(torch::Tensor& out,
                   torch::Tensor& input,
                   torch::Tensor& temperatures,
                   float lambd                            = 1.0,
                   std::optional<at::Generator> generator = std::nullopt,
                   float eps                              = 1e-10);

void mixed_sample_outer_exponential(torch::Tensor& out,
                                    torch::Tensor& input,
                                    torch::Tensor& exponentials,
                                    torch::Tensor& temperatures,
                                    float eps = 1e-10);

void mixed_sample(torch::Tensor& out,
                  torch::Tensor& input,
                  torch::Tensor& temperatures,
                  float lambd                            = 1.0,
                  std::optional<at::Generator> generator = std::nullopt,
                  float eps                              = 1e-10);

void exponential(torch::Tensor& out,
                 float lambd                            = 1.0,
                 std::optional<at::Generator> generator = std::nullopt,
                 float eps                              = 1e-10);
} // namespace aiter
