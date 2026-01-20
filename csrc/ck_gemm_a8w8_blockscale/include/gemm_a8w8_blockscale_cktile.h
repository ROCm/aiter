#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <torch/extension.h>

torch::Tensor gemm_a8w8_blockscale_cktile(torch::Tensor& XQ,
                                          torch::Tensor& WQ,
                                          torch::Tensor& x_scale,
                                          torch::Tensor& w_scale,
                                          torch::Tensor& Y,
                                          bool preshuffleB);

torch::Tensor gemm_a8w8_blockscale_cktile_tune(torch::Tensor& XQ,
                                               torch::Tensor& WQ,
                                               torch::Tensor& x_scale,
                                               torch::Tensor& w_scale,
                                               torch::Tensor& Y,
                                               int kernelId,
                                               int splitK,
                                               bool preshuffleB);
