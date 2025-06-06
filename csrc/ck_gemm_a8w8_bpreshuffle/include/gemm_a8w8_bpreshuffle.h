#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
torch::Tensor gemm_a8w8_bpreshuffle(torch::Tensor& XQ,
                                    torch::Tensor& WQ,
                                    torch::Tensor& x_scale,
                                    torch::Tensor& w_scale,
                                    torch::Tensor& Y);

torch::Tensor gemm_a8w8_bpreshuffle_tune(torch::Tensor& XQ,
                                         torch::Tensor& WQ,
                                         torch::Tensor& x_scale,
                                         torch::Tensor& w_scale,
                                         torch::Tensor& Y,
                                         int kernelId,
                                         int splitK);
