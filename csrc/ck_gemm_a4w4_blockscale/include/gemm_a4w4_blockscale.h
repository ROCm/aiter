#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
torch::Tensor gemm_a4w4_blockscale(torch::Tensor& A,
                                   torch::Tensor& B,
                                   torch::Tensor& a_scale,
                                   torch::Tensor& b_scale,
                                   torch::Tensor& C);

torch::Tensor gemm_a4w4_blockscale_tune(torch::Tensor& XQ,
                                        torch::Tensor& WQ,
                                        torch::Tensor& x_scale,
                                        torch::Tensor& w_scale,
                                        torch::Tensor& Y,
                                        int kernelId,
                                        int splitK);
