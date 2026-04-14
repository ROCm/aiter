// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/all.h>
#include <torch/extension.h>

torch::Tensor opus_gemm(torch::Tensor& XQ,
                        torch::Tensor& WQ,
                        torch::Tensor& Y,
                        std::optional<torch::Tensor> group_layout,
                        std::optional<torch::Tensor> x_scale,
                        std::optional<torch::Tensor> w_scale);

torch::Tensor opus_gemm_a16w16_tune(torch::Tensor& XQ,
                                    torch::Tensor& WQ,
                                    torch::Tensor& Y,
                                    int kernelId,
                                    int splitK);
