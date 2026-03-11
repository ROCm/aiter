#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "py_itfs_common.h"
#include <torch/all.h>
#include <torch/extension.h>

torch::Tensor deepgemm(torch::Tensor& XQ,
                       torch::Tensor& WQ,
                       torch::Tensor& Y,
                       torch::Tensor& group_layout,
                       std::optional<torch::Tensor> x_scale,
                       std::optional<torch::Tensor> w_scale);
