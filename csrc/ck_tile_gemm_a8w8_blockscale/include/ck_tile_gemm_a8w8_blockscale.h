#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
torch::Tensor ck_tile_gemm_a8w8_blockscale(
        torch::Tensor &XQ,
        torch::Tensor &WQ,
        torch::Tensor &x_scale,
        torch::Tensor &w_scale,
        torch::Tensor &Y);
