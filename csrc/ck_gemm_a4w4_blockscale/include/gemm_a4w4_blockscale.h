#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>
torch::Tensor gemm_a4w4_blockscale(
    torch::Tensor &A,
    torch::Tensor &B,
    torch::Tensor &a_scale,
    torch::Tensor &b_scale,
    torch::Tensor &C);
