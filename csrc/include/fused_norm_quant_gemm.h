// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

namespace aiter {

torch::Tensor fused_rmsnorm_quant_gemm(
    const torch::Tensor& input,
    const torch::Tensor& weight_fp8,
    const torch::Tensor& norm_w,
    double eps,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_w,
    torch::Tensor& fp8_workspace,
    int solution_index);

} // namespace aiter
