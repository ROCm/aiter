#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {

void gdr_decode_packed_bf16(const torch::Tensor& mixed_qkv,
                            const torch::Tensor& a,
                            const torch::Tensor& b,
                            const torch::Tensor& dt_bias,
                            const torch::Tensor& A_log,
                            const torch::Tensor& indices,
                            torch::Tensor& state,
                            torch::Tensor& out,
                            double scale);

} // namespace aiter
