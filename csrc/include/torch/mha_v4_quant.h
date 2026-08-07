#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

void rotate_activation_mxfp6_quant(at::Tensor& out,
                                   at::Tensor& scale,
                                   const at::Tensor& input,
                                   double multiplier);

void rotate_activation_mxfp4_quant(at::Tensor& out,
                                   at::Tensor& scale,
                                   const at::Tensor& input,
                                   double multiplier);

} // namespace torch_itfs
} // namespace aiter
