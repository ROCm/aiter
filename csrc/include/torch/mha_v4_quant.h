#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

// Apply normalized Walsh-Hadamard rotation to contiguous hd128 rows.
void rotate_activation_hd128(at::Tensor& out, const at::Tensor& input);

// Rotate hd128 rows and emit token-major MX data plus one E8M0 scale per 32 values.
void rotate_activation_mxfp8_quant(at::Tensor& out,
                                   at::Tensor& scale,
                                   const at::Tensor& input,
                                   double multiplier);

void rotate_activation_mxfp6_quant(at::Tensor& out,
                                   at::Tensor& scale,
                                   const at::Tensor& input,
                                   double multiplier);

// The K variants write the coalesced tile layouts consumed directly by MHA v4 code objects.
void rotate_activation_mxfp6_quant_k(at::Tensor& out,
                                     at::Tensor& scale,
                                     const at::Tensor& input);

void rotate_activation_mxfp4_quant(at::Tensor& out,
                                   at::Tensor& scale,
                                   const at::Tensor& input,
                                   double multiplier);

void rotate_activation_mxfp4_quant_k(at::Tensor& out,
                                     at::Tensor& scale,
                                     const at::Tensor& input);

} // namespace torch_itfs
} // namespace aiter
