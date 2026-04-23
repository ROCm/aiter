// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

namespace aiter {

/**
 * Fused Gemma RMSNorm + FP8 Group Quantization
 *
 * Operations:
 * 1. Optional residual add: x = x + residual (written back inplace)
 * 2. Gemma RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + weight)
 *    - Variance computed over full hidden_size
 *    - Gemma-style weight: (1 + weight) instead of weight
 * 3. FP8 group quantization with group_size=128
 * 4. Optional: also write unquantized normed output
 *
 * Constraints:
 * - hidden_size must be a multiple of 128
 * - group_size must be 128
 *
 * Args:
 *   out: Output quantized tensor [num_tokens, hidden_size] (FP8)
 *   scale: Quantization scales [num_tokens, num_groups] or transposed
 *   x: Input tensor [num_tokens, hidden_size] (bf16/fp16)
 *   weight: RMSNorm weight [hidden_size] (bf16/fp16)
 *   epsilon: Small value for numerical stability
 *   group_size: Quantization group size (MUST be 128)
 *   transpose_scale: If true, store scales in [num_groups, num_tokens] layout
 *   residual: Optional residual tensor [num_tokens, hidden_size] (bf16/fp16)
 *             If provided, computes x = x + residual and writes back inplace
 *   out_normed: Optional tensor [num_tokens, hidden_size] (bf16/fp16)
 *               If provided, also writes the unquantized normed output
 */
void gemma_rmsnorm_fp8_group_quant(
    torch::Tensor& out,
    torch::Tensor& scale,
    torch::Tensor const& x,
    torch::Tensor const& weight,
    double epsilon,
    int group_size,
    bool transpose_scale = false,
    c10::optional<torch::Tensor> residual = c10::nullopt,
    c10::optional<torch::Tensor> out_normed = c10::nullopt);

} // namespace aiter
