// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>

namespace aiter {

/**
 * Fused Gated RMSNorm + FP8 Group Quantization
 *
 * Operations:
 * 1. Per-head Gated RMSNorm: norm(x) * silu(z) where:
 *    - norm(x) = x * weight / sqrt(variance + eps) (standard RMSNorm)
 *    - silu(z) = z / (1 + exp(-z))
 * 2. Flatten: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim]
 * 3. FP8 group quantization with group_size=128
 *
 * Constraints:
 * - ONLY supports head_dim=128 and group_size=128
 * - Each head is exactly one quantization group
 *
 * Args:
 *   out: Output quantized tensor [num_tokens, num_heads * head_dim] (FP8)
 *   scale: Quantization scales [num_heads, num_tokens] (transposed) or [num_tokens, num_heads]
 *   x: Input tensor to normalize [num_tokens, num_heads, head_dim] (bf16/fp16)
 *   z: Gating tensor [num_tokens, num_heads, head_dim] (bf16/fp16)
 *   weight: RMSNorm weight [head_dim] (bf16/fp16)
 *   epsilon: Small value for numerical stability
 *   group_size: Quantization group size (MUST be 128)
 *   transpose_scale: If true, store scales in [num_heads, num_tokens] layout
 */
void gated_rmsnorm_fp8_group_quant(
    torch::Tensor& out,           // [num_tokens, num_heads * head_dim]
    torch::Tensor& scale,          // [num_heads, num_tokens] or [num_tokens, num_heads]
    torch::Tensor const& x,        // [num_tokens, num_heads, head_dim] - input to normalize
    torch::Tensor const& z,        // [num_tokens, num_heads, head_dim] - gating tensor
    torch::Tensor const& weight,   // [head_dim] - RMSNorm weight
    double epsilon,
    int group_size,
    bool transpose_scale = false);


} // namespace aiter
