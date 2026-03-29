#pragma once
// SPDX-License-Identifier: MIT
#include <torch/extension.h>

torch::Tensor turboquant_gemm_cktile(
    torch::Tensor x_rot,        // (B, K) fp32
    torch::Tensor packed_idx,   // (N, K/2) uint8
    torch::Tensor packed_idx_T, // (K/2, N) uint8 transposed
    torch::Tensor codebook,     // (16,) fp32
    torch::Tensor norms,        // (N,) or (N, n_groups) fp32
    int group_size
);
