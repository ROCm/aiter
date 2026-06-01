// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/all.h>
#include <string>

void mxfp4_moe_gemm1_a4w4_kernel(
    torch::Tensor& cumsum_tensor,
    torch::Tensor& a_quant,
    torch::Tensor& a_scale_sorted_shuffled,
    torch::Tensor& w12_shuffled_quant,
    torch::Tensor& w12_shuffled_scale,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& m_indices,
    torch::Tensor& inter_sorted_quant,
    torch::Tensor& inter_sorted_shuffled_scale,
    torch::Tensor& hidden_states,
    const std::string& kernelName);

void mxfp4_moe_gemm2_a4w4_kernel(
    torch::Tensor& cumsum_tensor,
    torch::Tensor& inter_sorted_quant,
    torch::Tensor& inter_sorted_shuffled_scale,
    torch::Tensor& w3_shuffled_quant,
    torch::Tensor& w3_shuffled_scale,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& sorted_weights,
    torch::Tensor& flat_out,
    int64_t M_logical,
    int64_t max_sorted,
    const std::string& kernelName);
