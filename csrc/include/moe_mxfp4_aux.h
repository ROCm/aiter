// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

// torch/extension.h (not torch/all.h) registers the at::Tensor → torch.Tensor
// pybind type caster — without it, aiter's compile_ops auto-signature parser
// fails on `at::Tensor`.
#include <torch/extension.h>

#include <string>

void mxfp4_moe_sort_quant_kernel(
    torch::Tensor& a_input,
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weight,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& cumsum_tensor,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& a_quant,
    torch::Tensor& a_scale,
    torch::Tensor& m_indices,
    torch::Tensor& bf16_zero_out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB);

void mxfp4_moe_sort_kernel(
    torch::Tensor& topk_ids,
    torch::Tensor& topk_weight,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& sorted_expert_ids,
    torch::Tensor& cumsum_tensor,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& m_indices,
    torch::Tensor& bf16_zero_out,
    torch::Tensor& bf16_zero_workspace,
    int64_t M_logical,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t D_INTER,
    int64_t MB,
    int64_t prologue);  // 0 = inline_quant, 1 = threestage

void mxfp4_moe_quant_kernel(
    torch::Tensor& a_input,
    torch::Tensor& a_quant,
    torch::Tensor& a_scale,
    torch::Tensor& bf16_zero_out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB);

void mxfp4_moe_sort_scales_kernel(
    torch::Tensor& a_scale,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& cumsum_tensor,
    torch::Tensor& a_scale_sorted_shuffled,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t D_INTER,
    int64_t MB,
    int64_t max_sorted);

void mxfp4_moe_scatter_reduce_kernel(
    torch::Tensor& flat_out,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB);

void mxfp4_moe_scatter_reduce_q_kernel(
    torch::Tensor& flat_out_q,
    torch::Tensor& flat_out_scale,
    torch::Tensor& reverse_sorted,
    torch::Tensor& sorted_weights,
    torch::Tensor& out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB);
