// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"

#include <string>

void mxfp4_moe_sort_quant_kernel(
    aiter_tensor_t& a_input,
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& topk_weight,
    aiter_tensor_t& sorted_token_ids,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& cumsum_tensor,
    aiter_tensor_t& reverse_sorted,
    aiter_tensor_t& sorted_weights,
    aiter_tensor_t& a_quant,
    aiter_tensor_t& a_scale,
    aiter_tensor_t& m_indices,
    aiter_tensor_t& bf16_zero_out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB);

void mxfp4_moe_sort_kernel(
    aiter_tensor_t& topk_ids,
    aiter_tensor_t& topk_weight,
    aiter_tensor_t& sorted_token_ids,
    aiter_tensor_t& sorted_expert_ids,
    aiter_tensor_t& cumsum_tensor,
    aiter_tensor_t& reverse_sorted,
    aiter_tensor_t& sorted_weights,
    aiter_tensor_t& m_indices,
    aiter_tensor_t& bf16_zero_out,
    aiter_tensor_t& bf16_zero_workspace,
    aiter_tensor_t& sort3stage_ws,
    int64_t M_logical,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t D_INTER,
    int64_t MB,
    int64_t prologue);  // 0 = inline_quant, 1 = threestage

void mxfp4_moe_quant_kernel(
    aiter_tensor_t& a_input,
    aiter_tensor_t& a_quant,
    aiter_tensor_t& a_scale,
    aiter_tensor_t& bf16_zero_out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB);

void mxfp4_moe_sort_scales_kernel(
    aiter_tensor_t& a_scale,
    aiter_tensor_t& sorted_token_ids,
    aiter_tensor_t& cumsum_tensor,
    aiter_tensor_t& a_scale_sorted_shuffled,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB,
    int64_t max_sorted);

void mxfp4_moe_scatter_reduce_kernel(
    aiter_tensor_t& flat_out,
    aiter_tensor_t& reverse_sorted,
    aiter_tensor_t& sorted_weights,
    aiter_tensor_t& out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB);

void mxfp4_moe_scatter_reduce_q_kernel(
    aiter_tensor_t& flat_out_q,
    aiter_tensor_t& flat_out_scale,
    aiter_tensor_t& reverse_sorted,
    aiter_tensor_t& sorted_weights,
    aiter_tensor_t& out,
    int64_t NE,
    int64_t TOPK,
    int64_t D_HIDDEN,
    int64_t MB);
