#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_enum.h"
#include "aiter_tensor.h"
#include <string>

void biased_grouped_topk(const aiter_tensor_t& gating_output,   // [num_tokens, num_experts]
                         const aiter_tensor_t& correction_bias, // [num_expert]
                         const aiter_tensor_t& topk_weights,    // [num_tokens, topk]
                         const aiter_tensor_t& topk_ids,        // [num_tokens, topk]
                         int num_expert_group,
                         int topk_group,
                         bool renormalize,
                         const float routed_scaling_factor = 1.);

void grouped_topk(const aiter_tensor_t& gating_output, // [num_tokens, num_experts]
                  const aiter_tensor_t& topk_weights,  // [num_tokens, topk]
                  const aiter_tensor_t& topk_ids,      // [num_tokens, topk]
                  int num_expert_group,
                  int topk_grp,
                  bool need_renorm,
                  bool is_softmax                   = true,
                  const float routed_scaling_factor = 1.);

void moe_fused_gate(const aiter_tensor_t& input,
                    const aiter_tensor_t& bias,
                    const aiter_tensor_t& topk_weights,
                    const aiter_tensor_t& topk_ids,
                    int64_t num_expert_group,
                    int64_t topk_group,
                    int64_t topk,
                    int64_t n_share_experts_fusion,
                    double routed_scaling_factor);

namespace aiter {

void topk_softmax(const aiter_tensor_t& topk_weights,
                  const aiter_tensor_t& topk_indices,
                  const aiter_tensor_t& token_expert_indices,
                  const aiter_tensor_t& gating_output,
                  const aiter_tensor_t& softmax_workspace,
                  bool need_renorm,
                  int num_shared_experts                        = 0,
                  const std::string& shared_expert_scoring_func = "");

// Option A ("fuse-gate") SEPARATE OP: always fuse-gate. The shared-expert weight is
// computed in-kernel as sigmoid(hidden_states @ gate_weight.T) * shared_expert_scale
// (scale applied AFTER the sigmoid) and the shared id (shared_expert_base + s) is
// written by the kernel. gating_output holds routed experts only. hidden_states /
// gate_weight are REQUIRED. Physically duplicated from topk_softmax so base callers
// carry no shared-gate LDS penalty.
void topk_softmax_fused_shared_gate(
    const aiter_tensor_t& topk_weights,          // [num_tokens, topk + num_shared_experts]
    const aiter_tensor_t& topk_indices,          // [num_tokens, topk + num_shared_experts]
    const aiter_tensor_t& token_expert_indices,  // [num_tokens, topk]  (written stride = topk)
    const aiter_tensor_t& gating_output,         // [num_tokens, num_experts]  routed only
    const aiter_tensor_t& softmax_workspace,
    bool need_renorm,
    int num_shared_experts,
    const std::string& shared_expert_scoring_func,
    const aiter_tensor_t& hidden_states,         // [num_tokens, hidden]
    const aiter_tensor_t& gate_weight,           // [num_shared_experts, hidden]
    float shared_expert_scale = 1.0f,
    int shared_expert_base    = -1);

void moe_align_block_size(const aiter_tensor_t& topk_ids,
                          int64_t num_experts,
                          int64_t block_size,
                          const aiter_tensor_t& sorted_token_ids,
                          const aiter_tensor_t& experts_ids,
                          const aiter_tensor_t& token_nums,
                          const aiter_tensor_t& num_tokens_post_pad);

void moe_sum(const aiter_tensor_t& input, const aiter_tensor_t& output);

} // namespace aiter
