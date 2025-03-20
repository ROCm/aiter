#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void topk_softmax(torch::Tensor &topk_weights, torch::Tensor &topk_indices,
                  torch::Tensor &token_expert_indices,
                  torch::Tensor &gating_output,
                  bool need_renorm);

void biased_grouped_topk(
    torch::Tensor &gating_output,   // [num_tokens, num_experts]
    torch::Tensor &correction_bias, // [num_expert]
    torch::Tensor &topk_weights,    // [num_tokens, topk]
    torch::Tensor &topk_ids,        // [num_tokens, topk]
    int num_expert_group,
    int topk_group,
    bool renormalize,
    const float routed_scaling_factor = 1.);

void grouped_topk(
    torch::Tensor &gating_output, // [num_tokens, num_experts]
    torch::Tensor &topk_weights,  // [num_tokens, topk]
    torch::Tensor &topk_ids,      // [num_tokens, topk]
    int num_expert_group,
    int topk_grp,
    bool need_renorm,
    std::string scoring_func = "softmax",
    const float routed_scaling_factor = 1.);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor token_nums,
                          torch::Tensor num_tokens_post_pad);

void fmoe(torch::Tensor &out,               // [token_cnt, dim]
          torch::Tensor &input,             // [token_cnt, dim] M,K
          torch::Tensor &gate,              // [expert, hidden_dim, dim] N,K
          torch::Tensor &down,              // [expert, hidden_dim, dim]
          torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
          torch::Tensor &sorted_weight_buf, // [max_num_tokens_padded]
          torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
          torch::Tensor &num_valid_ids,     // [1]
          uint32_t topk                     //
);

namespace py = pybind11;
enum class ActivationType : int
{
    Silu,
    Gelu
};

void fmoe_int8_g1u0(torch::Tensor &out,               // [token_cnt, dim]
                    torch::Tensor &input,             // [token_cnt, dim] M,K
                    torch::Tensor &gate,              // [expert, hidden_dim, dim] N,K
                    torch::Tensor &down,              // [expert, hidden_dim, dim]
                    torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                    torch::Tensor &sorted_weight_buf, // [max_num_tokens_padded]
                    torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                    torch::Tensor &num_valid_ids,     // [1]
                    uint32_t topk,                    //
                    torch::Tensor &input_scale,       // [token_cnt, 1]
                    torch::Tensor &fc1_scale,         // [expert, 1, hidden_dim]
                    torch::Tensor &fc2_scale,         // [expert, 1, dim]
                    torch::Tensor &fc2_smooth_scale,  // [expert, 1, hidden_dim]
                    ActivationType activation = ActivationType::Silu);

void fmoe_g1u1(torch::Tensor &out,                                           // [token_cnt, dim]
               torch::Tensor &input,                                         // [token_cnt, dim] M,K
               torch::Tensor &gate,                                          // [expert, hidden_dim*2, dim] N,K
               torch::Tensor &down,                                          // [expert, hidden_dim, dim]
               torch::Tensor &sorted_token_ids,                              // [max_num_tokens_padded]
               torch::Tensor &sorted_weight_buf,                             // [max_num_tokens_padded]
               torch::Tensor &sorted_expert_ids,                             // [max_num_m_blocks]
               torch::Tensor &num_valid_ids,                                 // [1]
               uint32_t topk,                                                //
               torch::Tensor &input_scale,                                   // [token_cnt, 1]
               torch::Tensor &fc1_scale,                                     // [expert, 1, hidden_dim]
               torch::Tensor &fc2_scale,                                     // [expert, 1, dim]
               std::optional<torch::Tensor> fc2_smooth_scale = std::nullopt, // [expert, 1, hidden_dim]
               ActivationType activation = ActivationType::Silu);

void fmoe_int8_g1u0_a16(torch::Tensor &out,               // [token_cnt, dim]
                        torch::Tensor &input,             // [token_cnt, dim] M,K
                        torch::Tensor &gate,              // [expert, inter_dim, dim] N,K
                        torch::Tensor &down,              // [expert, dim, inter_dim]
                        torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                        torch::Tensor &sorted_weight_buf, // [max_num_tokens_padded]
                        torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                        torch::Tensor &num_valid_ids,     // [1]
                        uint32_t topk,                    //
                        torch::Tensor &fc1_scale,         // [expert, 1, hidden_dim]
                        torch::Tensor &fc2_scale,         // [expert, 1, dim]
                        torch::Tensor &fc1_smooth_scale,  // [expert, 1, hidden_dim]
                        torch::Tensor &fc2_smooth_scale   // [expert, 1, hidden_dim]
);
void fmoe_g1u1_a16(torch::Tensor &out,               // [token_cnt, dim]
                   torch::Tensor &input,             // [token_cnt, dim] M,K
                   torch::Tensor &gate,              // [expert, inter_dim, dim] N,K
                   torch::Tensor &down,              // [expert, dim, inter_dim]
                   torch::Tensor &sorted_token_ids,  // [max_num_tokens_padded]
                   torch::Tensor &sorted_weight_buf, // [max_num_tokens_padded]
                   torch::Tensor &sorted_expert_ids, // [max_num_m_blocks]
                   torch::Tensor &num_valid_ids,     // [1]
                   uint32_t topk,                    //
                   torch::Tensor &fc1_scale,         // [expert, 1, hidden_dim]
                   torch::Tensor &fc2_scale,         // [expert, 1, dim]
                   torch::Tensor &fc1_smooth_scale,  // [expert, 1, hidden_dim]
                   torch::Tensor &fc2_smooth_scale   // [expert, 1, hidden_dim]
);

void fmoe_fp8_blockscale_g1u1(torch::Tensor &out,                           // [token_cnt, dim]
                              torch::Tensor &input,                         // [token_cnt, dim] M,K
                              torch::Tensor &gate,                          // [expert, inter_dim*2, dim] N,K
                              torch::Tensor &down,                          // [expert, dim, inter_dim]
                              torch::Tensor &sorted_token_ids,              // [max_num_tokens_padded]
                              torch::Tensor &sorted_weight_buf,             // [max_num_tokens_padded]
                              torch::Tensor &sorted_expert_ids,             // [max_num_m_blocks]
                              torch::Tensor &num_valid_ids,                 // [1]
                              uint32_t topk,                                //
                              torch::Tensor &fc1_scale,                     // [expert, 1, inter_dim]
                              torch::Tensor &fc2_scale,                     // [expert, 1, dim]
                              torch::Tensor input_scale,                    // [expert, 1, dim]
                              int fc_scale_blkn,                            // = 128,
                              int fc_scale_blkk,                            // = 128
                              std::optional<torch::Tensor> fc2_smooth_scale // [expert, 1, inter_dim]
);

void moe_sum(torch::Tensor &input, torch::Tensor &output);
