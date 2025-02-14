#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void moe_sorting_fwd(torch::Tensor &topk_ids,              // [m, topk]
                     torch::Tensor &topk_weights,          // [m, topk]
                     torch::Tensor &sorted_token_ids,      // [max_num_tokens_padded]
                     torch::Tensor &sorted_weights,        // [max_num_tokens_padded]
                     torch::Tensor &sorted_expert_ids,     // [max_num_m_blocks]
                     torch::Tensor &total_tokens_post_pad, // [1]
                     torch::Tensor &moe_buf,               // [max_num_tokens_padded]
                     int num_experts,
                     int unit_size);