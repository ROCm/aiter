// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"

#include "moe_sorting_api.hpp"

void moe_sorting_fwd(torch::Tensor &topk_ids,              // [m, topk]
                     torch::Tensor &topk_weights,          // [m, topk]
                     torch::Tensor &sorted_token_ids,      // [max_num_tokens_padded]
                     torch::Tensor &sorted_weights,        // [max_num_tokens_padded]
                     torch::Tensor &sorted_expert_ids,     // [max_num_m_blocks]
                     torch::Tensor &total_tokens_post_pad, // [1]
                     torch::Tensor &moe_buf,               // [max_num_tokens_padded]
                     int num_experts,
                     int unit_size)
{
    auto dtype = topk_ids.dtype();

    auto dtype_str = torchDTypeToStr(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk = topk_ids.size(1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    moe_sorting({
                    dtype_str, // index_type
                    "fp32"     // weight_type; // currently always float
                },
                {topk_ids.data_ptr(),              // p_topk_ids
                 topk_weights.data_ptr(),          // p_weights
                 sorted_token_ids.data_ptr(),      // p_sorted_token_ids
                 sorted_weights.data_ptr(),        // p_sorted_weights
                 sorted_expert_ids.data_ptr(),     // p_sorted_expert_ids
                 total_tokens_post_pad.data_ptr(), // p_total_tokens_post_pad
                 moe_buf.data_ptr(),               // p_moe_buf
                 num_tokens, unit_size, num_experts, topk, (int)moe_buf.nbytes()},
                {stream});
}
