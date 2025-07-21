// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "py_itfs_common.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "moe_sorting_api.hpp"

void moe_sorting_fwd(torch::Tensor& topk_ids,          // [m, topk]
                     torch::Tensor& topk_weights,      // [m, topk]
                     torch::Tensor& sorted_token_ids,  // [max_num_tokens_padded]
                     torch::Tensor& sorted_weights,    // [max_num_tokens_padded]
                     torch::Tensor& sorted_expert_ids, // [max_num_m_blocks]
                     torch::Tensor& num_valid_ids,     // [1]
                     torch::Tensor& moe_buf,           // [max_num_tokens_padded]
                     int num_experts,
                     int unit_size,
                     std::optional<torch::Tensor> local_expert_mask = std::nullopt,
                     std::optional<torch::Tensor> num_local_tokens  = std::nullopt,
                     int dispatch_policy = 0)
{
    // Ensure that the incoming top‑k weights tensor is FP32
    TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
                "topk_weights must be FP32 (float32)");

    auto dtype = topk_ids.dtype();

    auto dtype_str = torchDTypeToStr(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk       = topk_ids.size(1);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(topk_ids));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int workspace_size = moe_sorting_get_workspace_size(num_tokens, num_experts, topk, dispatch_policy);
    void* ws_ptr       = nullptr;
    if(workspace_size > 0)
    {
        auto ws = torch::empty({workspace_size},
                               torch::TensorOptions().dtype(dtype).device(device_of(topk_ids)));
        ws_ptr  = ws.data_ptr();
    }

    moe_sorting(
        {
            dtype_str,                     // index_type
            "fp32",                        // weight_type; // currently always float
            local_expert_mask.has_value(), // if mask experts as local expert
            true, // if true, no need clear workspace outsize (will take care of it inside API)
            dispatch_policy
        },
        {topk_ids.data_ptr(),     // p_topk_ids
         topk_weights.data_ptr(), // p_weights
         local_expert_mask.has_value() ? local_expert_mask.value().data_ptr() : nullptr,
         num_local_tokens.has_value() ? num_local_tokens.value().data_ptr() : nullptr,
         sorted_token_ids.data_ptr(),  // p_sorted_token_ids
         sorted_weights.data_ptr(),    // p_sorted_weights
         sorted_expert_ids.data_ptr(), // p_sorted_expert_ids
         num_valid_ids.data_ptr(),     // p_total_tokens_post_pad
         moe_buf.data_ptr(),           // p_moe_buf
         ws_ptr,                       // p_workspace
         num_tokens,
         unit_size,
         num_experts,
         topk,
         static_cast<int>(moe_buf.size(-1)),
         static_cast<int>(moe_buf.itemsize())},
        {stream});
}
