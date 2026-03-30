// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include "py_itfs_common.h"

#include "topk_softmax_decode_api.hpp"

namespace aiter
{

void topk_softmax_decode(torch::Tensor gating_output,      // [1, E]
                         torch::Tensor sorted_token_ids,    // [max_num_tokens_padded]
                         torch::Tensor sorted_weights,      // [max_num_tokens_padded]
                         torch::Tensor sorted_expert_ids,   // [max_num_m_blocks]
                         torch::Tensor num_valid_ids,       // [2]
                         torch::Tensor moe_buf,             // [1, model_dim]
                         int num_experts,
                         int topk,
                         int unit_size,
                         bool renormalize)
{
    TORCH_CHECK(gating_output.size(0) == 1,
                "topk_softmax_decode only supports M=1 (decode), got M=",
                gating_output.size(0));
    TORCH_CHECK(sorted_weights.scalar_type() == at::ScalarType::Float,
                "sorted_weights must be FP32");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(gating_output));

    auto dtype_to_string = [](const auto dtype) -> std::string {
        if(dtype == torch::kFloat16)
            return "fp16";
        else if(dtype == torch::kBFloat16)
            return "bf16";
        else if(dtype == torch::kFloat32)
            return "fp32";
        else
            throw std::runtime_error(
                "invalid datatype for topk_softmax_decode: only fp16/bf16/fp32!");
    };

    std::string input_prec = dtype_to_string(gating_output.dtype());

    topk_softmax_decode_trait trait{input_prec, "fp32", num_experts, "softmax"};

    topk_softmax_decode_kargs karg{
        gating_output.data_ptr(),           // p_input
        num_experts,                        // num_experts
        topk,                               // topk
        num_experts,                        // stride_input (contiguous row)
        renormalize,                        // renormalize
        sorted_token_ids.data_ptr(),        // p_sorted_token_ids
        sorted_weights.data_ptr(),          // p_sorted_weights
        sorted_expert_ids.data_ptr(),       // p_sorted_expert_ids
        num_valid_ids.data_ptr(),           // p_total_tokens_post_pad
        moe_buf.data_ptr(),                 // p_moe_buf
        unit_size,                          // unit_size
        static_cast<int>(moe_buf.size(-1)), // moe_buf_interm_dim
        static_cast<int>(moe_buf.itemsize()) // moe_buf_elem_bytes
    };

    ck_tile::stream_config sc{at::hip::getCurrentHIPStream()};

    topk_softmax_decode(trait, karg, sc);
}

} // namespace aiter
