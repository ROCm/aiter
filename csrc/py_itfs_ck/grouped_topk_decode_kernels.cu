// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include "py_itfs_common.h"

#include "topk_grouped_decode_api.hpp"

namespace aiter
{

void grouped_topk_decode(torch::Tensor gating_output,      // [1, E]
                         torch::Tensor sorted_token_ids,    // [max_num_tokens_padded]
                         torch::Tensor sorted_weights,      // [max_num_tokens_padded]
                         torch::Tensor sorted_expert_ids,   // [max_num_m_blocks]
                         torch::Tensor num_valid_ids,       // [2]
                         torch::Tensor moe_buf,             // [1, model_dim]
                         int num_experts,
                         int topk,
                         int unit_size,
                         bool renormalize,
                         int num_expert_group,
                         int topk_group,
                         c10::optional<torch::Tensor> correction_bias,
                         double routed_scaling_factor)
{
    TORCH_CHECK(gating_output.size(0) == 1,
                "grouped_topk_decode only supports M=1 (decode), got M=",
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
                "invalid datatype for grouped_topk_decode: only fp16/bf16/fp32!");
    };

    std::string input_prec = dtype_to_string(gating_output.dtype());
    std::string activation = "sigmoid";

    topk_grouped_decode_trait trait{input_prec, "fp32", num_experts, activation};

    topk_grouped_decode_kargs karg{
        gating_output.data_ptr(),
        num_experts,
        topk,
        num_experts,
        renormalize,
        num_expert_group,
        topk_group,
        correction_bias.has_value() ? correction_bias->data_ptr() : nullptr,
        static_cast<float>(routed_scaling_factor),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        moe_buf.data_ptr(),
        unit_size,
        static_cast<int>(moe_buf.size(-1)),
        static_cast<int>(moe_buf.itemsize())};

    ck_tile::stream_config sc{at::hip::getCurrentHIPStream()};

    topk_grouped_decode(trait, karg, sc);
}

} // namespace aiter
