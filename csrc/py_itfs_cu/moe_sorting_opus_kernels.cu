// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based MOE sorting torch binding.
// Self-contained: no CK header dependency.

#define MOE_SORTING_OPUS_IMPL
#include "moe_sorting_opus.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

namespace {
inline std::string torchDTypeToStr(caffe2::TypeMeta dtype)
{
    switch (dtype.toScalarType())
    {
    case torch::kFloat:            return "fp32";
    case torch::kHalf:             return "fp16";
    case torch::kBFloat16:         return "bf16";
    case torch::kInt32:            return "int32";
    case torch::kInt8:             return "int8";
    case torch::kFloat8_e4m3fnuz:  return "fp8";
    case torch::kFloat8_e4m3fn:    return "fp8";
    default:
        throw std::runtime_error("moe_sorting_opus: unsupported dtype " +
                                 std::to_string((int8_t)(dtype.toScalarType())));
    }
}
} // namespace

void moe_sorting_opus_fwd(torch::Tensor& topk_ids,
                          torch::Tensor& topk_weights,
                          torch::Tensor& sorted_token_ids,
                          torch::Tensor& sorted_weights,
                          torch::Tensor& sorted_expert_ids,
                          torch::Tensor& num_valid_ids,
                          torch::Tensor& moe_buf,
                          int num_experts,
                          int unit_size,
                          std::optional<torch::Tensor> local_expert_mask,
                          std::optional<torch::Tensor> num_local_tokens,
                          int dispatch_policy)
{
    TORCH_CHECK(topk_weights.scalar_type() == at::ScalarType::Float,
                "topk_weights must be FP32 (float32)");

    auto dtype     = topk_ids.dtype();
    auto dtype_str = torchDTypeToStr(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk       = topk_ids.size(1);

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(topk_ids));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    int workspace_size = moe_sorting_opus_get_workspace_size(num_tokens, num_experts, topk, dispatch_policy);
    void* ws_ptr = nullptr;
    if (workspace_size > 0)
    {
        auto ws = torch::empty({workspace_size},
                               torch::TensorOptions().dtype(dtype).device(device_of(topk_ids)));
        ws_ptr = ws.data_ptr();
    }

    moe_sorting_opus(
        {
            dtype_str,
            "fp32",
            local_expert_mask.has_value(),
            true,
            dispatch_policy
        },
        {topk_ids.data_ptr(),
         topk_weights.data_ptr(),
         local_expert_mask.has_value() ? local_expert_mask.value().data_ptr() : nullptr,
         num_local_tokens.has_value() ? num_local_tokens.value().data_ptr() : nullptr,
         sorted_token_ids.data_ptr(),
         sorted_weights.data_ptr(),
         sorted_expert_ids.data_ptr(),
         num_valid_ids.data_ptr(),
         moe_buf.data_ptr(),
         ws_ptr,
         num_tokens,
         unit_size,
         num_experts,
         topk,
         static_cast<int>(moe_buf.size(-1)),
         static_cast<int>(moe_buf.itemsize())},
        {stream});
}
