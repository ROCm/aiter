// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus-based MOE sorting torch-free binding.
// Self-contained: no CK header dependency.

#define MOE_SORTING_OPUS_IMPL
#include "moe_sorting_opus.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

void moe_sorting_opus_fwd(aiter_tensor_t& topk_ids,
                          aiter_tensor_t& topk_weights,
                          aiter_tensor_t& sorted_token_ids,
                          aiter_tensor_t& sorted_weights,
                          aiter_tensor_t& sorted_expert_ids,
                          aiter_tensor_t& num_valid_ids,
                          aiter_tensor_t& moe_buf,
                          int num_experts,
                          int unit_size,
                          std::optional<aiter_tensor_t> local_expert_mask,
                          std::optional<aiter_tensor_t> num_local_tokens,
                          int dispatch_policy)
{
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32,
                "topk_weights must be FP32 (float32)");

    auto dtype_str = AiterDtype_to_str(topk_ids.dtype());
    int num_tokens = topk_ids.size(0);
    int topk       = topk_ids.size(1);

    HipDeviceGuard device_guard(topk_ids.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();

    int workspace_size = moe_sorting_opus_get_workspace_size(num_tokens, num_experts, topk, dispatch_policy);
    std::optional<AiterTensor> ws;
    void* ws_ptr = nullptr;
    if (workspace_size > 0)
    {
        ws.emplace(AiterTensor::empty(
            {workspace_size}, topk_ids.dtype(), topk_ids.device_id, stream));
        ws_ptr = ws->data_ptr();
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
         static_cast<int>(moe_buf.element_size())},
        {stream});
}
