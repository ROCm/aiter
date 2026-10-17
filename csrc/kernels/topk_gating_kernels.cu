// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Public entry point for the fused topk gating kernel (MoE routing).
//
// Scoring functions (selected by string):
//   "sqrtsoftplus"  -> sqrt(softplus(x))   - DeepSeek V4-Pro default
//   "sigmoid"       -> sigmoid(x)          - Llama4
//   "softmax"       -> softmax(x)          - DeepSeek V3 / classic MoE
//
// The kernels live in csrc/include/topk_gating_kernels.cuh, instantiated by the
// TUs under csrc/kernels/topk_gating/. This file only resolves the runtime
// dtype/score_func pair to one of those instantiations.

// This translation unit is torch-free: define AITER_NO_TORCH_TYPES before any
// aiter header so topk_gating_kernels.cuh's aiter_opus_plus.h include does not
// pull in the c10 half/bfloat16 headers (the kernels use aiter::hip2opus, never
// the t2opus<c10::*> specializations).
#define AITER_NO_TORCH_TYPES
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "moe_topk_op.h"
#include "topk_gating_kernels.cuh"

#include <string>
#include <type_traits>

namespace aiter {

// Resolve "sqrtsoftplus"/"sigmoid"/"softmax" -> SCORE_* enum, or AITER_CHECK fail.
static inline int parse_score_func(const std::string& s)
{
    if(s == "sqrtsoftplus") return SCORE_SQRTSOFTPLUS;
    if(s == "sigmoid")      return SCORE_SIGMOID;
    if(s == "softmax")      return SCORE_SOFTMAX;
    AITER_CHECK(false, "unknown score_func: ", s,
                " (expected sqrtsoftplus|sigmoid|softmax)");
    return SCORE_SQRTSOFTPLUS;  // unreachable
}

void topk_gating(aiter_tensor_t& topk_weights,
                 aiter_tensor_t& topk_indices,
                 aiter_tensor_t& gating_output,
                 aiter_tensor_t& correction_bias,
                 bool need_renorm,
                 float routed_scaling_factor,
                 const std::string& score_func,
                 int num_shared_experts)
{
    AITER_CHECK(topk_weights.dtype() == AITER_DTYPE_fp32,
                "topk_weights must be float32");
    AITER_CHECK(topk_indices.dtype() == AITER_DTYPE_i32,
                "topk_indices must be int32");
    AITER_CHECK(num_shared_experts == 0 || num_shared_experts == 1 ||
                    num_shared_experts == 2 || num_shared_experts == 4 ||
                    num_shared_experts == 8,
                "num_shared_experts must be 0, 1, 2, 4 or 8, got ",
                num_shared_experts);

    HipDeviceGuard device_guard(gating_output.device_id);

    const int sf_code       = parse_score_func(score_func);
    const int gating_cols   = static_cast<int>(gating_output.size(1));
    const int num_experts   = gating_cols - num_shared_experts;
    const int topk          = static_cast<int>(topk_indices.size(1));
    const bool has_bias     = correction_bias.numel() > 0;

    AITER_CHECK(num_experts > 0,
                "gating_output has no routed experts after ", num_shared_experts,
                " shared expert column(s)");
    AITER_CHECK(gating_output.stride(1) == 1,
                "gating_output rows must be contiguous");
    AITER_CHECK(topk <= static_cast<int>(WARP_SIZE),
                "topk (", topk, ") exceeds WARP_SIZE (", WARP_SIZE, ")");
    AITER_CHECK(topk <= num_experts,
                "topk (", topk, ") exceeds num_experts (", num_experts, ")");
    AITER_CHECK(topk_weights.stride(0) == topk_indices.stride(0),
                "topk_weights and topk_indices must share a row stride");
    if(num_shared_experts > 0)
    {
        AITER_CHECK(topk_weights.size(1) >= topk + num_shared_experts,
                    "topk_weights needs ", topk + num_shared_experts,
                    " columns for topk plus shared experts, got ",
                    topk_weights.size(1));
    }

    topk_gating_params p{};
    p.gating                = gating_output.data_ptr();
    p.bias                  = has_bias ? correction_bias.data_ptr() : nullptr;
    p.weights               = reinterpret_cast<float*>(topk_weights.data_ptr());
    p.ids                   = reinterpret_cast<int*>(topk_indices.data_ptr());
    p.stride_tk             = static_cast<size_t>(topk_indices.stride(0));
    p.num_experts           = num_experts;
    p.gating_stride         = static_cast<int>(gating_output.stride(0));
    p.num_shared_experts    = num_shared_experts;
    p.topk                  = topk;
    p.num_tokens            = static_cast<int>(gating_output.size(0));
    p.routed_scaling_factor = routed_scaling_factor;
    p.need_renorm           = need_renorm;
    p.stream                = aiter::getCurrentHIPStream();

    const auto gating_st = gating_output.dtype();
    // Unbiased: the pointer is never dereferenced, so pick an instantiated DTYPE_B
    // rather than depending on the placeholder tensor's dtype.
    const auto bias_st = has_bias ? correction_bias.dtype() : AITER_DTYPE_fp32;

    // Three-level dispatch: gating dtype -> score_func -> bias dtype. See
    // _AITER_TOPK_GATING_SLICE for which bias dtypes are instantiated.
    auto dispatch_bias = [&](auto gating_tag, auto sf_tag) {
        using scalar_t   = decltype(gating_tag);
        constexpr int SF = decltype(sf_tag)::value;
        if(bias_st == AITER_DTYPE_fp32)
        {
            topk_gating_launch<scalar_t, float, SF>(p);
            return;
        }
        if constexpr(!std::is_same_v<scalar_t, __half>)
        {
            if(bias_st == AITER_DTYPE_bf16)
            {
                topk_gating_launch<scalar_t, hip_bfloat16, SF>(p);
                return;
            }
        }
        AITER_CHECK(false,
                    "correction_bias dtype must be float32, or bfloat16 when "
                    "gating_output is not float16");
    };

    auto dispatch_sf = [&](auto gating_tag) {
        switch(sf_code)
        {
        case SCORE_SIGMOID:
            dispatch_bias(gating_tag, std::integral_constant<int, SCORE_SIGMOID>{}); break;
        case SCORE_SOFTMAX:
            dispatch_bias(gating_tag, std::integral_constant<int, SCORE_SOFTMAX>{}); break;
        default:
            dispatch_bias(gating_tag, std::integral_constant<int, SCORE_SQRTSOFTPLUS>{}); break;
        }
    };

    switch(gating_st)
    {
    case AITER_DTYPE_fp32: dispatch_sf(float{});        break;
    case AITER_DTYPE_fp16: dispatch_sf(__half{});       break;
    case AITER_DTYPE_bf16: dispatch_sf(hip_bfloat16{}); break;
    default: AITER_CHECK(false, "unsupported gating_output dtype"); break;
    }
}

} // namespace aiter
