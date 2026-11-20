// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

// Fused MoE sort + MXFP8 activation quant for decode-sized batches.
// Takes the router's top-k as input and emits, in one launch, the standard
// sorted routing map (sorted_ids / sorted_weights / sorted_expert_ids /
// num_valid_ids), the per-token MXFP8 activations and the e8m0 scales already
// swizzled per sorted row -- the same outputs moe_sorting followed by
// fused_dynamic_mx_quant_moe_sort produce -- and optionally zeroes moe_buf.
void moe_sort_mx_quant_fwd(aiter_tensor_t& hidden,            // [M, cols] bf16
                           aiter_tensor_t& topk_ids,          // [M, topk] int32
                           aiter_tensor_t& topk_weights,      // [M, topk] fp32
                           aiter_tensor_t& sorted_ids,        // [max_rows] int32
                           aiter_tensor_t& sorted_weights,    // [max_rows] fp32
                           aiter_tensor_t& sorted_expert_ids, // [max_rows / unit_size] int32
                           aiter_tensor_t& num_valid_ids,     // [2] int32
                           aiter_tensor_t& out,               // [M, cols] fp8
                           aiter_tensor_t& out_scale, // [pad32(max_rows), pad8(cols / 32)] e8m0
                           int64_t num_experts,
                           int64_t unit_size,
                           std::optional<aiter_tensor_t> moe_buf = std::nullopt);

} // namespace aiter
