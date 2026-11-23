#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <optional>

namespace aiter {

// Router GEMM (bf16 x @ w^T, the FlyDSL gfx1250 split-K accumulation order) +
// biased sigmoid top-k gating (topk_gating_kernel_opt) in one launch. See
// csrc/kernels/moe_router_topk_fused.cu.
void moe_router_topk_fused(aiter_tensor_t& topk_weights,
                           aiter_tensor_t& topk_ids,
                           aiter_tensor_t& x,
                           aiter_tensor_t& w,
                           aiter_tensor_t& correction_bias,
                           aiter_tensor_t& workspace,
                           aiter_tensor_t& semaphore,
                           int split_k,
                           int waves,
                           int chunk,
                           bool need_renorm,
                           float routed_scaling_factor,
                           std::optional<aiter_tensor_t> num_valid_rows = std::nullopt,
                           std::optional<aiter_tensor_t> logits_out     = std::nullopt);

} // namespace aiter
