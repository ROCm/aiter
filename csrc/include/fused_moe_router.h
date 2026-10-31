#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <optional>

namespace aiter {

// Bytes the caller must allocate to serve every M up to max_tokens. Exposed so
// the Python side sizes the workspace without duplicating the layout.
int64_t fused_moe_router_workspace_size(int64_t max_tokens);

void fused_moe_router_impl(aiter_tensor_t& gating,
                           aiter_tensor_t& bias,
                           aiter_tensor_t& hidden,
                           aiter_tensor_t& topk_ids,
                           aiter_tensor_t& topk_weights,
                           aiter_tensor_t& sorted_ids,
                           aiter_tensor_t& sorted_weights,
                           aiter_tensor_t& sorted_expert_ids,
                           aiter_tensor_t& num_valid_ids,
                           aiter_tensor_t& out_fp4,
                           aiter_tensor_t& out_scale,
                           int64_t num_experts,
                           int64_t topk,
                           int64_t unit_size,
                           int64_t group_size,
                           bool need_renorm,
                           double routed_scaling_factor,
                           // Caller-owned scratch, >=
                           // fused_moe_router_workspace_size(M) bytes, with its
                           // first 4 bytes zeroed once at allocation.
                           // Caller-owned so that a CUDA-graph capture, which
                           // bakes this pointer into the replayed launch, cannot
                           // have it reallocated out from under it.
                           aiter_tensor_t& workspace,
                           // [num_experts] int32, nonzero iff this rank owns the
                           // expert; empty/None = no EP
                           std::optional<aiter_tensor_t> expert_mask = std::nullopt,
                           // [M, model_dim] bf16 stage2 accumulator, zeroed by
                           // the kernel (stage2 accumulates atomically).
                           // None = caller zeroed it already.
                           std::optional<aiter_tensor_t> moe_buf = std::nullopt,
                           // Fused shared experts appended after the routed ids,
                           // 0 = none. Each token gives every shared expert
                           // shared_expert_weight. Under EP the shared weights
                           // are replicated on every rank while every rank sees
                           // every token, so token ownership round-robins over
                           // ep_size to keep the post-MoE all-reduce from summing
                           // ep_size copies; ep_size == 1 is the non-EP case.
                           int64_t num_fused_shared_experts = 0,
                           double shared_expert_weight      = 1.0,
                           int64_t ep_rank                  = 0,
                           int64_t ep_size                  = 1);

} // namespace aiter
