// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"
#include <optional>
#include <string>

namespace aiter {

void fused_qknorm_idxrqknorm(
    aiter_tensor_t& qkv,
    const aiter_tensor_t& q_norm_weight,
    const aiter_tensor_t& k_norm_weight,
    const aiter_tensor_t& cos_sin_cache,
    const aiter_tensor_t& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<aiter_tensor_t> index_q_norm_weight,
    std::optional<aiter_tensor_t> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<aiter_tensor_t> slot_mapping,
    std::optional<aiter_tensor_t> kv_cache_k,
    std::optional<aiter_tensor_t> kv_cache_v,
    std::optional<aiter_tensor_t> index_cache,
    int64_t block_size,
    std::optional<aiter_tensor_t> q_out,
    std::optional<aiter_tensor_t> index_q_out,
    std::optional<aiter_tensor_t> index_slot_mapping,
    const std::string& kv_cache_dtype = "auto",
    const std::string& index_cache_dtype = "auto",
    std::optional<aiter_tensor_t> k_scale = std::nullopt,
    std::optional<aiter_tensor_t> v_scale = std::nullopt,
    bool asm_layout = false,
    bool skip_index_branch = false,
    // Write the index cache in the FlyDSL index scorers' shuffled page layout
    // (minimax_m3_index_score.shuffle_cache) rather than row major. Requires
    // whole 128x128 index pages -- the page size is the layout's own, not the
    // block_size argument. Off by default: it changes the bytes a consumer sees.
    bool index_shuffled = false);

} // namespace aiter
