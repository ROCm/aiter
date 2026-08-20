#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <optional>
#include <string>

#include "aiter_tensor.h"

namespace aiter {

// Fused MiniMax-M3 attention pre-processing that writes the paged K/V caches
// directly in the page-16 SHUFFLE layout consumed by pa_decode_gluon.
//
//   k_cache: [num_pages, num_kv_heads, head_dim/x,  page_size, x]
//   v_cache: [num_pages, num_kv_heads, page_size/x, head_dim,  x]   x = 16/elem_size
//
// Replaces the three-kernel sequence (fused QK-norm/RoPE -> reshape_and_cache
// with asm_layout=True -> index-cache scatter) with a single pass over the
// fused ``qkv`` row [q | k | v | index_q | index_k].
void minimax_m3_qknorm_rope_cache_shuffle_insert(
    aiter_tensor_t& qkv,           // [num_tokens, qkv_row] fp16/bf16
    aiter_tensor_t& q_norm_weight, // [head_dim]
    aiter_tensor_t& k_norm_weight, // [head_dim]
    aiter_tensor_t& cos_sin_cache, // [max_pos, rotary_dim]
    aiter_tensor_t& positions,     // [num_tokens] i64
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t num_index_heads,
    int64_t rotary_dim,
    double eps,
    aiter_tensor_t& slot_mapping, // [num_tokens] i64
    aiter_tensor_t& k_cache,
    aiter_tensor_t& v_cache,
    aiter_tensor_t& q_out, // [num_tokens, num_heads * head_dim]
    std::optional<aiter_tensor_t> index_q_norm_weight,
    std::optional<aiter_tensor_t> index_k_norm_weight,
    std::optional<aiter_tensor_t> index_slot_mapping, // [num_tokens] i64
    std::optional<aiter_tensor_t> index_cache,        // [pages, page_size, head_dim]
    std::optional<aiter_tensor_t> index_q_out,        // [num_tokens, niq*head_dim]
    const std::string& kv_cache_dtype,
    std::optional<aiter_tensor_t> k_scale, // fp32 device scalar
    std::optional<aiter_tensor_t> v_scale, // fp32 device scalar
    bool skip_index_branch);

} // namespace aiter
