// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>
#include <string>

namespace aiter {

void fused_qk_norm_rope_cache_quant_shuffle(
    aiter_tensor_t& q,
    aiter_tensor_t& k,
    aiter_tensor_t& v,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    aiter_tensor_t& q_weight,
    aiter_tensor_t& k_weight,
    aiter_tensor_t& cos_sin_cache,
    bool is_neox,
    aiter_tensor_t& position_ids,
    aiter_tensor_t& k_cache,
    aiter_tensor_t& v_cache,
    aiter_tensor_t& slot_mapping,
    const std::string& kv_cache_dtype,
    std::optional<aiter_tensor_t> k_scale,
    std::optional<aiter_tensor_t> v_scale);

void fused_qk_norm_rope_cache_pts_quant_shuffle(aiter_tensor_t& qkv,
                                                aiter_tensor_t& qw,
                                                aiter_tensor_t& kw,
                                                aiter_tensor_t& cos_sin,
                                                aiter_tensor_t& positions,
                                                int64_t num_tokens,
                                                int64_t num_heads_q,
                                                int64_t num_heads_k,
                                                int64_t num_heads_v,
                                                int64_t head_size,
                                                bool is_neox_style,
                                                double eps,
                                                aiter_tensor_t& q_out,
                                                aiter_tensor_t& k_cache,
                                                aiter_tensor_t& v_cache,
                                                aiter_tensor_t& slot_mapping,
                                                aiter_tensor_t& per_tensor_k_scale,
                                                aiter_tensor_t& per_tensor_v_scale,
                                                std::optional<aiter_tensor_t> k_out,
                                                std::optional<aiter_tensor_t> v_out,
                                                bool return_kv,
                                                bool use_shuffle_layout,
                                                int64_t block_size,
                                                int64_t x,
                                                int64_t rotary_dim = 0);

void fused_qk_norm_rope_2way(aiter_tensor_t& q0,
                             aiter_tensor_t& k0,
                             aiter_tensor_t& q1,
                             aiter_tensor_t& k1,
                             aiter_tensor_t& w_q0,
                             aiter_tensor_t& w_k0,
                             aiter_tensor_t& w_q1,
                             aiter_tensor_t& w_k1,
                             aiter_tensor_t& cos_sin0,
                             aiter_tensor_t& cos_sin1,
                             int64_t batch_size,
                             int64_t num_tokens0,
                             int64_t num_tokens1,
                             int64_t num_heads_q,
                             int64_t num_heads_k,
                             int64_t head_size,
                             bool is_interleaved,
                             double eps,
                             aiter_tensor_t& out_q01,
                             aiter_tensor_t& out_k01);

void fused_qk_norm_rope_1way(aiter_tensor_t& q,
                             aiter_tensor_t& k,
                             aiter_tensor_t& w_q,
                             aiter_tensor_t& w_k,
                             aiter_tensor_t& cos_sin,
                             int64_t batch_size,
                             int64_t num_tokens,
                             int64_t num_heads_q,
                             int64_t num_heads_k,
                             int64_t head_size,
                             bool is_interleaved,
                             double eps,
                             aiter_tensor_t& out_q,
                             aiter_tensor_t& out_k);

// Same signature as the pertensor variant, but writes per-(batch, head) descales:
//   q_descale shape [batch_size, num_heads_q]
//   k_descale shape [batch_size, num_heads_k]
// These shapes match what CK FP8 flash attention accepts natively.
void fused_qk_norm_rope_2way_fp8_perhead_quant(aiter_tensor_t& q0,
                                               aiter_tensor_t& k0,
                                               aiter_tensor_t& q1,
                                               aiter_tensor_t& k1,
                                               aiter_tensor_t& w_q0,
                                               aiter_tensor_t& w_k0,
                                               aiter_tensor_t& w_q1,
                                               aiter_tensor_t& w_k1,
                                               aiter_tensor_t& cos_sin0,
                                               aiter_tensor_t& cos_sin1,
                                               int64_t batch_size,
                                               int64_t num_tokens0,
                                               int64_t num_tokens1,
                                               int64_t num_heads_q,
                                               int64_t num_heads_k,
                                               int64_t head_size,
                                               bool is_interleaved,
                                               double eps,
                                               aiter_tensor_t& q_fp8,
                                               aiter_tensor_t& k_fp8,
                                               aiter_tensor_t& q_descale,
                                               aiter_tensor_t& k_descale,
                                               aiter_tensor_t& q_unquantized,
                                               aiter_tensor_t& k_unquantized);

// Per-(batch, head) FP8 quant for concatenated [v0, v1] without a bf16 cat.
// v0/v1: [B, T0/T1, H, D]; v_fp8: [B, T0+T1, H, D]; v_descale: [B, H].
void v_2way_per_head_fp8_quant(aiter_tensor_t& v0,
                               aiter_tensor_t& v1,
                               aiter_tensor_t& v_fp8,
                               aiter_tensor_t& v_descale);

void fused_qk_rmsnorm(aiter_tensor_t& q,
                      aiter_tensor_t& q_weight,
                      double q_eps,
                      aiter_tensor_t& k,
                      aiter_tensor_t& k_weight,
                      double k_eps,
                      aiter_tensor_t& q_out,
                      aiter_tensor_t& k_out);

void fused_qk_norm_rope_cache_block_quant_shuffle(
    aiter_tensor_t& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    aiter_tensor_t& q_weight,
    aiter_tensor_t& k_weight,
    aiter_tensor_t& cos_sin_cache,
    bool is_neox,
    aiter_tensor_t& position_ids,
    aiter_tensor_t& k_cache,
    aiter_tensor_t& v_cache,
    aiter_tensor_t& slot_mapping,
    aiter_tensor_t& cu_q_len,
    const std::string& kv_cache_dtype,
    std::optional<aiter_tensor_t> k_scale,
    std::optional<aiter_tensor_t> v_scale,
    int64_t max_tokens_per_batch = 0);

} // namespace aiter
