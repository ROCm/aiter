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

void fused_qk_norm_rope_group_quant(
    aiter_tensor_t& q,                  // [num_tokens, num_heads, head_dim]
    aiter_tensor_t& kv,                 // [num_tokens, (k_num_heads,) head_dim]
    aiter_tensor_t& k_rope_buff,        // [num_tokens, (k_num_heads,) pe_dim] bf16 (RoPE'd K-PE)
    aiter_tensor_t& k_weight,           // [head_dim] RMSNorm weights
    aiter_tensor_t& k_nope_scale_buff,  // [num_tokens, (k_num_heads,) entry_bytes] K nope+scale
    aiter_tensor_t& q_nope_scale_buff,  // [num_tokens, num_heads, head_dim] bf16 (full Q) OR fp8 (nope+scale)
    aiter_tensor_t& positions,          // [num_tokens]
    aiter_tensor_t& cos_cache,          // [max_position, rot_dim//2]
    aiter_tensor_t& sin_cache,          // [max_position, rot_dim//2]
    double eps,                         // epsilon for RMS norm
    bool is_neox,
    // q_weight: optional per-channel RMSNorm weight for Q [head_dim]. nullopt = weightless (V4-Pro).
    std::optional<aiter_tensor_t> q_weight = std::nullopt,
    // q_scale: legacy separate Q scale. Shape [num_tokens, num_heads, head_dim/quant_group_size].
    // dtype: fp32 when scale_dtype="fp32", u8 (e8m0) when scale_dtype="e8m0".
    std::optional<aiter_tensor_t> q_scale = std::nullopt,
    // quant_group_size: width of the 1xG scale block applied to Q. Must be one of {32, 64, 128}
    // and divide head_dim. Default 64 (matches existing K-side hard-coded group). When
    // q_nope_scale_buff is bf16 this is ignored.
    int64_t quant_group_size = 64,
    // scale_dtype: "e8m0" (1-byte MX) or "fp32" (4-byte). Ignored when q_nope_scale_buff is bf16.
    const std::string& scale_dtype = "e8m0",
    // q_rope_buff: rotated Q-PE (bf16) [num_tokens, num_heads, pe_dim], required when Q is fp8
    // (Q mirrors K: nope fp8 + inline scale in q_nope_scale_buff, PE bf16 here). Unused for bf16 Q.
    std::optional<aiter_tensor_t> q_rope_buff = std::nullopt);

// K-only fused RMSNorm + GPT-J/NeoX RoPE + 1xG e8m0 group-quant for the
// V4-Pro Attention.forward inference path (no Q wave). Scatters into a PAGED
// KV cache via slot_mapping: per token the destination is the flat slot
// slot_mapping[token] = physical_block*page_size + offset, split inside the
// kernel against the cache's [num_blocks, page_size, (NK,) entry] strides.
// k_nope_scale_buff holds nope fp8 + inline duplicated e8m0 scale + pad;
// k_rope_buff holds the rotated K-PE (bf16, NOT quantized). MQA: num_kv_heads == 1,
// so the paged caches carry no num_kv_heads dim (one head_dim vector per slot).
void fused_kv_norm_rope_group_quant(
    aiter_tensor_t& kv,                 // [num_tokens, (NK=1,) head_dim]
    aiter_tensor_t& k_rope_buff,        // paged [num_blocks, page_size, rot_dim] bf16
    aiter_tensor_t& k_weight,           // [head_dim] RMSNorm gamma
    aiter_tensor_t& k_nope_scale_buff,  // paged [num_blocks, page_size, head_dim] fp8
    aiter_tensor_t& positions,          // [num_tokens]
    aiter_tensor_t& slot_mapping,       // [num_tokens] int64 flat slot
    aiter_tensor_t& cos_cache,          // [max_position, rot_dim//2]
    aiter_tensor_t& sin_cache,          // [max_position, rot_dim//2]
    double eps,
    bool is_neox,
    int64_t quant_group_size = 64,
    const std::string& scale_dtype = "e8m0");

} // namespace aiter
