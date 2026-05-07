// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <optional>
#include <string>
#include <vector>

namespace aiter {

void swap_blocks(aiter_tensor_t& src, aiter_tensor_t& dst, const aiter_tensor_t& block_mapping);

void copy_blocks(std::vector<aiter_tensor_t> const& key_caches,
                 std::vector<aiter_tensor_t> const& value_caches,
                 const aiter_tensor_t& block_mapping);

void reshape_and_cache(aiter_tensor_t& key,
                       aiter_tensor_t& value,
                       aiter_tensor_t& key_cache,
                       aiter_tensor_t& value_cache,
                       aiter_tensor_t& slot_mapping,
                       const std::string& kv_cache_dtype,
                       std::optional<aiter_tensor_t> k_scale,
                       std::optional<aiter_tensor_t> v_scale,
                       const bool asm_layout);

void reshape_and_cache_flash(aiter_tensor_t& key,
                             aiter_tensor_t& value,
                             aiter_tensor_t& key_cache,
                             aiter_tensor_t& value_cache,
                             aiter_tensor_t& slot_mapping,
                             const std::string& kv_cache_dtype,
                             aiter_tensor_t& k_scale,
                             aiter_tensor_t& v_scale);

void reshape_and_cache_with_pertoken_quant(aiter_tensor_t& key,
                                           aiter_tensor_t& value,
                                           aiter_tensor_t& key_cache,
                                           aiter_tensor_t& value_cache,
                                           aiter_tensor_t& k_dequant_scales,
                                           aiter_tensor_t& v_dequant_scales,
                                           aiter_tensor_t& slot_mapping,
                                           const bool asm_layout);

void reshape_and_cache_with_block_quant(aiter_tensor_t& key,
                                        aiter_tensor_t& value,
                                        aiter_tensor_t& key_cache,
                                        aiter_tensor_t& value_cache,
                                        aiter_tensor_t& k_dequant_scales,
                                        aiter_tensor_t& v_dequant_scales,
                                        aiter_tensor_t& slot_mapping,
                                        const bool asm_layout);

void reshape_and_cache_with_block_quant_for_asm_pa(
    aiter_tensor_t& key,              // [batch_size, seq_len, num_heads, head_size]
    aiter_tensor_t& value,            // [batch_size, seq_len, num_heads, head_size]
    aiter_tensor_t& key_cache,        // [num_blocks, num_heads, head_size/x, block_size:16, x]
    aiter_tensor_t& value_cache,      // [num_blocks, num_heads, head_size, block_size:16]
    aiter_tensor_t& k_dequant_scales, // [num_heads, num_blocks/(ori_block_size/block_size:16)]
    aiter_tensor_t& v_dequant_scales, // [num_heads, num_blocks/(ori_block_size/block_size:16)]
    aiter_tensor_t& slot_mapping,     // [num_tokens]
    const bool asm_layout,
    const int ori_block_size = 128);

void concat_and_cache_mla(aiter_tensor_t& kv_c,         // [num_tokens, kv_lora_rank]
                          aiter_tensor_t& k_pe,         // [num_tokens, pe_dim]
                          aiter_tensor_t& kv_cache,     // [num_blocks, block_size, (kv_lora_rank +
                                                        // pe_dim)]
                          aiter_tensor_t& slot_mapping,  // [num_tokens] or [num_actual_tokens]
                          const std::string& kv_cache_dtype,
                          aiter_tensor_t& scale);


void indexer_k_quant_and_cache(aiter_tensor_t& k,            // [num_tokens, head_dim]
                               aiter_tensor_t& kv_cache,     // [num_blocks, block_size, cache_stride]
                               aiter_tensor_t& slot_mapping,  // [num_tokens]
                               int64_t quant_block_size,
                               const std::string& scale_fmt,
                               bool preshuffle = false);

void cp_gather_indexer_k_quant_cache(
    const aiter_tensor_t& kv_cache,     // [num_blocks, block_size, cache_stride]
    aiter_tensor_t& dst_k,              // [num_tokens, head_dim]
    aiter_tensor_t& dst_scale,          // [num_tokens, head_dim / quant_block_size * 4]
    const aiter_tensor_t& block_table,  // [batch_size, num_blocks]
    const aiter_tensor_t& cu_seq_lens,  // [batch_size + 1]
    bool preshuffle = false);

void fused_qk_rope_concat_and_cache_mla(
    aiter_tensor_t& q_nope,       // [num_tokens, num_heads, qk_lora_rank]
    aiter_tensor_t& q_pe,         // [num_tokens, num_heads, pe_dim]
    aiter_tensor_t& kv_c,         // [num_tokens, kv_lora_rank]
    aiter_tensor_t& k_pe,         // [num_tokens, pe_dim]
    aiter_tensor_t& kv_cache,     // [num_blocks, block_size, (kv_lora_rank + pe_dim)]
    aiter_tensor_t& q_out,        // [num_tokens, num_heads, qk_lora_rank+pe_dim]
    aiter_tensor_t& slot_mapping, // [num_tokens] or [num_actual_tokens]
    aiter_tensor_t& k_scale,
    aiter_tensor_t& q_scale,
    aiter_tensor_t& positions, // [num_tokens]
    aiter_tensor_t& cos_cache, // [max_positions, pe_dim//2]
    aiter_tensor_t& sin_cache, // [max_positions, pe_dim//2]
    bool is_neox,
    bool is_nope_first);

} // namespace aiter
