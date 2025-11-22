#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

#include <map>
#include <vector>

namespace aiter {

void swap_blocks(torch::Tensor& src, torch::Tensor& dst, const torch::Tensor& block_mapping);

// Note: the key_caches and value_caches vectors are constant but
// not the Tensors they contain. The vectors need to be const refs
// in order to satisfy pytorch's C++ operator registration code.
void copy_blocks(std::vector<torch::Tensor> const& key_caches,
                 std::vector<torch::Tensor> const& value_caches,
                 const torch::Tensor& block_mapping);

void reshape_and_cache(torch::Tensor& key,
                       torch::Tensor& value,
                       torch::Tensor& key_cache,
                       torch::Tensor& value_cache,
                       torch::Tensor& slot_mapping,
                       const std::string& kv_cache_dtype,
                       std::optional<torch::Tensor> k_scale,
                       std::optional<torch::Tensor> v_scale,
                       const bool asm_layout);

void reshape_and_cache_flash(torch::Tensor& key,
                             torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             const std::string& kv_cache_dtype,
                             torch::Tensor& k_scale,
                             torch::Tensor& v_scale);

void reshape_and_cache_with_pertoken_quant(torch::Tensor& key,
                                           torch::Tensor& value,
                                           torch::Tensor& key_cache,
                                           torch::Tensor& value_cache,
                                           torch::Tensor& k_dequant_scales,
                                           torch::Tensor& v_dequant_scales,
                                           torch::Tensor& slot_mapping,
                                           const bool asm_layout);

void reshape_and_cache_with_block_quant(torch::Tensor& key,
                                        torch::Tensor& value,
                                        torch::Tensor& key_cache,
                                        torch::Tensor& value_cache,
                                        torch::Tensor& k_dequant_scales,
                                        torch::Tensor& v_dequant_scales,
                                        torch::Tensor& slot_mapping,
                                        const bool asm_layout);

void reshape_and_cache_with_block_quant_for_asm_pa(
    torch::Tensor& key,              // [batch_size, seq_len, num_heads, head_size]
    torch::Tensor& value,            // [batch_size, seq_len, num_heads, head_size]
    torch::Tensor& key_cache,        // [num_blocks, num_heads, head_size/x, block_size:16, x]
    torch::Tensor& value_cache,      // [num_blocks, num_heads, head_size, block_size:16]
    torch::Tensor& k_dequant_scales, // [num_heads, num_blocks/(ori_block_size/block_size:16)]
    torch::Tensor& v_dequant_scales, // [num_heads, num_blocks/(ori_block_size/block_size:16)]
    torch::Tensor& slot_mapping,     // [num_tokens]
    const bool asm_layout,
    const int ori_block_size = 128);

void concat_and_cache_mla(torch::Tensor& kv_c,         // [num_tokens, kv_lora_rank]
                          torch::Tensor& k_pe,         // [num_tokens, pe_dim]
                          torch::Tensor& kv_cache,     // [num_blocks, block_size, (kv_lora_rank +
                                                       // pe_dim)]
                          torch::Tensor& slot_mapping, // [num_tokens] or [num_actual_tokens]
                          const std::string& kv_cache_dtype,
                          torch::Tensor& scale);

void indexer_k_quant_and_cache(torch::Tensor& k,        // [num_tokens, head_dim]
                               torch::Tensor& kv_cache, // [num_blocks, block_size, cache_stride]
                               torch::Tensor& slot_mapping, // [num_tokens]
                               int64_t quant_block_size,    // quantization block size
                               const std::string& scale_fmt);

void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,     // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,              // [num_tokens, head_dim]
    torch::Tensor& dst_scale,          // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens); // [batch_size + 1]
} // namespace aiter
