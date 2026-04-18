#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace aiter {

void swap_blocks(const aiter_tensor_t& src, const aiter_tensor_t& dst, const aiter_tensor_t& block_mapping);

void copy_blocks(std::vector<aiter_tensor_t> const& key_caches,
                 std::vector<aiter_tensor_t> const& value_caches,
                 const aiter_tensor_t& block_mapping);

void reshape_and_cache(const aiter_tensor_t& key,
                       const aiter_tensor_t& value,
                       const aiter_tensor_t& key_cache,
                       const aiter_tensor_t& value_cache,
                       const aiter_tensor_t& slot_mapping,
                       const std::string& kv_cache_dtype,
                       std::optional<aiter_tensor_t> k_scale,
                       std::optional<aiter_tensor_t> v_scale,
                       const bool asm_layout);

void reshape_and_cache_flash(const aiter_tensor_t& key,
                             const aiter_tensor_t& value,
                             const aiter_tensor_t& key_cache,
                             const aiter_tensor_t& value_cache,
                             const aiter_tensor_t& slot_mapping,
                             const std::string& kv_cache_dtype,
                             const aiter_tensor_t& k_scale,
                             const aiter_tensor_t& v_scale);

void reshape_and_cache_with_pertoken_quant(const aiter_tensor_t& key,
                                           const aiter_tensor_t& value,
                                           const aiter_tensor_t& key_cache,
                                           const aiter_tensor_t& value_cache,
                                           const aiter_tensor_t& k_dequant_scales,
                                           const aiter_tensor_t& v_dequant_scales,
                                           const aiter_tensor_t& slot_mapping,
                                           const bool asm_layout);

void reshape_and_cache_with_block_quant(const aiter_tensor_t& key,
                                        const aiter_tensor_t& value,
                                        const aiter_tensor_t& key_cache,
                                        const aiter_tensor_t& value_cache,
                                        const aiter_tensor_t& k_dequant_scales,
                                        const aiter_tensor_t& v_dequant_scales,
                                        const aiter_tensor_t& slot_mapping,
                                        const bool asm_layout);

void reshape_and_cache_with_block_quant_for_asm_pa(
    const aiter_tensor_t& key,
    const aiter_tensor_t& value,
    const aiter_tensor_t& key_cache,
    const aiter_tensor_t& value_cache,
    const aiter_tensor_t& k_dequant_scales,
    const aiter_tensor_t& v_dequant_scales,
    const aiter_tensor_t& slot_mapping,
    const bool asm_layout,
    const int ori_block_size = 128);

void concat_and_cache_mla(const aiter_tensor_t& kv_c,
                          const aiter_tensor_t& k_pe,
                          const aiter_tensor_t& kv_cache,
                          const aiter_tensor_t& slot_mapping,
                          const std::string& kv_cache_dtype,
                          const aiter_tensor_t& scale);

void indexer_k_quant_and_cache(const aiter_tensor_t& k,
                               const aiter_tensor_t& kv_cache,
                               const aiter_tensor_t& slot_mapping,
                               int64_t quant_block_size,
                               const std::string& scale_fmt);

void cp_gather_indexer_k_quant_cache(
    const aiter_tensor_t& kv_cache,
    const aiter_tensor_t& dst_k,
    const aiter_tensor_t& dst_scale,
    const aiter_tensor_t& block_table,
    const aiter_tensor_t& cu_seq_lens);

void fused_qk_rope_concat_and_cache_mla(
    const aiter_tensor_t& q_nope,
    const aiter_tensor_t& q_pe,
    const aiter_tensor_t& kv_c,
    const aiter_tensor_t& k_pe,
    const aiter_tensor_t& kv_cache,
    const aiter_tensor_t& q_out,
    const aiter_tensor_t& slot_mapping,
    const aiter_tensor_t& k_scale,
    const aiter_tensor_t& q_scale,
    const aiter_tensor_t& positions,
    const aiter_tensor_t& cos_cache,
    const aiter_tensor_t& sin_cache,
    bool is_neox,
    bool is_nope_first);

} // namespace aiter
