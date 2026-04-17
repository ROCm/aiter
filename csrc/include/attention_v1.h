#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <optional>
#include <string>

void paged_attention_v1(
    const aiter_tensor_t &out, // [num_seqs, num_heads, head_size]
    const aiter_tensor_t &workspace_buffer,
    const aiter_tensor_t &query, // [num_seqs, num_heads, head_size]
    const aiter_tensor_t &key_cache,
    const aiter_tensor_t &value_cache,
    double scale,
    const aiter_tensor_t &block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const aiter_tensor_t *cu_query_lens,  // [num_seqs+1]
    const aiter_tensor_t &context_lens,   // [num_seqs]
    int64_t max_context_len,
    const aiter_tensor_t *alibi_slopes,
    const std::string &kv_cache_dtype, const std::string &kv_cache_layout,
    float logits_soft_cap, const aiter_tensor_t &k_scale, const aiter_tensor_t &v_scale,
    const aiter_tensor_t *fp8_out_scale, int64_t partition_size);
