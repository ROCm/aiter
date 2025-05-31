#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& tmp_out,     // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor& value_cache, // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables, // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& context_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const std::optional<torch::Tensor>& fp8_out_scale,
    int partition_size);
