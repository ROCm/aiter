#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void paged_attention_ragged(
    torch::Tensor& out, // [num_seqs, num_heads, head_size]
    torch::Tensor& workspace_buffer,
    torch::Tensor& query,       // [num_seqs, num_heads, head_size]
    torch::Tensor& key_cache,   // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache, // [num_blocks, num_heads, block_size, head_size] or
                                // [num_blocks, block_size, num_heads, head_size]
    double scale,
    torch::Tensor& kv_indptr,         // [num_seqs + 1]
    torch::Tensor& kv_page_indices,   // [max_num_blocks]
    std::optional<torch::Tensor>& kv_last_page_lens, // [num_seqs]
    int64_t block_size,
    int64_t max_num_partitions,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype,
    const std::string& kv_cache_layout,
    float logits_soft_cap,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale,
    const c10::optional<torch::Tensor>& fp8_out_scale,
    int64_t partition_size);