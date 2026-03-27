#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void unified_attention_fwd(
    torch::Tensor& output,          // [num_tokens, num_heads_q, head_size]
    torch::Tensor& query,           // [num_tokens, num_heads_q, head_size]
    torch::Tensor& key_cache,       // [num_blks, blk_size, num_kv_heads, head_size]
    torch::Tensor& value_cache,     // [num_blks, blk_size, num_kv_heads, head_size]
    torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,        // [num_seqs]
    torch::Tensor& query_start_len, // [num_seqs + 1]
    int mask_type,
    float scale_s,
    float scale,
    float scale_k,
    float scale_v,
    float scale_out);
