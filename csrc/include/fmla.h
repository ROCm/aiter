// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

// Returns
//   [0] tile_scheduler_metadata: [num cu parts, metadata size]
//   [1] num_splits:              [batch size + 1]
std::vector<torch::Tensor> get_mla_metadata(
    const torch::Tensor& p_seqlens_k,               // [batch size]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k
);

// Returns
//   [0] output:      [batch size, seqlen of q,     head count of q, head dim of v]
//   [1] softmax_lse: [batch size, head count of q, seqlen of q]
std::vector<torch::Tensor> flash_mla_fwd_with_kvcache_impl(
    torch::Tensor& query,                           // [batch size,  seqlen of q, head count of q,  head dim of qk]
    const torch::Tensor& key_cache,                 // [block count, block size,  head count of kv, head dim of qk]
    const torch::Tensor& value_cache,               // [block count, block size,  head count of kv, head dim of v ]
    const int32_t        head_size_v,
    const torch::Tensor& seqlens_k,                 // [batch size]
    const torch::Tensor& block_table,               // [batch size, max blocks per seq]
    const float          softmax_scale,
    const bool           is_causal,
    const torch::Tensor& tile_scheduler_metadata,   // [num cu parts, metadata size]
    const torch::Tensor& num_splits                 // [batch size + 1]
);

std::vector<torch::Tensor> flash_mla_fwd_prefill_with_kvcache_impl(
    torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const int32_t        head_size_v,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& block_table,
    const float          softmax_scale,
    const bool           is_causal);