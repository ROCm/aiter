// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

std::vector<torch::Tensor> flash_mla_fwd_inline_impl(
    torch::Tensor&       query,
    const torch::Tensor& key_cache,
    const torch::Tensor& qo_indptr,
    const torch::Tensor& cache_seqlens,
    const torch::Tensor& block_table,
    const torch::Tensor& kv_last_page_lens,
    const torch::Tensor& num_kv_splits_indptr,

    const int            max_seqlen_q,
    const float          softmax_scale,

    torch::Tensor&       split_data,
    torch::Tensor&       split_lse,
    std::optional<torch::Tensor>&       query_rope,
    const std::optional<torch::Tensor>& key_rope_cache,
    const std::optional<torch::Tensor>& batch_split_table,
    const std::optional<torch::Tensor>& split_table,
    std::optional<torch::Tensor>&       out,
    const int            num_splits_ = 1);

std::vector<torch::Tensor> get_mla_metadata_impl(
    const torch::Tensor& kv_indptr,            // [batch size + 1]
    torch::Tensor&       num_kv_splits_indptr, // [batch size + 1]
    torch::Tensor&       batch_split_table,    // [max_cu_num]
    torch::Tensor&       split_table);         // [max_cu_num]
