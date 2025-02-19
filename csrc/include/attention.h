#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void paged_attention(
    torch::Tensor &out, torch::Tensor &workspace_buffer,
    torch::Tensor &query, torch::Tensor &key_cache,
    torch::Tensor &value_cache, double scale,
    torch::Tensor &kv_indptr, torch::Tensor &kv_page_indices,
    c10::optional<torch::Tensor> &kv_last_page_lens,
    int64_t block_size, int64_t max_num_partitions,
    const c10::optional<torch::Tensor> &alibi_slopes,
    const std::string &kv_cache_dtype, const std::string &kv_cache_layout,
    float logits_soft_cap, torch::Tensor& k_scale, torch::Tensor& v_scale,
    const c10::optional<torch::Tensor> &fp8_out_scale, int64_t partition_size);