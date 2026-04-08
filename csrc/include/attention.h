#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <string>

void paged_attention(
    const aiter_tensor_t &out, const aiter_tensor_t &exp_sums, const aiter_tensor_t &max_logits,
    const aiter_tensor_t &tmp_out, const aiter_tensor_t &query, const aiter_tensor_t &key_cache,
    const aiter_tensor_t &value_cache, int64_t num_kv_heads, double scale,
    const aiter_tensor_t &block_tables, const aiter_tensor_t &context_lens,
    int64_t block_size, int64_t max_context_len,
    const aiter_tensor_t *alibi_slopes,
    const std::string &kv_cache_dtype, double k_scale, double v_scale,
    const aiter_tensor_t *fp8_out_scale, int64_t partition_size);
