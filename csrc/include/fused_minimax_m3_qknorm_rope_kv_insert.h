// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2024-2026, The vLLM team.
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <optional>
#include <string>
#include <torch/extension.h>

namespace aiter {

void fused_minimax_m3_qknorm_rope_kv_insert(
    at::Tensor& qkv,
    const at::Tensor& q_norm_weight,
    const at::Tensor& k_norm_weight,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    std::optional<at::Tensor> index_q_norm_weight,
    std::optional<at::Tensor> index_k_norm_weight,
    int64_t num_index_heads,
    std::optional<at::Tensor> slot_mapping,
    std::optional<at::Tensor> kv_cache,
    std::optional<at::Tensor> index_cache,
    int64_t block_size,
    std::optional<at::Tensor> q_out,
    std::optional<at::Tensor> index_q_out,
    std::optional<at::Tensor> index_slot_mapping);

void fused_minimax_m3_qknorm_rope_kv_insert_fp8(
    at::Tensor& qkv,
    const at::Tensor& q_norm_weight,
    const at::Tensor& k_norm_weight,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& positions,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t rotary_dim,
    double eps,
    const at::Tensor& index_q_norm_weight,
    const at::Tensor& index_k_norm_weight,
    int64_t num_index_heads,
    const at::Tensor& slot_mapping,
    at::Tensor& kv_cache,
    at::Tensor& index_cache,
    int64_t block_size,
    at::Tensor& q_out,
    at::Tensor& index_q_out,
    const at::Tensor& index_slot_mapping,
    const std::string& kv_cache_dtype,
    const at::Tensor& k_scale,
    const at::Tensor& v_scale);

} // namespace aiter
