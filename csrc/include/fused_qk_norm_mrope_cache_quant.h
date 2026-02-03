// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

using namespace at;

void fused_qk_norm_mrope_3d_cache_quant_shuffle(Tensor& qkv,
                               Tensor& qw,
                               Tensor& kw,
                               Tensor& cos_sin,
                               Tensor& positions,
                               int64_t num_tokens,
                               int64_t num_heads_q,
                               int64_t num_heads_k,
                               int64_t num_heads_v,
                               int64_t head_size,
                               bool is_neox_style,
                               std::vector<int64_t> mrope_section_,
                               bool is_interleaved,
                               double eps,
                               Tensor& q,
                               Tensor& k_cache,
                               Tensor& v_cache,
                               Tensor& kv_loc,
                               Tensor& k_scale,
                               Tensor& v_scale,
                               std::optional<Tensor> k_out,
                               std::optional<Tensor> v_out,
                               bool return_kv,
                               bool use_shuffle_layout,
                               int64_t block_size,
                               int64_t x);
