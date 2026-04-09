// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"
#include <optional>
#include <string>
#include <vector>

void fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(const aiter_tensor_t& qkv,
                                                    const aiter_tensor_t& qw,
                                                    const aiter_tensor_t& kw,
                                                    const aiter_tensor_t& cos_sin,
                                                    const aiter_tensor_t& positions,
                                                    int64_t num_tokens,
                                                    int64_t num_heads_q,
                                                    int64_t num_heads_k,
                                                    int64_t num_heads_v,
                                                    int64_t head_size,
                                                    bool is_neox_style,
                                                    std::vector<int64_t> mrope_section_,
                                                    bool is_interleaved,
                                                    double eps,
                                                    const aiter_tensor_t& q_out,
                                                    const aiter_tensor_t& k_cache,
                                                    const aiter_tensor_t& v_cache,
                                                    const aiter_tensor_t& slot_mapping,
                                                    const aiter_tensor_t& per_tensor_k_scale,
                                                    const aiter_tensor_t& per_tensor_v_scale,
                                                    std::optional<aiter_tensor_t> k_out,
                                                    std::optional<aiter_tensor_t> v_out,
                                                    bool return_kv,
                                                    bool use_shuffle_layout,
                                                    int64_t block_size,
                                                    int64_t x,
                                                    int64_t rotary_dim = 0);
