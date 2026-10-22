// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include "aiter_tensor.h"
#include <optional>

void fused_norm_rope_attn_rope_cast_fwd(const aiter_tensor_t& q,
                                        const aiter_tensor_t& pool,
                                        const aiter_tensor_t& prefix,
                                        const aiter_tensor_t& prefix_indptr,
                                        const aiter_tensor_t& cos,
                                        const aiter_tensor_t& sin,
                                        const aiter_tensor_t& positions,
                                        const aiter_tensor_t& out,
                                        const aiter_tensor_t& out_scale,
                                        const aiter_tensor_t& max_logits,
                                        const aiter_tensor_t& lse,
                                        const std::optional<aiter_tensor_t>& extend_kv,
                                        const std::optional<aiter_tensor_t>& extend,
                                        const std::optional<aiter_tensor_t>& extend_indptr,
                                        const std::optional<aiter_tensor_t>& attn_sink,
                                        const std::optional<aiter_tensor_t>& debug_out,
                                        float softmax_scale,
                                        bool q_is_rotated,
                                        const std::optional<aiter_tensor_t>& cache_scales,
                                        int64_t cache_page_size,
                                        int64_t cache_num_rows,
                                        int64_t cache_data_page_stride,
                                        int64_t cache_scale_page_stride,
                                        bool fp8_qk,
                                        const std::optional<aiter_tensor_t>& q_scale,
                                        bool q_norm,
                                        float norm_eps);
