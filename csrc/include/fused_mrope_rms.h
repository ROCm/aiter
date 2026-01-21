// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>
#include <optional>

using namespace at;

void fused_mrope_3d_rms(Tensor& qkv,
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
                        double eps);

void fused_mrope_3d_rms_set_kv(Tensor& qkv,
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
                               double k_scale,
                               double v_scale,
                               std::optional<Tensor> k_out,
                               std::optional<Tensor> v_out,
                               bool return_kv,
                               bool use_shuffle_layout,
                               int64_t block_size,
                               int64_t x);

void fused_rope_rms(Tensor& qkv,
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
                    double eps);

void fused_rope_rms_set_kv(Tensor& qkv,
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
                           double eps,
                           Tensor& q,
                           Tensor& k_cache,
                           Tensor& v_cache,
                           Tensor& kv_loc,
                           double k_scale,
                           double v_scale,
                           std::optional<Tensor> k_out,
                           std::optional<Tensor> v_out,
                           bool return_kv,
                           bool use_shuffle_layout,
                           int64_t block_size,
                           int64_t x);

void fused_rope_rms_2way(
    Tensor &q0, Tensor &k0, Tensor &q1, Tensor &k1,
    Tensor &w_q0, Tensor &w_k0, Tensor &w_q1, Tensor &w_k1,
    Tensor &cos_sin0, Tensor &cos_sin1,
    int64_t batch_size, int64_t num_tokens0, int64_t num_tokens1,
    int64_t num_heads_q, int64_t num_heads_k,
    int64_t head_size, bool is_interleaved, double eps,
    Tensor &out_q01, Tensor &out_k01);
