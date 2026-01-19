// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

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
