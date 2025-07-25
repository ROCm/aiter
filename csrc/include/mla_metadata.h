// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

std::vector<torch::Tensor> get_mla_metadata_v0(const torch::Tensor& p_seqlens_k,
                                               const int32_t num_heads_per_head_k,
                                               const int32_t num_heads_k);

std::vector<torch::Tensor>
get_mla_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                    const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                    const int32_t num_heads_per_head_k,
                    const int32_t num_heads_k,
                    const bool is_causal);
