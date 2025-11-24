// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

void get_pa_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                        const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                        const int32_t num_heads_per_head_k,
                        const int32_t num_heads_k,
                        const bool is_causal,
                        torch::Tensor& work_metadata_ptrs,
                        torch::Tensor& work_indptr,
                        torch::Tensor& work_info,
                        torch::Tensor& reduce_indptr,
                        torch::Tensor& reduce_final_map,
                        torch::Tensor& reduce_partial_map,
                        const int32_t kv_granularity,
                        const int32_t max_seqlen_qo,
                        const int32_t uni_seqlen_qo,
                        const bool fast_mode,
                        const int32_t topk,
                        const int32_t max_split_per_batch);
