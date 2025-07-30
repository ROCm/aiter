// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

union MlaWorkInfo
{
    struct
    {
        int32_t bs_index;
        int32_t partial_index;
        int32_t q_start;
        int32_t q_end;
        int32_t kv_start;
        int32_t kv_end;
        int32_t kv_offset;
        int32_t padding[1];
    };
    uint32_t u32All[8];
};

union MlaPartialTileInfo
{
    struct
    {
        int32_t q_start;
        int32_t q_end;
    };
    uint32_t u32All[2];
};

std::vector<torch::Tensor> get_mla_metadata_v0(const torch::Tensor& p_seqlens_k, // [batch size + 1]
                                               const int32_t num_heads_per_head_k,
                                               const int32_t num_heads_k);

std::vector<torch::Tensor>
get_mla_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                    const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                    const int32_t num_heads_per_head_k,
                    const int32_t num_heads_k,
                    const bool is_causal);

void mla_reduce_v1(torch::Tensor& final_lse,
                   torch::Tensor& final_output,
                   const torch::Tensor& partial_lse,
                   const torch::Tensor& partial_output,
                   const torch::Tensor& reduce_indptr,
                   const torch::Tensor& reduce_final_map,
                   const torch::Tensor& reduce_partial_map);
