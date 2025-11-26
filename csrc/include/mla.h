// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

union MlaWorkInfo
{
    struct
    {
        int32_t batch_idx;
        int32_t partial_qo_loc;
        int32_t qo_start;
        int32_t qo_end;
        int32_t kv_start;
        int32_t kv_end;
        int32_t kv_offset;
        int32_t padding[1];
    };
    uint32_t u32All[8];
};
constexpr size_t kSizeMlaWorkInfoInDw = sizeof(MlaWorkInfo) / sizeof(uint32_t);
static_assert(kSizeMlaWorkInfoInDw == 8);

union MlaPartialTileInfo
{
    struct
    {
        int32_t q_start;
        int32_t q_end;
    };
    uint32_t u32All[2];
};
constexpr size_t kSizeMlaPartialTileInfoInDw = sizeof(MlaPartialTileInfo) / sizeof(uint32_t);
static_assert(kSizeMlaPartialTileInfoInDw == 2);

void get_mla_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
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
                         const int32_t max_split_per_batch,
                         const std::optional<at::ScalarType> dtype_q,
                         const std::optional<at::ScalarType> dtype_kv);

std::vector<torch::Tensor>
get_mla_metadata_v1_no_redundant(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                                 const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                                 const int32_t num_heads_per_head_k,
                                 const int32_t num_heads_k,
                                 const bool is_causal,
                                 const int32_t kv_granularity);

void mla_reduce_v1(const torch::Tensor& partial_output,
                   const torch::Tensor& partial_lse,
                   const torch::Tensor& reduce_indptr,
                   const std::optional<torch::Tensor>& reduce_final_map,
                   const torch::Tensor& reduce_partial_map,
                   torch::Tensor& final_output,
                   std::optional<torch::Tensor>& final_lse);
