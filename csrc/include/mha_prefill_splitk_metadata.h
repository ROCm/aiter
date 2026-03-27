// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

union MhaPrefillSplitKWorkInfo
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
constexpr size_t kSizeMhaPrefillSplitKWorkInfoInDw =
    sizeof(MhaPrefillSplitKWorkInfo) / sizeof(uint32_t);
static_assert(kSizeMhaPrefillSplitKWorkInfoInDw == 8);

union MhaPrefillSplitKPartialTileInfo
{
    struct
    {
        int32_t q_start;
        int32_t q_end;
    };
    uint32_t u32All[2];
};
constexpr size_t kSizeMhaPrefillSplitKPartialTileInfoInDw =
    sizeof(MhaPrefillSplitKPartialTileInfo) / sizeof(uint32_t);
static_assert(kSizeMhaPrefillSplitKPartialTileInfoInDw == 2);

void get_mha_prefill_splitk_metadata_v1(
    const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
    const bool is_causal,
    const int32_t q_tile_size,
    const int32_t split_k_size,
    torch::Tensor& work_metadata_ptrs,
    torch::Tensor& work_info_set,
    torch::Tensor& work_indptr,
    torch::Tensor& reduce_indptr,
    torch::Tensor& reduce_final_map,
    torch::Tensor& reduce_partial_map,
    const int32_t max_seqlen_qo,
    const int32_t uni_seqlen_qo);
