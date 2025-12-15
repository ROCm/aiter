// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

union PaWorkInfo
{
    struct
    {
        int32_t batch_idx;
        int32_t partial_qo_loc;
        int32_t qo_start;      // global qo len idx
        int32_t qo_end;        // global qo len idx
        int32_t kv_start;      // global kv block idx
        int32_t kv_end;        // global kv block idx
        int32_t kv_offset;     // not used
        int32_t q_head_range;  // q_head_start(low 16bits) | q_head_end(high 16bits)
                               // for asm kernel & future divide qheads
    };
    uint32_t u32All[8];
};
constexpr size_t kSizePaWorkInfoInDw = sizeof(PaWorkInfo) / sizeof(uint32_t);
static_assert(kSizePaWorkInfoInDw == 8);


union PaPartialTileInfo
{
    struct
    {
        int32_t q_start;
        int32_t q_end;
    };
    uint32_t u32All[2];
};
constexpr size_t kSizePaPartialTileInfoInDw = sizeof(PaPartialTileInfo) / sizeof(uint32_t);
static_assert(kSizePaPartialTileInfoInDw == 2);

void get_pa_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                        const torch::Tensor& pages_kv_indptr, // [batch size + 1]
                        const torch::Tensor& context_lens,
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
                        const int32_t block_size,
                        const int32_t max_seqlen_qo,
                        const int32_t uni_seqlen_qo,
                        const bool    fast_mode,
                        const int32_t topk,
                        const int32_t max_split_per_batch);
