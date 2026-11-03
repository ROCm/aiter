// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

union PaWorkInfo
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
        int32_t q_head_range;
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

void get_pa_metadata_v1(const aiter_tensor_t& seqlens_qo_indptr, // [batch size + 1]
                        const aiter_tensor_t& pages_kv_indptr, // [batch size + 1]
                        const aiter_tensor_t& context_lens,
                        const int32_t num_heads_per_head_k,
                        const int32_t num_heads_k,
                        const bool is_causal,
                        aiter_tensor_t& work_metadata_ptrs,
                        aiter_tensor_t& work_indptr,
                        aiter_tensor_t& work_info,
                        aiter_tensor_t& reduce_indptr,
                        aiter_tensor_t& reduce_final_map,
                        aiter_tensor_t& reduce_partial_map,
                        const int32_t kv_granularity,
                        const int32_t block_size,
                        const int32_t max_seqlen_qo,
                        const int32_t uni_seqlen_qo,
                        const bool    fast_mode,
                        const int32_t topk,
                        const int32_t max_split_per_batch);
