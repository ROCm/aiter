// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#define PRINT_DBG 0

#include <torch/extension.h>

template <typename T>
inline T pack_dword(const T low_part, const T high_part)
{
    T dw = (high_part << 16) | (low_part & 0xFFFF);
    return dw;
}

template <typename T>
inline std::tuple<T, T> unpack_dword(const T dw)
{
    T high_part = (dw >> 16) & 0xFFFF;
    T low_part  = dw & 0xFFFF;
    return std::make_tuple(low_part, high_part);
}

union WorkInfo
{
    struct
    {
        int32_t batch_idx;
        int32_t partial_o_loc;
        int32_t qo_start;
        int32_t qo_end;
        int32_t kv_start;
        int32_t kv_end;
        int32_t kv_offset;
        int32_t q_head_range;
    };
    uint32_t u32All[8];
};
constexpr size_t kSizeWorkInfoInDw = sizeof(WorkInfo) / sizeof(uint32_t);
static_assert(kSizeWorkInfoInDw == 8);

union FinalLoc
{
    struct
    {
        int32_t qo_start;
        int32_t qo_end;
    };
    uint32_t u32All[2];
};
constexpr size_t kSizeFinalLocInDw = sizeof(FinalLoc) / sizeof(uint32_t);
static_assert(kSizeFinalLocInDw == 2);

struct QTile
{
    int32_t batch_idx;
    int32_t qo_start; // global
    int32_t qo_end;   // global
    int32_t num_blocks;
    int32_t effective_kv_length;
};

struct PsMetadataV1KernelParameter
{
    // Outputs
    int32_t* p_work_indptr;
    WorkInfo* p_work_info;
    int32_t* p_reduce_indptr;
    FinalLoc* p_reduce_final_map;
    int32_t* p_reduce_partial_map;

    // Inputs
    const int32_t* p_seqlens_qo_indptr;
    const int32_t* p_pages_kv_indptr;
    const int32_t* p_context_lens;
    int32_t batch_size;
    int32_t gqa_ratio;
    int32_t num_heads_k;
    int32_t qhead_granularity;
    int32_t qlen_granularity;
    int32_t kvlen_granularity;
    int32_t block_size;
    int32_t available_tgs;
    int32_t num_clusters;
    int32_t tgs_per_cluster;
    int32_t kheads_per_cluster;
    bool is_causal;
};

void get_ps_metadata_v1(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                        const torch::Tensor& pages_kv_indptr,   // [batch size + 1]
                        const torch::Tensor& context_lens,      // [batch size]
                        const int32_t gqa_ratio,
                        const int32_t num_heads_k,
                        torch::Tensor& work_indptr,
                        torch::Tensor& work_info,
                        torch::Tensor& reduce_indptr,
                        torch::Tensor& reduce_final_map,
                        torch::Tensor& reduce_partial_map,
                        const int32_t qhead_granularity,
                        const int32_t qlen_granularity,
                        const int32_t kvlen_granlarity,
                        const int32_t block_size,
                        const bool is_causal);
