// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/extension.h>

#include <cstddef>
#include <cstdint>

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

void mla_decode_fwd_opus_stage1(
    torch::Tensor& q,  // [B, H, 576]           fp8 (merged nope+rope)
    torch::Tensor& kv, // [total_tokens, 576]   fp8 (merged nope+rope)
    const torch::Tensor& qo_indptr,
    const torch::Tensor& kv_indptr,
    const torch::Tensor& kv_indices,
    const torch::Tensor& kv_last_page_lens,
    const torch::Tensor& work_indptr,
    const torch::Tensor& work_info_set,
    const int max_seqlen_q,
    const int page_size,
    const int nhead_kv,
    const double softmax_scale,
    torch::Tensor& logits,   // aiter split_output [num_partials,1,H,512] fp32
    torch::Tensor& attn_lse, // aiter split_lse    [num_partials,1,H,1]   fp32
    torch::Tensor& out,      // final [B, H, 512] bf16
    torch::Tensor& final_lse,
    torch::Tensor& q_scale,   // float[1] per-tensor descale
    torch::Tensor& kv_scale); // float[1] per-tensor descale
