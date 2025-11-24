// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "pa.h"
#include "mla/metadata/v1_2_device.cuh"
#include "mla/metadata/v1_1_device.cuh"`

void get_pa_metadata_v1_2_device(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
    const int32_t num_heads_per_head_k,
    const int32_t num_heads_k,
    const bool is_causal,
    const int32_t kv_granularity,
    const int32_t max_seqlen_qo,
    const int32_t ori_uni_seqlen_qo,
    const int32_t topk,
    const int32_t max_split_per_batch,
    torch::Tensor& work_metadata_ptrs,
    torch::Tensor& work_info_set,
    torch::Tensor& work_indptr,
    torch::Tensor& reduce_indptr,
    torch::Tensor& reduce_final_map,
    torch::Tensor& reduce_partial_map)
{
    get_mla_metadata_v1_2_device(
        seqlens_qo_indptr,
        seqlens_kv_indptr,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        kv_granularity,
        max_seqlen_qo,
        ori_uni_seqlen_qo,
        topk,
        max_split_per_batch,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map);
}

void get_pa_metadata_v1_1_device(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const bool           no_redundant,
    const int32_t        kv_granularity,
    const int32_t        max_seqlen_qo,
    const int32_t        ori_uni_seqlen_qo,
    const int32_t        topk,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_info_set,
    torch::Tensor&       work_indptr,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map)
{
    get_mla_metadata_v1_1_device(
        seqlens_qo_indptr,
        seqlens_kv_indptr,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        no_redundant,
        kv_granularity,
        max_seqlen_qo,
        ori_uni_seqlen_qo,
        topk,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map);
}



void get_pa_metadata_v1(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_info_set,
    torch::Tensor&       work_indptr,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map,
    const int32_t        kv_granularity,
    const int32_t        max_seqlen_qo,
    const int32_t        uni_seqlen_qo,
    const bool           fast_mode,
    const int32_t        topk,
    const int32_t        max_split_per_batch)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(seqlens_kv_indptr));

    TORCH_CHECK((kv_granularity & (kv_granularity - 1)) == 0,
                __func__, ": kv_granularity Must be power of 2!");
    TORCH_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__, ": seqlens_qo_indptr should be continuous!");
    TORCH_CHECK(seqlens_qo_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_qo_indptr's element type should be int!");
    TORCH_CHECK(seqlens_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    TORCH_CHECK(seqlens_kv_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_kv_indptr's element type should be int!");

    if (fast_mode)
    {
        get_pa_metadata_v1_2_device(
            seqlens_qo_indptr,
            seqlens_kv_indptr,
            num_heads_per_head_k,
            num_heads_k,
            is_causal,
            kv_granularity,
            max_seqlen_qo,
            uni_seqlen_qo,
            topk,
            max_split_per_batch,
            work_metadata_ptrs,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map);
    }
    else
    {
        get_pa_metadata_v1_1_device(
            seqlens_qo_indptr,
            seqlens_kv_indptr,
            num_heads_per_head_k,
            num_heads_k,
            is_causal,
            false,
            kv_granularity,
            max_seqlen_qo,
            uni_seqlen_qo,
            topk,
            work_metadata_ptrs,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map);
    }

}