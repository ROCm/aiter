// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "metadata/v1_0_device.cuh"
#include "metadata/v1_1_device.cuh"
#include "metadata/v1_1_host.cuh"
#include "metadata/v1_2_device.cuh"
#include "metadata/v1_2_pa_device.cuh"
#include "metadata/v1_2_host.cuh"

// MLA Metadata V1

// Persistent thread group solution; accounts for variable query/output lengths.
//
// Returns
//   [0] work_metadata_ptrs  (2)                 Two 64-bit pointers to the 1st element of work_indptr and work_info.
//   [1] work_info           (#work, 8)
//   [1.0] bs_index:         (#work),            Index of batch handled by each work.
//   [1.1] partial_index:    (#work),            Index of tile in output buffer when split; -1 means no split.
//   [1.2] q_start:          (#work),            Global seq index where q/o starts (global index avoids extra
//                                               memory access in kernel).
//   [1.3] q_end:            (#work),            Global seq index where q/o ends (not included).
//   [1.4] kv_start:         (#work),            Global seq index where k/v starts.
//   [1.5] kv_end:           (#work),            Global seq index where k/v ends (not included).
//   [1.6] pad               (#work, 2),         Pad to 8 DWs.
//   [2] work_indptr:        (#cu_part + 1),     IDs of work handled by each cu_part.
//   [3] reduce_indptr:      (sum(qo_seqlen_blk_count) + 1),
//                                               IDs in reduce_partial_map indicating tiles to merge together.
//   [4] reduce_final_map:   (sum(qo_seqlen_blk_count)),
//                                               Final output location of each group of tiles.
//   [5] reduce_partial_map: (#partial_tiles),   Locations in partial buffer of partial tiles awaiting reduction.
//
void get_mla_metadata_v1(
    const torch::Tensor&                seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor&                seqlens_kv_indptr,     // [batch size + 1]
    const torch::Tensor&                kv_last_page_lens,     // [batch size]
    const int32_t                       num_heads_per_head_k,
    const int32_t                       num_heads_k,
    const bool                          is_causal,
    torch::Tensor&                      work_metadata_ptrs,
    torch::Tensor&                      work_info_set,
    torch::Tensor&                      work_indptr,
    torch::Tensor&                      reduce_indptr,
    torch::Tensor&                      reduce_final_map,
    torch::Tensor&                      reduce_partial_map,
    const int32_t                       page_size,
    const int32_t                       kv_granularity,
    const int32_t                       max_seqlen_qo,
    const int32_t                       uni_seqlen_qo,
    const bool                          fast_mode,
    const int32_t                       topk,
    const int32_t                       max_split_per_batch,
    const bool                          intra_batch_mode,
    const std::optional<at::ScalarType> dtype_q,
    const std::optional<at::ScalarType> dtype_kv,
    const bool                          is_cp_round_robin)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(seqlens_kv_indptr));

    TORCH_CHECK((kv_granularity & (kv_granularity - 1)) == 0,
                __func__, ": kv_granularity Must be power of 2!");
    TORCH_CHECK((page_size & (page_size - 1)) == 0,
                __func__, ": page_size Must be power of 2!");
    TORCH_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__, ": seqlens_qo_indptr should be continuous!");
    TORCH_CHECK(seqlens_qo_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_qo_indptr's element type should be int!");
    TORCH_CHECK(seqlens_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    TORCH_CHECK(seqlens_kv_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_kv_indptr's element type should be int!");
    TORCH_CHECK(kv_last_page_lens.stride(0) == 1,
                __func__, ": kv_last_page_lens should be continuous!");
    TORCH_CHECK(kv_last_page_lens.scalar_type() == at::ScalarType::Int,
                __func__, ": kv_last_page_lens's element type should be int!");

    at::ScalarType q_dtype = dtype_q.has_value() ? dtype_q.value() : at::ScalarType::BFloat16;
    at::ScalarType kv_dtype = dtype_kv.has_value() ? dtype_kv.value() : at::ScalarType::BFloat16;

    if (fast_mode)
    {
        get_mla_metadata_v1_2_device(
            seqlens_qo_indptr,
            seqlens_kv_indptr,
            kv_last_page_lens,
            num_heads_per_head_k,
            num_heads_k,
            is_causal,
            page_size,
            kv_granularity,
            max_seqlen_qo,
            uni_seqlen_qo,
            topk,
            max_split_per_batch,
            q_dtype,
            kv_dtype,
            is_cp_round_robin,
            work_metadata_ptrs,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map);
    }
    else if (intra_batch_mode)
    {
        get_mla_metadata_v1_0_device(
            seqlens_qo_indptr,
            seqlens_kv_indptr,
            num_heads_per_head_k,
            num_heads_k,
            is_causal,
            kv_granularity,
            max_seqlen_qo,
            uni_seqlen_qo,
            max_split_per_batch,
            q_dtype,
            work_metadata_ptrs,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map);
    }
    else
    {
        get_mla_metadata_v1_1_device(
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

std::vector<torch::Tensor> get_mla_metadata_v1_no_redundant(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const int32_t        kv_granularity)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(seqlens_kv_indptr));

    // Defaults for our ASM MLA decode kernel: num_heads=16, qo size 1-4, no per-workgroup qo split,
    // so kPackedQoLenPerWg = 4*16 = 64 to avoid splitting in any case it supports.
    //                                PackedQoLenPerWg, MaxClusterSize
    using Traits = MlaMetadataV11Traits<64,               1>;

    return get_mla_metadata_v1_1_host<Traits>(
        seqlens_qo_indptr,
        seqlens_kv_indptr,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        kv_granularity,
        true);
}


void get_pa_metadata_v1(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& pages_kv_indptr,       // [batch size + 1]
    const torch::Tensor& context_lens,          // [batch size]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_indptr,
    torch::Tensor&       work_info_set,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map,
    const int32_t        kv_granularity,
    const int32_t        block_size,
    const int32_t        max_seqlen_qo,
    const int32_t        uni_seqlen_qo,
    const bool           fast_mode,
    const int32_t        topk,
    const int32_t        max_split_per_batch)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(pages_kv_indptr));

    TORCH_CHECK((kv_granularity & (kv_granularity - 1)) == 0,
                __func__, ": kv_granularity Must be power of 2!");
    TORCH_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__, ": seqlens_qo_indptr should be continuous!");
    TORCH_CHECK(seqlens_qo_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_qo_indptr's element type should be int!");
    TORCH_CHECK(pages_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    TORCH_CHECK(pages_kv_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_kv_indptr's element type should be int!");

    get_pa_metadata_v1_2_device(
        seqlens_qo_indptr,
        pages_kv_indptr,
        context_lens,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        kv_granularity,
        block_size,
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


void get_ps_metadata_v1(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& pages_kv_indptr,       // [batch size + 1]
    const torch::Tensor& context_lens,          // [batch size]
    const int32_t        gqa_ratio,
    const int32_t        num_heads_k,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_indptr,
    torch::Tensor&       work_info,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map,
    const int32_t        qhead_granularity,
    const int32_t        qlen_granularity,
    const int32_t        kvlen_granlarity,
    const int32_t        block_size,
    const bool           is_causal)
{
    // const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(pages_kv_indptr));

    TORCH_CHECK((kvlen_granlarity & (kvlen_granlarity - 1)) == 0,
                __func__, ": kvlen_granlarity Must be power of 2!");
    TORCH_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__, ": seqlens_qo_indptr should be continuous!");
    TORCH_CHECK(seqlens_qo_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_qo_indptr's element type should be int!");
    TORCH_CHECK(pages_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    TORCH_CHECK(pages_kv_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_kv_indptr's element type should be int!");

    get_ps_metadata_v1_2_host(
        seqlens_qo_indptr,
        pages_kv_indptr,
        context_lens,
        gqa_ratio,
        num_heads_k,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        qhead_granularity,
        qlen_granularity,
        kvlen_granlarity,
        block_size,
        is_causal);

}
