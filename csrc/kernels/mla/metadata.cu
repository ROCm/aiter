// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include "metadata/v1_0_device.cuh"
#include "metadata/v1_1_device.cuh"
#include "metadata/v1_1_host.cuh"
#include "metadata/v1_2_device.cuh"
#include "metadata/v1_2_pa_device.cuh"

// ===================================================================================================================
// MLA Metadata V1
// ===================================================================================================================

//
// Persistent thread group solution which take variable query/output lengths into consideration as well.
//
// Returns
//   [0] work_metadata_ptrs  (2)                 Two 64-bits pointers point to the 1st element of work_indptr and
//                                               work_info.
//   [1] work_info           (#work, 8)
//   [1.0] bs_index:         (#work),            The index of batch handled by each work.
//   [1.1] partial_index:    (#work),            The index of tile in output buffer when splits. -1 means no split.
//   [1.2] q_start:          (#work),            The global index in seq where q/o starts. Use global index here can
//                                               reduce memory access count in kernel.
//   [1.3] q_end:            (#work),            The global index in seq where q/o ends (not included).
//   [1.4] kv_start:         (#work),            The global index in seq where k/v starts.
//   [1.5] kv_end:           (#work),            The global index in seq where k/v ends (not included).
//   [1.6] pad               (#work, 2),         Pad to 8 DWs.
//   [2] work_indptr:        (#cu_part + 1),     The IDs of work handled by each cu_part.
//   [3] reduce_indptr:      (sum(qo_seqlen_blk_count) + 1),
//                                               The IDs in reduce_partial_map indicates the tiles should be merged
//                                               together.
//   [4] reduce_final_map:   (sum(qo_seqlen_blk_count)),
//                                               The final output location of each group of tiles.
//   [5] reduce_partial_map: (#partial_tiles),   The locations in partial buffer of partial tiles waiting for being
//                                               reduced.
//
void get_mla_metadata_v1(
    const torch::Tensor&                seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor&                seqlens_kv_indptr,     // [batch size + 1]
    const int32_t                       num_heads_per_head_k,
    const int32_t                       num_heads_k,
    const bool                          is_causal,
    torch::Tensor&                      work_metadata_ptrs,
    torch::Tensor&                      work_info_set,
    torch::Tensor&                      work_indptr,
    torch::Tensor&                      reduce_indptr,
    torch::Tensor&                      reduce_final_map,
    torch::Tensor&                      reduce_partial_map,
    const int32_t                       kv_granularity,
    const int32_t                       max_seqlen_qo,
    const int32_t                       uni_seqlen_qo,
    const bool                          fast_mode,
    const int32_t                       topk,
    const int32_t                       max_split_per_batch,
    const bool                          intra_batch_mode,
    const std::optional<at::ScalarType> dtype_q,
    const std::optional<at::ScalarType> dtype_kv)
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

    at::ScalarType q_dtype = dtype_q.has_value() ? dtype_q.value() : at::ScalarType::BFloat16;
    at::ScalarType kv_dtype = dtype_kv.has_value() ? dtype_kv.value() : at::ScalarType::BFloat16;

    if (fast_mode)
    {
        get_mla_metadata_v1_2_device(
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
            q_dtype,
            kv_dtype,
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

    // This default settings is for our ASM MLA decode kernel. This kernel supports num_heads=16 and qo size from 1 to 4
    // without support to split qo for each workgroup. This means that kPackedQoLenPerWg should be 4*16=64 to prevent
    // spliting in any case supported by it.
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
