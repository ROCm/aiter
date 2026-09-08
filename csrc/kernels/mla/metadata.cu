// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "mla_metadata.h"
#include "metadata/v1_0_device.cuh"
#include "metadata/v1_1_device.cuh"
#include "metadata/v1_2_device.cuh"
#include "metadata/v1_2_pa_device.cuh"
#include "metadata/v1_2_host.cuh"

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
    const aiter_tensor_t&              seqlens_qo_indptr,     // [batch size + 1]
    const aiter_tensor_t&              seqlens_kv_indptr,     // [batch size + 1]
    const aiter_tensor_t&              kv_last_page_lens,     // [batch size]
    const int32_t                     num_heads_per_head_k,
    const int32_t                     num_heads_k,
    const bool                        is_causal,
    aiter_tensor_t&                   work_metadata_ptrs,
    aiter_tensor_t&                   work_info_set,
    aiter_tensor_t&                   work_indptr,
    aiter_tensor_t&                   reduce_indptr,
    aiter_tensor_t&                   reduce_final_map,
    aiter_tensor_t&                   reduce_partial_map,
    const int32_t                     page_size,
    const int32_t                     kv_granularity,
    const int32_t                     max_seqlen_qo,
    const int32_t                     uni_seqlen_qo,
    const bool                        fast_mode,
    const int32_t                     topk,
    const int32_t                     max_split_per_batch,
    const bool                        intra_batch_mode,
    const bool                        is_cp_round_robin,
    const int64_t                     mla_version,
    const std::optional<int64_t>      dtype_q_nope,
    const std::optional<int64_t>      dtype_q_rope,
    const std::optional<int64_t>      dtype_kv_nope,
    const std::optional<int64_t>      dtype_kv_rope)
{
    const HipDeviceGuard device_guard(seqlens_kv_indptr.device_id);

    AITER_CHECK((kv_granularity & (kv_granularity - 1)) == 0,
                __func__, ": kv_granularity Must be power of 2!");
    AITER_CHECK((page_size & (page_size - 1)) == 0,
                __func__, ": page_size Must be power of 2!");
    AITER_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__, ": seqlens_qo_indptr should be continuous!");
    AITER_CHECK(seqlens_qo_indptr.dtype() == AITER_DTYPE_i32,
                __func__, ": seqlens_qo_indptr's element type should be int!");
    AITER_CHECK(seqlens_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    AITER_CHECK(seqlens_kv_indptr.dtype() == AITER_DTYPE_i32,
                __func__, ": seqlens_kv_indptr's element type should be int!");
    AITER_CHECK(kv_last_page_lens.stride(0) == 1,
                __func__, ": kv_last_page_lens should be continuous!");
    AITER_CHECK(kv_last_page_lens.dtype() == AITER_DTYPE_i32,
                __func__, ": kv_last_page_lens's element type should be int!");

    const MlaVersion mla_version_enum = static_cast<MlaVersion>(mla_version);

    AiterDtype q_nope_dtype =
        dtype_q_nope.has_value() ? static_cast<AiterDtype>(dtype_q_nope.value()) : AITER_DTYPE_bf16;
    AiterDtype kv_nope_dtype =
        dtype_kv_nope.has_value() ? static_cast<AiterDtype>(dtype_kv_nope.value()) : AITER_DTYPE_bf16;
    // rope dtypes default to their corresponding nope dtype when unspecified.
    AiterDtype q_rope_dtype =
        dtype_q_rope.has_value() ? static_cast<AiterDtype>(dtype_q_rope.value()) : q_nope_dtype;
    AiterDtype kv_rope_dtype =
        dtype_kv_rope.has_value() ? static_cast<AiterDtype>(dtype_kv_rope.value()) : kv_nope_dtype;

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
            q_nope_dtype,
            kv_nope_dtype,
            q_rope_dtype,
            kv_rope_dtype,
            is_cp_round_robin,
            mla_version_enum,
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
            q_nope_dtype,
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


void get_pa_metadata_v1(
    const aiter_tensor_t& seqlens_qo_indptr,     // [batch size + 1]
    const aiter_tensor_t& pages_kv_indptr,       // [batch size + 1]
    const aiter_tensor_t& context_lens,          // [batch size]
    const int32_t         num_heads_per_head_k,
    const int32_t         num_heads_k,
    const bool            is_causal,
    aiter_tensor_t&       work_metadata_ptrs,
    aiter_tensor_t&       work_indptr,
    aiter_tensor_t&       work_info_set,
    aiter_tensor_t&       reduce_indptr,
    aiter_tensor_t&       reduce_final_map,
    aiter_tensor_t&       reduce_partial_map,
    const int32_t         kv_granularity,
    const int32_t         block_size,
    const int32_t         max_seqlen_qo,
    const int32_t         uni_seqlen_qo,
    const bool            fast_mode,
    const int32_t         topk,
    const int32_t         max_split_per_batch)
{
    const HipDeviceGuard device_guard(pages_kv_indptr.device_id);

    AITER_CHECK((kv_granularity & (kv_granularity - 1)) == 0,
                __func__, ": kv_granularity Must be power of 2!");
    AITER_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__, ": seqlens_qo_indptr should be continuous!");
    AITER_CHECK(seqlens_qo_indptr.dtype() == AITER_DTYPE_i32,
                __func__, ": seqlens_qo_indptr's element type should be int!");
    AITER_CHECK(pages_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    AITER_CHECK(pages_kv_indptr.dtype() == AITER_DTYPE_i32,
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
    const aiter_tensor_t& seqlens_qo_indptr,     // [batch size + 1]
    const aiter_tensor_t& pages_kv_indptr,       // [batch size + 1]
    const aiter_tensor_t& context_lens,          // [batch size]
    const int32_t         gqa_ratio,
    const int32_t         num_heads_k,
    aiter_tensor_t&       work_metadata_ptrs,
    aiter_tensor_t&       work_indptr,
    aiter_tensor_t&       work_info,
    aiter_tensor_t&       reduce_indptr,
    aiter_tensor_t&       reduce_final_map,
    aiter_tensor_t&       reduce_partial_map,
    const int32_t         qhead_granularity,
    const int32_t         qlen_granularity,
    const int32_t         kvlen_granlarity,
    const int32_t         block_size,
    const bool            is_causal)
{
    // const HipDeviceGuard device_guard(pages_kv_indptr.device_id);

    AITER_CHECK((kvlen_granlarity & (kvlen_granlarity - 1)) == 0,
                __func__, ": kvlen_granlarity Must be power of 2!");
    AITER_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__, ": seqlens_qo_indptr should be continuous!");
    AITER_CHECK(seqlens_qo_indptr.dtype() == AITER_DTYPE_i32,
                __func__, ": seqlens_qo_indptr's element type should be int!");
    AITER_CHECK(pages_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    AITER_CHECK(pages_kv_indptr.dtype() == AITER_DTYPE_i32,
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
