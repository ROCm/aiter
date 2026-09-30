// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "mla_decode_shape.h"
#include "mla_metadata.h"
#include "metadata/v1_0_device.cuh"
#include "metadata/v1_1_device.cuh"
#include "metadata/v1_2_device.cuh"
#include "metadata/v1_2_pa_device.cuh"
#include "metadata/v1_2_host.cuh"
#include <cstdlib>
#include <stdexcept>

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


std::tuple<std::string,
           bool,
           bool,
           int64_t,
           int64_t,
           bool,
           int64_t,
           int64_t,
           int64_t,
           int64_t,
           bool,
           int64_t>
get_mla_decode_head_plan_v1(const int64_t num_heads_k,
                            const int64_t num_heads_per_head_k,
                            const int64_t max_seqlen_qo,
                            const int64_t uni_seqlen_qo,
                            const int64_t dtype_q_nope,
                            const int64_t dtype_kv_nope,
                            const bool fast_mode,
                            const bool intra_batch_mode,
                            const std::string& arch,
                            const int64_t enable_experimental,
                            const int64_t flydsl_ps1,
                            const int64_t v_head_dim,
                            const int64_t page_size,
                            const int64_t cp_world_size,
                            const bool cp_round_robin,
                            const bool has_scales,
                            const bool use_opus,
                            const bool use_ps1_asm)
{
    if(num_heads_k < 1 || num_heads_per_head_k < 1)
    {
        throw std::invalid_argument("get_mla_decode_head_plan_v1: #heads must be >= 1");
    }
    if(max_seqlen_qo < 1)
    {
        throw std::invalid_argument("get_mla_decode_head_plan_v1: max_seqlen_qo must be >= 1");
    }

    auto env_flag = [](const char* name) {
        return std::getenv(name) != nullptr && std::atoi(std::getenv(name)) != 0;
    };
    const std::string arch_id = arch.empty() ? get_gpu_arch() : arch;
    const MlaPlannerFlags flags{
        enable_experimental < 0 ? env_flag("AITER_ENABLE_EXPERIMENTAL") : enable_experimental != 0,
        flydsl_ps1 < 0 ? env_flag("AITER_MLA_DECODE_PS1_FLYDSL") : flydsl_ps1 != 0};
    const AiterDtype q_dtype  = static_cast<AiterDtype>(dtype_q_nope);
    const AiterDtype kv_dtype = static_cast<AiterDtype>(dtype_kv_nope);

    const MlaDecodeHeadPlan p = mla_decode_head_plan(arch_id,
                                                     static_cast<int32_t>(num_heads_k),
                                                     static_cast<int32_t>(num_heads_per_head_k),
                                                     static_cast<int32_t>(max_seqlen_qo),
                                                     static_cast<int32_t>(uni_seqlen_qo),
                                                     q_dtype,
                                                     kv_dtype,
                                                     fast_mode,
                                                     intra_batch_mode,
                                                     flags);
    const bool reduce_supported =
        mla_reduce_v1_supports(p.kernel_num_heads, static_cast<int32_t>(v_head_dim));
    const MlaBackendId backend =
        mla_decode_backend(arch_id,
                           static_cast<int32_t>(num_heads_k * num_heads_per_head_k),
                           p.kernel_num_heads,
                           static_cast<int32_t>(max_seqlen_qo),
                           q_dtype,
                           kv_dtype,
                           static_cast<int32_t>(page_size),
                           static_cast<int32_t>(cp_world_size),
                           cp_round_robin,
                           intra_batch_mode,
                           has_scales,
                           use_opus,
                           use_ps1_asm,
                           flags);

    return {arch_id,
            flags.enable_experimental,
            flags.flydsl_ps1,
            static_cast<int64_t>(p.planner),
            static_cast<int64_t>(p.plan),
            p.natively_supported,
            p.qk_batch_ratio,
            p.kernel_num_heads,
            p.seqlen_fold,
            p.packed_qo_len_per_wg,
            reduce_supported,
            static_cast<int64_t>(backend)};
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
    const bool            is_causal,
    const bool            need_lse)
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
        is_causal,
        need_lse);

}
