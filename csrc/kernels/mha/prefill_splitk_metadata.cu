// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <algorithm>
#include "mha_prefill_splitk_metadata.h"

void get_mha_prefill_splitk_metadata_v1(
    const torch::Tensor& seqlens_qo_indptr,
    const torch::Tensor& seqlens_kv_indptr,
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
    const int32_t uni_seqlen_qo)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(seqlens_kv_indptr));

    TORCH_CHECK(q_tile_size > 0, __func__, ": q_tile_size must be > 0");
    TORCH_CHECK(split_k_size > 0, __func__, ": split_k_size must be > 0");
    TORCH_CHECK(max_seqlen_qo > 0 || uni_seqlen_qo > 0,
                __func__,
                ": max_seqlen_qo or uni_seqlen_qo must be > 0");
    TORCH_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__,
                ": seqlens_qo_indptr should be continuous!");
    TORCH_CHECK(seqlens_qo_indptr.scalar_type() == at::ScalarType::Int,
                __func__,
                ": seqlens_qo_indptr's element type should be int!");
    TORCH_CHECK(seqlens_kv_indptr.stride(0) == 1,
                __func__,
                ": seqlens_kv_indptr should be continuous!");
    TORCH_CHECK(seqlens_kv_indptr.scalar_type() == at::ScalarType::Int,
                __func__,
                ": seqlens_kv_indptr's element type should be int!");
    TORCH_CHECK(work_indptr.numel() >= 2,
                __func__,
                ": work_indptr must have at least 2 elements in simplified mode");

    auto seqlens_qo_cpu = seqlens_qo_indptr.to(torch::kCPU);
    auto seqlens_kv_cpu = seqlens_kv_indptr.to(torch::kCPU);

    const int32_t* p_seqlens_qo = seqlens_qo_cpu.data_ptr<int32_t>();
    const int32_t* p_seqlens_kv = seqlens_kv_cpu.data_ptr<int32_t>();
    const int32_t num_batches = static_cast<int32_t>(seqlens_qo_cpu.size(0)) - 1;

    auto work_metadata_ptrs_cpu =
        torch::zeros(work_metadata_ptrs.sizes(), work_metadata_ptrs.options().device(torch::kCPU));
    auto work_info_set_cpu =
        torch::zeros(work_info_set.sizes(), work_info_set.options().device(torch::kCPU));
    auto work_indptr_cpu =
        torch::zeros(work_indptr.sizes(), work_indptr.options().device(torch::kCPU));
    auto reduce_indptr_cpu =
        torch::zeros(reduce_indptr.sizes(), reduce_indptr.options().device(torch::kCPU));
    auto reduce_final_map_cpu =
        torch::zeros(reduce_final_map.sizes(), reduce_final_map.options().device(torch::kCPU));
    auto reduce_partial_map_cpu =
        torch::zeros(reduce_partial_map.sizes(), reduce_partial_map.options().device(torch::kCPU));

    auto* p_work_metadata_ptrs = work_metadata_ptrs_cpu.data_ptr<uint64_t>();
    auto* p_work_info_set =
        reinterpret_cast<MhaPrefillSplitKWorkInfo*>(work_info_set_cpu.data_ptr<int32_t>());
    auto* p_work_indptr = work_indptr_cpu.data_ptr<int32_t>();
    auto* p_reduce_indptr = reduce_indptr_cpu.data_ptr<int32_t>();
    auto* p_reduce_final_map = reduce_final_map_cpu.data_ptr<int32_t>();
    auto* p_reduce_partial_map = reduce_partial_map_cpu.data_ptr<int32_t>();

    const int64_t work_capacity = work_info_set.size(0);
    const int64_t reduce_group_capacity = reduce_final_map.size(0);
    const int64_t reduce_partial_capacity = reduce_partial_map.numel();

    int32_t work_idx = 0;
    int32_t reduce_group_idx = 0;
    int32_t partial_qo_loc = 0;
    p_reduce_indptr[0] = 0;

    for(int32_t bid = 0; bid < num_batches; ++bid)
    {
        const int32_t qo_begin = p_seqlens_qo[bid];
        const int32_t qo_end = p_seqlens_qo[bid + 1];
        const int32_t kv_begin = p_seqlens_kv[bid];
        const int32_t kv_end_total = p_seqlens_kv[bid + 1];

        for(int32_t qo_start = qo_begin; qo_start < qo_end; qo_start += q_tile_size)
        {
            const int32_t qo_end_tile = std::min(qo_start + q_tile_size, qo_end);
            const int32_t local_q_end = qo_end_tile - qo_begin;
            const int32_t allowed_kv_end =
                is_causal ? std::min(kv_begin + local_q_end, kv_end_total) : kv_end_total;

            TORCH_CHECK(reduce_group_idx < reduce_group_capacity,
                        __func__,
                        ": reduce_final_map is too small for requested metadata");

            for(int32_t kv_start = kv_begin; kv_start < allowed_kv_end; kv_start += split_k_size)
            {
                TORCH_CHECK(work_idx < work_capacity,
                            __func__,
                            ": work_info_set is too small for requested metadata");
                TORCH_CHECK(work_idx < reduce_partial_capacity,
                            __func__,
                            ": reduce_partial_map is too small for requested metadata");

                const int32_t kv_end = std::min(kv_start + split_k_size, allowed_kv_end);

                MhaPrefillSplitKWorkInfo work_info{};
                work_info.batch_idx = bid;
                work_info.partial_qo_loc = partial_qo_loc;
                work_info.qo_start = qo_start;
                work_info.qo_end = qo_end_tile;
                work_info.kv_start = kv_start;
                work_info.kv_end = kv_end;
                work_info.kv_offset = 0;
                p_work_info_set[work_idx] = work_info;
                p_reduce_partial_map[work_idx] = partial_qo_loc;
                ++work_idx;
            }

            p_reduce_indptr[reduce_group_idx + 1] = work_idx;
            p_reduce_final_map[reduce_group_idx * 2] = qo_start;
            p_reduce_final_map[reduce_group_idx * 2 + 1] = qo_end_tile;

            partial_qo_loc += (qo_end_tile - qo_start);
            ++reduce_group_idx;
        }
    }

    for(int64_t idx = 1; idx < work_indptr.numel(); ++idx)
    {
        p_work_indptr[idx] = work_idx;
    }
    for(int64_t idx = reduce_group_idx + 1; idx < reduce_indptr.numel(); ++idx)
    {
        p_reduce_indptr[idx] = work_idx;
    }

    p_work_metadata_ptrs[0] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(work_indptr.data_ptr<int32_t>()));
    p_work_metadata_ptrs[1] =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(work_info_set.data_ptr<int32_t>()));

    work_metadata_ptrs.copy_(work_metadata_ptrs_cpu);
    work_info_set.copy_(work_info_set_cpu);
    work_indptr.copy_(work_indptr_cpu);
    reduce_indptr.copy_(reduce_indptr_cpu);
    reduce_final_map.copy_(reduce_final_map_cpu);
    reduce_partial_map.copy_(reduce_partial_map_cpu);
}
