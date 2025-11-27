// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "v1_comm.cuh"

__launch_bounds__(ck_tile::get_warp_size(), 1) __global__
void kn_get_mla_metadata_v1_0(MlaMetadataV1KernelParameter params)
{
    const int32_t lane_idx = ck_tile::get_lane_id();

    MlaWorkInfo* p_work_info_set = reinterpret_cast<MlaWorkInfo*>(params.p_work_info_set_raw);

    if(lane_idx == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_indptr[0]   = 0;
        params.p_work_metadata_ptrs[0] =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
        params.p_work_metadata_ptrs[1] =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p_work_info_set));
    }
    extern __shared__ uint8_t p_smem[];
    int32_t* p_lds_shift = reinterpret_cast<int32_t*>(p_smem);
    int32_t* p_lds_split = p_lds_shift + params.num_batches;
    int32_t* p_lds_payload = p_lds_split  + params.num_batches;
    int32_t* p_lds_kv_seqlen = p_lds_payload + params.num_batches;

    for(int32_t bid = lane_idx; bid < params.num_batches; bid += ck_tile::get_warp_size())
    {
        const int32_t kv_begin  = params.p_seqlens_kv_indptr[bid];
        const int32_t kv_end    = params.p_seqlens_kv_indptr[bid + 1];
        const int32_t seqlen_kv = kv_end - kv_begin;

        const int32_t num_blocks = integer_divide_ceil_power2(
            seqlen_kv, params.kv_granularity, params.kv_granularity_log2);
        const int32_t payload = ck_tile::integer_divide_ceil(num_blocks, params.num_splits);
        const int32_t split_local = ck_tile::integer_divide_ceil(num_blocks, payload);
        p_lds_split[bid] = split_local;
        p_lds_payload[bid] = payload;
        p_lds_kv_seqlen[bid + 1] = kv_end;
    }

    __syncthreads();
    if (lane_idx == 0)
    {
        p_lds_shift[0] = 0;
        p_lds_kv_seqlen[0] = 0;
        for (int32_t bid = 1; bid < params.num_batches; bid++)
        {
            p_lds_shift[bid] = p_lds_shift[bid - 1] + p_lds_split[bid - 1];
        }
    }
    __syncthreads();

    for(int32_t bid = lane_idx; bid < params.num_batches; bid += ck_tile::get_warp_size())
    {
        const int32_t kv_begin  = p_lds_kv_seqlen[bid];
        const int32_t kv_end    = p_lds_kv_seqlen[bid + 1];
        MlaWorkInfo work_info{};
        const int32_t split_start = p_lds_shift[bid];
        const int32_t split_local = p_lds_split[bid];
        const int32_t payload = p_lds_payload[bid];

        for(int32_t sid = 0; sid < split_local; sid++)
        {
            const int32_t work_index = split_start + sid;

            work_info.batch_idx = bid;
            work_info.partial_qo_loc = split_local == 1 ? -1 : work_index * params.uni_seqlen_qo;
            work_info.qo_start  = bid * params.uni_seqlen_qo;
            work_info.qo_end    = work_info.qo_start + params.uni_seqlen_qo;
            work_info.kv_start  = kv_begin + (sid * payload * params.kv_granularity);
            work_info.kv_end    = ck_tile::min(work_info.kv_start + payload * params.kv_granularity, kv_end);
            work_info.kv_offset = kv_end - work_info.kv_end;
            p_work_info_set[work_index] = work_info;
            params.p_reduce_partial_map[work_index] = work_info.partial_qo_loc;
        }

        params.p_reduce_indptr[bid + 1] = split_start + split_local;
        params.p_reduce_final_map[bid * 2]     = work_info.qo_start;
        params.p_reduce_final_map[bid * 2 + 1] = work_info.qo_end;
    }



    int32_t work_end = p_lds_shift[params.num_batches - 1] + p_lds_split[params.num_batches - 1];
    int32_t work_per_cu = (work_end + params.num_cu - 1) / params.num_cu;
    int32_t used_cu = (work_end + work_per_cu - 1) / work_per_cu;

    int32_t reduce_end = params.p_reduce_indptr[params.num_batches];
    for (int32_t work_id = lane_idx + 1; work_id < used_cu; work_id += ck_tile::get_warp_size())
    {
        params.p_work_indptr[work_id] = min(work_id * work_per_cu, work_end);
    }

    for (int32_t work_id = used_cu + lane_idx; work_id < params.num_cu + 1; work_id += ck_tile::get_warp_size())
    {
        params.p_work_indptr[work_id] = work_end;
    }

    for (int32_t reduce_id = params.num_batches + lane_idx; reduce_id <= params.fixed_num_batches; reduce_id += ck_tile::get_warp_size())
    {
        params.p_reduce_indptr[reduce_id] = reduce_end;
    }
}

void get_mla_metadata_v1_0_device(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
                                  const torch::Tensor& seqlens_kv_indptr, // [batch size + 1]
                                  const int32_t num_heads_per_head_k,
                                  const int32_t num_heads_k,
                                  const bool is_causal,
                                  const int32_t kv_granularity,
                                  const int32_t max_seqlen_qo,
                                  const int32_t ori_uni_seqlen_qo,
                                  const int32_t num_splits,
                                  torch::Tensor& work_metadata_ptrs,
                                  torch::Tensor& work_info_set,
                                  torch::Tensor& work_indptr,
                                  torch::Tensor& reduce_indptr,
                                  torch::Tensor& reduce_final_map,
                                  torch::Tensor& reduce_partial_map)
{
    constexpr int32_t kPackedQoLenPerWg = 128;

    const hipStream_t stream = at::hip::getCurrentHIPStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);

    const int32_t num_clusters = dev_prop.multiProcessorCount / num_heads_k;

    int32_t num_batches    = seqlens_kv_indptr.size(0) - 1;
    int32_t num_heads      = num_heads_k * num_heads_per_head_k;
    int32_t qk_batch_ratio = 1;
    int32_t uni_seqlen_qo  = ori_uni_seqlen_qo;

    int32_t fixed_num_batches = reduce_indptr.size(0) - 1;
    // In the following cases, we use #head=16 to simulate cases which is not natively supported by
    // mla main kernel.
    if((num_heads != 16) &&
       (num_heads != 128) && // main kernel natively supports #head=16 or #head=128
       (num_heads % 16 == 0) && (num_heads < 128))
    {
        qk_batch_ratio = num_heads / 16;
        num_heads      = 16;
        num_batches *= qk_batch_ratio;
    }

    TORCH_CHECK((num_heads == 16) || (num_heads == 128),
                __func__,
                ": only supports #heads in [16, 128], or (#head, uni_seqlen_qo) = (16*N, 1) where "
                "N is in [2, 8).")

    MlaMetadataV1KernelParameter params = {};
    params.p_work_metadata_ptrs         = work_metadata_ptrs.data_ptr<uint64_t>();
    params.p_work_indptr                = work_indptr.data_ptr<int32_t>();
    params.p_work_info_set_raw          = work_info_set.data_ptr<int32_t>();
    params.p_reduce_indptr              = reduce_indptr.data_ptr<int32_t>();
    params.p_reduce_final_map           = reduce_final_map.data_ptr<int32_t>();
    params.p_reduce_partial_map         = reduce_partial_map.data_ptr<int32_t>();
    params.p_seqlens_qo_indptr          = seqlens_qo_indptr.data_ptr<int32_t>();
    params.p_seqlens_kv_indptr          = seqlens_kv_indptr.data_ptr<int32_t>();
    params.num_batches                  = num_batches;
    params.num_heads                    = num_heads_k * num_heads_per_head_k;
    params.num_cu                       = num_clusters;
    params.num_splits                   = num_splits;
    params.fixed_num_batches            = fixed_num_batches;
    params.reduce_indptr_size           = reduce_indptr.size(0);
    params.kv_granularity               = kv_granularity;
    params.kv_granularity_log2          = __builtin_ctz(kv_granularity);
    params.uni_seqlen_qo                = uni_seqlen_qo;
    params.ori_seqlen_qo                = ori_uni_seqlen_qo;
    params.is_causal                    = is_causal;
    params.qk_batch_ratio               = qk_batch_ratio;

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
	kn_get_mla_metadata_v1_0<<<grid, dev_prop.warpSize, dev_prop.maxSharedMemoryPerMultiProcessor, stream>>>(params);
}
