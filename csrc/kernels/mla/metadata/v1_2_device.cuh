// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "v1_comm.cuh"


template <int32_t kPackedQoLenPerWg_>
struct FlashMlaKernelTrait
{
    static constexpr int32_t kPackedQoLenPerWg       = kPackedQoLenPerWg_;
    static constexpr int32_t kFixedOverheadNumBlocks = 16;
};

template <typename Traits>
__launch_bounds__(ck_tile::get_warp_size(), 1)
__global__ void kn_get_mla_metadata_v1_2(
    MlaMetadataV1KernelParameter params)
{
    extern __shared__ uint8_t p_smem[];
    int32_t* p_lds_seqlens_qo = reinterpret_cast<int32_t*>(p_smem);
    int32_t* p_lds_seqlens_kv = p_lds_seqlens_qo + params.num_batches;

    const int32_t lane_idx = ck_tile::get_lane_id();

    MlaWorkInfo* p_work_info_set = reinterpret_cast<MlaWorkInfo*>(params.p_work_info_set_raw);

    int32_t sum_blocks = 0;
    for (int32_t bid = lane_idx; bid < params.num_batches; bid += ck_tile::get_warp_size())
    {
        const int32_t seqlen_qo = params.p_seqlens_qo_indptr[bid + 1] - params.p_seqlens_qo_indptr[bid];
        const int32_t seqlen_kv = params.p_seqlens_kv_indptr[bid + 1] - params.p_seqlens_kv_indptr[bid];
        p_lds_seqlens_kv[bid] = seqlen_kv;
        p_lds_seqlens_qo[bid] = seqlen_qo;
        const int32_t num_blocks = ck_tile::integer_divide_ceil(seqlen_kv, params.kv_granularity);
        sum_blocks += num_blocks;
    }

    sum_blocks = aiter::warpReduce<aiter::AddFunctor, decltype(sum_blocks), ck_tile::get_warp_size()>(sum_blocks);
    sum_blocks += params.num_batches * Traits::kFixedOverheadNumBlocks;

    if (lane_idx == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_indptr[0]   = 0;
        params.p_work_metadata_ptrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
        params.p_work_metadata_ptrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p_work_info_set));
    }

    // expected payload handled by each cu part.
    const int32_t payload = ck_tile::integer_divide_ceil(sum_blocks, params.num_cu) +
                            Traits::kFixedOverheadNumBlocks;

    int32_t curr_batch = 0;         // batch ID of the batch which is under review
    int32_t curr_kv_block = 0;      // #blocks handled by previous cu part(s)
    int32_t curr_n_split_idx = 0;   // #cu parts used to handle current batch

    int32_t num_works = 0;
    int32_t partial_idx = 0;
    int32_t tot_qo_tiles = 0;
    int32_t last_reduce_indptr = 0;

    for (int32_t cid = 0; cid < params.num_cu; ++cid)
    {
        int32_t remain_payload = payload;
        while (curr_batch < params.num_batches)
        {
            const int32_t seqlen_kv     = p_lds_seqlens_kv[curr_batch];
            const int32_t packed_qo_len = p_lds_seqlens_qo[curr_batch] * params.num_heads;
            const int32_t num_qo_tiles  = ck_tile::integer_divide_ceil(packed_qo_len, Traits::kPackedQoLenPerWg);
            const int32_t qo_tile_size  = ck_tile::integer_divide_ceil(p_lds_seqlens_qo[curr_batch], num_qo_tiles);
            const int32_t num_kv_blocks = ck_tile::integer_divide_ceil(seqlen_kv, params.kv_granularity);
            const int32_t remain_kv_blocks = num_kv_blocks - curr_kv_block;

            // If current cu part is able to handle this batch of seqences
            if (remain_payload >= (remain_kv_blocks + Traits::kFixedOverheadNumBlocks))
            {
                const int32_t num_splits = curr_n_split_idx + 1;

                // record a work in work_info_set
                for (int32_t idx = lane_idx; idx < num_splits * num_qo_tiles; idx += ck_tile::get_warp_size())
                {
                    const int32_t qo_tile_idx        = idx % num_qo_tiles;
                    const int32_t global_qo_tile_idx = tot_qo_tiles + qo_tile_idx;
                    const int32_t split_idx          = idx / num_qo_tiles;

                    MlaWorkInfo work_info{};
                    work_info.batch_idx = curr_batch;
                    work_info.qo_start = params.p_seqlens_qo_indptr[curr_batch] + qo_tile_idx * qo_tile_size;
                    work_info.qo_end =
                        ck_tile::min(work_info.qo_start + qo_tile_size, params.p_seqlens_qo_indptr[curr_batch + 1]);
                    work_info.kv_start = params.p_seqlens_kv_indptr[curr_batch] + curr_kv_block * params.kv_granularity;
                    work_info.kv_end =
                        ck_tile::min(work_info.kv_start + remain_kv_blocks * params.kv_granularity,
                                     params.p_seqlens_kv_indptr[curr_batch+1]) -
                        (num_qo_tiles - 1 - qo_tile_idx);
                    work_info.kv_offset = params.p_seqlens_kv_indptr[curr_batch+1] - work_info.kv_end;

                    // split related info
                    if (curr_n_split_idx > 0)
                    {
                        // set work info
                        work_info.partial_qo_loc = partial_idx + qo_tile_idx * qo_tile_size;

                        // set reduce info
                        if (split_idx == 0)
                        {
                            params.p_reduce_indptr[global_qo_tile_idx + 1] =
                                last_reduce_indptr + (qo_tile_idx + 1) * num_splits;
                            params.p_reduce_final_map[global_qo_tile_idx * 2] = work_info.qo_start;
                            params.p_reduce_final_map[global_qo_tile_idx * 2 + 1] = work_info.qo_end;
                        }

                        params.p_reduce_partial_map[last_reduce_indptr + qo_tile_idx * num_splits + split_idx] =
                            partial_idx - (curr_n_split_idx - split_idx) * qo_tile_size * num_qo_tiles;
                    }
                    else
                    {
                        work_info.partial_qo_loc = -1;
                        if (split_idx == 0)
                        {
                            params.p_reduce_indptr[global_qo_tile_idx + 1] = last_reduce_indptr;
                            params.p_reduce_final_map[global_qo_tile_idx * 2] = -1;
                            params.p_reduce_final_map[global_qo_tile_idx * 2 + 1] = -2;
                        }
                    }

                    if (split_idx == 0)
                    {
                        p_work_info_set[num_works + qo_tile_idx] = work_info;
                    }
                }

                tot_qo_tiles += num_qo_tiles;
                num_works += num_qo_tiles;
                if (curr_n_split_idx > 0)
                {
                    partial_idx += num_qo_tiles * qo_tile_size;
                    last_reduce_indptr += num_qo_tiles * num_splits;
                }

                // update state
                remain_payload -= (remain_kv_blocks + Traits::kFixedOverheadNumBlocks);
                ++curr_batch;
                curr_kv_block = 0;
                curr_n_split_idx = 0;
            }
            else
            {
                if (remain_payload > Traits::kFixedOverheadNumBlocks)
                {
                    const int32_t consuming_blks = remain_payload - Traits::kFixedOverheadNumBlocks;

                    // record a work in work_info_set
                    for (int32_t qo_tile_idx = lane_idx;
                         qo_tile_idx < num_qo_tiles;
                         qo_tile_idx += ck_tile::get_warp_size())
                    {
                        MlaWorkInfo work_info{};
                        work_info.batch_idx = curr_batch;
                        work_info.qo_start = params.p_seqlens_qo_indptr[curr_batch] + qo_tile_idx * qo_tile_size;
                        work_info.qo_end =
                            ck_tile::min(work_info.qo_start + qo_tile_size, params.p_seqlens_qo_indptr[curr_batch + 1]);
                        work_info.kv_start = params.p_seqlens_kv_indptr[curr_batch] + curr_kv_block * params.kv_granularity;
                        work_info.kv_end = work_info.kv_start + consuming_blks * params.kv_granularity; // TODO: we need to take casual mask into consideration
                        work_info.kv_offset = params.p_seqlens_kv_indptr[curr_batch+1] - work_info.kv_end;
                        work_info.partial_qo_loc = partial_idx + qo_tile_idx * qo_tile_size;
                        p_work_info_set[num_works + qo_tile_idx] = work_info;
                    }
                    partial_idx += num_qo_tiles * qo_tile_size;
                    num_works += num_qo_tiles;

                    // update state
                    curr_kv_block += consuming_blks;
                    ++curr_n_split_idx;
                }
                break;
            }
        }

        if (lane_idx == 0)
        {
            params.p_work_indptr[cid + 1] = num_works;
        }
    }

    for (int32_t i = params.num_batches + lane_idx; i < params.reduce_indptr_size; i += ck_tile::get_warp_size())
    {
        params.p_reduce_indptr[i] = last_reduce_indptr;
    }
}

void get_mla_metadata_v1_2_device(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const int32_t        kv_granularity,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_info_set,
    torch::Tensor&       work_indptr,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map)
{
    //                                 kPackedQoLenPerWg
    using Traits = FlashMlaKernelTrait<128>;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);

    const int32_t num_clusters = dev_prop.multiProcessorCount / num_heads_k;

    MlaMetadataV1KernelParameter params = {};
    params.p_work_metadata_ptrs = work_metadata_ptrs.data_ptr<uint64_t>();
    params.p_work_indptr        = work_indptr.data_ptr<int32_t>();
    params.p_work_info_set_raw  = work_info_set.data_ptr<int32_t>();
    params.p_reduce_indptr      = reduce_indptr.data_ptr<int32_t>();
    params.p_reduce_final_map   = reduce_final_map.data_ptr<int32_t>();
    params.p_reduce_partial_map = reduce_partial_map.data_ptr<int32_t>();
    params.p_seqlens_qo_indptr  = seqlens_qo_indptr.data_ptr<int32_t>();
    params.p_seqlens_kv_indptr  = seqlens_kv_indptr.data_ptr<int32_t>();
    params.num_batches          = seqlens_kv_indptr.size(0) - 1;
    params.num_heads            = num_heads_k * num_heads_per_head_k;
    params.num_cu               = num_clusters;
    params.reduce_indptr_size   = reduce_indptr.size(0);
    params.kv_granularity       = ck_tile::integer_least_multiple(kv_granularity, 16);
    params.is_causal            = is_causal;

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity
    kn_get_mla_metadata_v1_2<Traits><<<grid, num_thr, dev_prop.maxSharedMemoryPerMultiProcessor, stream>>>(params);
}
