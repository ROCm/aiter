#pragma once

#include "aiter_hip_common.h"
#include "ps.h"
#include "v1_comm.cuh"

#define SPLIT_KV_OVERHEAD 0

inline int gcd(int a, int b)
{
    while(b != 0)
    {
        int temp = b;
        b        = a % b;
        a        = temp;
    }
    return a;
}

std::pair<int32_t, int32_t> kn_generate_ps_metadata(const PsMetadataV1KernelParameter& params,
                                                    const int32_t cluster_id,
                                                    int32_t current_work_idx)
{
    const int32_t batch_size = params.batch_size;

    assert(params.kvlen_granularity % params.block_size == 0);
    const int32_t blocks_per_unit = params.kvlen_granularity / params.block_size;

    // Step 1: count split units
    std::vector<QTile> query_tiles;
    int32_t total_units = 0; // split units

    for(int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) // parallel
    {
        const int32_t qo_length =
            params.p_seqlens_qo_indptr[batch_idx + 1] - params.p_seqlens_qo_indptr[batch_idx];
        const int32_t kv_length = params.p_context_lens[batch_idx];
        // Split query sequence into tiles
        // TODO: optimize ping-pong allocate
        std::vector<std::pair<int32_t, int32_t>> query_tile_ranges;
        for(int32_t q_offset = 0; q_offset < qo_length; q_offset += params.qlen_granularity)
        {
            const int32_t local_qo_start = q_offset;
            const int32_t local_qo_end   = std::min(q_offset + params.qlen_granularity, qo_length);
            query_tile_ranges.push_back({local_qo_start, local_qo_end});
        }

        int num_query_tile_ranges = query_tile_ranges.size();

        for(int32_t i = 0; i < num_query_tile_ranges; ++i)
        {
            // ping-pong allocate between head & tail: 0, n-1, 1, n-2, 2, n-3, ...
            int32_t idx = (i % 2 == 0) ? (i / 2) : (num_query_tile_ranges - 1 - i / 2);

            const int32_t local_qo_start = query_tile_ranges[idx].first;
            const int32_t local_qo_end   = query_tile_ranges[idx].second;

            // For causal attention, each query position can only attend to
            // earlier positions, limiting the KV range
            const int32_t effective_kv_length =
                params.is_causal ? std::min(kv_length - qo_length + local_qo_end, kv_length)
                                 : kv_length;
            const int32_t num_units =
                ck_tile::integer_divide_ceil(effective_kv_length, params.kvlen_granularity);
            query_tiles.push_back({batch_idx,
                                   local_qo_start + params.p_seqlens_qo_indptr[batch_idx],
                                   local_qo_end + params.p_seqlens_qo_indptr[batch_idx],
                                   num_units * blocks_per_unit,
                                   effective_kv_length});
            total_units += num_units;
        }
    }
    const int32_t average  = total_units / params.available_tgs;
    const int32_t reminder = total_units % params.available_tgs;
    // TODO: sort by num_units

    // Step 2: distribute split units
    int32_t current_tile_idx  = 0; // index of query_tile
    int32_t current_block_idx = 0; // index of split blocks within a query_tile
    int32_t partial_tile_idx  = 0; // index of partial_tile for reduce
    int32_t final_tile_idx    = 0; // index of final_tile for reduce
    for(int32_t tg_idx = 0; tg_idx < params.available_tgs; ++tg_idx)
    {
        // dupclicate (parallal)
        for(int32_t k_head_offset = 0; k_head_offset < params.kheads_per_cluster; k_head_offset++)
        {
            const int32_t k_head_idx       = cluster_id * params.kheads_per_cluster + k_head_offset;
            const int32_t qhead_range      = pack_dword(k_head_idx * params.qhead_granularity,
                                                   (k_head_idx + 1) * params.qhead_granularity);
            int32_t saved_tile_idx         = current_tile_idx;
            int32_t saved_block_idx        = current_block_idx;
            int32_t saved_partial_tile_idx = partial_tile_idx;

            auto allocate_work = [&]() mutable {
                int32_t blocks_capacity = (tg_idx < reminder) ? (average + 1) * blocks_per_unit
                                                              : average * blocks_per_unit;
                // Allocate KV units to this TG until quota is filled
                while(current_tile_idx < static_cast<int>(query_tiles.size()) &&
                      blocks_capacity > 0)
                {
                    const QTile& current_tile = query_tiles[current_tile_idx];
                    // OPT: add split_kvlen_overhead
                    const int32_t remaining_blocks = current_tile.num_blocks - current_block_idx;
                    const int32_t remaining_kv_len =
                        current_tile.effective_kv_length - current_block_idx * params.block_size;

                    int32_t consuming_blocks = 0;
                    const int32_t kv_start =
                        current_block_idx + params.p_pages_kv_indptr[current_tile.batch_idx];
                    if(remaining_kv_len <= blocks_capacity * params.block_size + SPLIT_KV_OVERHEAD)
                    {
                        consuming_blocks = remaining_blocks;
                        // This TG can process all of this qo_tile's remaining_blocks to the causal
                        // boundary
                        int32_t partial_o_loc = -1;
                        if(current_block_idx != 0)
                        {
                            partial_o_loc = params.qlen_granularity * partial_tile_idx;
                            if(k_head_offset == 0)
                            {
                                params.p_reduce_partial_map[partial_tile_idx] = partial_o_loc;
                                params.p_reduce_indptr[final_tile_idx + 1] = partial_tile_idx + 1;
                                params.p_reduce_final_map[final_tile_idx] =
                                    FinalLoc{current_tile.qo_start, current_tile.qo_end};
                                final_tile_idx++;
                            }
                            partial_tile_idx++;
                        }
                        const int32_t kv_end =
                            std::min(kv_start + consuming_blocks,
                                     params.p_pages_kv_indptr[current_tile.batch_idx + 1]);

                        params.p_work_info[current_work_idx++] = {current_tile.batch_idx,
                                                                  partial_o_loc,
                                                                  current_tile.qo_start,
                                                                  current_tile.qo_end,
                                                                  kv_start,
                                                                  kv_end,
                                                                  0,
                                                                  qhead_range};
                        current_tile_idx++;
                        current_block_idx = 0;
                    }
                    else
                    {
                        // This TG can only process part of this qotile's KV units under
                        // blocks_capacity
                        consuming_blocks = blocks_capacity;

                        const int32_t partial_o_loc = params.qlen_granularity * partial_tile_idx;
                        if(k_head_offset == 0)
                        {
                            params.p_reduce_partial_map[partial_tile_idx] = partial_o_loc;
                        }
                        partial_tile_idx++;

                        const int32_t kv_end =
                            std::min(kv_start + consuming_blocks,
                                     params.p_pages_kv_indptr[current_tile.batch_idx + 1]);
                        const int32_t kv_length = params.p_context_lens[current_tile.batch_idx];
                        const int32_t kv_offset =
                            kv_length -
                            (kv_end - params.p_pages_kv_indptr[current_tile.batch_idx]) *
                                params.block_size;

                        params.p_work_info[current_work_idx++] = {current_tile.batch_idx,
                                                                  partial_o_loc,
                                                                  current_tile.qo_start,
                                                                  current_tile.qo_end,
                                                                  kv_start,
                                                                  kv_end,
                                                                  kv_offset,
                                                                  qhead_range};
                        current_block_idx += consuming_blocks;
                    }
                    blocks_capacity -= consuming_blocks;
                }
            };

            allocate_work();

            if(k_head_offset != params.kheads_per_cluster - 1)
            {
                current_tile_idx  = saved_tile_idx;
                current_block_idx = saved_block_idx;
                partial_tile_idx  = saved_partial_tile_idx;
            }
        }
        params.p_work_indptr[cluster_id * params.tgs_per_cluster + tg_idx + 1] = current_work_idx;
    }
    return std::make_pair(final_tile_idx, partial_tile_idx);
}

void get_ps_metadata_v1_2_host(const torch::Tensor& seqlens_qo_indptr, // [batch size + 1]
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
                               const int32_t kvlen_granularity,
                               const int32_t block_size,
                               const bool is_causal)
{
    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    const int32_t available_tgs = dev_prop.multiProcessorCount;

    const int32_t batch_size = context_lens.size(0);

    // 1. divide
    const int32_t num_clusters       = gcd(num_heads_k, available_tgs);
    const int32_t tgs_per_cluster    = available_tgs / num_clusters;
    const int32_t kheads_per_cluster = num_heads_k / num_clusters;

    // prepare pointers
    int32_t* p_seqlens_qo_indptr = seqlens_qo_indptr.data_ptr<int32_t>();
    int32_t* p_pages_kv_indptr   = pages_kv_indptr.data_ptr<int32_t>();
    int32_t* p_context_lens      = context_lens.data_ptr<int32_t>();

    int32_t* p_work_indptr   = work_indptr.data_ptr<int32_t>();
    WorkInfo* p_work_info    = reinterpret_cast<WorkInfo*>(work_info.data_ptr<int32_t>());
    int32_t* p_reduce_indptr = reduce_indptr.data_ptr<int32_t>();
    FinalLoc* p_reduce_final_map =
        reinterpret_cast<FinalLoc*>(reduce_final_map.data_ptr<int32_t>());
    int32_t* p_reduce_partial_map = reduce_partial_map.data_ptr<int32_t>();

    PsMetadataV1KernelParameter params = {};
    params.batch_size                  = batch_size;
    params.gqa_ratio                   = gqa_ratio;
    params.num_heads_k                 = num_heads_k;
    params.qhead_granularity           = qhead_granularity;
    params.qlen_granularity            = qlen_granularity;
    params.kvlen_granularity           = kvlen_granularity;
    params.block_size                  = block_size;
    params.is_causal                   = is_causal;
    params.available_tgs               = available_tgs;
    params.num_clusters                = num_clusters;
    params.tgs_per_cluster             = tgs_per_cluster;
    params.kheads_per_cluster          = kheads_per_cluster;
    params.p_seqlens_qo_indptr         = p_seqlens_qo_indptr;
    params.p_pages_kv_indptr           = p_pages_kv_indptr;
    params.p_context_lens              = p_context_lens;
    params.p_work_indptr               = p_work_indptr;
    params.p_work_info                 = p_work_info;
    params.p_reduce_indptr             = p_reduce_indptr;
    params.p_reduce_final_map          = p_reduce_final_map;
    params.p_reduce_partial_map        = p_reduce_partial_map;

    p_work_indptr[0]          = 0;
    p_reduce_indptr[0]        = 0;
    int32_t num_final_tiles   = 0;
    int32_t num_partial_tiles = 0;
    // 2. conquer(parallel)
    for(int32_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
    {
        // OPT: consider XCC L2 cache
        int32_t current_work_idx = p_work_indptr[cluster_id * tgs_per_cluster];
        std::tie(num_final_tiles, num_partial_tiles) =
            kn_generate_ps_metadata(params, cluster_id, current_work_idx);
    }

    for(auto i = num_final_tiles + 1; i < reduce_indptr.size(0); i++)
    {
        p_reduce_indptr[i] = num_partial_tiles;
    }
    return;
}
