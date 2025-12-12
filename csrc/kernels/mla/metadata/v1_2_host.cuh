#pragma once

#include "aiter_hip_common.h"
#include "v1_comm.cuh"
#include "ps.h"

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


void generate_reduce_info(int32_t num_work,
                          std::vector<WorkInfo>& work_info,
                          std::vector<int32_t>& reduce_indptr,
                          std::vector<FinalLoc>& reduce_final_map,
                          std::vector<int32_t>& reduce_partial_map)
{
    std::map<std::vector<int32_t>, std::set<int32_t>> reduce_map; // 2d
    int32_t q_head_start, q_head_end;
    for(int32_t i = 0; i < num_work; ++i)
    {
        auto work = work_info[i];
        if(work.partial_o_loc == -1)
            continue;
        std::tie(q_head_start, q_head_end) = unpack_dword(work.q_head_range);

        auto final_loc   = std::vector<int32_t>({work.qo_start, work.qo_end});
        auto partial_loc = work.partial_o_loc;
        reduce_map[final_loc].insert(partial_loc);
    }

    int32_t final_idx = 0;
    int32_t partial_idx = 0;
    for(auto it = reduce_map.begin(); it != reduce_map.end(); ++it)
    {
        auto final_loc = it->first;
        std::vector<uint32_t> partial_loc_vec(it->second.begin(), it->second.end());
        const int32_t num_partials = partial_loc_vec.size();
        reduce_indptr[final_idx + 1] = reduce_indptr[final_idx] + num_partials;
        reduce_final_map[final_idx] = FinalLoc{final_loc[0], final_loc[1]};
        std::copy(partial_loc_vec.begin(), partial_loc_vec.end(),
            reduce_partial_map.begin() + partial_idx);
        final_idx++;
        partial_idx += partial_loc_vec.size();
    }
}


void kn_generate_ps_metadata(std::vector<int32_t>& seqlens_qo_indptr,
                             std::vector<int32_t>& pages_kv_indptr,
                             std::vector<int32_t>& context_lens,
                             const int32_t gqa_ratio,
                             const int32_t num_heads_k,
                             const int32_t qhead_granularity,
                             const int32_t qlen_granularity,
                             const int32_t kvlen_granularity,
                             const int32_t block_size,
                             const bool is_causal,
                             std::vector<int32_t>& work_indptr,
                             std::vector<WorkInfo>& work_info,
                             const int32_t available_tgs = 256,
                             const int32_t cluster_id    = 0,
                             int32_t current_work_idx    = 0)
{
    const int32_t batch_size = seqlens_qo_indptr.size() - 1;

    assert(kvlen_granularity % block_size == 0);
    const int32_t blocks_per_unit = kvlen_granularity / block_size;

    // Step 1: count split units
    std::vector<QTile> query_tiles;
    int32_t total_units = 0; // split units

    for(int32_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) // parallel
    {
        const int32_t qo_length = seqlens_qo_indptr[batch_idx + 1] - seqlens_qo_indptr[batch_idx];
        const int32_t kv_length = context_lens[batch_idx];
        // Split query sequence into tiles
        for(int32_t q_offset = 0; q_offset < qo_length; q_offset += qlen_granularity)
        {
            const int32_t local_qo_start = q_offset;
            const int32_t local_qo_end   = std::min(local_qo_start + qlen_granularity, qo_length);

            // For causal attention, each query position can only attend to
            // earlier positions, limiting the KV range
            const int32_t effective_kv_length =
                is_causal ? std::min(kv_length - qo_length + local_qo_end, kv_length) : kv_length;
            const int32_t num_units = ck_tile::integer_divide_ceil(effective_kv_length, kvlen_granularity);
            // const int32_t num_units =
            //     offset_div(effective_kv_length, kvlen_granularity, SPLIT_KV_OVERHEAD);
            query_tiles.push_back({batch_idx,
                                   local_qo_start + seqlens_qo_indptr[batch_idx],
                                   local_qo_end + seqlens_qo_indptr[batch_idx],
                                   num_units * blocks_per_unit,
                                   effective_kv_length});
            total_units += num_units;
        }
    }
    const int32_t average  = total_units / available_tgs;
    const int32_t reminder = total_units % available_tgs;

#if PRINT_DBG
    std::cout << "++++num_heads_k: " << num_heads_k << std::endl
              << "++++available_tgs: " << available_tgs << std::endl
              << "++++blocks per split unit: " << blocks_per_unit << std::endl
              << "++++total split units: " << total_units << std::endl
              << "++++average split units: " << average << std::endl
              << "++++remining split units: " << reminder << std::endl;
    std::cout << std::setw(14) << "" << std::setw(11) << "batch_idx" << std::setw(11)
              << "[qo_start" << std::setw(11) << "qo_end)" << std::setw(11) << "num_blocks"
              << std::setw(11) << "effect_kv" << std::endl;
    std::cout << "query_tiles: (" << query_tiles.size() << ") [" << std::endl;
    for(size_t qtile_idx = 0; qtile_idx < query_tiles.size(); qtile_idx++)
        std::cout << "query_tiles[" << qtile_idx << "]:" << query_tiles[qtile_idx] << std::endl;
    std::cout << "]" << std::endl;
#endif
    // OPT: sort by num_units

    // Step 2: distribute split units
    int32_t current_tile_idx  = 0; // index of query_tile
    int32_t current_block_idx = 0; // index of split blocks within a query_tile
    int32_t partial_tile_idx  = 0; // index of partial_tile for reduce
    for(int32_t tg_idx = 0; tg_idx < available_tgs; ++tg_idx)
    {
        // dupclicate (parallal)
        for(int32_t k_head_offset = 0; k_head_offset < num_heads_k; k_head_offset++)
        {
            const int32_t k_head_idx = cluster_id * num_heads_k + k_head_offset;
            const int32_t qhead_range =
                pack_dword(k_head_idx * qhead_granularity, (k_head_idx + 1) * qhead_granularity);
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
                        current_tile.effective_kv_length - current_block_idx * block_size;

                    int32_t consuming_blocks = 0;
                    const int32_t kv_start =
                        current_block_idx + pages_kv_indptr[current_tile.batch_idx];
                    if(remaining_kv_len <= blocks_capacity * block_size + SPLIT_KV_OVERHEAD)
                    {
                        consuming_blocks = remaining_blocks;
                        // This TG can process all of this qo_tile's remaining_blocks to the causal
                        // boundary
                        const int32_t partial_o_loc =
                            (current_block_idx == 0)
                                ? -1
                                : (qlen_granularity * partial_tile_idx++); // -1 - no split
                        const int32_t kv_end =
                            std::min(kv_start + consuming_blocks,
                                     pages_kv_indptr[current_tile.batch_idx + 1]);

                        work_info[current_work_idx++] = {current_tile.batch_idx,
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
                        consuming_blocks            = blocks_capacity;
                        const int32_t partial_o_loc = qlen_granularity * partial_tile_idx++;

                        const int32_t kv_end =
                            std::min(kv_start + consuming_blocks,
                                     pages_kv_indptr[current_tile.batch_idx + 1]);
                        const int32_t kv_length = context_lens[current_tile.batch_idx];
                        const int32_t kv_offset =
                            kv_length -
                            (kv_end - pages_kv_indptr[current_tile.batch_idx]) * block_size;

                        work_info[current_work_idx++] = {current_tile.batch_idx,
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

            if(k_head_offset != num_heads_k - 1)
            {
                current_tile_idx  = saved_tile_idx;
                current_block_idx = saved_block_idx;
                partial_tile_idx  = saved_partial_tile_idx;
            }
        }
        work_indptr[cluster_id * available_tgs + tg_idx + 1] = current_work_idx;
    }
}


void get_ps_metadata_v1_2_host(
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
    const int32_t        kvlen_granularity,
    const int32_t        block_size,
    const bool           is_causal)
{

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
    const int32_t available_tgs = dev_prop.multiProcessorCount;

    const int32_t batch_size = seqlens_qo_indptr.size(0) - 1;
    const int32_t num_heads_q   = num_heads_k * gqa_ratio;

    // 1. divide
    const int32_t num_clusters       = gcd(num_heads_k, available_tgs);
    const int32_t tgs_per_cluster    = available_tgs / num_clusters;
    const int32_t kheads_per_cluster = num_heads_k / num_clusters;

    // prepare host buffer
    auto seqlens_qo_indptr_vec = seqlens_qo_indptr.to(at::DeviceType::CPU);
    auto pages_kv_indptr_vec = pages_kv_indptr.to(at::DeviceType::CPU);
    auto context_lens_vec = context_lens.to(at::DeviceType::CPU);

    std::vector<int32_t> p_seqlens_qo_indptr = {
        seqlens_qo_indptr_vec.data_ptr<int32_t>(),
        seqlens_qo_indptr_vec.data_ptr<int32_t>() + batch_size + 1};
    std::vector<int32_t> p_pages_kv_indptr = {
        pages_kv_indptr_vec.data_ptr<int32_t>(),
        pages_kv_indptr_vec.data_ptr<int32_t>() + batch_size + 1};
    std::vector<int32_t> p_context_lens = {
        context_lens_vec.data_ptr<int32_t>(),
        context_lens_vec.data_ptr<int32_t>() + batch_size};

    std::vector<int32_t> work_indptr_vec(work_indptr.numel(), 0);
    std::vector<WorkInfo> work_info_vec(work_info.numel() / kSizeWorkInfoInDw, WorkInfo());
    std::vector<int32_t> reduce_indptr_vec(reduce_indptr.numel(), 0);
    std::vector<FinalLoc> reduce_final_map_vec(reduce_final_map.numel() / 2, FinalLoc());
    std::vector<int32_t> reduce_partial_map_vec(reduce_partial_map.numel(), 0);

    // 2. conquer(parallel)
    for(int32_t cluster_id = 0; cluster_id < num_clusters; cluster_id++)
    {
        // OPT: consider XCC L2 cache
        int32_t current_work_idx = work_indptr_vec[cluster_id * tgs_per_cluster];
        kn_generate_ps_metadata(p_seqlens_qo_indptr,
                                p_pages_kv_indptr,
                                p_context_lens,
                                gqa_ratio,
                                kheads_per_cluster,
                                qhead_granularity,
                                qlen_granularity,
                                kvlen_granularity,
                                block_size,
                                is_causal,
                                work_indptr_vec,
                                work_info_vec,
                                tgs_per_cluster,
                                cluster_id,
                                current_work_idx);
    }

    // TODO: fix reduce info
    const int32_t actual_works = work_indptr_vec.back();
    generate_reduce_info(actual_works, work_info_vec, reduce_indptr_vec, reduce_final_map_vec, reduce_partial_map_vec);

    // H2D
    auto input_options = work_indptr.options();
    auto input_dtype   = torch::TensorOptions().dtype(torch::kInt32);
    auto work_indptr_cpu = torch::from_blob(
        work_indptr_vec.data(),
        {work_indptr.numel()},
        input_options.device(torch::kCPU));
    work_indptr.copy_(work_indptr_cpu);

    auto work_info_cpu = torch::from_blob(
        work_info_vec.data(),
        {static_cast<int64_t>(work_info_vec.size()), kSizeWorkInfoInDw},
        input_options.device(torch::kCPU));
    work_info.copy_(work_info_cpu);

    auto reduce_indptr_cpu = torch::from_blob(
        reduce_indptr_vec.data(),
        {reduce_indptr.numel()},
        input_options.device(torch::kCPU));
    reduce_indptr.copy_(reduce_indptr_cpu);

    auto reduce_final_map_cpu = torch::from_blob(
        reduce_final_map_vec.data(),
        {static_cast<int64_t>(reduce_final_map_vec.size()), kSizeFinalLocInDw},
        input_options.device(torch::kCPU));
    reduce_final_map.copy_(reduce_final_map_cpu);

    auto reduce_partial_map_cpu = torch::from_blob(
        reduce_partial_map_vec.data(),
        {reduce_partial_map.numel()},
        input_options.device(torch::kCPU));
    reduce_partial_map.copy_(reduce_partial_map_cpu);

#if PRINT_DBG
    // print_metadata(work_indptr_vec, work_info_vec);
    // print_reduce_info(reduce_indptr_vec, reduce_final_map_vec, reduce_partial_map_vec);
#endif
    return;
}
