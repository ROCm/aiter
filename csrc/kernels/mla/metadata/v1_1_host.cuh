#pragma once

#include <queue>
#include "aiter_hip_common.h"
#include "v1_comm.cuh"

template <typename Traits>
std::vector<torch::Tensor> get_mla_metadata_v1_1_host(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const int32_t        kv_granularity,
    const bool           no_redundant)
{
    using index_t = uint32_t;

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t num_batches = seqlens_qo_indptr.size(0) - 1;
    const int32_t num_heads  = num_heads_k * num_heads_per_head_k;

    auto seqlens_qo_indptr_cpu = seqlens_qo_indptr.to(at::DeviceType::CPU);
    auto seqlens_kv_indptr_cpu = seqlens_kv_indptr.to(at::DeviceType::CPU);

    const int32_t* p_seqlens_qo_indptr = seqlens_qo_indptr_cpu.data_ptr<int32_t>();
    const int32_t* p_seqlens_kv_indptr = seqlens_kv_indptr_cpu.data_ptr<int32_t>();

    // Step.0. Get sequence lengths of query/output and key/value for each batch.
    std::vector<BatchInfo> batch_infos;
    batch_infos.reserve(num_batches);
    int32_t sum_packed_qo_len = 0;
    for (int32_t bid = 0; bid < num_batches; ++bid)
    {
        const int32_t qo_len = p_seqlens_qo_indptr[bid + 1] - p_seqlens_qo_indptr[bid];
        const int32_t kv_len = p_seqlens_kv_indptr[bid + 1] - p_seqlens_kv_indptr[bid];
        TORCH_CHECK((qo_len > 0) && (kv_len > 0), __func__, ": Invalid qo_len or/and kv_len!");

        const int32_t packed_qo_len = qo_len * num_heads;
        sum_packed_qo_len += packed_qo_len;

        batch_infos.push_back({bid, qo_len, kv_len});
    }
    std::sort(batch_infos.begin(), batch_infos.end(), std::greater<BatchInfo>());

    // Step.1. Calculate the size of cluster and some related information. The size is the number of workgroups
    //         composing each cluster. The size is determined by average packed qo length.
    const int32_t cluster_size =
    [&]() {
        const int32_t avg_packed_qo_len = sum_packed_qo_len / num_batches;
        const int32_t cluster_size =
            ck_tile::integer_divide_ceil(avg_packed_qo_len, Traits::kPackedQoLenPerWg);
        return ck_tile::min(cluster_size, Traits::kMaxClusterSize);
    }();
    TORCH_CHECK((dev_prop.multiProcessorCount % cluster_size) == 0, __func__, ": Invalid cluster_size!");
    const int32_t num_clusters  = dev_prop.multiProcessorCount / cluster_size;
    const int32_t cluster_len_q = cluster_size * Traits::kPackedQoLenPerWg;

    // Step.2.
    //   a. Get total valid (after causal masking) kv lengths and the maximun workload handled by each cluster
    //   b. Get a indptr array about #cluster for each batch in direction of qo.
    int32_t workload_sum = 0;
    std::vector<int32_t> num_qo_clusters_indptr;
    num_qo_clusters_indptr.reserve(num_batches + 1);
    num_qo_clusters_indptr.push_back(0);
    for (const auto& binfo : batch_infos)
    {
        const int32_t packed_qo_len      = binfo.qo_len * num_heads;
        const int32_t num_qo_tiles       = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);
        const int32_t packed_qo_tile_len = ck_tile::min(packed_qo_len, cluster_len_q);

        num_qo_clusters_indptr.push_back(num_qo_clusters_indptr.back() + num_qo_tiles);

        for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
        {
            const int32_t kv_len_valid =
                cal_packed_causal_kv_len(
                    binfo.qo_len, binfo.kv_len, tid, packed_qo_tile_len, num_qo_tiles, num_heads, is_causal);
            // always assume that each batch of tile will be splited once along kv.
            const int32_t kv_len_splited =
                ck_tile::integer_least_multiple(ck_tile::integer_divide_ceil(kv_len_valid, 2), kv_granularity);
            workload_sum += 2 * cal_cost(packed_qo_tile_len, kv_len_splited) + kv_granularity;
        }
    }

    const int32_t workload_limit_global = cal_workload_limit_global_v0(workload_sum, num_clusters, kv_granularity);
#if PRINT_DBG
    printf("[metadata] workload_limit_global=%d\n", workload_limit_global);
#endif

    // Step.3.1. Allocates output buffers except indptrs
    std::vector<std::vector<MlaWorkInfo>> work_info_set(num_clusters, std::vector<MlaWorkInfo>());
    std::vector<std::vector<index_t>> reduce_partial_map(num_qo_clusters_indptr.back(), std::vector<index_t>());
    std::vector<MlaPartialTileInfo> reduce_partial_info(num_qo_clusters_indptr.back(), {{-1, -2}});

    // Step.3.2. Declare priority queue
    using ClusterCost = std::tuple<int32_t, int32_t>; // cluster_id(cid), cost
    auto pq_cmp = [](const ClusterCost& l, const ClusterCost& r) { return std::get<1>(l) > std::get<1>(r); };
    std::priority_queue<ClusterCost, std::vector<ClusterCost>, decltype(pq_cmp)> cost_heap(pq_cmp);
    for (int32_t cid = 0; cid < num_clusters; ++cid) { cost_heap.push(std::tuple{cid, 0}); }

    // Step.4. Fill the output buffers except indptrs
    int32_t num_reduce_row      = 0;
    int32_t num_partial_outputs = 0;
    int32_t loc_partial_outputs = 0;
    for (const auto& binfo : batch_infos)
    {
        const int32_t bid            = binfo.batch_idx;
        const int32_t qo_len         = binfo.qo_len;
        const int32_t kv_len         = binfo.kv_len;
        const int32_t packed_qo_len  = qo_len * num_heads;
        const int32_t num_qo_tiles   = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);
        const int32_t qo_batch_start = p_seqlens_qo_indptr[bid];
        const int32_t kv_batch_start = p_seqlens_kv_indptr[bid];
        const int32_t kv_batch_end   = p_seqlens_kv_indptr[bid + 1];
#if PRINT_DBG
        printf("[metadata] Dividing batch=%d, qo_len=%d, kv_len=%d\n", bid, qo_len, kv_len);
#endif

        for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
        {
            const int32_t global_cluster_q_idx = num_qo_clusters_indptr[bid] + tid;

            int32_t remaining_kv_len =
                cal_packed_causal_kv_len(qo_len, kv_len, tid, cluster_len_q, num_qo_tiles, num_heads, is_causal);
            int32_t kv_start_local = 0;

            const auto [cid_top, accum_cost_top] = cost_heap.top();
            const int32_t remaining_capability_top = cal_kv_len(workload_limit_global - accum_cost_top, cluster_len_q);
            const int32_t num_splits_estimated =
                ck_tile::integer_divide_ceil(remaining_kv_len, remaining_capability_top);
            // For the case of #splits==2, make sure that the tailing tile is smaller than Traits::kSplitTolerance.
            const bool split_kv = (num_splits_estimated == 2) ?
                ((remaining_kv_len - remaining_capability_top) > Traits::kSplitTolerance) : (num_splits_estimated > 1);
            const int32_t kv_len_limit_floor =
                ck_tile::integer_least_multiple(ck_tile::integer_divide_ceil(kv_len, num_clusters), kv_granularity);

            do
            {
                // Check and update cost_heap
                auto [cid, accum_cost] = cost_heap.top();
                cost_heap.pop();
                const int32_t remaining_capability = cal_kv_len(workload_limit_global - accum_cost, cluster_len_q);
                const int32_t kv_len_limit_local =
                [&]() {
                    const int32_t limit_ori = ck_tile::max(remaining_capability, kv_len_limit_floor);
                    const int32_t tail_size = (remaining_kv_len > limit_ori) ? (remaining_kv_len - limit_ori) : 0x7fffffff;
                    const int32_t limit_fin = (tail_size <= Traits::kSplitTolerance) ? remaining_kv_len : limit_ori;
                    return limit_fin;
                }();
                const int32_t kv_len_consuming = ck_tile::min(remaining_kv_len, kv_len_limit_local);
                const int32_t cost = cal_cost(cluster_len_q, kv_len_consuming);
#if PRINT_DBG
                printf("[metadata] cost heap updated: cid=%d, pre_cost=%d, new_cost=%d, tot_cost=%d, kv_len_cons=%d\n",
                       cid, accum_cost, cost, accum_cost+cost, kv_len_consuming);
#endif
                const int32_t new_cost = accum_cost + cost;
                cost_heap.push(std::tuple{cid, new_cost});

                // Record work
                MlaWorkInfo work_info{};
                work_info.batch_idx = bid;
                work_info.qo_start  = tid * cluster_len_q + qo_batch_start;
                work_info.qo_end    = ck_tile::min(work_info.qo_start + cluster_len_q, qo_batch_start + qo_len);
                work_info.kv_start  = kv_start_local + kv_batch_start;
                work_info.kv_end    = work_info.kv_start + kv_len_consuming;
                work_info.kv_offset = kv_batch_end - work_info.kv_end;
                if (split_kv)
                {
                    work_info.partial_qo_loc = loc_partial_outputs;
                    if (reduce_partial_map[global_cluster_q_idx].empty())
                    {
                        ++num_reduce_row;
                        reduce_partial_info[global_cluster_q_idx] = {{ work_info.qo_start, work_info.qo_end }};
                    }
                    reduce_partial_map[global_cluster_q_idx].push_back(loc_partial_outputs);
                    ++num_partial_outputs;
                    loc_partial_outputs += (work_info.qo_end - work_info.qo_start);
                }
                else
                {
                    work_info.partial_qo_loc = -1;
                }
                work_info_set[cid].push_back(work_info);

                // Update state
                remaining_kv_len -= kv_len_consuming;
                kv_start_local += kv_len_consuming;
            }
            while (remaining_kv_len > 0);
        }
    }

#if PRINT_DBG
    printf("[metadata] Final Cost Heap Status: %zu elements\n", cost_heap.size());
    while (cost_heap.empty() == false)
    {
        auto [id, cost] = cost_heap.top();
        cost_heap.pop();
        printf("[metadata] - cid=%d, cost=%d\n", id, cost);
    }
#endif

    // Step.5. Allocate and fill indptrs
    std::vector<index_t> work_indptr;
    work_indptr.reserve(num_clusters + 1);
    work_indptr.push_back(0);
    for (int32_t cid = 0; cid < num_clusters; ++cid)
    {
        if ((work_info_set[cid].empty() == false) || (no_redundant == false))
        {
            work_indptr.push_back(work_indptr.back() + work_info_set[cid].size());
        }
    }
    const int32_t num_works = work_indptr.back();

    const int32_t reduce_final_map_size = no_redundant ? num_reduce_row : num_qo_clusters_indptr.back();
    const int32_t reduce_indptr_size = reduce_final_map_size + 1;
    std::vector<MlaPartialTileInfo> reduce_final_map;
    std::vector<index_t> reduce_indptr;
    reduce_final_map.reserve(reduce_final_map_size);
    reduce_indptr.reserve(reduce_indptr_size);
    reduce_indptr.push_back(0);
    for (auto [global_cluster_q_idx ,rid] = std::tuple{0, 0};
         (global_cluster_q_idx < num_qo_clusters_indptr.back()) && ((rid < num_reduce_row) || (no_redundant == false));
         ++global_cluster_q_idx)
    {
        if ((reduce_partial_map[global_cluster_q_idx].empty() == false) || (no_redundant == false))
        {
            reduce_indptr.push_back(reduce_indptr.back() + reduce_partial_map[global_cluster_q_idx].size());
            reduce_final_map.push_back(reduce_partial_info[global_cluster_q_idx]);
            ++rid;
        }
    }

    // Step.6. Flatten 2D arries
    auto work_info_set_flatten = flatten(work_info_set, num_works);
    auto reduce_partial_map_flatten = flatten(reduce_partial_map, num_partial_outputs);

    // Step.7. Create tensors.
    auto input_opts = seqlens_qo_indptr.options();
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32);
    auto work_metadata_ptrs_tsr = torch::empty({2}, torch::TensorOptions().dtype(torch::kUInt64));
    auto work_info_set_tsr = torch::from_blob(work_info_set_flatten.data(), {num_works, kSizeMlaWorkInfoInDw}, int_opts).to(input_opts);
    auto work_indptr_tsr = torch::from_blob(work_indptr.data(), {static_cast<int32_t>(work_indptr.size())}, int_opts).to(input_opts);
    auto reduce_indptr_tsr = torch::from_blob(reduce_indptr.data(), {reduce_indptr_size}, int_opts).to(input_opts);
    auto reduce_final_map_tsr = torch::from_blob(reduce_final_map.data(), {reduce_final_map_size, kSizeMlaPartialTileInfoInDw}, int_opts).to(input_opts);
    auto reduce_partial_map_tsr = torch::from_blob(reduce_partial_map_flatten.data(), {num_partial_outputs}, int_opts).to(input_opts);

    work_metadata_ptrs_tsr.index_put_({0}, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(work_indptr_tsr.data_ptr())));
    work_metadata_ptrs_tsr.index_put_({1}, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(work_info_set_tsr.data_ptr())));

    // Last step. Copy to the device of input and return the results.
    return {work_metadata_ptrs_tsr.to(input_opts),
            work_indptr_tsr,
            work_info_set_tsr,
            reduce_indptr_tsr,
            reduce_final_map_tsr,
            reduce_partial_map_tsr};
}
