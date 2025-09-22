// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_hip_common.h"
#include "v1_comm.cuh"

#define PRINT_DBG 0

CK_TILE_DEVICE auto get_cost_top(
    const int32_t* p_cost_heap,
    const int32_t  num_clusters)
{
    int32_t cid_min = -1;
    int32_t cost_min = 0x7fffffff;

    // Get local top
    for (int32_t cid = ck_tile::get_lane_id(); cid < num_clusters; cid += ck_tile::get_warp_size())
    {
        const int32_t cost = p_cost_heap[cid];
        if (cost < cost_min)
        {
            cost_min = cost;
            cid_min = cid;
        }
    }

    // Get global top
    #pragma unroll
    for (int32_t offset = (ck_tile::get_warp_size() >> 1); offset > 0; offset >>= 1)
    {
        const int32_t srd_lane    = (offset ^ ck_tile::get_warp_size()) ^ ck_tile::get_lane_id();
        const int32_t cid_remote  = ck_tile::warp_shuffle(cid_min,  srd_lane);
        const int32_t cost_remote = ck_tile::warp_shuffle(cost_min, srd_lane);
        if ((cost_remote < cost_min) || ((cost_remote == cost_min) && (cid_remote < cid_min)))
        {
            cost_min = cost_remote;
            cid_min  = cid_remote;
        }
    }

    return std::make_tuple(cid_min, cost_min);
}

template<int32_t kPackedQoLenPerWg_,
         int32_t kMaxClusterSize_>
struct MlaMetadataTraits
{
    static constexpr int32_t kPackedQoLenPerWg  = kPackedQoLenPerWg_;
    static constexpr int32_t kMaxClusterSize    = kMaxClusterSize_;
    static constexpr int32_t kSplitTolerance    = 16;
};

// This version just follows Flashinfer
CK_TILE_HOST_DEVICE int32_t cal_workload_limit_global_v0(
    const int32_t cum_workload,
    const int32_t num_clusters,
    const int32_t kv_granularity)
{
    int32_t limit;

    const int32_t avg_workload = ck_tile::max(ck_tile::integer_divide_ceil(cum_workload, num_clusters), 1);
    if (avg_workload <= 8) limit = 32;
    else if (avg_workload <= 16) limit = 64;
    else if (avg_workload <= 32) limit = 128;
    else if (avg_workload <= 64) limit = 192;
    else limit = avg_workload;

    return ck_tile::integer_least_multiple(limit, kv_granularity);
}

// This version estimate the total #workload (especially for estimating global #splits) plus an amplifier
CK_TILE_HOST_DEVICE int32_t cal_workload_limit_global_v1(
    const int32_t avg_workload,
    const int32_t var_workload,
    const int32_t cluster_len_q,
    const int32_t num_qo_tiles,
    const int32_t num_clusters,
    const int32_t kv_granularity)
{
    auto estimate_total_workload = [](
        const int32_t avg_workload,
        const int32_t var_workload,
        const int32_t cluster_len_q,
        const int32_t num_qo_tiles,
        const int32_t num_clusters)
    {
        const int32_t qo_factor = cluster_len_q >> 2;
        const float cv = sqrtf(float(var_workload)) / float(avg_workload); // coefficient of variation
        const int32_t A = int32_t(12.0f - (2.0f * cv + 1.0f)) * qo_factor;

        const int32_t workload_tot = (avg_workload + A) * num_qo_tiles;

        return workload_tot;
    };

    auto estimate_workload_per_cluster = [](
        const int32_t workload_tot,
        const int32_t cluster_len_q,
        const int32_t num_qo_tiles,
        const int32_t num_clusters)
    {
        // The following coefficients come from regression analysis. They may need to be changed with platform and MLA
        // kernels.
        const float workload_per_cluster_raw = float(workload_tot) / float(num_clusters);
        const float factor = -0.00000113f * powf(num_qo_tiles, 3) +
                              0.00025215f * powf(num_qo_tiles, 2) +
                             -0.00546401f * float(num_qo_tiles) +
                              0.40612956f;
        const float amplifier = ck_tile::max(factor * num_clusters / num_qo_tiles, 1.0f);
        const int32_t workload_per_cluster = int32_t(workload_per_cluster_raw * amplifier);

        return workload_per_cluster;
    };

    const int32_t tot_workload_estimated =
        estimate_total_workload(avg_workload, var_workload, cluster_len_q, num_qo_tiles, num_clusters);
    const int32_t workload_per_cluster_estimated =
        estimate_workload_per_cluster(tot_workload_estimated, cluster_len_q, num_qo_tiles, num_clusters);

    return ck_tile::integer_least_multiple(workload_per_cluster_estimated, kv_granularity);
}

// Just calculate the final value from features of kv seqlens. All the coefficients comes from linear-regression
// analysis.
CK_TILE_HOST_DEVICE int32_t cal_workload_limit_global_v2(
    const int32_t num_clusters,
    const int32_t num_batches,
    const int32_t avg_workload,
    const int32_t var_workload,
    const int32_t kv_granularity)
{
    const float len = float(num_batches);
    const float len_2 = len * len;
    const float len_3 = len_2 * len;
    const float len_4 = len_2 * len_2;
    const float std = sqrtf(var_workload);
    const float mean = float(avg_workload);
    const float len_mean = len * mean;

    float limit;
    if (num_batches <= (num_clusters / 2))
    {
        constexpr float coef[] =
            { 5.99938779e+01, -8.60897143e-01,  9.98730735e-02, -3.41580286e-03,
              3.20742779e-02, -1.21532871e-03,  2.06024521e-05,  6.86938582e-04,
             -2.06666512e-05,  4.66947341e-05, -2.76209318e+01,  1.71001402e+03,
              1.14275189e+02, -9.70046247e+02, -7.45775345e+01 };
        limit =
            1.0          * coef[0]  + len             * coef[1]  + mean         * coef[2]  + std         * coef[3]  +
            len_2        * coef[4]  + len_mean        * coef[5]  + len * std    * coef[6]  + len_3       * coef[7]  +
            len_4        * coef[8]  + len_2 * mean    * coef[9]  + 1.0f / len   * coef[10] + 1.0f / mean * coef[11] +
            1.0f / len_2 * coef[12] + 1.0f / len_mean * coef[13] + 1.0f / len_3 * coef[14];
    }
    else
    {
        constexpr float coef[] =
            { 1.47512976e+03, -9.29222552e+01,  8.58853904e-03, -5.97940105e-04,
              2.24873043e+00,  2.79397812e-03, -7.44154816e-05, -2.39139141e-02,
              9.52088885e-05,  2.39891509e-06,  1.27530223e+02,  7.68369536e+03,
              6.70734798e+00, -2.30288307e+05,  2.78653564e-01 };
        limit =
            1.0          * coef[0]  + len             * coef[1]  + mean         * coef[2]  + std         * coef[3]  +
            len_2        * coef[4]  + len_mean        * coef[5]  + len * std    * coef[6]  + len_3       * coef[7]  +
            len_4        * coef[8]  + len_2 * mean    * coef[9]  + 1.0f / len   * coef[10] + 1.0f / mean * coef[11] +
            1.0f / len_2 * coef[12] + 1.0f / len_mean * coef[13] + 1.0f / len_3 * coef[14];
    }

    return ck_tile::integer_least_multiple(ck_tile::max(int32_t(limit), kv_granularity), kv_granularity);
}

template <typename Traits, bool kOnlyGatherWorkCount>
CK_TILE_DEVICE void generate_work(
    const int32_t       batch_idx,
    const int32_t       tile_idx,
    const int32_t       qo_len,
    const int32_t       kv_len,
    const int32_t       qo_tile_len,
    const int32_t       packed_qo_tile_len,
    const int32_t       qo_batch_start,
    const int32_t       kv_batch_start,
    const int32_t       kv_batch_end,
    const int32_t       workload_limit_global,
    const int32_t       num_clusters,
    const int32_t       kv_granularity,
    const int32_t*      p_work_indptr,
    const int32_t*      p_num_qo_clusters_indptr,
    int32_t*            p_loc_partial_outputs,
    int32_t*            p_num_partial_outputs,
    MlaWorkInfo*        p_work_info_set,
    MlaPartialTileInfo* p_reduce_final_map,
    MlaPartialTileInfo* p_reduce_partial_map,
    int32_t*            p_cost_heap,
    int32_t*            p_cluster_work_counter)
{
    int32_t remaining_kv_len = kv_len;
    int32_t kv_start_local = 0;

    const int32_t kv_len_limit_floor =
        ck_tile::integer_least_multiple(ck_tile::integer_divide_ceil(kv_len, num_clusters), kv_granularity);
    const auto [cid_top, accum_cost_top] = get_cost_top(p_cost_heap, num_clusters);
    const int32_t remaining_capability_top =
        ck_tile::max(cal_kv_len(workload_limit_global - accum_cost_top, packed_qo_tile_len), kv_len_limit_floor);
    const int32_t num_splits_estimated =
        ck_tile::integer_divide_ceil(remaining_kv_len, remaining_capability_top);
    // For the case of #splits==2, make sure that the tailing tile is smaller than Traits::kSplitTolerance.
    const bool split_kv = (num_splits_estimated == 2) ?
        ((remaining_kv_len - remaining_capability_top) > Traits::kSplitTolerance) :
                                                            (num_splits_estimated > 1);

    do
    {
        // Check and update cost_heap
        auto [cid, accum_cost] = get_cost_top(p_cost_heap, num_clusters);
        const int32_t remaining_capability = cal_kv_len(workload_limit_global - accum_cost, packed_qo_tile_len);
        const int32_t kv_len_limit_local =
        [&]() {
            const int32_t limit_ori = ck_tile::max(remaining_capability, kv_len_limit_floor);
            const int32_t tail_size = (remaining_kv_len > limit_ori) ? (remaining_kv_len - limit_ori) : 0x7fffffff;
            const int32_t limit_fin = (tail_size <= Traits::kSplitTolerance) ? remaining_kv_len : limit_ori;
            return limit_fin;
        }();
        const int32_t kv_len_consuming = ck_tile::min(remaining_kv_len, kv_len_limit_local);

        if (ck_tile::get_lane_id() == 0)
        {
            const int32_t cost = cal_cost(packed_qo_tile_len, kv_len_consuming);
            const int32_t new_cost = accum_cost + cost;
            p_cost_heap[cid] = new_cost;

            if constexpr (kOnlyGatherWorkCount == false)
            {
                // Record work
                MlaWorkInfo work_info{};
                work_info.batch_idx = batch_idx;
                work_info.qo_start  = tile_idx * qo_tile_len + qo_batch_start;
                work_info.qo_end    = ck_tile::min(work_info.qo_start + qo_tile_len, qo_batch_start + qo_len);
                work_info.kv_start  = kv_start_local + kv_batch_start;
                work_info.kv_end    = work_info.kv_start + kv_len_consuming;
                work_info.kv_offset = kv_batch_end - work_info.kv_end;
                if (split_kv)
                {
                    const int32_t global_cluster_q_idx = p_num_qo_clusters_indptr[batch_idx] + tile_idx;
                    work_info.partial_qo_loc = *p_loc_partial_outputs;
                    if (p_reduce_partial_map[global_cluster_q_idx].q_start == -1)
                    {
                        p_reduce_partial_map[global_cluster_q_idx].q_start = *p_loc_partial_outputs;
                        p_reduce_final_map[global_cluster_q_idx] = { work_info.qo_start, work_info.qo_end };
                    }
                    ++(*p_num_partial_outputs);
                    *p_loc_partial_outputs += (work_info.qo_end - work_info.qo_start);
                    p_reduce_partial_map[global_cluster_q_idx].q_end = *p_loc_partial_outputs;
                }
                else
                {
                    work_info.partial_qo_loc = -1;
                }

                const int32_t work_info_set_idx = p_work_indptr[cid] + p_cluster_work_counter[cid];
                p_work_info_set[work_info_set_idx] = work_info;

#if PRINT_DBG
                printf("[metadata] - cost heap updated: work_loc=%d, cid=%d, pre_cost=%d, new_cost=%d, tot_cost=%d, kv_len_cons=%d\n",
                        work_info_set_idx, cid, accum_cost, cost, accum_cost+cost, kv_len_consuming);
#endif
            }

            ++p_cluster_work_counter[cid];
        }

        // Update state
        remaining_kv_len -= kv_len_consuming;
        kv_start_local += kv_len_consuming;
    }
    while (remaining_kv_len > 0);
}

template <typename Traits>
__launch_bounds__(ck_tile::get_warp_size(), 1)
__global__ void kn_get_mla_metadata_v1_1(
    const MlaMetadataV1KernelParameter params)
{
    extern __shared__ uint8_t p_smem[];

    const int32_t lane_idx = ck_tile::get_lane_id();

    // Step.0. Get sequence lengths of query/output and key/value for each batch.
    int32_t* p_batch_idx = reinterpret_cast<int32_t*>(p_smem);
    int32_t* p_qo_lens   = p_batch_idx + params.num_batches;
    int32_t* p_kv_lens   = p_qo_lens + params.num_batches;
    for (int32_t bid = lane_idx; bid < params.num_batches; bid += ck_tile::get_warp_size())
    {
        p_batch_idx[bid] = bid;
        p_qo_lens[bid] = params.p_seqlens_qo_indptr[bid + 1] - params.p_seqlens_qo_indptr[bid];
        p_kv_lens[bid] = params.p_seqlens_kv_indptr[bid + 1] - params.p_seqlens_kv_indptr[bid];
    }

    // Step.1. Calculate the size of cluster and some related information. The size is the number of workgroups
    //         composing each cluster. The size is determined by average packed qo length.
    const int32_t sum_qo_len = warp_sum(p_qo_lens, params.num_batches);
    const int32_t cluster_size =
    [&]() {
        const int32_t avg_qo_len = sum_qo_len / params.num_batches;
        const int32_t cluster_size =
            ck_tile::integer_divide_ceil(avg_qo_len, Traits::kPackedQoLenPerWg);
        return ck_tile::min(cluster_size, Traits::kMaxClusterSize);
    }();
    // assert((params.num_cu % cluster_size) == 0);
    const int32_t num_clusters  = params.num_cu / cluster_size;
    const int32_t cluster_len_q = cluster_size * Traits::kPackedQoLenPerWg;

    // Step.2.
    //   a. Get total valid (after causal masking) kv lengths and the maximun workload handled by each cluster
    //   b. Get a indptr array about #cluster for each batch in direction of qo.
    int32_t* p_num_qo_clusters_indptr = p_kv_lens + params.num_batches;
    if (lane_idx == 0)
    {
        p_num_qo_clusters_indptr[0] = 0;
    }

    int32_t scan_base = 0;
    int32_t workload_sum = 0;
    int64_t workload_square_sum = 0;
    const int32_t num_loop_batch = ck_tile::integer_divide_ceil(params.num_batches, ck_tile::get_warp_size());
    // lds pointed by p_qo_tiles will be reused by p_sort_workspace later
    int32_t* p_qo_tiles  = p_num_qo_clusters_indptr + params.num_batches + 1;
    for (int32_t loop_idx = 0; loop_idx < num_loop_batch; ++loop_idx)
    {
        const int32_t bid = lane_idx + loop_idx * ck_tile::get_warp_size();
        int32_t num_qo_tiles = 0;
        int32_t workload = 0;

        if (bid < params.num_batches)
        {
            const int32_t kv_len = p_kv_lens[bid];
            const int32_t qo_len = p_qo_lens[bid];
            const int32_t packed_qo_len = qo_len * params.num_heads;
            num_qo_tiles = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);
            p_qo_tiles[bid] = num_qo_tiles;
            const int32_t packed_qo_tile_len = ck_tile::min(packed_qo_len, cluster_len_q);

            for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
            {
                const int32_t kv_len_valid =
                    cal_packed_causal_kv_len(
                        qo_len, kv_len, tid, packed_qo_tile_len, num_qo_tiles, params.num_heads, params.is_causal);
                workload += cal_cost(packed_qo_tile_len, kv_len_valid);
            }
        }

        const int32_t prefix_sum_qo_tiles = warp_prefix_sum(num_qo_tiles, ck_tile::get_warp_size());
        const int32_t global_sum_qo_tiles = prefix_sum_qo_tiles + scan_base;
        if (bid < params.num_batches)
        {
            p_num_qo_clusters_indptr[bid + 1] = global_sum_qo_tiles;
        }
        scan_base = ck_tile::warp_shuffle(global_sum_qo_tiles, ck_tile::get_warp_size() - 1);

        workload_sum += aiter::warpReduce<aiter::AddFunctor, decltype(workload), ck_tile::get_warp_size()>(workload);
        workload_square_sum +=
            aiter::warpReduce<aiter::AddFunctor, decltype(workload), ck_tile::get_warp_size()>(workload * workload);
    }
    const int32_t num_qo_tiles = scan_base;
    const int32_t workload_avg = workload_sum / params.num_batches;
    const int32_t workload_var = workload_square_sum / params.num_batches - workload_avg * workload_avg;
    const int32_t tot_qo_tiles = warp_sum(p_qo_tiles, params.num_batches);

    const int32_t workload_limit_global =
        params.num_heads == 16 ?
            cal_workload_limit_global_v1(
                workload_avg, workload_var, cluster_len_q, num_qo_tiles, num_clusters, params.kv_granularity) :
            cal_workload_limit_global_v0(workload_sum, num_clusters, params.kv_granularity);
#if PRINT_DBG
    if (lane_idx == 0)
    {
        printf("[metadata] workload_limit_global=%d\n", workload_limit_global);
    }
#endif

    // Step.3. Sort batch idx based on cost. High cost batch first.
    int32_t *p_sort_workspace = p_num_qo_clusters_indptr + params.num_batches + 1; // will be reused later.
    warp_sort(p_batch_idx, p_sort_workspace, p_qo_lens, p_kv_lens, params.num_batches);

    // Step.4.1. Initialize lds
    int32_t* p_cost_heap = p_sort_workspace;
    int32_t* p_cluster_work_counter = p_cost_heap + num_clusters + 1;
    for (int32_t cid = lane_idx; cid < num_clusters; cid += ck_tile::get_warp_size())
    {
        p_cost_heap[cid] = 0;
        p_cluster_work_counter[cid] = 0;
    }

    // Step.5. Fill the output buffers except indptrs

    // Step.5.1. Get total work for each cluster
    for (int32_t idx = 0; idx < params.num_batches; ++idx)
    {
        const int32_t bid                = p_batch_idx[idx];
        const int32_t qo_batch_start     = params.p_seqlens_qo_indptr[bid];
        const int32_t kv_batch_start     = params.p_seqlens_kv_indptr[bid];
        const int32_t kv_batch_end       = params.p_seqlens_kv_indptr[bid + 1];
        const int32_t kv_len             = kv_batch_end - kv_batch_start;
        const int32_t qo_len             = params.p_seqlens_qo_indptr[bid + 1] - qo_batch_start;
        const int32_t packed_qo_len      = qo_len * params.num_heads;
        const int32_t num_qo_tiles       = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);
        const int32_t packed_qo_tile_len = ck_tile::min(packed_qo_len, cluster_len_q);
        const int32_t qo_tile_len        = ck_tile::integer_divide_ceil(packed_qo_tile_len, params.num_heads);

        for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
        {
            const int32_t tile_kv_len =
                cal_packed_causal_kv_len(
                    qo_len, kv_len, tid, packed_qo_tile_len, num_qo_tiles, params.num_heads, params.is_causal);

            generate_work<Traits, true>(
                bid, tid, qo_len, tile_kv_len, qo_tile_len, packed_qo_tile_len, qo_batch_start, kv_batch_start,
                kv_batch_end, workload_limit_global, num_clusters, params.kv_granularity, nullptr,
                p_num_qo_clusters_indptr, nullptr, nullptr, nullptr, nullptr, nullptr, p_cost_heap,
                p_cluster_work_counter);
        }
    }

    // Step.5.2. Re-init cost heap and cumulative sum cluster_work_tot
    scan_base = 0;
    const int32_t num_loop_clusters = ck_tile::integer_divide_ceil(num_clusters, ck_tile::get_warp_size());
    for (int32_t loop_idx = 0; loop_idx < num_loop_clusters; ++loop_idx)
    {
        const int32_t cid = lane_idx + loop_idx * ck_tile::get_warp_size();

        const int32_t cluster_work = (cid < num_clusters) ? p_cluster_work_counter[cid] : 0;
        const int32_t cum_cluster_work = warp_prefix_sum(cluster_work, ck_tile::get_warp_size()) + scan_base;
        scan_base = ck_tile::warp_shuffle(cum_cluster_work, ck_tile::get_warp_size() - 1);

        if (cid < num_clusters)
        {
            params.p_work_indptr[cid + 1] = cum_cluster_work;
            p_cost_heap[cid] = 0;
            p_cluster_work_counter[cid] = 0;
        }
    }
    if (lane_idx == 0)
    {
        params.p_work_indptr[0] = 0;
    }

    MlaPartialTileInfo* p_reduce_partial_map =
        reinterpret_cast<MlaPartialTileInfo*>(p_cluster_work_counter + num_clusters);
    MlaPartialTileInfo* p_reduce_final_map = p_reduce_partial_map + tot_qo_tiles;
    for (int32_t cluster_q_idx = threadIdx.x; cluster_q_idx < tot_qo_tiles; cluster_q_idx += ck_tile::get_warp_size())
    {
        p_reduce_partial_map[cluster_q_idx] = MlaPartialTileInfo{-1, -2};
        p_reduce_final_map[cluster_q_idx] = MlaPartialTileInfo{-1, -2};
    }

    // Step.5.3. Output work info
    int32_t num_partial_outputs = 0;
    int32_t loc_partial_outputs = 0;
    MlaWorkInfo* p_work_info_set = reinterpret_cast<MlaWorkInfo*>(params.p_work_info_set_raw);
    for (int32_t idx = 0; idx < params.num_batches; ++idx)
    {
        const int32_t bid                = p_batch_idx[idx];
        const int32_t qo_batch_start     = params.p_seqlens_qo_indptr[bid];
        const int32_t kv_batch_start     = params.p_seqlens_kv_indptr[bid];
        const int32_t kv_batch_end       = params.p_seqlens_kv_indptr[bid + 1];
        const int32_t kv_len             = kv_batch_end - kv_batch_start;
        const int32_t qo_len             = params.p_seqlens_qo_indptr[bid + 1] - qo_batch_start;
        const int32_t packed_qo_len      = qo_len * params.num_heads;
        const int32_t num_qo_tiles       = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);
        const int32_t packed_qo_tile_len = ck_tile::min(packed_qo_len, cluster_len_q);
        const int32_t qo_tile_len        = ck_tile::integer_divide_ceil(packed_qo_tile_len, params.num_heads);

#if PRINT_DBG
        if (lane_idx == 0)
        {
            printf("[metadata] Dividing batch=%d, qo_len=%d, kv_len=%d\n", bid, qo_len, kv_len);
        }
#endif

        for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
        {
            const int32_t tile_kv_len =
                cal_packed_causal_kv_len(
                    qo_len, kv_len, tid, packed_qo_tile_len, num_qo_tiles, params.num_heads, params.is_causal);

            generate_work<Traits, false>(
                bid, tid, qo_len, tile_kv_len, qo_tile_len, packed_qo_tile_len, qo_batch_start, kv_batch_start,
                kv_batch_end, workload_limit_global, num_clusters, params.kv_granularity, params.p_work_indptr,
                p_num_qo_clusters_indptr, &loc_partial_outputs, &num_partial_outputs, p_work_info_set,
                p_reduce_final_map, p_reduce_partial_map, p_cost_heap, p_cluster_work_counter);
        }
    }

    // Step.6. Output metadata for reduce kernel
    scan_base = 0;
    const int32_t num_loop_reduce = ck_tile::integer_divide_ceil(tot_qo_tiles, ck_tile::get_warp_size());
    for (int32_t loop_idx = 0; loop_idx < num_loop_reduce; ++loop_idx)
    {
        const int32_t global_cluster_q_idx = lane_idx + loop_idx * ck_tile::get_warp_size();

        MlaPartialTileInfo final_info;
        MlaPartialTileInfo partial_range;
        int32_t reduce_tile_size;
        int32_t num_reduce_tiles = 0;

        if (global_cluster_q_idx < tot_qo_tiles)
        {
            final_info = p_reduce_final_map[global_cluster_q_idx];
            partial_range = p_reduce_partial_map[global_cluster_q_idx];
            reduce_tile_size = (final_info.q_start == -1) ? 0 : (final_info.q_end - final_info.q_start);
            num_reduce_tiles =
                (reduce_tile_size == 0) ? 0 : ((partial_range.q_end - partial_range.q_start) / reduce_tile_size);
        }

        const int32_t curr_cum_reduce_tiles = warp_prefix_sum(num_reduce_tiles, ck_tile::get_warp_size()) + scan_base;
        const int32_t prev_cum_reduce_tiles = curr_cum_reduce_tiles - num_reduce_tiles;
        scan_base = ck_tile::warp_shuffle(curr_cum_reduce_tiles, ck_tile::get_warp_size() - 1);

        if (global_cluster_q_idx < tot_qo_tiles)
        {
            for (int32_t tid = prev_cum_reduce_tiles; tid < curr_cum_reduce_tiles; ++tid)
            {
                const int32_t local_tid = tid - prev_cum_reduce_tiles;
                params.p_reduce_partial_map[tid] = partial_range.q_start + local_tid * reduce_tile_size;
            }

            params.p_reduce_indptr[global_cluster_q_idx + 1] = curr_cum_reduce_tiles;
            params.p_reduce_final_map[2 * global_cluster_q_idx] = final_info.q_start;
            params.p_reduce_final_map[2 * global_cluster_q_idx + 1] = final_info.q_end;
        }
    }

    // reduce_indptr may be larger than #clusters.
    const int32_t num_reduce_tiles = scan_base;
    for (int32_t idx = tot_qo_tiles + 1 + lane_idx; idx < params.reduce_indptr_size; idx += ck_tile::get_warp_size())
    {
        params.p_reduce_indptr[idx] = num_reduce_tiles;
    }

    // Step.7. Fill metadata pointers for MLA kernel and the 1st element of reduce_indptr.
    if (lane_idx == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_metadata_ptrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
        params.p_work_metadata_ptrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_info_set_raw));
    }

#if PRINT_DBG
    if (lane_idx == 0)
    {
        printf("[metadata] Final Cost Heap Status:\n");
        for (int32_t cid = 0; cid < num_clusters; ++cid)
        {
            printf("[metadata] - cid=%d, cost=%d\n", cid, p_cost_heap[cid]);
        }
    }
#endif
}

template <typename Traits>
void get_mla_metadata_v1_1_device(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const bool           no_redundant,
    const int32_t        kv_granularity,
    const int32_t        max_seqlen_qo,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_info_set,
    torch::Tensor&       work_indptr,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t num_batches = seqlens_qo_indptr.size(0) - 1;
    const int32_t num_cu      = dev_prop.multiProcessorCount;
    const int32_t num_heads   = num_heads_per_head_k * num_heads_k;

    const int32_t lds_size_in_bytes = [&]()
    {
        const int32_t qo_tile_per_batch =
            ck_tile::integer_divide_ceil(ck_tile::max(max_seqlen_qo, 1) * num_heads, Traits::kPackedQoLenPerWg);
        const int32_t tot_qo_tiles      = num_batches * qo_tile_per_batch;
        // this is maximun #clusters
        const int32_t num_clusters = dev_prop.multiProcessorCount;

        int32_t lds_size = 0;

        // Stores batch_id, qo_len and kv_len
        lds_size += 3 * num_batches * sizeof(int32_t);
        // Memory for indptr about #cluster for each batch in direction of qo
        lds_size += (num_batches + 1) * sizeof(int32_t);
        // LDS for sorting
        const int32_t power_2_num_batches = (num_batches <= 1) ? num_batches : ck_tile::next_power_of_two(num_batches);
        const int32_t lds_sort_size =
            lds_size +
            ck_tile::integer_least_multiple(power_2_num_batches, ck_tile::get_warp_size()) * 2 * sizeof(int32_t);
        // Memory for cost. Its size should be the same as #clusters
        lds_size += num_clusters * sizeof(int32_t);
        // Memory for counter of #works for each cluster.
        lds_size += num_clusters * sizeof(int32_t);
        // Memory for range of partial memory
        lds_size += tot_qo_tiles * sizeof(MlaPartialTileInfo);
        // Memory for range of output of partial memory
        lds_size += tot_qo_tiles * sizeof(MlaPartialTileInfo);

        return ck_tile::max(lds_size, lds_sort_size);
    }();

    TORCH_CHECK(lds_size_in_bytes <= dev_prop.maxSharedMemoryPerMultiProcessor,
                __func__, ": There is no enough LDS.");

    // auto opts = seqlens_kv_indptr.options();
    // auto work_ptrs          = torch::empty({2}, opts.dtype(torch::kUInt64));
    // auto work_indptr        = torch::empty({num_cu + 1}, opts);
    // auto work_info_set      = torch::empty({max_works, kSizeMlaWorkInfoInDw}, opts);
    // auto reduce_indptr      = torch::empty({max_qo_tiles + 1}, opts);
    // auto reduce_final_map   = torch::empty({max_qo_tiles, kSizeMlaPartialTileInfoInDw}, opts);
    // auto reduce_partial_map = torch::empty({max_works}, opts);

    // kernel input parameters
    MlaMetadataV1KernelParameter params = {};
    params.p_work_metadata_ptrs = work_metadata_ptrs.data_ptr<uint64_t>();
    params.p_work_indptr        = work_indptr.data_ptr<int32_t>();
    params.p_work_info_set_raw  = work_info_set.data_ptr<int32_t>();
    params.p_reduce_indptr      = reduce_indptr.data_ptr<int32_t>();
    params.p_reduce_final_map   = reduce_final_map.data_ptr<int32_t>();
    params.p_reduce_partial_map = reduce_partial_map.data_ptr<int32_t>();
    params.p_seqlens_qo_indptr  = seqlens_qo_indptr.data_ptr<int32_t>();
    params.p_seqlens_kv_indptr  = seqlens_kv_indptr.data_ptr<int32_t>();
    params.num_batches          = num_batches;
    params.num_heads            = num_heads;
    params.num_cu               = num_cu;
    params.reduce_indptr_size   = reduce_indptr.size(0);
    params.kv_granularity       = kv_granularity;
    params.is_causal            = is_causal;

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity
    kn_get_mla_metadata_v1_1<Traits><<<grid, num_thr, dev_prop.maxSharedMemoryPerMultiProcessor, stream>>>(params);
}
