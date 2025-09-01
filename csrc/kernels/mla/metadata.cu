// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <queue>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "custom_all_reduce.cuh"
#include "mla.h"


#define PRINT_DBG 0

// ===================================================================================================================
// MLA Metadata V0
// ===================================================================================================================

CK_TILE_HOST_DEVICE float get_overhead(
    const int32_t num_cu,
    const int32_t num_batches,
    const int32_t seqlen,
    const int32_t num_splits)
{
    constexpr float kSplitOverhead = 84.1f;

    const float bs_ratio = float(num_batches * num_splits) /
                           float((num_batches * num_splits + num_cu - 1) / num_cu) * float(num_cu);
    const float sq_ratio = float(seqlen) / float(seqlen + kSplitOverhead * num_splits);
    const float overhead = bs_ratio * sq_ratio;

    return overhead;
}

__launch_bounds__(ck_tile::get_warp_size())
__global__ void kn_get_mla_metadata_v0(
    int32_t*       p_num_kv_splits,
    int32_t*       p_max_num_splits,
    const int32_t* p_seqlens,
    const int32_t  num_cu,
    const int32_t  num_batches,
    const int32_t  num_heads_per_head_k,
    const int32_t  num_heads_k)
{
    constexpr int32_t kMaxSplits = 16;
    constexpr int32_t kWarpSize  = ck_tile::get_warp_size();

    int32_t base_scan  = 0;
    int32_t max_splits = 1;

    const int32_t num_loops = ck_tile::integer_divide_ceil(num_batches, kWarpSize);
    for (int32_t i = 0; i < num_loops; ++i)
    {
        const int32_t seqlen_idx = threadIdx.x + i * kWarpSize;
        int32_t splits = 0;

        if (seqlen_idx < num_batches)
        {
            const int32_t seqlen = p_seqlens[seqlen_idx + 1] - p_seqlens[seqlen_idx];
            float min_overhead   = std::numeric_limits<float>::max();
            #pragma unroll
            for (int32_t test_splits = 1; test_splits <= kMaxSplits; ++test_splits)
            {
                const float overhead = get_overhead(num_cu, num_batches, seqlen, test_splits);
                if (overhead < min_overhead)
                {
                    min_overhead = overhead;
                    splits = test_splits;
                }
            }

            max_splits = (max_splits > splits) ? max_splits : splits;
        }

        // prefix sum
        int32_t scan = splits;
        #pragma unroll
        for (int32_t offset = 1; offset <= (kWarpSize >> 1) ; offset *= 2)
        {
            const int32_t remote = ck_tile::warp_shuffle_up(scan, offset);
            scan += (threadIdx.x >= offset) ? remote : 0;
        }

        const int32_t global_scan = scan + base_scan;

        if (seqlen_idx < num_batches)
        {
            p_num_kv_splits[seqlen_idx + 1] = global_scan;
        }

        // update base_scan
        base_scan = ck_tile::warp_shuffle(global_scan, kWarpSize - 1);
    }

    // Reduce max_num_split
    for (int32_t mask = (kWarpSize >> 1); mask > 0; mask >>= 1)
    {
        const int32_t remote_max = __shfl_xor(max_splits, mask);
        max_splits = (max_splits > remote_max) ? max_splits : remote_max;
    }

    if (threadIdx.x == 0)
    {
        p_num_kv_splits[0] = 0;
        p_max_num_splits[0] = max_splits;
    }
}

//
// Get per batch kv split count for ASM MLA without persistent thread
// group support.
//
// Returns
//   [0] num_kv_splits:  (num_batches + 1), dtype torch.int32.
//   [1] max_num_splits: (1), dtype torch.int32.
//
std::vector<torch::Tensor> get_mla_metadata_v0(
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k)
{
    TORCH_CHECK(seqlens_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    TORCH_CHECK(seqlens_kv_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_kv_indptr's element type should be int!");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(seqlens_kv_indptr));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t num_batches = seqlens_kv_indptr.size(0) - 1;

    // declare outputs
    auto num_kv_splits = torch::empty({num_batches + 1}, seqlens_kv_indptr.options());
    auto max_num_splits = torch::empty({1}, seqlens_kv_indptr.options());

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity
    kn_get_mla_metadata_v0<<<grid, num_thr, 0, stream>>>(
        num_kv_splits.data_ptr<int32_t>(),
        max_num_splits.data_ptr<int32_t>(),
        seqlens_kv_indptr.data_ptr<int32_t>(),
        dev_prop.multiProcessorCount,
        num_batches,
        num_heads_per_head_k,
        num_heads_k);

    return {num_kv_splits, max_num_splits};
}

// ===================================================================================================================
// MLA Metadata V1
// ===================================================================================================================

CK_TILE_HOST_DEVICE int32_t cal_packed_causal_kv_len(
    const int32_t qo_len,
    const int32_t kv_len,
    const int32_t qo_tile_idx,
    const int32_t block_size_q,
    const int32_t num_qo_tiles,
    const int32_t num_heads,
    const bool    is_causal)
{
    int result = kv_len;

    if (is_causal && (qo_tile_idx < num_qo_tiles))
    {
        const int kv_len_init = kv_len - qo_len;
        const int kv_len_slop = ck_tile::integer_divide_ceil((qo_tile_idx + 1) * block_size_q, num_heads);
        result = ck_tile::min(kv_len_init + kv_len_slop, kv_len);
    }

    return result;
}

CK_TILE_HOST_DEVICE int32_t cal_cost(
    const int32_t qo_len,
    const int32_t kv_len)
{
    return 2 * qo_len + kv_len;
}

CK_TILE_HOST_DEVICE int32_t cal_kv_len(
    const int32_t cost,
    const int32_t qo_len)
{
    return cost - 2 * qo_len;
}

CK_TILE_HOST_DEVICE int32_t get_remaining_kv_capability(
    const int32_t kv_len_upper_bound,
    const int32_t kv_len_used)
{
    return ck_tile::integer_least_multiple(kv_len_upper_bound - kv_len_used, 16);
}

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

// Warp level customized bitonic sort for sorting batch idx based on cost. High cost first.
CK_TILE_DEVICE void warp_sort(
    int32_t*       p_batch_idx,
    int32_t*       p_workspace,
    const int32_t* p_qo_lens,
    const int32_t* p_kv_lens,
    const int32_t  num_batches)
{
    const int32_t lane_idx = ck_tile::get_lane_id();

    const int32_t num_batches_padded =
        ck_tile::integer_least_multiple(ck_tile::next_power_of_two(num_batches), ck_tile::get_warp_size());
    const int32_t warp_loops = num_batches_padded / ck_tile::get_warp_size();
    int32_t* p_costs = p_workspace;
    int32_t* p_indices = p_costs + num_batches_padded;

    auto check_and_swap = [&](const int32_t idx0, const int32_t idx1, const bool dir) {
        const int32_t cost0 = p_costs[idx0];
        const int32_t cost1 = p_costs[idx1];
        if ((cost0 > cost1) == dir)
        {
            int32_t temp_idx = p_indices[idx0];
            p_indices[idx0] = p_indices[idx1];
            p_indices[idx1] = temp_idx;
            p_costs[idx1] = cost0;
            p_costs[idx0] = cost1;
        }
    };

    // Initialize smem
    // Pre-calculate cost for each batch
    for (int32_t bid = lane_idx; bid < num_batches; bid += ck_tile::get_warp_size())
    {
        p_costs[bid] = cal_cost(p_qo_lens[bid], p_kv_lens[bid]);
        p_indices[bid] = bid;
    }
    for (int32_t bid = lane_idx + num_batches; bid < num_batches_padded; bid += ck_tile::get_warp_size())
    {
        p_costs[bid] = 0;
        p_indices[bid] = bid;
    }

    for (int32_t size = 2; size < num_batches_padded; size <<= 1)
    {
        const int32_t max_stride = size >> 1;
        for (int32_t loop_idx = 0; loop_idx < warp_loops; ++loop_idx)
        {
            const int32_t thr_idx = lane_idx + loop_idx * ck_tile::get_warp_size();
            if (thr_idx * 2 < num_batches_padded)
            {
                const bool dir = ((thr_idx & max_stride) == 0);
                for (int32_t stride = max_stride; stride > 0; stride >>= 1)
                {
                    const int32_t stride_m1 = stride - 1;
                    const int32_t idx = 2 * thr_idx - (thr_idx & stride_m1);
                    check_and_swap(idx, idx + stride, dir);
                }
            }
        }
    }

    for (int32_t stride = num_batches_padded >> 1; stride > 0; stride >>= 1)
    {
        const int32_t stride_m1 = stride - 1;
        for (int32_t loop_idx = 0; loop_idx < warp_loops; ++loop_idx)
        {
            const int32_t thr_idx = lane_idx + loop_idx * ck_tile::get_warp_size();
            if (thr_idx * 2 < num_batches_padded)
            {
                const int32_t idx = 2 * thr_idx - (thr_idx & stride_m1);
                check_and_swap(idx, idx + stride, false);
            }
        }
    }

    // Output results
    for (int32_t bid = lane_idx; bid < num_batches; bid += ck_tile::get_warp_size())
    {
        p_batch_idx[bid] = p_indices[bid];
    }
}

template <typename T>
std::vector<T> flatten(
    const std::vector<std::vector<T>>& vec,
    const int size_after_flatten)
{
    std::vector<T> result;
    result.reserve(size_after_flatten);

    for (const auto& inner_vec : vec)
    {
        result.insert(result.end(), inner_vec.begin(), inner_vec.end());
    }

    return result;
}

struct BatchInfo
{
    int32_t batch_idx;
    int32_t qo_len;
    int32_t kv_len;

    int32_t get_cost() const
    {
        return cal_cost(qo_len, kv_len);
    }

    bool operator > (const BatchInfo& rhs) const
    {
        return get_cost() > rhs.get_cost();
    }
};

template<int32_t kPackedQoLenPerWg_,
         int32_t kMaxClusterSize_>
struct MlaMetadataTraits
{
    static constexpr int32_t kPackedQoLenPerWg  = kPackedQoLenPerWg_;
    static constexpr int32_t kMaxClusterSize    = kMaxClusterSize_;
    static constexpr int32_t kSplitTolerance    = 16;
    static constexpr int32_t kMaxQoTilePerBatch = 1;
};

struct MlaMetadataV1KernelParameter
{
    // Outputs
    uint64_t* p_work_metadata_ptrs;
    int32_t*  p_work_indptr;
    int32_t*  p_work_info_set_raw;
    int32_t*  p_reduce_indptr;
    int32_t*  p_reduce_final_map;
    int32_t*  p_reduce_partial_map;

    // Inputs
    const int32_t* p_seqlens_qo_indptr;
    const int32_t* p_seqlens_kv_indptr;
    int32_t        num_batches;
    int32_t        num_heads;
    int32_t        num_cu;
    int32_t        reduce_indptr_size;
    bool           is_causal;
};

CK_TILE_DEVICE int32_t estimate_total_workload(
    const int32_t workload_avg,
    const int32_t workload_var,
    const int32_t cluster_len_q,
    const int32_t num_qo_tiles,
    const int32_t num_clusters)
{
    const int32_t qo_factor = cluster_len_q >> 2;
    const float cv = sqrtf(float(workload_var)) / float(workload_avg); // coefficient of variation
    const int32_t A = int32_t(12.0f - (2.0f * cv + 1.0f)) * qo_factor;

    const int32_t workload_tot = (workload_avg + A) * num_qo_tiles;

    return workload_tot;
}

CK_TILE_DEVICE int32_t estimate_workload_per_cluster(
    const int32_t workload_tot,
    const int32_t cluster_len_q,
    const int32_t num_qo_tiles,
    const int32_t num_clusters)
{
    // The following magic numbers come from regression analysis. They may need to be changed with platform and MLA
    // kernels.
    const float workload_per_cluster_raw = float(workload_tot) / float(num_clusters);
    const float factor = -0.00000113f * powf(num_qo_tiles, 3) +
                          0.00025215f * powf(num_qo_tiles, 2) +
                         -0.00546401f * float(num_qo_tiles) +
                          0.40612956f;
    const float amplifier = ck_tile::max(factor * num_clusters / num_qo_tiles, 1.0f);
    const int32_t workload_per_cluster = int32_t(workload_per_cluster_raw * amplifier);

    return workload_per_cluster;
}

template <typename Traits, bool kGatherWorkCount>
CK_TILE_DEVICE void generate_work(
    const int32_t       batch_idx,
    const int32_t       tile_idx,
    const int32_t       qo_len,
    const int32_t       kv_len,
    const int32_t       cluster_len_q,
    const int32_t       qo_batch_start,
    const int32_t       kv_batch_start,
    const int32_t       kv_batch_end,
    const int32_t       workload_limit_global,
    const int32_t       num_clusters,
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
        ck_tile::integer_least_multiple(ck_tile::integer_divide_ceil(kv_len, num_clusters), 16);
    const auto [cid_top, accum_cost_top] = get_cost_top(p_cost_heap, num_clusters);
    const int32_t remaining_capability_top =
        ck_tile::max(cal_kv_len(workload_limit_global - accum_cost_top, cluster_len_q), kv_len_limit_floor);
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
        const int32_t remaining_capability = cal_kv_len(workload_limit_global - accum_cost, cluster_len_q);
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
            const int32_t cost = cal_cost(cluster_len_q, kv_len_consuming);
            const int32_t new_cost = accum_cost + cost;
            p_cost_heap[cid] = new_cost;

            if constexpr (kGatherWorkCount == false)
            {
                // Record work
                MlaWorkInfo work_info{};
                work_info.bs_index  = batch_idx;
                work_info.q_start   = tile_idx * cluster_len_q + qo_batch_start;
                work_info.q_end     = ck_tile::min(work_info.q_start + cluster_len_q, qo_batch_start + qo_len);
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
                        p_reduce_final_map[global_cluster_q_idx] = { work_info.q_start, work_info.q_end };
                    }
                    ++(*p_num_partial_outputs);
                    *p_loc_partial_outputs += (work_info.q_end - work_info.q_start);
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

template <typename T>
CK_TILE_DEVICE T warp_sum(const T* p_data, const int32_t size)
{
    T sum = T(0);

    for (int32_t idx = ck_tile::get_lane_id(); idx < size; idx += ck_tile::get_warp_size())
    {
        sum += p_data[idx];
    }

    sum = aiter::warpReduce<aiter::AddFunctor, T, ck_tile::get_warp_size()>(sum);

    return sum;
}

template <typename T>
CK_TILE_DEVICE T warp_prefix_sum(T value, const int32_t size)
{
    // Always assume that size is power of 2
    #pragma unroll
    for (int32_t offset = 1; offset <= (ck_tile::get_warp_size() >> 1) ; offset *= 2)
    {
        const T remote = ck_tile::warp_shuffle_up(value, offset);
        value += (ck_tile::get_lane_id() >= offset) ? remote : 0;
    }
    return value;
}

template <typename Traits>
__launch_bounds__(ck_tile::get_warp_size(), 1)
__global__ void kn_get_mla_metadata_v1(
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
    int32_t sum_packed_qo_len = warp_sum(p_qo_lens, params.num_batches);
    const int32_t cluster_size =
    [&]() {
        const int32_t avg_packed_qo_len = sum_packed_qo_len / params.num_batches;
        const int32_t cluster_size =
            ck_tile::integer_divide_ceil(avg_packed_qo_len, Traits::kPackedQoLenPerWg);
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

            for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
            {
                const int32_t kv_len_valid =
                    cal_packed_causal_kv_len(
                        qo_len, kv_len, tid, cluster_len_q, num_qo_tiles, params.num_heads, params.is_causal);
                workload += cal_cost(cluster_len_q, kv_len_valid);
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

    const int32_t workload_limit_global =
    [&]() {
        const int32_t tot_workload_estimated =
            estimate_total_workload(workload_avg, workload_var, cluster_len_q, num_qo_tiles, num_clusters);
        const int32_t workload_per_cluster_estimated =
            estimate_workload_per_cluster(tot_workload_estimated, cluster_len_q, num_qo_tiles, num_clusters);
        const int32_t limit = ck_tile::integer_least_multiple(workload_per_cluster_estimated, 16);
        return limit;
    }();
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
    const int32_t max_qo_tiles = params.num_batches * Traits::kMaxQoTilePerBatch;

    // Step.5.1. Get total work for each cluster
    for (int32_t idx = 0; idx < params.num_batches; ++idx)
    {
        const int32_t bid            = p_batch_idx[idx];
        const int32_t qo_batch_start = params.p_seqlens_qo_indptr[bid];
        const int32_t kv_batch_start = params.p_seqlens_kv_indptr[bid];
        const int32_t kv_batch_end   = params.p_seqlens_kv_indptr[bid + 1];
        const int32_t kv_len         = kv_batch_end - kv_batch_start;
        const int32_t qo_len         = params.p_seqlens_qo_indptr[bid + 1] - qo_batch_start;
        const int32_t packed_qo_len  = qo_len * params.num_heads;
        const int32_t num_qo_tiles   = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);

        for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
        {
            const int32_t tile_kv_len =
                cal_packed_causal_kv_len(
                    qo_len, kv_len, tid, cluster_len_q, num_qo_tiles, params.num_heads, params.is_causal);

            generate_work<Traits, true>(
                bid, tid, qo_len, tile_kv_len, cluster_len_q, qo_batch_start, kv_batch_start, kv_batch_end,
                workload_limit_global, num_clusters, nullptr, p_num_qo_clusters_indptr,
                nullptr, nullptr, nullptr, nullptr, nullptr, p_cost_heap, p_cluster_work_counter);
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
    MlaPartialTileInfo* p_reduce_final_map = p_reduce_partial_map + max_qo_tiles;
    for (int32_t cluster_q_idx = threadIdx.x; cluster_q_idx < max_qo_tiles; cluster_q_idx += ck_tile::get_warp_size())
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
        const int32_t bid            = p_batch_idx[idx];
        const int32_t qo_batch_start = params.p_seqlens_qo_indptr[bid];
        const int32_t kv_batch_start = params.p_seqlens_kv_indptr[bid];
        const int32_t kv_batch_end   = params.p_seqlens_kv_indptr[bid + 1];
        const int32_t kv_len         = kv_batch_end - kv_batch_start;
        const int32_t qo_len         = params.p_seqlens_qo_indptr[bid + 1] - qo_batch_start;
        const int32_t packed_qo_len  = qo_len * params.num_heads;
        const int32_t num_qo_tiles   = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);

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
                    qo_len, kv_len, tid, cluster_len_q, num_qo_tiles, params.num_heads, params.is_causal);

            generate_work<Traits, false>(
                bid, tid, qo_len, tile_kv_len, cluster_len_q, qo_batch_start, kv_batch_start, kv_batch_end,
                workload_limit_global, num_clusters, params.p_work_indptr, p_num_qo_clusters_indptr,
                &loc_partial_outputs, &num_partial_outputs, p_work_info_set, p_reduce_final_map, p_reduce_partial_map,
                p_cost_heap, p_cluster_work_counter);
        }
    }

    // Step.6. Output metadata for reduce kernel
    scan_base = 0;
    const int32_t num_loop_reduce = ck_tile::integer_divide_ceil(max_qo_tiles, ck_tile::get_warp_size());
    for (int32_t loop_idx = 0; loop_idx < num_loop_reduce; ++loop_idx)
    {
        const int32_t global_cluster_q_idx = lane_idx + loop_idx * ck_tile::get_warp_size();

        MlaPartialTileInfo final_info;
        MlaPartialTileInfo partial_range;
        int32_t reduce_tile_size;
        int32_t num_reduce_tiles = 0;

        if (global_cluster_q_idx < max_qo_tiles)
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

        if (global_cluster_q_idx < max_qo_tiles)
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
    for (int32_t idx = max_qo_tiles + 1 + lane_idx; idx < params.reduce_indptr_size; idx += ck_tile::get_warp_size())
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
void get_mla_metadata_v1_device(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const bool           no_redundant,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_info_set,
    torch::Tensor&       work_indptr,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map)
{
    TORCH_CHECK(seqlens_qo_indptr.stride(0) == 1,
                __func__, ": seqlens_qo_indptr should be continuous!");
    TORCH_CHECK(seqlens_qo_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_qo_indptr's element type should be int!");
    TORCH_CHECK(seqlens_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    TORCH_CHECK(seqlens_kv_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_kv_indptr's element type should be int!");

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto opts = seqlens_kv_indptr.options();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t num_batches  = seqlens_qo_indptr.size(0) - 1;
    const int32_t num_cu       = dev_prop.multiProcessorCount;
    const int32_t max_works    = num_cu + num_batches * Traits::kMaxQoTilePerBatch;
    const int32_t max_qo_tiles = num_batches * Traits::kMaxQoTilePerBatch;

    const int32_t lds_size_in_bytes = [&]()
    {
        int32_t lds_size = 0;

        // this is maximun #clusters
        const int32_t num_clusters = dev_prop.multiProcessorCount;
        // Stores batch_id, qo_len and kv_len
        lds_size += 3 * num_batches * sizeof(int32_t);
        // Memory for indptr about #cluster for each batch in direction of qo
        lds_size += (num_batches + 1) * sizeof(int32_t);
        // LDS for sorting
        const int32_t lds_sort_size =
            lds_size +
            ck_tile::integer_least_multiple(ck_tile::next_power_of_two(num_batches),
                                            ck_tile::get_warp_size()) * 2 * sizeof(int32_t);
        // Memory for cost. Its size should be the same as #clusters
        lds_size += num_clusters * sizeof(int32_t);
        // Memory for counter of #works for each cluster.
        lds_size += num_clusters * sizeof(int32_t);
        // Memory for range of partial memory
        lds_size += max_qo_tiles * sizeof(MlaPartialTileInfo);
        // Memory for range of output of partial memory
        lds_size += max_qo_tiles * sizeof(MlaPartialTileInfo);

        return ck_tile::max(lds_size, lds_sort_size);
    }();

    TORCH_CHECK(lds_size_in_bytes <= dev_prop.maxSharedMemoryPerMultiProcessor,
                __func__, ": There is no enough LDS.");

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
    params.num_heads            = num_heads_per_head_k * num_heads_k;
    params.num_cu               = num_cu;
    params.reduce_indptr_size   = reduce_indptr.size(0);
    params.is_causal            = is_causal;

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity
    kn_get_mla_metadata_v1<Traits><<<grid, num_thr, dev_prop.maxSharedMemoryPerMultiProcessor, stream>>>(params);
}

template <typename Traits>
std::vector<torch::Tensor> get_mla_metadata_v1_host(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
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
        const int32_t packed_qo_len = binfo.qo_len * num_heads;
        const int32_t num_qo_tiles  = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);

        num_qo_clusters_indptr.push_back(num_qo_clusters_indptr.back() + num_qo_tiles);

        for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
        {
            const int32_t kv_len_valid =
                cal_packed_causal_kv_len(binfo.qo_len, binfo.kv_len, tid, cluster_len_q, num_qo_tiles, num_heads, is_causal);
            // always assume that each batch of tile will be splited once along kv.
            const int32_t kv_len_splited =
                ck_tile::integer_least_multiple(ck_tile::integer_divide_ceil(kv_len_valid, 2), 16);
            workload_sum += 2 * cal_cost(cluster_len_q, kv_len_splited) + 16;
        }
    }
    const int32_t workload_limit_global =
    [&]() {
        const int32_t avg_workload = ck_tile::max(ck_tile::integer_divide_ceil(workload_sum, num_clusters), 1);
        // TODO: The following code just follow FlashInfer. Further tune may be required for AMD GPU.
        int32_t limit;
        if (avg_workload <= 8) limit = 32;
        else if (avg_workload <= 16) limit = 64;
        else if (avg_workload <= 32) limit = 128;
        else if (avg_workload <= 64) limit = 192;
        else limit = ck_tile::integer_least_multiple(avg_workload, 16);
        return limit;
    }();
#if PRINT_DBG
    printf("[metadata] workload_limit_global=%d\n", workload_limit_global);
#endif

    // Step.3.1. Allocates output buffers except indptrs
    std::vector<std::vector<MlaWorkInfo>> work_info_set(num_clusters, std::vector<MlaWorkInfo>());
    std::vector<std::vector<index_t>> reduce_partial_map(num_qo_clusters_indptr.back(), std::vector<index_t>());
    std::vector<MlaPartialTileInfo> reduce_partial_info(num_qo_clusters_indptr.back(), {-1, -2});

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
                ck_tile::integer_least_multiple(ck_tile::integer_divide_ceil(kv_len, num_clusters), 16);

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
                work_info.bs_index  = bid;
                work_info.q_start   = tid * cluster_len_q + qo_batch_start;
                work_info.q_end     = ck_tile::min(work_info.q_start + cluster_len_q, qo_batch_start + qo_len);
                work_info.kv_start  = kv_start_local + kv_batch_start;
                work_info.kv_end    = work_info.kv_start + kv_len_consuming;
                work_info.kv_offset = kv_batch_end - work_info.kv_end;
                if (split_kv)
                {
                    work_info.partial_qo_loc = loc_partial_outputs;
                    if (reduce_partial_map[global_cluster_q_idx].empty())
                    {
                        ++num_reduce_row;
                        reduce_partial_info[global_cluster_q_idx] = { work_info.q_start, work_info.q_end };
                    }
                    reduce_partial_map[global_cluster_q_idx].push_back(loc_partial_outputs);
                    ++num_partial_outputs;
                    loc_partial_outputs += (work_info.q_end - work_info.q_start);
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
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_info_set,
    torch::Tensor&       work_indptr,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(seqlens_kv_indptr));

    // This default settings is for our ASM MLA decode kernel. This kernel supports num_heads=16 and qo size from 1 to 4
    // without support to split qo for each workgroup. This means that kPackedQoLenPerWg should be 4*16=64 to prevent
    // spliting in any case supported by it.
    //                                PackedQoLenPerWg, MaxClusterSize
    using Traits  = MlaMetadataTraits<64,               1>;

    get_mla_metadata_v1_device<Traits>(
        seqlens_qo_indptr,
        seqlens_kv_indptr,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        false,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map);
}

std::vector<torch::Tensor> get_mla_metadata_v1_no_redundant(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(seqlens_kv_indptr));

    // This default settings is for our ASM MLA decode kernel. This kernel supports num_heads=16 and qo size from 1 to 4
    // without support to split qo for each workgroup. This means that kPackedQoLenPerWg should be 4*16=64 to prevent
    // spliting in any case supported by it.
    //                                PackedQoLenPerWg, MaxClusterSize
    using Traits  = MlaMetadataTraits<64,               1>;

    return get_mla_metadata_v1_host<Traits>(
        seqlens_qo_indptr,
        seqlens_kv_indptr,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        true);
}