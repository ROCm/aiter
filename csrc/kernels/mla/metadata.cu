// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <queue>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "mla.h"


#define PRINT_DBG 0


__device__ constexpr int32_t get_warp_size()
{
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
    return 64;
#else
    return 32;
#endif
}

__device__ float get_overhead(
    const int32_t num_cu,
    const int32_t batch_size,
    const int32_t seqlen,
    const int32_t num_splits)
{
    constexpr float kSplitOverhead = 84.1f;

    const float bs_ratio = float(batch_size * num_splits) /
                           float((batch_size * num_splits + num_cu - 1) / num_cu) * float(num_cu);
    const float sq_ratio = float(seqlen) / float(seqlen + kSplitOverhead * num_splits);
    const float overhead = bs_ratio * sq_ratio;

    return overhead;
}

__launch_bounds__(get_warp_size())
__global__ void kn_get_mla_metadata_v0(
    int32_t*       p_num_kv_splits,
    int32_t*       p_max_num_splits,
    const int32_t* p_seqlens,
    const int32_t  num_cu,
    const int32_t  batch_size,
    const int32_t  num_heads_per_head_k,
    const int32_t  num_heads_k)
{
    constexpr int32_t kMaxSplits = 16;
    constexpr int32_t kWarpSize  = get_warp_size();

    int32_t base_scan  = 0;
    int32_t max_splits = 1;

    const int32_t num_loops = (batch_size + kWarpSize - 1) / kWarpSize;
    for (int32_t i = 0; i < num_loops; ++i)
    {
        const int32_t seqlen_idx = threadIdx.x + i * kWarpSize;
        int32_t splits = 0;

        if (seqlen_idx < batch_size)
        {
            const int32_t seqlen = p_seqlens[seqlen_idx + 1] - p_seqlens[seqlen_idx];
            float min_overhead   = std::numeric_limits<float>::max();
            #pragma unroll
            for (int32_t test_splits = 1; test_splits <= kMaxSplits; ++test_splits)
            {
                const float overhead = get_overhead(num_cu, batch_size, seqlen, test_splits);
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
            const int32_t remote = __shfl_up(scan, offset);
            scan += (threadIdx.x >= offset) ? remote : 0;
        }

        const int32_t global_scan = scan + base_scan;

        if (seqlen_idx < batch_size)
        {
            p_num_kv_splits[seqlen_idx + 1] = global_scan;
        }

        // update base_scan
        base_scan = __shfl(global_scan, kWarpSize - 1);
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
//   [0] num_kv_splits: (batch_size + 1), dtype torch.int32.
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

    const int32_t batch_size = seqlens_kv_indptr.size(0) - 1;

    // declare outputs
    auto num_kv_splits = torch::empty({batch_size + 1}, seqlens_kv_indptr.options());
    auto max_num_splits = torch::empty({1}, seqlens_kv_indptr.options());

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity
    kn_get_mla_metadata_v0<<<grid, num_thr, 0, stream>>>(
        num_kv_splits.data_ptr<int32_t>(),
        max_num_splits.data_ptr<int32_t>(),
        seqlens_kv_indptr.data_ptr<int32_t>(),
        dev_prop.multiProcessorCount,
        batch_size,
        num_heads_per_head_k,
        num_heads_k);

    return {num_kv_splits, max_num_splits};
}

inline int32_t cal_packed_causal_kv_len(
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

inline int32_t cal_cost(
    const int32_t qo_len,
    const int32_t kv_len)
{
    return 2 * qo_len + kv_len;
}

inline int32_t cal_kv_len(
    const int32_t cost,
    const int32_t qo_len)
{
    return cost - 2 * qo_len;
}

inline int32_t get_remaining_kv_capability(
    const int32_t kv_len_upper_bound,
    const int32_t kv_len_used)
{
    return ck_tile::integer_least_multiple(kv_len_upper_bound - kv_len_used, 16);
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

    bool operator > (const BatchInfo& rhs) const
    {
        return cal_cost(qo_len, kv_len) > cal_cost(rhs.qo_len, rhs.kv_len);
    }
};

template<int32_t kPackedQoLenPerWg_,
         int32_t kMaxClusterSize_>
struct MlaMetadataTraits
{
    static constexpr int32_t kPackedQoLenPerWg = kPackedQoLenPerWg_;
    static constexpr int32_t kMaxClusterSize   = kMaxClusterSize_;
    static constexpr int32_t kSplitTolerance   = 16;
};

//
// Persistent thread group solution which take variable query/output lengths into consideration as well.
//
// Returns
//   [0] work_indptr:        (#cu_part + 1),      The IDs of work handled by each cu_part.
//   [1] work information    (#work, 8)
//   [1.0] bs_index:         (#work),             The index of batch handled by each work.
//   [1.1] partial_qo_loc:   (#work),             The location in qo of tile in output buffer when splits. -1 means no split.
//   [1.2] q_start:          (#work),             The global index in seq where q/o starts.
//   [1.3] q_end:            (#work),             The global index in seq where q/o ends (not included).
//   [1.4] kv_start:         (#work),             The global index in seq where k/v starts.
//   [1.5] kv_end:           (#work),             The global index in seq where k/v ends (not included).
//   [1.6] kv_offset:        (#work),             The delta between kv_end and seqlens_kv_indptr[batch_idx].
//   [1.7] paddings:         (#work, 1),          Pad to 8 DWs.
//   [2] reduce_indptr:      (#reduce_tiles + 1), The IDs in reduce_partial_map indicates the tiles should be merged
//                                                together.
//   [3] reduce_final_map:   (#reduce_tiles, 2),  The final output location and length of each group of tiles.
//   [3.0] q_start           (#reduce_tiles),     The global index in seq where q/o starts.
//   [3.1] q_end             (#reduce_tiles),     The global index in seq where q/o ends (not included).
//   [4] reduce_partial_map: (#partial_tiles),    The locations in partial buffer of partial tiles waiting for being
//                                                reduced.
//
std::vector<torch::Tensor> get_mla_metadata_v1(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const bool           no_redundant)
{
    // This default settings is for our ASM MLA decode kernel. This kernel supports num_heads=16 and qo size from 1 to 4
    // without support to split qo for each workgroup. This means that kPackedQoLenPerWg should be 4*16=64 to prevent 
    // spliting in any case supported by it.
    //                                PackedQoLenPerWg, MaxClusterSize
    using Traits  = MlaMetadataTraits<64,               1>;
    using index_t = uint32_t;

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t batch_size = seqlens_qo_indptr.size(0) - 1;
    const int32_t num_heads  = num_heads_k * num_heads_per_head_k;

    auto seqlens_qo_indptr_cpu = seqlens_qo_indptr.to(at::DeviceType::CPU);
    auto seqlens_kv_indptr_cpu = seqlens_kv_indptr.to(at::DeviceType::CPU);

    const int32_t* p_seqlens_qo_indptr = seqlens_qo_indptr_cpu.data_ptr<int32_t>();
    const int32_t* p_seqlens_kv_indptr = seqlens_kv_indptr_cpu.data_ptr<int32_t>();

    // Step.0. Get sequence lengths of query/output and key/value for each batch.
    std::vector<BatchInfo> batch_infos;
    batch_infos.reserve(batch_size);
    int32_t sum_packed_qo_len = 0;
    for (int32_t bid = 0; bid < batch_size; ++bid)
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
        const int32_t avg_packed_qo_len = sum_packed_qo_len / batch_size;
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
    int32_t sum_kv_lens = 0;
    std::vector<int32_t> num_qo_clusters_indptr;
    num_qo_clusters_indptr.reserve(batch_size + 1);
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
            sum_kv_lens += kv_len_valid;
        }
    }
    const int32_t kv_len_limit_global =
    [&]() {
        const int32_t avg_kv_lens = ck_tile::max(ck_tile::integer_divide_ceil(sum_kv_lens, num_clusters), 1);
        // TODO: The following code just follow FlashInfer. Further tune may be required for AMD GPU.
        int32_t limit;
        if (avg_kv_lens <= 8) limit = 32;
        else if (avg_kv_lens <= 16) limit = 64;
        else if (avg_kv_lens <= 32) limit = 128;
        else if (avg_kv_lens <= 64) limit = 192;
        else limit = ck_tile::integer_least_multiple(avg_kv_lens, 16);
        return limit;
    }();
#if PRINT_DBG
    printf("[metadata] kv_len_limit_global=%d\n",
           kv_len_limit_global);
#endif

    // Step.3.1. Allocates output buffers except indptrs
    std::vector<std::vector<MlaWorkInfo>> work_info_set(num_clusters, std::vector<MlaWorkInfo>());
    std::vector<std::vector<index_t>> reduce_partial_map(num_qo_clusters_indptr.back(), std::vector<index_t>());
    std::vector<MlaPartialTileInfo> reduce_partial_info(num_qo_clusters_indptr.back(), {-1, -2});

    // Step.3.2. Declare priority queue
    using ClusterCost = std::tuple<int32_t, int32_t>; // cluster_id(cid), cost
    auto pq_cmp = [](const ClusterCost& l, const ClusterCost& r) { return std::get<1>(l) > std::get<1>(r); };
    std::priority_queue<ClusterCost, std::vector<ClusterCost>, decltype(pq_cmp)> cost_heap(pq_cmp);
    for (int32_t cid = 0; cid < num_clusters; ++cid) { cost_heap.push(std::tuple{cid, 0.0f}); }

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
            int32_t remaining_kv_len = 
                cal_packed_causal_kv_len(qo_len, kv_len, tid, cluster_len_q, num_qo_tiles, num_heads, is_causal);
            int32_t kv_start_local = 0;

            const auto [cid_top, accum_cost_top] = cost_heap.top();
            const int32_t remaining_capability_top =
                get_remaining_kv_capability(kv_len_limit_global, cal_kv_len(accum_cost_top, cluster_len_q));
            const int32_t num_splits_estimated =
                ck_tile::integer_divide_ceil(remaining_kv_len, remaining_capability_top);
            // For the case of #splits==2, make sure that the tailing tile is smaller than Traits::kSplitTolerance.
            const bool split_kv = (num_splits_estimated == 2) ?
                ((remaining_kv_len - remaining_capability_top) > Traits::kSplitTolerance) : (num_splits_estimated > 1);

            do
            {
                // Check and update cost_heap
                auto [cid, accum_cost] = cost_heap.top();
                cost_heap.pop();
                const int32_t remaining_capability =
                    get_remaining_kv_capability(kv_len_limit_global, cal_kv_len(accum_cost, cluster_len_q));
                const int32_t kv_len_limit_local =
                [&]() {
                    const int32_t limit_ori = remaining_capability > 64 ? remaining_capability : 128;
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
                    const int32_t global_cluster_q_idx = num_qo_clusters_indptr[bid] + tid;
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
    auto int_opts = torch::TensorOptions().dtype(torch::kInt32);
    auto work_info_set_tsr = torch::from_blob(work_info_set_flatten.data(), {num_works, kSizeMlaWorkInfoInDw}, int_opts);
    auto work_indptr_tsr = torch::from_blob(work_indptr.data(), {static_cast<int32_t>(work_indptr.size())}, int_opts);
    auto reduce_indptr_tsr = torch::from_blob(reduce_indptr.data(), {reduce_indptr_size}, int_opts);
    auto reduce_final_map_tsr = torch::from_blob(reduce_final_map.data(), {reduce_final_map_size, kSizeMlaPartialTileInfoInDw}, int_opts);
    auto reduce_partial_map_tsr = torch::from_blob(reduce_partial_map_flatten.data(), {num_partial_outputs}, int_opts);

    // Last step. Copy to the device of input and return the results.
    auto input_opts = seqlens_qo_indptr.options();
    return {work_indptr_tsr.to(input_opts),
            work_info_set_tsr.to(input_opts),
            reduce_indptr_tsr.to(input_opts),
            reduce_final_map_tsr.to(input_opts),
            reduce_partial_map_tsr.to(input_opts)};
}
