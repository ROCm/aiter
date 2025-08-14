// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <queue>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
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

    const int32_t num_loops = (num_batches + kWarpSize - 1) / kWarpSize;
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
            const int32_t remote = __shfl_up(scan, offset);
            scan += (threadIdx.x >= offset) ? remote : 0;
        }

        const int32_t global_scan = scan + base_scan;

        if (seqlen_idx < num_batches)
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
    for (int32_t cid = 0; cid < num_clusters; ++cid) {
        int32_t cost = p_cost_heap[cid];
        if (cost < cost_min) {
            cost_min = cost;
            cid_min = cid;
        }
    }
    return std::make_tuple(cid_min, cost_min);
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
    int32_t        max_reduce_size;
    int32_t        num_heads;
    int32_t        num_cu;
    bool           is_causal;
};

template <typename Traits>
__launch_bounds__(ck_tile::get_warp_size(), 1)
__global__ void kn_get_mla_metadata_v1(
    const MlaMetadataV1KernelParameter params)
{
    extern __shared__ uint8_t p_smem[];
 
    if (threadIdx.x == 0)
    {
        params.p_work_metadata_ptrs[0] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
        params.p_work_metadata_ptrs[1] = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_info_set_raw));

        // Step.1. Calculate the size of cluster and some related information. The size is the number of workgroups
        //         composing each cluster. The size is determined by average packed qo length.
        int32_t sum_packed_qo_len = 0;
        for (int32_t bid = 0; bid < params.num_batches; ++bid)
        {
            const int32_t qo_len = params.p_seqlens_qo_indptr[bid + 1] - params.p_seqlens_qo_indptr[bid];
            sum_packed_qo_len += qo_len * params.num_heads;
        }
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
        int32_t sum_kv_lens = 0;
        int32_t* p_num_qo_clusters_indptr = reinterpret_cast<int32_t*>(p_smem);
        p_num_qo_clusters_indptr[0] = 0;
        for (int32_t bid = 0; bid < params.num_batches; ++bid)
        {
            const int32_t kv_len = params.p_seqlens_kv_indptr[bid + 1] - params.p_seqlens_kv_indptr[bid];
            const int32_t qo_len = params.p_seqlens_qo_indptr[bid + 1] - params.p_seqlens_qo_indptr[bid];
            const int32_t packed_qo_len = qo_len * params.num_heads;
            const int32_t num_qo_tiles = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);

            p_num_qo_clusters_indptr[bid + 1] = (p_num_qo_clusters_indptr[bid] + num_qo_tiles);

            for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
            {
                const int32_t kv_len_valid =
                    cal_packed_causal_kv_len(
                        qo_len, kv_len, tid, cluster_len_q, num_qo_tiles, params.num_heads, params.is_causal);
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

        // Step.3.1. Initialize lds
        int32_t* p_cost_heap = p_num_qo_clusters_indptr + params.num_batches + 1;
        int32_t* p_cluster_work_counter = p_cost_heap + num_clusters + 1;
        for (int32_t cid = 0; cid < num_clusters; ++cid)
        {
            p_cost_heap[cid] = 0;
            p_cluster_work_counter[cid] = 0;
        }

        // Step.4. Fill the output buffers except indptrs
        MlaWorkInfo* p_work_info_set = reinterpret_cast<MlaWorkInfo*>(params.p_work_info_set_raw);

        // Step.4.1. Get total work for each cluster
        for (int32_t bid = 0; bid < params.num_batches; ++bid)
        {
            const int32_t qo_batch_start = params.p_seqlens_qo_indptr[bid];
            const int32_t kv_batch_start = params.p_seqlens_kv_indptr[bid];
            const int32_t kv_batch_end   = params.p_seqlens_kv_indptr[bid + 1];
            const int32_t kv_len         = kv_batch_end - kv_batch_start;
            const int32_t qo_len         = params.p_seqlens_qo_indptr[bid + 1] - qo_batch_start;
            const int32_t packed_qo_len  = qo_len * params.num_heads;
            const int32_t num_qo_tiles   = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);

            for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
            {
                int32_t remaining_kv_len =
                    cal_packed_causal_kv_len(qo_len, kv_len, tid, cluster_len_q, num_qo_tiles, params.num_heads, params.is_causal);

                do
                {
                    // Check and update cost_heap
                    auto [cid, accum_cost] = get_cost_top(p_cost_heap, num_clusters);
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
                    const int32_t new_cost = accum_cost + cost;
                    p_cost_heap[cid] = new_cost;

                    ++p_cluster_work_counter[cid];

                    // Update state
                    remaining_kv_len -= kv_len_consuming;
                }
                while (remaining_kv_len > 0);
            }
        }

        // Step.4.2. Re-init cost heap and cumulative sum cluster_work_tot
        int32_t cum_cluster_work = 0;
        params.p_work_indptr[0] = 0;
        for (int32_t cid = 0; cid < num_clusters; ++cid)
        {
            cum_cluster_work += p_cluster_work_counter[cid];
            params.p_work_indptr[cid + 1] = cum_cluster_work;
            p_cost_heap[cid] = 0;
            p_cluster_work_counter[cid] = 0;
        }
        const int32_t max_qo_tiles = params.num_batches * Traits::kMaxQoTilePerBatch;
        MlaPartialTileInfo* p_reduce_partial_map =
            reinterpret_cast<MlaPartialTileInfo*>(p_cluster_work_counter + num_clusters);
        MlaPartialTileInfo* p_reduce_final_map = p_reduce_partial_map + max_qo_tiles;
        for (int32_t global_cluster_q_idx = 0; global_cluster_q_idx < max_qo_tiles; ++global_cluster_q_idx)
        {
            p_reduce_partial_map[global_cluster_q_idx] = MlaPartialTileInfo{-1, -2};
            p_reduce_final_map[global_cluster_q_idx] = MlaPartialTileInfo{-1, -2};
        }

        // Step.4.3. Output work info
        int32_t num_partial_outputs = 0;
        int32_t loc_partial_outputs = 0;
        for (int32_t bid = 0; bid < params.num_batches; ++bid)
        {
            const int32_t qo_batch_start = params.p_seqlens_qo_indptr[bid];
            const int32_t kv_batch_start = params.p_seqlens_kv_indptr[bid];
            const int32_t kv_batch_end   = params.p_seqlens_kv_indptr[bid + 1];
            const int32_t kv_len         = kv_batch_end - kv_batch_start;
            const int32_t qo_len         = params.p_seqlens_qo_indptr[bid + 1] - qo_batch_start;
            const int32_t packed_qo_len  = qo_len * params.num_heads;
            const int32_t num_qo_tiles   = ck_tile::integer_divide_ceil(packed_qo_len, cluster_len_q);

            for (int32_t tid = 0; tid < num_qo_tiles; ++tid)
            {
                int32_t remaining_kv_len =
                    cal_packed_causal_kv_len(qo_len, kv_len, tid, cluster_len_q, num_qo_tiles, params.num_heads, params.is_causal);
                int32_t kv_start_local = 0;

                const auto [cid_top, accum_cost_top] = get_cost_top(p_cost_heap, num_clusters);
                const int32_t remaining_capability_top =
                    get_remaining_kv_capability(kv_len_limit_global, cal_kv_len(accum_cost_top, cluster_len_q));
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
                    const int32_t new_cost = accum_cost + cost;
                    p_cost_heap[cid] = new_cost;

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
                        const int32_t global_cluster_q_idx = p_num_qo_clusters_indptr[bid] + tid;
                        work_info.partial_qo_loc = loc_partial_outputs;
                        if (p_reduce_partial_map[global_cluster_q_idx].q_start == -1)
                        {
                            p_reduce_partial_map[global_cluster_q_idx].q_start = loc_partial_outputs;
                            p_reduce_final_map[global_cluster_q_idx] = { work_info.q_start, work_info.q_end };
                        }
                        ++num_partial_outputs;
                        loc_partial_outputs += (work_info.q_end - work_info.q_start);
                        p_reduce_partial_map[global_cluster_q_idx].q_end = loc_partial_outputs;
                    }
                    else
                    {
                        work_info.partial_qo_loc = -1;
                    }

                    const int32_t work_info_set_idx = params.p_work_indptr[cid] + p_cluster_work_counter[cid];
                    ++p_cluster_work_counter[cid];
                    p_work_info_set[work_info_set_idx] = work_info;

                    // Update state
                    remaining_kv_len -= kv_len_consuming;
                    kv_start_local += kv_len_consuming;
                }
                while (remaining_kv_len > 0);
            }
        }

        int32_t cum_reduce_tiles = 0;
        params.p_reduce_indptr[0] = cum_reduce_tiles;
        for (int32_t global_cluster_q_idx = 0; global_cluster_q_idx < max_qo_tiles; ++global_cluster_q_idx)
        {
            const MlaPartialTileInfo final_info = p_reduce_final_map[global_cluster_q_idx];
            const MlaPartialTileInfo partial_range = p_reduce_partial_map[global_cluster_q_idx];
            const int32_t reduce_tile_size = (final_info.q_start == -1) ? 0 : (final_info.q_end - final_info.q_start);
            const int32_t num_reduce_tiles =
                (reduce_tile_size == 0) ? 0 : ((partial_range.q_end - partial_range.q_start) / reduce_tile_size);

            for (int32_t tid = cum_reduce_tiles; tid < cum_reduce_tiles + num_reduce_tiles; ++tid)
            {
                const int32_t local_tid = tid - cum_reduce_tiles;
                params.p_reduce_partial_map[tid] = partial_range.q_start + local_tid * reduce_tile_size;
            }

            cum_reduce_tiles += num_reduce_tiles;
            params.p_reduce_indptr[global_cluster_q_idx + 1] = cum_reduce_tiles;
            params.p_reduce_final_map[2 * global_cluster_q_idx] = final_info.q_start;
            params.p_reduce_final_map[2 * global_cluster_q_idx + 1] = final_info.q_end;
        }
        for (int32_t global_cluster_q_idx = max_qo_tiles; global_cluster_q_idx < params.max_reduce_size; ++global_cluster_q_idx)
        {
            params.p_reduce_indptr[global_cluster_q_idx] = cum_reduce_tiles;
        }
    }
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
    torch::Tensor&       work_indptr,
    torch::Tensor&       work_info_set,
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
    const int32_t max_reduce_size = reduce_indptr.size(0);

    const int32_t lds_size_in_bytes = [&]()
    {
        int32_t lds_size = 0;
        // this is maximun #clusters
        const int32_t num_clusters = dev_prop.multiProcessorCount;
        // Memory for indptr about #cluster for each batch in direction of qo
        lds_size += (num_batches + 1) * sizeof(int32_t);
        // Memory for cost. Its size should be the same as #clusters
        lds_size += num_clusters * sizeof(int32_t);
        // Memory for counter of #works for each cluster.
        lds_size += num_clusters * sizeof(int32_t);
        // Memory for range of partial memory
        lds_size += max_qo_tiles * sizeof(MlaPartialTileInfo);
        // Memory for range of output of partial memory
        lds_size += max_qo_tiles * sizeof(MlaPartialTileInfo);
        return lds_size;
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
    params.max_reduce_size      = max_reduce_size;
    params.num_heads            = num_heads_per_head_k * num_heads_k;
    params.num_cu               = num_cu;
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
    int32_t sum_kv_lens = 0;
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
//   [1] work_indptr:        (#cu_part + 1),     The IDs of work handled by each cu_part.
//   [2] work_info           (#work, 8)
//   [2.0] bs_index:         (#work),            The index of batch handled by each work.
//   [2.1] partial_index:    (#work),            The index of tile in output buffer when splits. -1 means no split.
//   [2.2] q_start:          (#work),            The global index in seq where q/o starts. Use global index here can
//                                               reduce memory access count in kernel.
//   [2.3] q_end:            (#work),            The global index in seq where q/o ends (not included).
//   [2.4] kv_start:         (#work),            The global index in seq where k/v starts.
//   [2.5] kv_end:           (#work),            The global index in seq where k/v ends (not included).
//   [2.6] pad               (#work, 2),         Pad to 8 DWs.
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
    torch::Tensor&       work_info_set_tsr,
    torch::Tensor&       work_indptr_tsr,
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
        work_indptr_tsr,
        work_info_set_tsr,
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


// ======================================================= flash mla style metadata ==================================================

// =====================================================================================================================
// Definitions and helper structures
//

template <int32_t kSizeD_,
          int32_t kSizeDV_,
          int32_t kBlockM_,
          int32_t kBlockN_,
          int32_t kNumWarps_>
struct FlashMlaKernelTrait
{
    static constexpr int32_t kSizeD                  = kSizeD_;    // hidden dimension size of query and key
    static constexpr int32_t kSizeDV                 = kSizeDV_;   // hidden dimension size of value
    static constexpr int32_t kNumWarps               = kNumWarps_;
    static constexpr int32_t kNumThreads             = kNumWarps * warpSize;
    static constexpr int32_t kNumWarpsSoftmax        = 4;
    static constexpr int32_t kNumThreadsSoftmax      = kNumWarpsSoftmax * warpSize;
    static constexpr int32_t kBlockM                 = kBlockM_;
    static constexpr int32_t kBlockN                 = kBlockN_;
    static constexpr int32_t kFixedOverheadNumBlocks = 16;
    static constexpr int32_t kMaxBatchSize           = 4096;

    static_assert(kSizeD % 64 == 0);
    static_assert(kSizeDV % 64 == 0);
    static_assert(kSizeD >= kSizeDV);
};

// using FlashMlaKernelTraitsInstance = FlashMlaKernelTrait<576, 512, 64, 64, 4>;
using FlashMlaKernelTraitsInstance = FlashMlaKernelTrait<192, 128, 64, 16, 4>;

union TileSchedulerMetaData
{
    struct Core
    {
        int32_t batch_idx;
        int32_t partial_idx;
        int32_t begin_seqlen_q_idx;
        int32_t end_seqlen_q_idx;
        int32_t begin_seqlen_kv_idx;
        int32_t end_seqlen_kv_idx;
        int32_t end_seqlen_kv_idx_to_end;
    };
    uint32_t data[8];
};
constexpr size_t TileSchedulerMetaDataSizeInDw = sizeof(TileSchedulerMetaData) / sizeof(int32_t);


union FmlaTileSchedulerMetaData
{
    struct Core
    {
        int32_t begin_batch_idx;
        int32_t begin_seqlen_idx;
        int32_t end_batch_idx;
        int32_t end_seqlen_idx;
        int32_t begin_n_split_idx;
    };
    uint32_t data[8];
};

struct FlashMlaFwdParams
{
    int32_t* __restrict__ p_cu_seqlens_k;
    int32_t* __restrict__ p_block_table;
    int32_t* __restrict__ p_work_indptr;
    int32_t* __restrict__ p_tile_scheduler_metadata;
    
    void* __restrict__ p_query;
    void* __restrict__ p_key;
    void* __restrict__ p_value;
    void* __restrict__ p_output;
    void* __restrict__ p_softmax_lse;
    void* __restrict__ p_softmax_lseaccum;
    void* __restrict__ p_output_accum;

    int32_t size_b;
    int32_t size_s;
    int32_t size_h;
    int32_t hq_hk_ratio;
    int32_t num_groups;
    int32_t num_cu_parts;
    int64_t block_table_batch_stride;
    int32_t page_block_size;
    float   scale_softmax;
    float   scale_softmax_log2;
    bool    is_causal;

    // Use int64_t if there is int32 overflow case. For now, just use int32 to save sgpr and prevent using
    // spill table.
    using index_t = int32_t;

    index_t stride_b_q;     // stride in batch of query
    index_t stride_s_q;     //    ... in sequence ...
    index_t stride_h_q;     //    ... in head ...
    index_t stride_b_k;     // stride in batch of key
    index_t stride_s_k;     //    ... in sequence ...
    index_t stride_h_k;     //    ... in head ...
    index_t stride_b_v;     // stride in batch of value
    index_t stride_s_v;     //    ... in sequence ...
    index_t stride_h_v;     //    ... in head ...
    index_t stride_b_o;     // stride in batch of output
    index_t stride_s_o;     //    ... in sequence ...
    index_t stride_h_o;     //    ... in head ...
};

// =====================================================================================================================
// Kernel Entries
//


__inline__ __device__ void transfer_metadata_into_work_info(
    int32_t& info_idx,
    int32_t& partial_idx,
    FmlaTileSchedulerMetaData::Core& fmla_metadata,

    int32_t* p_reduce_indptr,
    int32_t* p_reduce_final_map,
    int32_t* p_reduce_partial_map,

    const int32_t* p_seqlens_k,
    const int32_t* p_seqlens_kv_indptr,
    const int32_t* p_seqlens_qo_indptr,
    int32_t* p_tile_scheduler_metadata)
{
    int32_t info_size = fmla_metadata.end_batch_idx - fmla_metadata.begin_batch_idx + 1;
    int32_t qo_len = p_seqlens_qo_indptr[1] - p_seqlens_qo_indptr[0];

    for (int i = 0; i < info_size; ++i)
    {
        int32_t batch_idx = fmla_metadata.begin_batch_idx + i;
        int32_t seqlen_k  = p_seqlens_k[batch_idx];
        TileSchedulerMetaData::Core metadata;
        metadata.batch_idx = batch_idx;
        metadata.begin_seqlen_q_idx = batch_idx * qo_len;
        metadata.end_seqlen_q_idx   = (batch_idx + 1) * qo_len;

        metadata.begin_seqlen_kv_idx = i == 0 ? fmla_metadata.begin_seqlen_idx : 0;
        metadata.end_seqlen_kv_idx = i == info_size - 1 ? fmla_metadata.end_seqlen_idx : seqlen_k;

        metadata.end_seqlen_kv_idx_to_end = seqlen_k - metadata.end_seqlen_kv_idx;


        metadata.partial_idx = metadata.end_seqlen_kv_idx == seqlen_k && metadata.begin_seqlen_kv_idx == 0 ? -1 : partial_idx;

        if (metadata.end_seqlen_kv_idx == seqlen_k && metadata.begin_seqlen_kv_idx == 0)
        {
            p_reduce_indptr[batch_idx + 1] = p_reduce_indptr[batch_idx];
            p_reduce_final_map[batch_idx * 2] = -2;
            p_reduce_final_map[batch_idx * 2 + 1] = -1;
        }
        else
        {
            int reduce_cur_idx = metadata.partial_idx / qo_len;
            p_reduce_indptr[batch_idx + 1] = reduce_cur_idx + 1;
            p_reduce_final_map[batch_idx * 2]     = p_seqlens_qo_indptr[batch_idx];
            p_reduce_final_map[batch_idx * 2 + 1] = p_seqlens_qo_indptr[batch_idx + 1];
            p_reduce_partial_map[reduce_cur_idx] = metadata.partial_idx;
            partial_idx += qo_len;
        }
        metadata.begin_seqlen_kv_idx += p_seqlens_kv_indptr[batch_idx];
        metadata.end_seqlen_kv_idx += p_seqlens_kv_indptr[batch_idx];

        *reinterpret_cast<TileSchedulerMetaData::Core*>(
            p_tile_scheduler_metadata + (i + info_idx)* TileSchedulerMetaDataSizeInDw) = metadata;

    }

    info_idx += info_size;
}

template <typename Traits>
__global__ void kn_get_mla_metadata(
    uint64_t*      p_work_metadata,
    int32_t*       p_tile_scheduler_metadata,
    int32_t*       p_work_indptr,
    int32_t*       p_reduce_indptr,
    int32_t*       p_reduce_final_map,
    int32_t*       p_reduce_partial_map,
    const int32_t* p_seqlens_qo_indptr,
    const int32_t* p_seqlens_kv_indptr,
    // int32_t*       p_tile_scheduler_metadata_ori,
    const int32_t  max_reduce_size,
    const int32_t  batch_size,
    const int32_t  num_cu_parts)
{
    __shared__ int lds_num_blocks[Traits::kMaxBatchSize];
    __shared__ int lds_num_splits[Traits::kMaxBatchSize];
    __shared__ int lds_seqlens_k[Traits::kMaxBatchSize];

    int32_t sum_blocks = 0;
    for (int32_t i = threadIdx.x; i < batch_size; i += warpSize)
    {
        int32_t seqlen_k = p_seqlens_kv_indptr[i + 1] - p_seqlens_kv_indptr[i];
        const int32_t num_blocks = ck_tile::integer_divide_ceil(seqlen_k, Traits::kBlockN);
        sum_blocks += num_blocks;
        lds_num_blocks[i] = num_blocks;
        lds_seqlens_k[i]  = seqlen_k;
    }
    __syncthreads();

    for (int32_t offset = 32; offset > 0; offset >>= 1)
    {
        sum_blocks += __shfl_xor(sum_blocks, offset);
    }
    __syncthreads();

    sum_blocks += batch_size * Traits::kFixedOverheadNumBlocks;

    if (threadIdx.x == 0)
    {
        // expected payload handled by each cu part.
        const int32_t payload = ck_tile::integer_divide_ceil(sum_blocks, num_cu_parts) +
                                Traits::kFixedOverheadNumBlocks;

        int32_t curr_batch = 0;         // batch ID of the batch which is under review
        int32_t curr_block = 0;         // #blocks handled by previous cu part(s)
        int32_t curr_n_split_idx = 0;   // #cu parts used to handle current batch
        int32_t cum_num_splits = 0;
        p_reduce_indptr[0] = 0;

        lds_num_splits[0] = 0;

        int info_idx = 0;
        int partial_idx = 0;
        for (int32_t i = 0; i < num_cu_parts; ++i)
        {
            FmlaTileSchedulerMetaData::Core metadata;
            metadata.begin_batch_idx   = curr_batch;
            metadata.begin_seqlen_idx  = curr_block * Traits::kBlockN;
            metadata.begin_n_split_idx = curr_n_split_idx;

            int remain_payload = payload;
            while (curr_batch < batch_size)
            {
                const int32_t num_blocks = lds_num_blocks[curr_batch];
                const int32_t curr_remain_blocks = num_blocks - curr_block;

                // If current cu part is able to handle this batch of seqences
                if (remain_payload >= (curr_remain_blocks + Traits::kFixedOverheadNumBlocks))
                {
                    cum_num_splits += (curr_n_split_idx + 1);
                    remain_payload -= (curr_remain_blocks + Traits::kFixedOverheadNumBlocks);
                    ++curr_batch;
                    curr_block = 0;
                    curr_n_split_idx = 0;
                }
                else
                {
                    if (remain_payload > Traits::kFixedOverheadNumBlocks)
                    {
                        curr_block += (remain_payload - Traits::kFixedOverheadNumBlocks);
                        ++curr_n_split_idx;
                    }
                    break;
                }
            }

            metadata.end_batch_idx  = curr_block > 0 ? curr_batch : (curr_batch - 1);
            metadata.end_seqlen_idx = curr_block > 0 ? curr_block * Traits::kBlockN : lds_seqlens_k[curr_batch - 1];

            transfer_metadata_into_work_info(
                info_idx,
                partial_idx,
                metadata,

                p_reduce_indptr,
                p_reduce_final_map,
                p_reduce_partial_map,

                lds_seqlens_k,
				p_seqlens_kv_indptr,
				p_seqlens_qo_indptr,
                p_tile_scheduler_metadata
            );

            lds_num_splits[i + 1] = info_idx;
            // *reinterpret_cast<FmlaTileSchedulerMetaData::Core*>(
            //     p_tile_scheduler_metadata_ori + i * TileSchedulerMetaDataSizeInDw) = metadata;
        }

        p_work_metadata[0] = (uint64_t)(p_work_indptr);
        p_work_metadata[1] = (uint64_t)(p_tile_scheduler_metadata);

        int32_t reduce_indptr_tail = p_reduce_indptr[batch_size];

        for (int32_t i = batch_size; i < max_reduce_size; ++i)
        {
            p_reduce_indptr[i] = reduce_indptr_tail;
        }
    }

    for (int32_t i = threadIdx.x; i <= num_cu_parts; i += warpSize)
    {
        p_work_indptr[i] = lds_num_splits[i];
    }
}





// =====================================================================================================================
// Dispatches
//

template <typename Traits>
void dispatch_get_mla_metadata(
    uint64_t*      p_work_metadata,
    int32_t*       p_tile_scheduler_metadata,
    int32_t*       p_work_indptr,
    int32_t*       p_reduce_indptr,
    int32_t*       p_reduce_final_map,
    int32_t*       p_reduce_partial_map,
    const int32_t* p_seqlens_qo_indptr,
    const int32_t* p_seqlens_kv_indptr,
    // int32_t*       p_tile_scheduler_metadata_ori,
    const int32_t  max_reduce_size,
    const int32_t  batch_size,
    const int32_t  num_cu_parts)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const uint32_t grid  = 1;
    const uint32_t block = warpSize;

    kn_get_mla_metadata<Traits><<<grid, block, 0, stream>>>(
        p_work_metadata,
        p_tile_scheduler_metadata,
        p_work_indptr,
        p_reduce_indptr,
        p_reduce_final_map,
        p_reduce_partial_map,
        p_seqlens_qo_indptr,
        p_seqlens_kv_indptr,
        // p_tile_scheduler_metadata_ori,
        max_reduce_size,
        batch_size,
        num_cu_parts);
}

template <typename Traits, typename scalar_t>
void dispatch_fmla_fwd_splictkv_combine(
    const FlashMlaFwdParams& params)
{

}

#define DISPATCH_TYPES(TYPE, NAME, ...)                 \
    switch ((TYPE))                                     \
    {                                                   \
        case at::ScalarType::BFloat16:                  \
        {                                               \
            using scalar_t = at::BFloat16;              \
            __VA_ARGS__;                                \
            break;                                      \
        }                                               \
        case at::ScalarType::Half:                      \
        {                                               \
            using scalar_t = at::Half;                  \
            __VA_ARGS__;                                \
            break;                                      \
        }                                               \
        default:                                        \
            TORCH_CHECK(false, NAME " does't support ", \
                        toString((TYPE)), ".");         \
    }

// =====================================================================================================================
// Interfaces
//

// std::vector<torch::Tensor> get_mla_metadata(
//     const torch::Tensor& cache_seqlens,
//     const int32_t        num_heads_per_head_k,
//     const int32_t        num_heads_k)
//
// std::vector<torch::Tensor> get_mla_metadata_v2(
void get_mla_metadata_v2(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,         // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    torch::Tensor& work_meta_data,
    torch::Tensor& work_info_set_tsr,
    torch::Tensor& work_indptr_tsr,
    torch::Tensor& reduce_indptr_tsr,
    torch::Tensor& reduce_final_map_tsr,
    torch::Tensor& reduce_partial_map_tsr)
{
    using Traits = FlashMlaKernelTraitsInstance;

    const torch::TensorOptions tensor_options = seqlens_kv_indptr.options();
    const int32_t batch_size = seqlens_kv_indptr.size(0) - 1;
    const int32_t max_reduce_size = reduce_indptr_tsr.size(0);
    assert(batch_size <= Traits::kMaxBatchSize);

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);
    // const int32_t cu_count = dev_prop.multiProcessorCount;
    const int32_t cu_count = dev_prop.multiProcessorCount;
    const int32_t cu_parts = cu_count / num_heads_k /
                             ck_tile::integer_divide_ceil(num_heads_per_head_k, Traits::kBlockM);

    // printf("cu_count %d \n", cu_count);
    // printf("num_heads_k %d \n", num_heads_k);
    // printf("ck_tile::integer_divide_ceil(num_heads_per_head_k, Traits::kBlockM)%d \n", ck_tile::integer_divide_ceil(num_heads_per_head_k, Traits::kBlockM));
    // printf("cu_parts %d \n", cu_parts);
    // auto tile_scheduler_metadata = torch::empty({cu_parts, TileSchedulerMetaDataSizeInDw}, tensor_options);

    dispatch_get_mla_metadata<Traits>(
        work_meta_data.data_ptr<uint64_t>(),
        work_info_set_tsr.data_ptr<int32_t>(),
        work_indptr_tsr.data_ptr<int32_t>(),
        reduce_indptr_tsr.data_ptr<int32_t>(),
        reduce_final_map_tsr.data_ptr<int32_t>(),
        reduce_partial_map_tsr.data_ptr<int32_t>(),
        seqlens_qo_indptr.data_ptr<int32_t>(),
        seqlens_kv_indptr.data_ptr<int32_t>(),
        // tile_scheduler_metadata.data_ptr<int32_t>(),
        max_reduce_size,
        batch_size,
        cu_parts);
}

