#pragma once

#include "v1_comm.cuh"

__device__ __forceinline__ int64_t pa_metadata_v1_1_prefix_sum(int64_t value)
{
    for(int32_t offset = 1; offset < opus::get_warp_size(); offset <<= 1)
    {
        const int64_t previous = __shfl_up(static_cast<long long>(value), offset);
        if(opus::lane_id() >= offset)
            value += previous;
    }
    return value;
}

__device__ __forceinline__ int32_t pa_metadata_v1_1_worker(
    int64_t cost, int64_t total_cost, int32_t workers)
{
    return static_cast<int32_t>(min(((cost + 1) * workers - 1) / total_cost,
                                     static_cast<int64_t>(workers - 1)));
}

__device__ __forceinline__ int32_t pa_metadata_v1_1_fragments(
    int64_t cost_begin, int64_t tiles, int32_t qo_length,
    int64_t total_cost, int32_t workers)
{
    if(tiles == 0 || qo_length == 0)
        return 0;
    const int32_t first = pa_metadata_v1_1_worker(cost_begin, total_cost, workers);
    const int32_t last = pa_metadata_v1_1_worker(
        cost_begin + (tiles - 1) * qo_length, total_cost, workers);
    return static_cast<int32_t>(min(tiles, static_cast<int64_t>(last - first + 1)));
}

__device__ __forceinline__ int64_t pa_metadata_v1_1_boundary(
    int32_t fragment, int32_t fragments, int64_t tiles, int64_t cost_begin,
    int32_t qo_length, int64_t total_cost, int32_t workers)
{
    if(fragment == 0)
        return 0;
    if(fragment == fragments)
        return tiles;
    if(fragments == tiles)
        return fragment;
    const int32_t first = pa_metadata_v1_1_worker(cost_begin, total_cost, workers);
    const int64_t boundary = static_cast<int64_t>(first + fragment) * total_cost / workers;
    return min(tiles, max(int64_t{0}, (boundary - cost_begin + qo_length - 1) / qo_length));
}

__launch_bounds__(opus::get_warp_size(), 1) __global__
void kn_get_pa_metadata_v1_1(PaMetadataV1KernelParameter params)
{
    const int32_t lane = opus::lane_id();
    const int32_t sequence = blockIdx.x;
    const int32_t wave_size = opus::get_warp_size();
    const int32_t workers_per_head = params.num_cu / params.num_heads_k;
    auto* works = reinterpret_cast<PaWorkInfo*>(params.p_work_info_set_raw);
    int64_t total_cost = 0;
    for(int32_t batch = lane; batch < params.num_batches; batch += wave_size)
    {
        const int32_t qo_length = params.p_seqlens_qo_indptr[batch + 1] -
                                  params.p_seqlens_qo_indptr[batch];
        const int64_t tiles =
            (static_cast<int64_t>(max(params.p_context_lens[batch], 0)) +
             params.kv_granularity - 1) / params.kv_granularity;
        const int64_t cost = tiles * qo_length;
        total_cost += cost;
    }
    total_cost = __shfl(static_cast<long long>(pa_metadata_v1_1_prefix_sum(total_cost)), wave_size - 1);
    const int64_t denominator = max(total_cost, int64_t{1});
    int64_t cost_base = 0;
    int32_t task_base = 0;
    int32_t reduce_base = 0;
    int32_t partial_base = 0;
    int64_t sequence_cost_before = 0;
    int32_t sequence_tasks_before = 0;
    int32_t sequence_reduce_before = 0;
    int32_t sequence_partial_before = 0;
    int32_t sequence_parts = 0;

    for(int32_t chunk = 0; chunk < params.num_batches; chunk += wave_size)
    {
        const int32_t batch = chunk + lane;
        const int32_t qo_length = batch < params.num_batches
            ? params.p_seqlens_qo_indptr[batch + 1] - params.p_seqlens_qo_indptr[batch] : 0;
        const int64_t tiles = batch < params.num_batches
            ? (static_cast<int64_t>(max(params.p_context_lens[batch], 0)) +
               params.kv_granularity - 1) / params.kv_granularity : 0;
        const int64_t cost = tiles * qo_length;
        const int64_t cost_prefix = cost_base + pa_metadata_v1_1_prefix_sum(cost);
        const int32_t parts = min(params.num_splits, pa_metadata_v1_1_fragments(
            cost_prefix - cost, tiles, qo_length, denominator, workers_per_head));
        const int32_t reduce_count = parts > 1 ? parts : 0;
        const int32_t partial_count = reduce_count * qo_length;
        const int32_t task_prefix = task_base + warp_prefix_sum(parts, wave_size);
        const int32_t reduce_prefix = reduce_base + warp_prefix_sum(reduce_count, wave_size);
        const int32_t partial_prefix = partial_base + warp_prefix_sum(partial_count, wave_size);
        if(batch == sequence)
        {
            sequence_cost_before = cost_prefix - cost;
            sequence_tasks_before = task_prefix - parts;
            sequence_reduce_before = reduce_prefix - reduce_count;
            sequence_partial_before = partial_prefix - partial_count;
            sequence_parts = parts;
        }
        cost_base = __shfl(cost_prefix, wave_size - 1);
        task_base = __shfl(task_prefix, wave_size - 1);
        reduce_base = __shfl(reduce_prefix, wave_size - 1);
        partial_base = __shfl(partial_prefix, wave_size - 1);
    }

    const int32_t owner_lane = sequence % wave_size;
    sequence_cost_before = __shfl(sequence_cost_before, owner_lane);
    sequence_tasks_before = __shfl(sequence_tasks_before, owner_lane);
    sequence_reduce_before = __shfl(sequence_reduce_before, owner_lane);
    sequence_partial_before = __shfl(sequence_partial_before, owner_lane);
    sequence_parts = __shfl(sequence_parts, owner_lane);
    const int32_t qo_start = params.p_seqlens_qo_indptr[sequence];
    const int32_t qo_end = params.p_seqlens_qo_indptr[sequence + 1];
    const int32_t qo_length = qo_end - qo_start;
    const int32_t kv_start = params.p_pages_kv_indptr[sequence];
    const int32_t kv_end = params.p_pages_kv_indptr[sequence + 1];
    const int64_t tiles = (static_cast<int64_t>(max(params.p_context_lens[sequence], 0)) +
                           params.kv_granularity - 1) / params.kv_granularity;
    const int32_t fragments = pa_metadata_v1_1_fragments(
        sequence_cost_before, tiles, qo_length, denominator, workers_per_head);

    if(lane == 0)
    {
        params.p_reduce_indptr[sequence + 1] = sequence_reduce_before +
                                             (sequence_parts > 1 ? sequence_parts : 0);
        params.p_reduce_final_map[sequence * 2] = qo_start;
        params.p_reduce_final_map[sequence * 2 + 1] = qo_end;
        if(sequence == 0)
        {
            params.p_reduce_indptr[0] = 0;
            params.p_work_indptr[params.num_cu] = task_base * params.num_heads_k;
            params.p_work_metadata_ptrs[0] = reinterpret_cast<uintptr_t>(params.p_work_indptr);
            params.p_work_metadata_ptrs[1] = reinterpret_cast<uintptr_t>(works);
        }
    }
    for(int32_t part = lane; part < sequence_parts; part += wave_size)
    {
        const int32_t fragment_begin = part * fragments / sequence_parts;
        const int32_t fragment_end = (part + 1) * fragments / sequence_parts;
        const int64_t tile_begin = pa_metadata_v1_1_boundary(
            fragment_begin, fragments, tiles, sequence_cost_before,
            qo_length, denominator, workers_per_head);
        const int64_t tile_end = pa_metadata_v1_1_boundary(
            fragment_end, fragments, tiles, sequence_cost_before,
            qo_length, denominator, workers_per_head);
        const int32_t local_worker = pa_metadata_v1_1_worker(
            sequence_cost_before + tile_begin * qo_length, denominator, workers_per_head);
        const int32_t partial = sequence_parts > 1
            ? sequence_partial_before + part * qo_length : -1;
        if(sequence_parts > 1)
            params.p_reduce_partial_map[sequence_reduce_before + part] = partial;

        for(int32_t head = 0; head < params.num_heads_k; ++head)
        {
            const int32_t extra_workers = params.num_cu % params.num_heads_k;
            const int32_t worker_begin = head * workers_per_head + min(head, extra_workers);
            const int32_t worker = worker_begin + local_worker;
            PaWorkInfo work{};
            work.batch_idx = sequence;
            work.partial_qo_loc = partial;
            work.qo_start = qo_start;
            work.qo_end = qo_end;
            work.kv_start = static_cast<int32_t>(min(
                static_cast<int64_t>(kv_start) + tile_begin * params.blocks_per_unit,
                static_cast<int64_t>(kv_end)));
            work.kv_end = static_cast<int32_t>(min(
                static_cast<int64_t>(kv_start) + tile_end * params.blocks_per_unit,
                static_cast<int64_t>(kv_end)));
            work.kv_offset = worker;
            work.q_head_range = static_cast<int32_t>(
                (static_cast<uint32_t>((head + 1) * params.qhead_granularity) << 16) |
                static_cast<uint32_t>(head * params.qhead_granularity));
            works[head * task_base + sequence_tasks_before + part] = work;
        }
    }
}

__launch_bounds__(opus::get_warp_size(), 1) __global__
void kn_finalize_pa_metadata_v1_1(PaMetadataV1KernelParameter params)
{
    auto* works = reinterpret_cast<PaWorkInfo*>(params.p_work_info_set_raw);
    const int32_t total_tasks = params.p_work_indptr[params.num_cu];
    for(int32_t worker = threadIdx.x; worker < params.num_cu; worker += blockDim.x)
    {
        int32_t begin = 0;
        int32_t end = total_tasks;
        while(begin < end)
        {
            const int32_t middle = begin + (end - begin) / 2;
            if(works[middle].kv_offset < worker)
                begin = middle + 1;
            else
                end = middle;
        }
        params.p_work_indptr[worker] = begin;
    }
    __syncthreads();
    for(int32_t task = threadIdx.x; task < total_tasks; task += blockDim.x)
        works[task].kv_offset = 0;
    const int32_t total_reduce = params.p_reduce_indptr[params.num_batches];
    for(int32_t tile = params.num_batches + 1 + threadIdx.x;
        tile < params.reduce_indptr_size; tile += blockDim.x)
        params.p_reduce_indptr[tile] = total_reduce;
}

void get_pa_metadata_v1_1_device(
    const aiter_tensor_t& seqlens_qo_indptr,
    const aiter_tensor_t& pages_kv_indptr,
    const aiter_tensor_t& context_lens,
    const int32_t num_heads_per_head_k,
    const int32_t num_heads_k,
    const bool is_causal,
    const int32_t kv_granularity,
    const int32_t block_size,
    const int32_t max_seqlen_qo,
    const int32_t ori_uni_seqlen_qo,
    const int32_t topk,
    const int32_t max_split_per_batch,
    aiter_tensor_t& work_metadata_ptrs,
    aiter_tensor_t& work_info_set,
    aiter_tensor_t& work_indptr,
    aiter_tensor_t& reduce_indptr,
    aiter_tensor_t& reduce_final_map,
    aiter_tensor_t& reduce_partial_map)
{
    hipDeviceProp_t properties;
    int device;
    HIP_CALL(hipGetDevice(&device));
    HIP_CALL(hipGetDeviceProperties(&properties, device));
    const int32_t workers = properties.multiProcessorCount;
    const int32_t batches = context_lens.size(0);
    AITER_CHECK(batches > 0 && batches <= 4096, "PA v1_1 supports batches in [1, 4096]");
    AITER_CHECK(num_heads_k > 0 && num_heads_k <= workers, "invalid PA v1_1 KV head count");
    AITER_CHECK(topk < 0, "PA v1_1 supports dense attention only");
    AITER_CHECK(max_seqlen_qo > 0 && num_heads_per_head_k > 0 &&
                static_cast<int64_t>(max_seqlen_qo) * num_heads_per_head_k <= 128,
                "PA v1_1 requires one Q tile: max_seqlen_qo * GQA <= 128");
    AITER_CHECK(static_cast<int64_t>(num_heads_k) * num_heads_per_head_k <= 65535,
                "PA Q head range exceeds the metadata field");
    AITER_CHECK(block_size > 0 && kv_granularity >= block_size &&
                kv_granularity % block_size == 0, "PA v1_1 requires page-aligned KV granularity");
    AITER_CHECK(max_split_per_batch == -1 || max_split_per_batch > 0,
                "max_split_per_batch must be -1 or positive");
    AITER_CHECK(seqlens_qo_indptr.size(0) == batches + 1 &&
                pages_kv_indptr.size(0) == batches + 1 &&
                context_lens.dtype() == AITER_DTYPE_i32 && context_lens.stride(0) == 1,
                "invalid PA v1_1 input lengths");
    const int32_t max_partitions = max_split_per_batch < 0
        ? workers : std::min(workers, max_split_per_batch);
    const int32_t capacity = static_cast<int32_t>(std::min(
        static_cast<int64_t>(batches) * max_partitions,
        static_cast<int64_t>(batches) + workers / num_heads_k - 1));
    AITER_CHECK(work_metadata_ptrs.numel() >= 2 && work_indptr.numel() >= workers + 1 &&
                work_info_set.numel() >= static_cast<int64_t>(capacity) * num_heads_k * 8 &&
                reduce_indptr.numel() >= batches + 1 && reduce_final_map.numel() >= batches * 2 &&
                reduce_partial_map.numel() >= capacity,
                "PA v1_1 buffers are too small; use get_pa_metadata_info_v1(..., fast_mode=False)");

    PaMetadataV1KernelParameter params{};
    params.p_work_metadata_ptrs = static_cast<uint64_t*>(work_metadata_ptrs.data_ptr());
    params.p_work_indptr = static_cast<int32_t*>(work_indptr.data_ptr());
    params.p_work_info_set_raw = static_cast<int32_t*>(work_info_set.data_ptr());
    params.p_reduce_indptr = static_cast<int32_t*>(reduce_indptr.data_ptr());
    params.p_reduce_final_map = static_cast<int32_t*>(reduce_final_map.data_ptr());
    params.p_reduce_partial_map = static_cast<int32_t*>(reduce_partial_map.data_ptr());
    params.p_seqlens_qo_indptr = static_cast<int32_t*>(seqlens_qo_indptr.data_ptr());
    params.p_pages_kv_indptr = static_cast<int32_t*>(pages_kv_indptr.data_ptr());
    params.p_context_lens = static_cast<int32_t*>(context_lens.data_ptr());
    params.num_batches = batches;
    params.num_heads_k = num_heads_k;
    params.num_cu = workers;
    params.num_splits = max_partitions;
    params.reduce_indptr_size = reduce_indptr.numel();
    params.qhead_granularity = num_heads_per_head_k;
    params.kv_granularity = kv_granularity;
    params.blocks_per_unit = kv_granularity / block_size;
    const hipStream_t stream = aiter::getCurrentHIPStream();
    kn_get_pa_metadata_v1_1<<<batches, properties.warpSize, 0, stream>>>(params);
    HIP_CALL_LAUNCH(hipGetLastError());
    kn_finalize_pa_metadata_v1_1<<<1, properties.warpSize, 0, stream>>>(params);
    HIP_CALL_LAUNCH(hipGetLastError());
}