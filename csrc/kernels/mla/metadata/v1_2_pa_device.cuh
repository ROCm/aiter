// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "v1_comm.cuh"
#include <cstdlib>

constexpr int32_t kPaMetadataSharedMaxBatches = 512;
constexpr int32_t kPaMetadataSharedMaxWaves = kPaMetadataSharedMaxBatches / opus::get_warp_size();
constexpr int32_t kPaMetadataMaxParallelBatches = 32768;

struct PaMetadataV12ChunkRecord
{
    int64_t pages;
    int32_t tasks;
    int32_t reductions;
    int32_t partials;
    int32_t reserved[3];
};

static_assert(sizeof(PaMetadataV12ChunkRecord) == sizeof(PaWorkInfo));

__device__ __forceinline__ int64_t pa_metadata_v1_2_prefix_sum(int64_t value)
{
    for(int32_t offset = 1; offset < opus::get_warp_size(); offset <<= 1)
    {
        const int64_t previous = __shfl_up(static_cast<long long>(value), offset);
        if(opus::lane_id() >= offset)
            value += previous;
    }
    return value;
}

__device__ __forceinline__ int64_t pa_metadata_v1_2_divide(int64_t numerator, int64_t denominator)
{
    if(numerator <= int64_t{0xffffffff} && denominator <= int64_t{0xffffffff})
        return static_cast<uint32_t>(numerator) / static_cast<uint32_t>(denominator);
    return numerator / denominator;
}

__device__ __forceinline__ int32_t pa_metadata_v1_2_worker(
    int64_t offset, int64_t total, int32_t workers)
{
    const int64_t average = pa_metadata_v1_2_divide(total, workers);
    const int64_t remainder = total - average * workers;
    const int64_t larger_payload = (average + 1) * remainder;
    const int64_t worker = offset < larger_payload
        ? pa_metadata_v1_2_divide(offset, average + 1)
        : remainder + pa_metadata_v1_2_divide(offset - larger_payload, max(average, int64_t{1}));
    return static_cast<int32_t>(min(worker, static_cast<int64_t>(workers - 1)));
}

__device__ __forceinline__ int32_t pa_metadata_v1_2_parts(
    int64_t begin, int64_t pages, int64_t total, int32_t workers)
{
    if(pages == 0)
        return 0;
    return pa_metadata_v1_2_worker(begin + pages - 1, total, workers) -
           pa_metadata_v1_2_worker(begin, total, workers) + 1;
}

__device__ __forceinline__ int64_t pa_metadata_v1_2_block_prefix(
    int64_t value, int64_t* wave_totals, bool wave_only)
{
    const int32_t lane = opus::lane_id();
    const int32_t wave = threadIdx.x / opus::get_warp_size();
    const int32_t waves = blockDim.x / opus::get_warp_size();
    const int64_t prefix = pa_metadata_v1_2_prefix_sum(value);
    if(wave_only)
        return prefix;
    if(lane == opus::get_warp_size() - 1)
        wave_totals[wave] = prefix;
    __syncthreads();
    if(wave == 0)
    {
        const int64_t wave_prefix = pa_metadata_v1_2_prefix_sum(
            lane < waves ? wave_totals[lane] : 0);
        if(lane < waves)
            wave_totals[lane] = wave_prefix;
    }
    __syncthreads();
    const int64_t result = prefix + (wave > 0 ? wave_totals[wave - 1] : 0);
    __syncthreads();
    return result;
}

__device__ __forceinline__ void pa_metadata_v1_2_emit(
    PaMetadataV1KernelParameter params, int32_t sequence, int32_t part,
    int32_t sequence_parts, int64_t pages_before, int32_t tasks_before,
    int32_t reductions_before, int32_t partials_before, int32_t total_tasks,
    int64_t total_pages)
{
    const int32_t workers = params.num_splits / params.num_heads_k;
    const int32_t first_worker = pa_metadata_v1_2_worker(pages_before, total_pages, workers);
    const int32_t worker = first_worker + part;
    const int64_t average = pa_metadata_v1_2_divide(total_pages, workers);
    const int64_t remainder = total_pages - average * workers;
    const int32_t pages = (static_cast<uint32_t>(max(params.p_context_lens[sequence], 0)) +
                           params.block_size - 1) >> (__ffs(params.block_size) - 1);
    const int64_t begin = max(pages_before, worker * average + min(int64_t{worker}, remainder));
    const int64_t end = min(pages_before + pages,
        (worker + 1) * average + min(int64_t{worker + 1}, remainder));
    const int32_t qo_start = params.p_seqlens_qo_indptr[sequence];
    const int32_t qo_end = params.p_seqlens_qo_indptr[sequence + 1];
    const int32_t partial = sequence_parts > 1 ? partials_before + part * (qo_end - qo_start) : -1;
    if(sequence_parts > 1)
        params.p_reduce_partial_map[reductions_before + part] = partial;
    auto* works = reinterpret_cast<PaWorkInfo*>(params.p_work_info_set_raw);
    for(int32_t head = 0; head < params.num_heads_k; ++head)
    {
        PaWorkInfo work{};
        work.batch_idx = sequence;
        work.partial_qo_loc = partial;
        work.qo_start = qo_start;
        work.qo_end = qo_end;
        work.kv_start = params.p_pages_kv_indptr[sequence] + static_cast<int32_t>(begin - pages_before);
        work.kv_end = params.p_pages_kv_indptr[sequence] + static_cast<int32_t>(end - pages_before);
        work.kv_offset = head * workers + worker;
        work.q_head_range = static_cast<int32_t>(
            (static_cast<uint32_t>((head + 1) * params.qhead_granularity) << 16) |
            static_cast<uint32_t>(head * params.qhead_granularity));
        works[head * total_tasks + tasks_before + part] = work;
    }
}

__device__ __forceinline__ void pa_metadata_v1_2_finalize(PaMetadataV1KernelParameter params)
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
    __shared__ int64_t wave_totals[kPaMetadataSharedMaxWaves];
    int32_t groups = 0;
    if(total_reduce > 0)
    {
        const int32_t wave_size = opus::get_warp_size();
        const bool wave_only = blockDim.x == wave_size;
        for(int32_t begin = 0; begin < params.num_batches; begin += blockDim.x)
        {
            const int32_t sequence = begin + threadIdx.x;
            const bool active = sequence < params.num_batches;
            const int32_t reduce_begin = active ? params.p_reduce_indptr[sequence] : 0;
            const int32_t reduce_end = active ? params.p_reduce_indptr[sequence + 1] : 0;
            const bool split = reduce_end > reduce_begin;
            const int32_t prefix = static_cast<int32_t>(pa_metadata_v1_2_block_prefix(
                split, wave_totals, wave_only));
            const int32_t count = wave_only ? __shfl(prefix, wave_size - 1) :
                static_cast<int32_t>(wave_totals[blockDim.x / wave_size - 1]);
            __syncthreads();
            if(split)
            {
                const int32_t group = groups + prefix - 1;
                params.p_reduce_indptr[group + 1] = reduce_end;
                params.p_reduce_final_map[group * 2] = params.p_seqlens_qo_indptr[sequence];
                params.p_reduce_final_map[group * 2 + 1] = params.p_seqlens_qo_indptr[sequence + 1];
            }
            groups += count;
            __syncthreads();
        }
    }
    for(int32_t tile = groups + 1 + threadIdx.x; tile < params.reduce_indptr_size; tile += blockDim.x)
        params.p_reduce_indptr[tile] = total_reduce;
}

__launch_bounds__(kPaMetadataSharedMaxBatches, 1) __global__
void kn_get_pa_metadata_v1_2_shared(PaMetadataV1KernelParameter params)
{
    __shared__ int64_t wave_totals[kPaMetadataSharedMaxWaves];
    __shared__ int64_t page_prefixes[kPaMetadataSharedMaxBatches];
    __shared__ int32_t task_prefixes[kPaMetadataSharedMaxBatches];
    __shared__ int32_t partial_prefixes[kPaMetadataSharedMaxBatches];
    __shared__ int64_t total_pages;
    __shared__ int32_t total_tasks;
    const int32_t sequence = threadIdx.x;
    const int32_t wave_size = opus::get_warp_size();
    const int32_t wave = sequence / wave_size;
    const int32_t waves = blockDim.x / wave_size;
    const bool wave_only = params.num_batches <= wave_size;
    if(!wave_only || wave == 0)
    {
        const bool active = sequence < params.num_batches;
        const int32_t qo_length = active ? params.p_seqlens_qo_indptr[sequence + 1] -
                                           params.p_seqlens_qo_indptr[sequence] : 0;
        const uint32_t context = active ? max(params.p_context_lens[sequence], 0) : 0;
        const int64_t pages = (context + params.block_size - 1) >> (__ffs(params.block_size) - 1);
        const int64_t page_prefix = pa_metadata_v1_2_block_prefix(pages, wave_totals, wave_only);
        const int64_t page_count = wave_only ? __shfl(static_cast<long long>(page_prefix), wave_size - 1) :
            wave_totals[waves - 1];
        if(!wave_only)
            __syncthreads();
        const int32_t parts = pa_metadata_v1_2_parts(page_prefix - pages, pages,
            max(page_count, int64_t{1}), params.num_splits / params.num_heads_k);
        const int32_t reductions = parts > 1 ? parts : 0;
        const int32_t task_prefix = static_cast<int32_t>(pa_metadata_v1_2_block_prefix(
            parts, wave_totals, wave_only));
        const int32_t task_count = wave_only ? __shfl(task_prefix, wave_size - 1) :
            static_cast<int32_t>(wave_totals[waves - 1]);
        if(!wave_only)
            __syncthreads();
        const int32_t reduce_prefix = static_cast<int32_t>(pa_metadata_v1_2_block_prefix(
            reductions, wave_totals, wave_only));
        const int32_t partial_prefix = static_cast<int32_t>(pa_metadata_v1_2_block_prefix(
            reductions * qo_length, wave_totals, wave_only));
        if(active)
        {
            page_prefixes[sequence] = page_prefix;
            task_prefixes[sequence] = task_prefix;
            partial_prefixes[sequence] = partial_prefix;
            params.p_reduce_indptr[sequence + 1] = reduce_prefix;
        }
        if(sequence == 0)
        {
            params.p_reduce_indptr[0] = 0;
            params.p_work_indptr[params.num_cu] = task_count * params.num_heads_k;
            params.p_work_metadata_ptrs[0] = reinterpret_cast<uintptr_t>(params.p_work_indptr);
            params.p_work_metadata_ptrs[1] = reinterpret_cast<uintptr_t>(params.p_work_info_set_raw);
            total_pages = max(page_count, int64_t{1});
            total_tasks = task_count;
        }
    }
    __syncthreads();
    for(int32_t task = threadIdx.x; task < total_tasks; task += blockDim.x)
    {
        int32_t begin = 0;
        int32_t end = params.num_batches;
        while(begin < end)
        {
            const int32_t middle = begin + (end - begin) / 2;
            if(task_prefixes[middle] <= task)
                begin = middle + 1;
            else
                end = middle;
        }
        const int32_t tasks_before = begin > 0 ? task_prefixes[begin - 1] : 0;
        pa_metadata_v1_2_emit(params, begin, task - tasks_before,
            task_prefixes[begin] - tasks_before, begin > 0 ? page_prefixes[begin - 1] : 0,
            tasks_before, params.p_reduce_indptr[begin], begin > 0 ? partial_prefixes[begin - 1] : 0,
            total_tasks, total_pages);
    }
    __syncthreads();
    pa_metadata_v1_2_finalize(params);
}

__device__ __forceinline__ PaMetadataV12ChunkRecord* pa_metadata_v1_2_scratch(
    PaMetadataV1KernelParameter params)
{
    const int64_t capacity = (static_cast<int64_t>(params.num_batches) +
                             params.num_splits / params.num_heads_k - 1) * params.num_heads_k;
    auto* works = reinterpret_cast<PaWorkInfo*>(params.p_work_info_set_raw);
    return reinterpret_cast<PaMetadataV12ChunkRecord*>(works + capacity);
}

__launch_bounds__(kPaMetadataSharedMaxBatches, 1) __global__
void kn_scan_pa_metadata_v1_2_chunks(PaMetadataV1KernelParameter params)
{
    __shared__ int64_t wave_totals[kPaMetadataSharedMaxWaves];
    const int32_t sequence = blockIdx.x * kPaMetadataSharedMaxBatches + threadIdx.x;
    const bool active = sequence < params.num_batches;
    const uint32_t context = active ? max(params.p_context_lens[sequence], 0) : 0;
    const int64_t pages = (context + params.block_size - 1) >> (__ffs(params.block_size) - 1);
    const int64_t prefix = pa_metadata_v1_2_block_prefix(pages, wave_totals, false);
    auto* rows = pa_metadata_v1_2_scratch(params);
    if(active)
        rows[sequence].pages = prefix;
    if(threadIdx.x == blockDim.x - 1)
        rows[params.num_batches + blockIdx.x].pages = prefix;
}

__launch_bounds__(kPaMetadataSharedMaxBatches, 1) __global__
void kn_plan_pa_metadata_v1_2_chunks(PaMetadataV1KernelParameter params)
{
    __shared__ int64_t wave_totals[kPaMetadataSharedMaxWaves];
    const int32_t sequence = blockIdx.x * kPaMetadataSharedMaxBatches + threadIdx.x;
    const bool active = sequence < params.num_batches;
    const int32_t chunks = (params.num_batches + kPaMetadataSharedMaxBatches - 1) /
                           kPaMetadataSharedMaxBatches;
    auto* rows = pa_metadata_v1_2_scratch(params);
    auto* totals = rows + params.num_batches;
    auto* counts = totals + chunks;
    int64_t total_pages = 0;
    int64_t page_base = 0;
    for(int32_t chunk = 0; chunk < chunks; ++chunk)
    {
        total_pages += totals[chunk].pages;
        if(chunk < blockIdx.x)
            page_base += totals[chunk].pages;
    }
    const int32_t qo_length = active ? params.p_seqlens_qo_indptr[sequence + 1] -
                                       params.p_seqlens_qo_indptr[sequence] : 0;
    const uint32_t context = active ? max(params.p_context_lens[sequence], 0) : 0;
    const int64_t pages = (context + params.block_size - 1) >> (__ffs(params.block_size) - 1);
    const int64_t pages_before = active ? page_base + rows[sequence].pages - pages : 0;
    const int32_t parts = pa_metadata_v1_2_parts(pages_before, pages,
        max(total_pages, int64_t{1}), params.num_splits / params.num_heads_k);
    const int32_t reductions = parts > 1 ? parts : 0;
    const int32_t task_prefix = static_cast<int32_t>(pa_metadata_v1_2_block_prefix(
        parts, wave_totals, false));
    const int32_t reduce_prefix = static_cast<int32_t>(pa_metadata_v1_2_block_prefix(
        reductions, wave_totals, false));
    const int32_t partial_prefix = static_cast<int32_t>(pa_metadata_v1_2_block_prefix(
        reductions * qo_length, wave_totals, false));
    if(active)
    {
        rows[sequence].pages = pages_before;
        rows[sequence].tasks = task_prefix;
        rows[sequence].reductions = reduce_prefix;
        rows[sequence].partials = partial_prefix;
    }
    if(threadIdx.x == blockDim.x - 1)
    {
        counts[blockIdx.x].tasks = task_prefix;
        counts[blockIdx.x].reductions = reduce_prefix;
        counts[blockIdx.x].partials = partial_prefix;
    }
}

__launch_bounds__(kPaMetadataSharedMaxBatches, 1) __global__
void kn_emit_pa_metadata_v1_2_chunks(PaMetadataV1KernelParameter params)
{
    const int32_t sequence = blockIdx.x * kPaMetadataSharedMaxBatches + threadIdx.x;
    if(sequence >= params.num_batches)
        return;
    const int32_t chunks = (params.num_batches + kPaMetadataSharedMaxBatches - 1) /
                           kPaMetadataSharedMaxBatches;
    auto* rows = pa_metadata_v1_2_scratch(params);
    auto* totals = rows + params.num_batches;
    auto* counts = totals + chunks;
    int64_t total_pages = 0;
    int32_t total_tasks = 0;
    int32_t task_base = 0;
    int32_t reduce_base = 0;
    int32_t partial_base = 0;
    for(int32_t chunk = 0; chunk < chunks; ++chunk)
    {
        total_pages += totals[chunk].pages;
        total_tasks += counts[chunk].tasks;
        if(chunk < blockIdx.x)
        {
            task_base += counts[chunk].tasks;
            reduce_base += counts[chunk].reductions;
            partial_base += counts[chunk].partials;
        }
    }
    const int32_t tasks_before = threadIdx.x > 0 ? rows[sequence - 1].tasks : 0;
    const int32_t reductions_before = threadIdx.x > 0 ? rows[sequence - 1].reductions : 0;
    const int32_t partials_before = threadIdx.x > 0 ? rows[sequence - 1].partials : 0;
    const int32_t parts = rows[sequence].tasks - tasks_before;
    params.p_reduce_indptr[sequence + 1] = reduce_base + rows[sequence].reductions;
    if(sequence == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_indptr[params.num_cu] = total_tasks * params.num_heads_k;
        params.p_work_metadata_ptrs[0] = reinterpret_cast<uintptr_t>(params.p_work_indptr);
        params.p_work_metadata_ptrs[1] = reinterpret_cast<uintptr_t>(params.p_work_info_set_raw);
    }
    for(int32_t part = 0; part < parts; ++part)
        pa_metadata_v1_2_emit(params, sequence, part, parts, rows[sequence].pages,
            task_base + tasks_before, reduce_base + reductions_before, partial_base + partials_before,
            total_tasks, max(total_pages, int64_t{1}));
}

__launch_bounds__(kPaMetadataSharedMaxBatches, 1) __global__
void kn_finalize_pa_metadata_v1_2(PaMetadataV1KernelParameter params)
{
    pa_metadata_v1_2_finalize(params);
}

template <int32_t kPackedQoLenPerWg_,
          bool kQoSplits_,
          int32_t kUniSeqlenQo_,
          bool kLdsBatchInfo_,
          bool kIsSparse_ = false>
struct PaMetadataV12Traits
{
    static constexpr int32_t kPackedQoLenPerWg      = kPackedQoLenPerWg_;
    static constexpr int32_t kPackedQoLenPerWg_log2 = __builtin_ctz(kPackedQoLenPerWg);
    static constexpr bool kQoSplits                 = kQoSplits_;
    // <= -1: read from seqlens_qo_indptr
    // ==  0: read from PaMetadataV1KernelParameter::uni_seqlen_qo
    // >=  1: read from PaMetadataV12Traits::kUniSeqlenQo
    static constexpr int32_t kUniSeqlenQo            = kUniSeqlenQo_;
    static constexpr int32_t kFixedOverheadNumBlocks = 0;
    static constexpr int32_t kIsSparse               = kIsSparse_;
    static constexpr int32_t kLdsBatchInfo           = kLdsBatchInfo_;
};

template <typename Traits>
__launch_bounds__(opus::get_warp_size(), 1) __global__
    void kn_get_pa_metadata_v1_2(PaMetadataV1KernelParameter params)
{
    using QoState = QoState<Traits>;

    extern __shared__ uint8_t p_smem[];
    int32_t* p_lds_seqlens_qo = reinterpret_cast<int32_t*>(p_smem);
    int32_t* p_lds_pages_kv   = p_lds_seqlens_qo + (QoState::is_unique() ? 0 : params.num_batches);
    int32_t* p_lds_partial_info = p_lds_pages_kv + (Traits::kLdsBatchInfo ? params.num_batches : 0);

    QoState qo_state(
        params.uni_seqlen_qo, params.ori_seqlen_qo, p_lds_seqlens_qo, params.p_seqlens_qo_indptr);

    const int32_t lane_idx = opus::lane_id();

    PaWorkInfo* p_work_info_set = reinterpret_cast<PaWorkInfo*>(params.p_work_info_set_raw);

    int32_t sum_blocks = 0;
    for(int32_t bid = lane_idx; bid < params.num_batches; bid += opus::get_warp_size())
    {
        const int32_t bid_ori = Traits::kIsSparse
                                    ? (bid / params.ori_seqlen_qo / params.qk_batch_ratio)
                                    : (bid / params.qk_batch_ratio);
        const int32_t qo_length =
            params.p_seqlens_qo_indptr[bid_ori + 1] - params.p_seqlens_qo_indptr[bid_ori];
        const int32_t kv_length = params.p_context_lens[bid_ori];

        int32_t num_blocks = 0;
        for(int32_t qo_offset = 0; qo_offset < qo_length; qo_offset += params.qlen_granularity)
        {
            const int32_t local_qo_start = qo_offset;
            const int32_t local_qo_end   = min(local_qo_start + params.qlen_granularity, qo_length);
            const int32_t effective_kv_length =
                params.is_causal ? std::min(kv_length - qo_length + local_qo_end, kv_length)
                                 : kv_length;
            num_blocks = integer_divide_ceil_power2(
                effective_kv_length, params.kv_granularity, params.kv_granularity_log2);
            if(Traits::kIsSparse)
                num_blocks = min(num_blocks, params.topk);
        }

        if constexpr(Traits::kLdsBatchInfo)
        {
            p_lds_pages_kv[bid] = num_blocks;
        }

        sum_blocks += num_blocks;

        if constexpr(QoState::is_unique() == false)
        {
            p_lds_seqlens_qo[bid] =
                params.p_seqlens_qo_indptr[bid_ori + 1] - params.p_seqlens_qo_indptr[bid_ori];
        }
    }

    sum_blocks =
        aiter::warpReduce<aiter::AddFunctor, decltype(sum_blocks), opus::get_warp_size()>(
            sum_blocks);
    sum_blocks += params.num_batches * Traits::kFixedOverheadNumBlocks;

    if(lane_idx == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_indptr[0]   = 0;
        params.p_work_metadata_ptrs[0] =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(params.p_work_indptr));
        params.p_work_metadata_ptrs[1] =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p_work_info_set));
    }

    // expected payload handled by each cu part.
    const int32_t num_splits_per_khead = params.num_splits / params.num_heads_k;
    const int32_t average              = sum_blocks / num_splits_per_khead;
    const int32_t reminder             = sum_blocks % num_splits_per_khead;

    int32_t cid                    = 0; // CU ID
    int32_t num_works              = 0;
    int32_t last_reduce_indptr     = 0;
    int32_t global_reduce_tile_idx = 0;

    for(int32_t khead_idx = 0; khead_idx < params.num_heads_k; ++khead_idx)
    {
        // Note:
        // To control divide strategy more flexibly, store khead info by metadata.qhead_range
        // instead of launch config. Considering current reduce implementation, khead loop should be
        // in the outer since different khead need share same divide strategy. Therefor need reset
        // state variables before allocate work by cu loop -> batch loop).
        int32_t curr_batch        = 0; // batch ID of the batch which is under review
        int32_t curr_kv_block     = 0; // #blocks handled by previous cu part(s)
        int32_t curr_n_split_idx  = 0; // #cu parts used to handle current batch
        int32_t curr_sub_head_idx = 0;

        int32_t curr_kv_begin = 0;
        // The size of 1st element equals to the end loc of the 1st element.
        int32_t curr_kv_end   = Traits::kLdsBatchInfo ? p_lds_pages_kv[0]
                                : Traits::kIsSparse ? min(params.p_pages_kv_indptr[1], params.topk)
                                                    : params.p_pages_kv_indptr[1];
        int32_t curr_kv_pages = curr_kv_end - curr_kv_begin;

        int32_t partial_idx    = 0;
        last_reduce_indptr     = 0;
        global_reduce_tile_idx = 0;
        // allocate work
        while(cid < params.num_cu)
        {
            int32_t remain_payload =
                ((cid % num_splits_per_khead) < reminder) ? (average + 1) : average;
            while(curr_batch < params.num_batches)
            {
                const int32_t packed_qo_len =
                    qo_state.get_seqlen(curr_batch) * params.qhead_granularity;
                const int32_t num_qo_tiles =
                    Traits::kQoSplits ? integer_divide_ceil_power2(packed_qo_len,
                                                                   Traits::kPackedQoLenPerWg,
                                                                   Traits::kPackedQoLenPerWg_log2)
                                      : 1;
                const int32_t qo_tile_size =
                    integer_divide_ceil(qo_state.get_seqlen(curr_batch), num_qo_tiles);
                const int32_t num_kv_blocks    = curr_kv_pages;
                const int32_t remain_kv_blocks = num_kv_blocks - curr_kv_block;

                // If current cu part is able to handle this batch of seqences
                if(remain_payload >= (remain_kv_blocks + Traits::kFixedOverheadNumBlocks))
                {
                    const int32_t consuming_blks = remain_kv_blocks;
                    const int32_t num_splits     = curr_n_split_idx + 1;

                    auto fill_work_info = [&](const int32_t qo_tile_idx,
                                              const int32_t split_idx,
                                              const int32_t khead_idx = 0) {
                        PaWorkInfo work_info{};
                        work_info.batch_idx = curr_batch;
                        work_info.qo_start =
                            qo_state.get_begin(curr_batch) + qo_tile_idx * qo_tile_size;
                        work_info.qo_end   = opus::min(work_info.qo_start + qo_tile_size,
                                                        qo_state.get_end(curr_batch));
                        work_info.kv_start = curr_kv_begin + curr_kv_block;
                        work_info.kv_end   = opus::min(work_info.kv_start + consuming_blks,
                        integer_divide_ceil_power2(curr_kv_end * params.kv_granularity -
                                                       (num_qo_tiles - 1 - qo_tile_idx),
                                                   params.kv_granularity,
                                                   params.kv_granularity_log2));
                        work_info.kv_offset = 0;
                        work_info.q_head_range =
                            qo_state.get_q_head_range(khead_idx * params.qhead_granularity,
                                                      (khead_idx + 1) * params.qhead_granularity);

                        // split related info
                        if(curr_n_split_idx > 0)
                        {
                            // set work info
                            work_info.partial_qo_loc = partial_idx + qo_tile_idx * qo_tile_size;

                            // set reduce info
                            if(split_idx == 0)
                            {
                                const int32_t reduce_tile_idx = global_reduce_tile_idx + qo_tile_idx;
                                params.p_reduce_indptr[reduce_tile_idx + 1] =
                                    last_reduce_indptr + (qo_tile_idx + 1) * num_splits;
                                params.p_reduce_final_map[reduce_tile_idx * 2] =
                                    work_info.qo_start;
                                params.p_reduce_final_map[reduce_tile_idx * 2 + 1] =
                                    work_info.qo_end;
                            }

                            if constexpr(Traits::kQoSplits)
                            {
                                const int32_t partial_qo_loc =
                                    (split_idx < (num_splits - 1))
                                        ? p_lds_partial_info[qo_tile_idx + split_idx * num_qo_tiles]
                                        : work_info.partial_qo_loc;
                                params.p_reduce_partial_map[last_reduce_indptr +
                                                            qo_tile_idx * num_splits + split_idx] =
                                    partial_qo_loc;
                            }
                            else
                            {
                                params.p_reduce_partial_map[last_reduce_indptr + split_idx] =
                                    partial_idx - (curr_n_split_idx - split_idx) * qo_tile_size;
                            }
                        }
                        else
                        {
                            work_info.partial_qo_loc = -1;
                        }

                        if(split_idx == 0)
                            p_work_info_set[num_works + qo_tile_idx] = work_info;
                    };

                    // record a work in work_info_set
                    if constexpr(Traits::kQoSplits)
                    {
                        if(curr_n_split_idx > 0)
                        {
                            for(int32_t idx = lane_idx; idx < num_splits * num_qo_tiles;
                                idx += opus::get_warp_size())
                            {
                                const int32_t qo_tile_idx = idx % num_qo_tiles;
                                const int32_t split_idx   = idx / num_qo_tiles;
                                fill_work_info(qo_tile_idx, split_idx, khead_idx);
                            }

                            partial_idx += num_qo_tiles * qo_tile_size;
                            last_reduce_indptr += num_qo_tiles * num_splits;
                        }
                        else
                        {
                            for(int32_t idx = lane_idx; idx < num_qo_tiles;
                                idx += opus::get_warp_size())
                            {
                                fill_work_info(idx, 0, khead_idx);
                            }
                        }
                    }
                    else
                    {
                        if(curr_n_split_idx > 0)
                        {
                            for(int32_t idx = lane_idx; idx < num_splits;
                                idx += opus::get_warp_size())
                            {
                                fill_work_info(0, idx, khead_idx);
                            }

                            partial_idx += qo_tile_size;
                            last_reduce_indptr += num_splits;
                        }
                        else
                        {
                            fill_work_info(0, 0, khead_idx);
                        }
                    }

                    if(curr_n_split_idx > 0)
                        global_reduce_tile_idx += num_qo_tiles;
                    num_works += num_qo_tiles;

                    // update state
                    remain_payload -= (remain_kv_blocks + Traits::kFixedOverheadNumBlocks);
                    ++curr_batch;
                    // same as curr_sub_head_idx = curr_batch % params.qk_batch_ratio;
                    curr_sub_head_idx = (curr_sub_head_idx == (params.qk_batch_ratio - 1))
                                            ? 0
                                            : (curr_sub_head_idx + 1);
                    if(curr_batch < params.num_batches)
                    {
                        if(curr_sub_head_idx == 0)
                        {
                            if constexpr(Traits::kLdsBatchInfo)
                            {
                                curr_kv_pages = p_lds_pages_kv[curr_batch];
                            }
                            else
                            {
                                const int32_t bid_ori = Traits::kIsSparse
                                                            ? (curr_batch / params.ori_seqlen_qo /
                                                               params.qk_batch_ratio)
                                                            : (curr_batch / params.qk_batch_ratio);
                                curr_kv_pages         = params.p_pages_kv_indptr[bid_ori + 1] -
                                                params.p_pages_kv_indptr[bid_ori];
                                curr_kv_pages = Traits::kIsSparse ? min(curr_kv_pages, params.topk)
                                                                  : curr_kv_pages;
                            }
                            curr_kv_begin =
                                Traits::kIsSparse ? (curr_kv_begin + params.topk) : curr_kv_end;
                            curr_kv_end = curr_kv_begin + curr_kv_pages;
                        }
                        curr_kv_block    = 0;
                        curr_n_split_idx = 0;
                    }
                }
                else
                {
                    if(remain_payload > Traits::kFixedOverheadNumBlocks)
                    {
                        const int32_t consuming_blks =
                            remain_payload - Traits::kFixedOverheadNumBlocks;

                        auto fill_work_info = [&](const int32_t qo_tile_idx,
                                                  const int32_t khead_idx = 0) {
                            PaWorkInfo work_info{};
                            work_info.batch_idx = curr_batch;
                            work_info.qo_start =
                                qo_state.get_begin(curr_batch) + qo_tile_idx * qo_tile_size;
                            work_info.qo_end   = opus::min(work_info.qo_start + qo_tile_size,
                                                            qo_state.get_end(curr_batch));
                            work_info.kv_start = curr_kv_begin + curr_kv_block;
                            work_info.kv_end   = opus::min(
                                work_info.kv_start + consuming_blks,
                                integer_divide_ceil_power2(curr_kv_end * params.kv_granularity -
                                                               (num_qo_tiles - 1 - qo_tile_idx),
                                                           params.kv_granularity,
                                                           params.kv_granularity_log2));
                            work_info.kv_offset    = 0;
                            work_info.q_head_range = qo_state.get_q_head_range(
                                khead_idx * params.qhead_granularity,
                                (khead_idx + 1) * params.qhead_granularity);
                            work_info.partial_qo_loc = partial_idx + qo_tile_idx * qo_tile_size;
                            p_work_info_set[num_works + qo_tile_idx] = work_info;

                            if constexpr(Traits::kQoSplits)
                            {
                                p_lds_partial_info[curr_n_split_idx * num_qo_tiles + qo_tile_idx] =
                                    work_info.partial_qo_loc;
                            }
                        };

                        // record a work in work_info_set
                        if constexpr(Traits::kQoSplits)
                        {
                            for(int32_t qo_tile_idx = lane_idx; qo_tile_idx < num_qo_tiles;
                                qo_tile_idx += opus::get_warp_size())
                            {
                                fill_work_info(qo_tile_idx, khead_idx);
                            }
                        }
                        else
                        {
                            fill_work_info(0, khead_idx);
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
            params.p_work_indptr[cid + 1] = num_works;
            ++cid;
            if(curr_batch == params.num_batches)
            {
                break;
            }
        }
    }

    for(int32_t i = cid + lane_idx; i <= params.num_cu; i += opus::get_warp_size())
    {
        params.p_work_indptr[i] = num_works;
    }

    global_reduce_tile_idx = __shfl(global_reduce_tile_idx, 0);
    for(int32_t i = global_reduce_tile_idx + lane_idx; i < params.reduce_indptr_size;
        i += opus::get_warp_size())
    {
        params.p_reduce_indptr[i] = last_reduce_indptr;
    }
}

template <int32_t kPackedQoLenPerWg, bool kQoSplits, int32_t kUniSeqlenQo, bool kIsSparse>
void dispatch_pa_metadata_v1_2_device(const PaMetadataV1KernelParameter& params,
                                      const hipStream_t stream,
                                      const int32_t max_seqlen_qo,
                                      const int32_t warp_size,
                                      const int32_t lds_size)
{
    const dim3 grid = dim3(1, 1, 1);

    using DummyTraits =
        PaMetadataV12Traits<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, true, kIsSparse>;
    const int32_t lds_bytes_per_batch =
        sizeof(int32_t) * (QoState<DummyTraits>::is_unique() ? 1 : 2);
    const int32_t max_qo_tiles =
        kQoSplits ? (integer_divide_ceil(max_seqlen_qo, kPackedQoLenPerWg)) : 1;
    const int32_t lds_bytes_partial_info =
        kQoSplits ? params.num_cu * max_qo_tiles * sizeof(int32_t) : 0;
    const int32_t max_lds_batch_size = (lds_size - lds_bytes_partial_info) / lds_bytes_per_batch;

    if(params.num_batches <= max_lds_batch_size)
    {
        using Traits =
            PaMetadataV12Traits<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, true, kIsSparse>;
        kn_get_pa_metadata_v1_2<Traits><<<grid, warp_size, lds_size, stream>>>(params);
    }
    else
    {
        using Traits =
            PaMetadataV12Traits<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, false, kIsSparse>;
        kn_get_pa_metadata_v1_2<Traits><<<grid, warp_size, lds_size, stream>>>(params);
    }
}

void get_pa_metadata_v1_2_device(const aiter_tensor_t& seqlens_qo_indptr, // [batch size + 1]
                                 const aiter_tensor_t& pages_kv_indptr,   // [batch size + 1]
                                 const aiter_tensor_t& context_lens,      // [batch size]
                                 const int32_t num_heads_per_head_k,
                                 const int32_t num_heads_k,
                                 const bool is_causal,
                                 const int32_t kv_granularity,
                                 const int32_t block_size,
                                 const int32_t max_seqlen_qo,
                                 const int32_t ori_uni_seqlen_qo,
                                 const bool fast_mode,
                                 const int32_t topk,
                                 const int32_t max_split_per_batch,
                                 aiter_tensor_t& work_metadata_ptrs,
                                 aiter_tensor_t& work_info_set,
                                 aiter_tensor_t& work_indptr,
                                 aiter_tensor_t& reduce_indptr,
                                 aiter_tensor_t& reduce_final_map,
                                 aiter_tensor_t& reduce_partial_map)
{
    constexpr int32_t kPackedQoLenPerWg = 128;

    const hipStream_t stream = aiter::getCurrentHIPStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);

    const int32_t num_cu = dev_prop.multiProcessorCount;

    int32_t num_batches    = context_lens.size(0);
    int32_t num_heads      = num_heads_k * num_heads_per_head_k;
    int32_t qk_batch_ratio = 1;
    int32_t uni_seqlen_qo  = ori_uni_seqlen_qo;

    int32_t num_splits = max_split_per_batch < 0 ? num_cu : static_cast<int32_t>(
        std::min<int64_t>(num_cu, static_cast<int64_t>(max_split_per_batch) * num_batches));

    AITER_CHECK(num_heads_k > 0 && num_splits > 0 && num_splits % num_heads_k == 0,
                __func__,
                ": only supports #num_splits evenly divisible by #kv_heads.");

    PaMetadataV1KernelParameter params = {};
    params.p_work_metadata_ptrs        = static_cast<uint64_t*>(work_metadata_ptrs.data_ptr());
    params.p_work_indptr               = static_cast<int32_t*>(work_indptr.data_ptr());
    params.p_work_info_set_raw         = static_cast<int32_t*>(work_info_set.data_ptr());
    params.p_reduce_indptr             = static_cast<int32_t*>(reduce_indptr.data_ptr());
    params.p_reduce_final_map          = static_cast<int32_t*>(reduce_final_map.data_ptr());
    params.p_reduce_partial_map        = static_cast<int32_t*>(reduce_partial_map.data_ptr());
    params.p_seqlens_qo_indptr         = static_cast<int32_t*>(seqlens_qo_indptr.data_ptr());
    params.p_pages_kv_indptr           = static_cast<int32_t*>(pages_kv_indptr.data_ptr());
    params.p_context_lens              = static_cast<int32_t*>(context_lens.data_ptr());
    params.num_batches                 = num_batches;
    params.num_heads                   = num_heads_k * num_heads_per_head_k;
    params.num_heads_k                 = num_heads_k;
    params.num_cu                      = num_cu;
    params.num_splits                  = num_splits;
    params.reduce_indptr_size          = reduce_indptr.size(0);
    params.qhead_granularity           = num_heads_per_head_k;
    params.qlen_granularity            = max_seqlen_qo;
    params.kv_granularity              = kv_granularity;
    params.kv_granularity_log2         = __builtin_ctz(kv_granularity);
    params.block_size                  = block_size;
    params.blocks_per_unit             = kv_granularity / block_size;
    params.uni_seqlen_qo               = uni_seqlen_qo;
    params.ori_seqlen_qo               = ori_uni_seqlen_qo;
    params.is_causal                   = is_causal;
    params.topk                        = topk;
    params.qk_batch_ratio              = qk_batch_ratio;

    const int64_t work_capacity = static_cast<int64_t>(num_batches + num_cu / num_heads_k - 1) *
                                  num_heads_k;
    const int32_t chunks =
        (num_batches + kPaMetadataSharedMaxBatches - 1) / kPaMetadataSharedMaxBatches;
    const int64_t scratch_rows =
        num_batches > kPaMetadataSharedMaxBatches ? static_cast<int64_t>(num_batches) + 2 * chunks
                                                  : 0;

    // The parallel kernels only reproduce the dense, single-Q-tile, full-CU, page-sized layout.
    const bool layout_supported =
        topk < 0 && num_splits == num_cu && kv_granularity == block_size && block_size > 0 &&
        (block_size & (block_size - 1)) == 0 && max_seqlen_qo > 0 && num_heads_per_head_k > 0 &&
        static_cast<int64_t>(max_seqlen_qo) * num_heads_per_head_k <= kPackedQoLenPerWg &&
        static_cast<int64_t>(num_heads_k) * num_heads_per_head_k <= 65535;
    // Parallel output needs scratch rows; smaller legacy allocations fall back to serial.
    const bool buffers_fit =
        work_info_set.numel() >= (work_capacity + scratch_rows) * 8 &&
        reduce_partial_map.numel() >=
            std::min<int64_t>(work_capacity / num_heads_k, 2 * (num_cu / num_heads_k - 1));
    if(fast_mode && num_batches > 0 && num_batches <= kPaMetadataMaxParallelBatches &&
       layout_supported && buffers_fit)
    {
        if(num_batches <= kPaMetadataSharedMaxBatches)
        {
            const int32_t threads = std::min(kPaMetadataSharedMaxBatches,
                std::max(dev_prop.warpSize, next_power_of_two(std::max(num_batches,
                    static_cast<int32_t>(work_capacity / num_heads_k)))));
            kn_get_pa_metadata_v1_2_shared<<<1, threads, 0, stream>>>(params);
            HIP_CALL_LAUNCH(hipGetLastError());
        }
        else
        {
            kn_scan_pa_metadata_v1_2_chunks<<<chunks, kPaMetadataSharedMaxBatches, 0, stream>>>(params);
            HIP_CALL_LAUNCH(hipGetLastError());
            kn_plan_pa_metadata_v1_2_chunks<<<chunks, kPaMetadataSharedMaxBatches, 0, stream>>>(params);
            HIP_CALL_LAUNCH(hipGetLastError());
            kn_emit_pa_metadata_v1_2_chunks<<<chunks, kPaMetadataSharedMaxBatches, 0, stream>>>(params);
            HIP_CALL_LAUNCH(hipGetLastError());
            kn_finalize_pa_metadata_v1_2<<<1, kPaMetadataSharedMaxBatches, 0, stream>>>(params);
            HIP_CALL_LAUNCH(hipGetLastError());
        }
        return;
    }

    // launch kernel
    MLA_METADATA_DISPATCHER(
        max_seqlen_qo * num_heads_per_head_k,
        kPackedQoLenPerWg,
        params.uni_seqlen_qo,
        topk,
        dispatch_pa_metadata_v1_2_device<kPackedQoLenPerWg, kQoSplits, kUniSeqlenQo, kIsSparse>(
            params,
            stream,
            max_seqlen_qo,
            dev_prop.warpSize,
            dev_prop.maxSharedMemoryPerMultiProcessor));
}
