#pragma once

#include "v1_comm.cuh"

constexpr int32_t kPaMetadataSharedMaxBatches = 512;
constexpr int32_t kPaMetadataSharedMaxWaves = kPaMetadataSharedMaxBatches / opus::get_warp_size();
constexpr int32_t kPaMetadataMaxBatches = 32768;

struct PaMetadataV1ChunkRecord
{
    int64_t cost;
    int32_t tasks;
    int32_t reductions;
    int32_t partials;
    int32_t reserved[3];
};

static_assert(sizeof(PaMetadataV1ChunkRecord) == sizeof(PaWorkInfo));

__device__ __forceinline__ void pa_metadata_v1_1_finalize(PaMetadataV1KernelParameter params);

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

__device__ __forceinline__ int64_t pa_metadata_v1_1_divide(int64_t numerator, int64_t denominator)
{
    if(numerator <= int64_t{0xffffffff} && denominator <= int64_t{0xffffffff})
        return static_cast<uint32_t>(numerator) / static_cast<uint32_t>(denominator);
    return numerator / denominator;
}

__device__ __forceinline__ int32_t pa_metadata_v1_1_worker(
    int64_t cost, int64_t total_cost, int32_t workers)
{
    return static_cast<int32_t>(min(pa_metadata_v1_1_divide((cost + 1) * workers - 1, total_cost),
                                     static_cast<int64_t>(workers - 1)));
}

__device__ __forceinline__ int32_t pa_metadata_v1_1_fragments(
    int64_t cost_begin, int64_t tiles, int64_t tile_cost,
    int64_t total_cost, int32_t workers)
{
    if(tiles == 0 || tile_cost == 0)
        return 0;
    const int32_t first = pa_metadata_v1_1_worker(cost_begin, total_cost, workers);
    const int32_t last = pa_metadata_v1_1_worker(
        cost_begin + (tiles - 1) * tile_cost, total_cost, workers);
    return static_cast<int32_t>(min(tiles, static_cast<int64_t>(last - first + 1)));
}

__device__ __forceinline__ int64_t pa_metadata_v1_1_boundary(
    int32_t fragment, int32_t fragments, int64_t tiles, int64_t cost_begin,
    int64_t tile_cost, int64_t total_cost, int32_t workers)
{
    if(fragment == 0)
        return 0;
    if(fragment == fragments)
        return tiles;
    if(fragments == tiles)
        return fragment;
    const int32_t first = pa_metadata_v1_1_worker(cost_begin, total_cost, workers);
    const int64_t boundary = pa_metadata_v1_1_divide(
        static_cast<int64_t>(first + fragment) * total_cost, workers);
    return min(tiles, max(int64_t{0}, pa_metadata_v1_1_divide(
        boundary - cost_begin + tile_cost - 1, tile_cost)));
}

__device__ __forceinline__ void pa_metadata_v1_1_emit(
    PaMetadataV1KernelParameter params, int32_t sequence, int32_t part,
    int32_t sequence_parts, int32_t sequence_units, int64_t sequence_cost_before,
    int32_t sequence_tasks_before, int32_t sequence_reduce_before,
    int32_t sequence_partial_before, int32_t total_tasks,
    bool proportional, int64_t denominator)
{
    const int32_t workers_per_head = params.num_cu / params.num_heads_k;
    const int32_t qo_start = params.p_seqlens_qo_indptr[sequence];
    const int32_t qo_end = params.p_seqlens_qo_indptr[sequence + 1];
    const int32_t qo_length = qo_end - qo_start;
    const int32_t kv_start = params.p_pages_kv_indptr[sequence];
    const int32_t page_shift = __ffs(params.block_size) - 1;
    const int32_t unit_shift = __ffs(params.blocks_per_unit) - 1;
    const int32_t kv_end = static_cast<int32_t>(min(
        static_cast<int64_t>(params.p_pages_kv_indptr[sequence + 1]),
        static_cast<int64_t>(kv_start) +
            ((static_cast<uint32_t>(max(params.p_context_lens[sequence], 0)) +
              params.block_size - 1) >> page_shift)));
    const int64_t tiles = (static_cast<uint32_t>(max(params.p_context_lens[sequence], 0)) +
                           params.kv_granularity - 1) >> (page_shift + unit_shift);
    const int64_t tile_cost = proportional ? 2 * tiles :
        static_cast<int64_t>(params.blocks_per_unit) * qo_length;
    const int64_t schedule_begin = proportional ?
        2 * sequence_units * sequence_cost_before + tiles : sequence_cost_before;
    const int64_t schedule_total = proportional ?
        2 * max(sequence_units, 1) * denominator : denominator;
    const int64_t schedule_units = proportional ? sequence_units : tiles;
    const int32_t fragments = proportional || params.num_splits >= workers_per_head
        ? sequence_parts : pa_metadata_v1_1_fragments(
            schedule_begin, schedule_units, tile_cost, schedule_total, workers_per_head);
    const int64_t unit_begin = pa_metadata_v1_1_boundary(
        part * fragments / sequence_parts, fragments, schedule_units, schedule_begin,
        tile_cost, schedule_total, workers_per_head);
    const int64_t unit_end = pa_metadata_v1_1_boundary(
        (part + 1) * fragments / sequence_parts, fragments, schedule_units, schedule_begin,
        tile_cost, schedule_total, workers_per_head);
    const int64_t tile_begin = proportional ?
        pa_metadata_v1_1_divide(unit_begin * tiles, sequence_units) : unit_begin;
    const int64_t tile_end = proportional ?
        pa_metadata_v1_1_divide(unit_end * tiles, sequence_units) : unit_end;
    const int32_t local_worker = pa_metadata_v1_1_worker(
        schedule_begin + unit_begin * tile_cost, schedule_total, workers_per_head);
    const int32_t partial = sequence_parts > 1
        ? sequence_partial_before + part * qo_length : -1;
    if(sequence_parts > 1)
        params.p_reduce_partial_map[sequence_reduce_before + part] = partial;
    auto* works = reinterpret_cast<PaWorkInfo*>(params.p_work_info_set_raw);
    for(int32_t head = 0; head < params.num_heads_k; ++head)
    {
        const int32_t extra_workers = params.num_cu % params.num_heads_k;
        const int32_t worker_begin = head * workers_per_head + min(head, extra_workers);
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
        work.kv_offset = worker_begin + local_worker;
        work.q_head_range = static_cast<int32_t>(
            (static_cast<uint32_t>((head + 1) * params.qhead_granularity) << 16) |
            static_cast<uint32_t>(head * params.qhead_granularity));
        works[head * total_tasks + sequence_tasks_before + part] = work;
    }
}

__device__ __forceinline__ int64_t pa_metadata_v1_1_block_prefix(
    int64_t value, int64_t* wave_totals, bool wave_only)
{
    const int32_t lane = opus::lane_id();
    const int32_t wave = threadIdx.x / opus::get_warp_size();
    const int32_t waves = blockDim.x / opus::get_warp_size();
    const int64_t prefix = pa_metadata_v1_1_prefix_sum(value);
    if(wave_only)
        return prefix;
    if(lane == opus::get_warp_size() - 1)
        wave_totals[wave] = prefix;
    __syncthreads();
    if(wave == 0)
    {
        const int64_t wave_prefix = pa_metadata_v1_1_prefix_sum(
            lane < waves ? wave_totals[lane] : 0);
        if(lane < waves)
            wave_totals[lane] = wave_prefix;
    }
    __syncthreads();
    const int64_t result = prefix + (wave > 0 ? wave_totals[wave - 1] : 0);
    __syncthreads();
    return result;
}

__launch_bounds__(kPaMetadataSharedMaxBatches, 1) __global__
void kn_get_pa_metadata_v1_1_shared(PaMetadataV1KernelParameter params)
{
    __shared__ int64_t wave_totals[kPaMetadataSharedMaxWaves];
    __shared__ int64_t maxima[kPaMetadataSharedMaxWaves];
    __shared__ int64_t cost_prefixes[kPaMetadataSharedMaxBatches];
    __shared__ int32_t task_prefixes[kPaMetadataSharedMaxBatches];
    __shared__ int32_t partial_prefixes[kPaMetadataSharedMaxBatches];
    __shared__ int32_t sequence_units[kPaMetadataSharedMaxBatches];
    __shared__ int64_t total_weight;
    __shared__ int32_t task_count;
    __shared__ bool use_proportional;
    const int32_t sequence = threadIdx.x;
    const int32_t lane = opus::lane_id();
    const int32_t wave_size = opus::get_warp_size();
    const int32_t wave = sequence / wave_size;
    const int32_t waves = blockDim.x / wave_size;
    const int32_t workers_per_head = params.num_cu / params.num_heads_k;
    const bool wave_only = params.num_batches <= wave_size;
    if(!wave_only || wave == 0)
    {
        const bool active = sequence < params.num_batches;
        const int32_t qo_start = active ? params.p_seqlens_qo_indptr[sequence] : 0;
        const int32_t qo_end = active ? params.p_seqlens_qo_indptr[sequence + 1] : 0;
        const int32_t qo_length = qo_end - qo_start;
        const uint32_t context = active ? max(params.p_context_lens[sequence], 0) : 0;
        const uint32_t pages = (context + params.block_size - 1) >> (__ffs(params.block_size) - 1);
        const int64_t tiles = (context + params.kv_granularity - 1) >>
                              (__ffs(params.kv_granularity) - 1);
        const int64_t cost = static_cast<int64_t>(pages) * qo_length;
        const int64_t cost_prefix = pa_metadata_v1_1_block_prefix(cost, wave_totals, wave_only);
        const int64_t total_cost = wave_only ?
            __shfl(static_cast<long long>(cost_prefix), wave_size - 1) : wave_totals[waves - 1];
        if(!wave_only)
            __syncthreads();
        const bool consider_proportional = params.num_heads_k == 1 &&
            params.uni_seqlen_qo > 0 && params.num_batches > 1 &&
            params.num_batches < 2 * workers_per_head;
        int64_t prefix = cost_prefix;
        int64_t denominator = max(total_cost, int64_t{1});
        int32_t nonempty = 0;
        bool proportional = false;
        if(consider_proportional)
        {
            int64_t max_cost = cost;
            for(int32_t offset = wave_size / 2; offset > 0; offset >>= 1)
                max_cost = max(max_cost, static_cast<int64_t>(__shfl_xor(
                    static_cast<long long>(max_cost), offset)));
            if(!wave_only)
            {
                if(lane == 0)
                    maxima[wave] = max_cost;
                __syncthreads();
                max_cost = 0;
                for(int32_t index = 0; index < waves; ++index)
                    max_cost = max(max_cost, maxima[index]);
            }
            proportional = max_cost * params.num_batches > 4 * total_cost;
            if(proportional)
            {
                prefix = pa_metadata_v1_1_block_prefix(tiles, wave_totals, wave_only);
                denominator = max(wave_only ?
                    static_cast<int64_t>(__shfl(static_cast<long long>(prefix), wave_size - 1)) :
                    wave_totals[waves - 1], int64_t{1});
                if(!wave_only)
                    __syncthreads();
                const int64_t nonempty_prefix = pa_metadata_v1_1_block_prefix(
                    pages > 0, wave_totals, wave_only);
                nonempty = static_cast<int32_t>(wave_only ?
                    __shfl(static_cast<long long>(nonempty_prefix), wave_size - 1) :
                    wave_totals[waves - 1]);
                if(!wave_only)
                    __syncthreads();
            }
        }
        const int64_t sequence_cost_before = prefix - (proportional ? tiles : cost);
        int32_t units = 0;
        int32_t parts;
        if(proportional)
        {
            const int32_t remaining = min(params.num_batches * params.num_splits,
                                         2 * workers_per_head) - nonempty;
            units = static_cast<int32_t>(min(static_cast<int64_t>(params.num_splits),
                min(tiles, static_cast<int64_t>(tiles > 0) +
                    pa_metadata_v1_1_divide(prefix * remaining, denominator) -
                    pa_metadata_v1_1_divide(sequence_cost_before * remaining, denominator))));
            parts = pa_metadata_v1_1_fragments(
                2 * units * sequence_cost_before + tiles, units, 2 * tiles,
                2 * max(units, 1) * denominator, workers_per_head);
        }
        else
            parts = min(params.num_splits, pa_metadata_v1_1_fragments(
                sequence_cost_before, tiles, static_cast<int64_t>(params.blocks_per_unit) * qo_length,
                denominator, workers_per_head));
        const int32_t reduce_count = parts > 1 ? parts : 0;
        const int32_t task_prefix = static_cast<int32_t>(pa_metadata_v1_1_block_prefix(
            parts, wave_totals, wave_only));
        const int32_t total_tasks = wave_only ? __shfl(task_prefix, wave_size - 1) :
            static_cast<int32_t>(wave_totals[waves - 1]);
        if(!wave_only)
            __syncthreads();
        const int32_t reduce_prefix = static_cast<int32_t>(pa_metadata_v1_1_block_prefix(
            reduce_count, wave_totals, wave_only));
        const int32_t partial_prefix = static_cast<int32_t>(pa_metadata_v1_1_block_prefix(
            reduce_count * qo_length, wave_totals, wave_only));
        if(active)
        {
            cost_prefixes[sequence] = prefix;
            task_prefixes[sequence] = task_prefix;
            partial_prefixes[sequence] = partial_prefix;
            sequence_units[sequence] = units;
            params.p_reduce_indptr[sequence + 1] = reduce_prefix;
            params.p_reduce_final_map[sequence * 2] = qo_start;
            params.p_reduce_final_map[sequence * 2 + 1] = qo_end;
        }
        if(sequence == 0)
        {
            params.p_reduce_indptr[0] = 0;
            params.p_work_indptr[params.num_cu] = total_tasks * params.num_heads_k;
            params.p_work_metadata_ptrs[0] = reinterpret_cast<uintptr_t>(params.p_work_indptr);
            params.p_work_metadata_ptrs[1] = reinterpret_cast<uintptr_t>(params.p_work_info_set_raw);
            total_weight = denominator;
            task_count = total_tasks;
            use_proportional = proportional;
        }
    }
    __syncthreads();
    const int64_t denominator = total_weight;
    const int32_t total_tasks = task_count;
    const bool proportional = use_proportional;
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
        pa_metadata_v1_1_emit(params, begin, task - tasks_before,
            task_prefixes[begin] - tasks_before, sequence_units[begin],
            begin > 0 ? cost_prefixes[begin - 1] : 0, tasks_before,
            params.p_reduce_indptr[begin], begin > 0 ? partial_prefixes[begin - 1] : 0,
            total_tasks, proportional, denominator);
    }
    __syncthreads();
    pa_metadata_v1_1_finalize(params);
}

__launch_bounds__(opus::get_warp_size(), 1) __global__
void kn_get_pa_metadata_v1_1(PaMetadataV1KernelParameter params)
{
    const int32_t lane = opus::lane_id();
    const int32_t wave_size = opus::get_warp_size();
    const int32_t sequence = blockIdx.x;
    const int32_t workers_per_head = params.num_cu / params.num_heads_k;
    const int32_t page_shift = __ffs(params.block_size) - 1;
    const int32_t unit_shift = __ffs(params.blocks_per_unit) - 1;
    auto* works = reinterpret_cast<PaWorkInfo*>(params.p_work_info_set_raw);
    int64_t total_cost = 0;
    int64_t total_tiles = 0;
    int64_t max_cost = 0;
    int32_t nonempty = 0;
    const bool consider_proportional = params.num_heads_k == 1 &&
        params.uni_seqlen_qo > 0 && params.num_batches > 1 &&
        params.num_batches < 2 * workers_per_head;
    for(int32_t batch = lane; batch < params.num_batches; batch += wave_size)
    {
        const int32_t qo_length = params.p_seqlens_qo_indptr[batch + 1] -
                                  params.p_seqlens_qo_indptr[batch];
        const uint32_t pages =
            (static_cast<uint32_t>(max(params.p_context_lens[batch], 0)) +
             params.block_size - 1) >> page_shift;
        const int64_t cost = static_cast<int64_t>(pages) * qo_length;
        total_cost += cost;
        if(consider_proportional)
        {
            total_tiles += (pages + params.blocks_per_unit - 1) >> unit_shift;
            max_cost = max(max_cost, cost);
            nonempty += pages > 0;
        }
    }
    total_cost = __shfl(pa_metadata_v1_1_prefix_sum(total_cost), wave_size - 1);
    if(consider_proportional)
    {
        total_tiles = __shfl(pa_metadata_v1_1_prefix_sum(total_tiles), wave_size - 1);
        nonempty = __shfl(warp_prefix_sum(nonempty, wave_size), wave_size - 1);
        for(int32_t offset = wave_size / 2; offset > 0; offset >>= 1)
            max_cost = max(max_cost, static_cast<int64_t>(__shfl_xor(
                static_cast<long long>(max_cost), offset)));
    }
    const bool proportional = consider_proportional &&
        max_cost * params.num_batches > 4 * total_cost;
    const int64_t denominator = max(proportional ? total_tiles : total_cost, int64_t{1});
    const int32_t remaining = min(params.num_batches * params.num_splits,
                                 2 * workers_per_head) - nonempty;
    int64_t cost_base = 0;
    int32_t task_base = 0;
    int32_t reduce_base = 0;
    int32_t partial_base = 0;
    int64_t sequence_cost_before = 0;
    int32_t sequence_tasks_before = 0;
    int32_t sequence_reduce_before = 0;
    int32_t sequence_partial_before = 0;
    int32_t sequence_parts = 0;
    int32_t sequence_units = 0;

    for(int32_t chunk = 0; chunk < params.num_batches; chunk += wave_size)
    {
        const int32_t batch = chunk + lane;
        const int32_t qo_length = batch < params.num_batches
            ? params.p_seqlens_qo_indptr[batch + 1] - params.p_seqlens_qo_indptr[batch] : 0;
        const uint32_t pages = batch < params.num_batches
            ? (static_cast<uint32_t>(max(params.p_context_lens[batch], 0)) +
             params.block_size - 1) >> page_shift : 0;
         const uint32_t tiles = (pages + params.blocks_per_unit - 1) >> unit_shift;
        const int64_t cost = proportional ? static_cast<int64_t>(tiles) :
                                           static_cast<int64_t>(pages) * qo_length;
        const int64_t tile_cost = static_cast<int64_t>(params.blocks_per_unit) * qo_length;
        const int64_t cost_prefix = cost_base + pa_metadata_v1_1_prefix_sum(cost);
        int32_t units = 0;
        int32_t parts;
        if(proportional)
        {
            units = static_cast<int32_t>(min(static_cast<int64_t>(params.num_splits),
                min(static_cast<int64_t>(tiles), static_cast<int64_t>(tiles > 0) +
                    pa_metadata_v1_1_divide(cost_prefix * remaining, denominator) -
                    pa_metadata_v1_1_divide((cost_prefix - tiles) * remaining, denominator))));
            parts = pa_metadata_v1_1_fragments(
                2 * units * (cost_prefix - tiles) + tiles, units, 2 * tiles,
                2 * max(units, 1) * denominator, workers_per_head);
        }
        else
            parts = min(params.num_splits, pa_metadata_v1_1_fragments(
                cost_prefix - cost, tiles, tile_cost, denominator, workers_per_head));
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
            sequence_units = units;
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
    sequence_units = __shfl(sequence_units, owner_lane);
    const int32_t qo_start = params.p_seqlens_qo_indptr[sequence];
    const int32_t qo_end = params.p_seqlens_qo_indptr[sequence + 1];

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
        pa_metadata_v1_1_emit(params, sequence, part, sequence_parts, sequence_units,
            sequence_cost_before, sequence_tasks_before, sequence_reduce_before,
            sequence_partial_before, task_base, proportional, denominator);
    }
}

__device__ __forceinline__ PaMetadataV1ChunkRecord* pa_metadata_v1_1_scratch(
    PaMetadataV1KernelParameter params)
{
    const int64_t capacity = min(
        static_cast<int64_t>(params.num_batches) * params.num_splits,
        static_cast<int64_t>(params.num_batches) + params.num_cu / params.num_heads_k - 1);
    auto* works = reinterpret_cast<PaWorkInfo*>(params.p_work_info_set_raw);
    return reinterpret_cast<PaMetadataV1ChunkRecord*>(works + capacity * params.num_heads_k);
}

__launch_bounds__(kPaMetadataSharedMaxBatches, 1) __global__
void kn_scan_pa_metadata_v1_1_chunks(PaMetadataV1KernelParameter params)
{
    __shared__ int64_t wave_totals[kPaMetadataSharedMaxWaves];
    const int32_t sequence = blockIdx.x * kPaMetadataSharedMaxBatches + threadIdx.x;
    const bool active = sequence < params.num_batches;
    const int32_t qo_length = active ?
        params.p_seqlens_qo_indptr[sequence + 1] - params.p_seqlens_qo_indptr[sequence] : 0;
    const uint32_t context = active ? max(params.p_context_lens[sequence], 0) : 0;
    const uint32_t pages = (context + params.block_size - 1) >> (__ffs(params.block_size) - 1);
    const int64_t cost_prefix = pa_metadata_v1_1_block_prefix(
        static_cast<int64_t>(pages) * qo_length, wave_totals, false);
    auto* rows = pa_metadata_v1_1_scratch(params);
    if(active)
        rows[sequence].cost = cost_prefix;
    if(threadIdx.x == blockDim.x - 1)
        rows[params.num_batches + blockIdx.x].cost = cost_prefix;
}

__launch_bounds__(kPaMetadataSharedMaxBatches, 1) __global__
void kn_plan_pa_metadata_v1_1_chunks(PaMetadataV1KernelParameter params)
{
    __shared__ int64_t wave_totals[kPaMetadataSharedMaxWaves];
    const int32_t sequence = blockIdx.x * kPaMetadataSharedMaxBatches + threadIdx.x;
    const bool active = sequence < params.num_batches;
    const int32_t chunks = (params.num_batches + kPaMetadataSharedMaxBatches - 1) /
                           kPaMetadataSharedMaxBatches;
    auto* rows = pa_metadata_v1_1_scratch(params);
    auto* costs = rows + params.num_batches;
    auto* counts = costs + chunks;
    int64_t total_cost = 0;
    int64_t cost_base = 0;
    for(int32_t chunk = 0; chunk < chunks; ++chunk)
    {
        total_cost += costs[chunk].cost;
        if(chunk < blockIdx.x)
            cost_base += costs[chunk].cost;
    }
    const int32_t qo_length = active ?
        params.p_seqlens_qo_indptr[sequence + 1] - params.p_seqlens_qo_indptr[sequence] : 0;
    const uint32_t context = active ? max(params.p_context_lens[sequence], 0) : 0;
    const uint32_t pages = (context + params.block_size - 1) >> (__ffs(params.block_size) - 1);
    const int64_t tiles = (context + params.kv_granularity - 1) >>
                          (__ffs(params.kv_granularity) - 1);
    const int64_t cost_before = active ?
        cost_base + rows[sequence].cost - static_cast<int64_t>(pages) * qo_length : 0;
    const int32_t parts = min(params.num_splits, pa_metadata_v1_1_fragments(
        cost_before, tiles, static_cast<int64_t>(params.blocks_per_unit) * qo_length,
        max(total_cost, int64_t{1}), params.num_cu / params.num_heads_k));
    const int32_t reductions = parts > 1 ? parts : 0;
    const int32_t task_prefix = static_cast<int32_t>(pa_metadata_v1_1_block_prefix(
        parts, wave_totals, false));
    const int32_t reduce_prefix = static_cast<int32_t>(pa_metadata_v1_1_block_prefix(
        reductions, wave_totals, false));
    const int32_t partial_prefix = static_cast<int32_t>(pa_metadata_v1_1_block_prefix(
        reductions * qo_length, wave_totals, false));
    if(active)
    {
        rows[sequence].cost = cost_before;
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
void kn_emit_pa_metadata_v1_1_chunks(PaMetadataV1KernelParameter params)
{
    const int32_t sequence = blockIdx.x * kPaMetadataSharedMaxBatches + threadIdx.x;
    if(sequence >= params.num_batches)
        return;
    const int32_t chunks = (params.num_batches + kPaMetadataSharedMaxBatches - 1) /
                           kPaMetadataSharedMaxBatches;
    auto* rows = pa_metadata_v1_1_scratch(params);
    auto* costs = rows + params.num_batches;
    auto* counts = costs + chunks;
    int64_t total_cost = 0;
    int32_t total_tasks = 0;
    int32_t task_base = 0;
    int32_t reduce_base = 0;
    int32_t partial_base = 0;
    for(int32_t chunk = 0; chunk < chunks; ++chunk)
    {
        total_cost += costs[chunk].cost;
        total_tasks += counts[chunk].tasks;
        if(chunk < blockIdx.x)
        {
            task_base += counts[chunk].tasks;
            reduce_base += counts[chunk].reductions;
            partial_base += counts[chunk].partials;
        }
    }
    const int32_t local_task_base = threadIdx.x > 0 ? rows[sequence - 1].tasks : 0;
    const int32_t local_reduce_base = threadIdx.x > 0 ? rows[sequence - 1].reductions : 0;
    const int32_t local_partial_base = threadIdx.x > 0 ? rows[sequence - 1].partials : 0;
    const int32_t parts = rows[sequence].tasks - local_task_base;
    params.p_reduce_indptr[sequence + 1] = reduce_base + rows[sequence].reductions;
    params.p_reduce_final_map[sequence * 2] = params.p_seqlens_qo_indptr[sequence];
    params.p_reduce_final_map[sequence * 2 + 1] = params.p_seqlens_qo_indptr[sequence + 1];
    if(sequence == 0)
    {
        params.p_reduce_indptr[0] = 0;
        params.p_work_indptr[params.num_cu] = total_tasks * params.num_heads_k;
        params.p_work_metadata_ptrs[0] = reinterpret_cast<uintptr_t>(params.p_work_indptr);
        params.p_work_metadata_ptrs[1] = reinterpret_cast<uintptr_t>(params.p_work_info_set_raw);
    }
    for(int32_t part = 0; part < parts; ++part)
        pa_metadata_v1_1_emit(params, sequence, part, parts, 0, rows[sequence].cost,
            task_base + local_task_base, reduce_base + local_reduce_base,
            partial_base + local_partial_base, total_tasks, false, max(total_cost, int64_t{1}));
}

__device__ __forceinline__ void pa_metadata_v1_1_finalize(PaMetadataV1KernelParameter params)
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

__launch_bounds__(opus::get_warp_size(), 1) __global__
void kn_finalize_pa_metadata_v1_1(PaMetadataV1KernelParameter params)
{
    pa_metadata_v1_1_finalize(params);
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
    AITER_CHECK(batches > 0 && batches <= kPaMetadataMaxBatches,
                "PA v1_1 supports batches in [1, 32768]");
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
    const int32_t reduce_capacity = max_partitions == 1 ? 0 :
        std::min(capacity, 2 * (workers / num_heads_k - 1));
    const int32_t chunks = (batches + kPaMetadataSharedMaxBatches - 1) /
                           kPaMetadataSharedMaxBatches;
    const int64_t scratch_rows = batches > kPaMetadataSharedMaxBatches ?
        static_cast<int64_t>(batches) + 2 * chunks : 0;
    AITER_CHECK(work_metadata_ptrs.numel() >= 2 && work_indptr.numel() >= workers + 1 &&
                work_info_set.numel() >= (static_cast<int64_t>(capacity) * num_heads_k +
                                         scratch_rows) * 8 &&
                reduce_indptr.numel() >= batches + 1 && reduce_final_map.numel() >= batches * 2 &&
                reduce_partial_map.numel() >= reduce_capacity,
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
    params.uni_seqlen_qo = ori_uni_seqlen_qo;
    params.block_size = block_size;
    params.kv_granularity = std::max(256, kv_granularity);
    params.blocks_per_unit = params.kv_granularity / block_size;
    const hipStream_t stream = aiter::getCurrentHIPStream();
    if(batches <= kPaMetadataSharedMaxBatches)
    {
        const int32_t threads = std::min(kPaMetadataSharedMaxBatches,
            std::max(properties.warpSize, next_power_of_two(std::max(batches, capacity))));
        kn_get_pa_metadata_v1_1_shared<<<1, threads, 0, stream>>>(params);
        HIP_CALL_LAUNCH(hipGetLastError());
        return;
    }
    if(num_heads_k == 1 && ori_uni_seqlen_qo > 0 && batches < 2 * workers)
    {
        kn_get_pa_metadata_v1_1<<<batches, properties.warpSize, 0, stream>>>(params);
        HIP_CALL_LAUNCH(hipGetLastError());
    }
    else
    {
        kn_scan_pa_metadata_v1_1_chunks<<<chunks, kPaMetadataSharedMaxBatches, 0, stream>>>(params);
        HIP_CALL_LAUNCH(hipGetLastError());
        kn_plan_pa_metadata_v1_1_chunks<<<chunks, kPaMetadataSharedMaxBatches, 0, stream>>>(params);
        HIP_CALL_LAUNCH(hipGetLastError());
        kn_emit_pa_metadata_v1_1_chunks<<<chunks, kPaMetadataSharedMaxBatches, 0, stream>>>(params);
        HIP_CALL_LAUNCH(hipGetLastError());
    }
    kn_finalize_pa_metadata_v1_1<<<1, properties.warpSize, 0, stream>>>(params);
    HIP_CALL_LAUNCH(hipGetLastError());
}