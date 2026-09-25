// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "hip_reduce.h"
#include "iq2r.h"
#include "mx_quant_utils.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace aiter {
namespace {

constexpr int kThreads                      = 256;
constexpr int kQuantThreads                 = 64;
constexpr int kSwiGLUQuantElementsPerThread = 8;
constexpr int kTaskColumns                  = 3;
constexpr int kMaxExperts                   = 512;
constexpr int kMaxRoutes                    = 131072;
constexpr int kSmallRoutes                  = 16;
constexpr int kQuantGroup                   = 32;
constexpr int kGptOssTopK                   = 4;
constexpr int kGptOssExperts                = 128;
constexpr int kGlmTopK                      = 8;
constexpr int kGlmExperts                   = 256;
constexpr int kFusedSortTokens              = 16;
constexpr int kFusedSortRoutes              = kFusedSortTokens * kGptOssTopK;
constexpr int kGlmFusedSortRoutes           = kFusedSortTokens * kGlmTopK;
constexpr int kFusedSortThreads             = 512;

template <typename RouterType>
__device__ __forceinline__ float iq2r_router_value(const RouterType* __restrict__ router_logits,
                                                   const opus::bf16_t* __restrict__ router_bias,
                                                   int64_t offset,
                                                   int expert)
{
    float value = static_cast<float>(router_logits[offset]);
    if(router_bias == nullptr)
        return value;
    value += static_cast<float>(router_bias[expert]);
    // ATOM's existing skinny-GEMV path materializes BF16 logits and then runs
    // an in-place BF16 bias add. Preserve that rounding point exactly when the
    // producer is BF16; FP32 logits retain FP32 addition semantics.
    if constexpr(std::is_same_v<RouterType, opus::bf16_t>)
        value = __bfloat162float(__float2bfloat16(value));
    return value;
}

template <typename RouterType>
__device__ __forceinline__ float iq2r_router_sigmoid(const RouterType* __restrict__ router_logits,
                                                     int64_t offset)
{
    constexpr float kLog2E = 1.4426950408889634f;
    const float value      = static_cast<float>(router_logits[offset]);
    return __builtin_amdgcn_rcpf(1.0f + exp2f(-kLog2E * value));
}

bool iq2r_valid_scale_shape(const aiter_tensor_t& scales, int64_t rows, int64_t groups_per_row)
{
    return (scales.dim() == 2 && scales.size(0) == rows && scales.size(1) == groups_per_row) ||
           (scales.dim() == 4 && scales.size(0) == (groups_per_row + 3) / 4 &&
            scales.size(1) >= (rows + 15) / 16 && scales.size(2) == 4 && scales.size(3) == 16);
}

__device__ __forceinline__ int64_t
iq2r_scale_offset(int row, int group, int groups_per_row, int m_blocks, bool tiled_scales)
{
    if(tiled_scales)
        return (static_cast<int64_t>(group / 4) * m_blocks + row / 16) * 64 + (group % 4) * 16 +
               row % 16;
    return static_cast<int64_t>(row) * groups_per_row + group;
}

__device__ __forceinline__ int iq2r_local_expert(int global_expert,
                                                 const int32_t* __restrict__ expert_map,
                                                 int expert_map_count,
                                                 int expert_start,
                                                 int expert_stride,
                                                 int expert_count)
{
    if(expert_map != nullptr)
    {
        if(global_expert < 0 || global_expert >= expert_map_count)
            return -1;
        const int local_expert = expert_map[global_expert];
        return local_expert >= 0 && local_expert < expert_count ? local_expert : -1;
    }
    const int offset = global_expert - expert_start;
    if(offset < 0)
        return -1;
    if(expert_stride == 1)
        return offset < expert_count ? offset : -1;
    if(offset % expert_stride != 0)
        return -1;
    const int local_expert = offset / expert_stride;
    return local_expert < expert_count ? local_expert : -1;
}

__device__ __forceinline__ int iq2r_block_inclusive_scan_256(int value, int* wave_totals)
{
    const int lane = static_cast<int>(threadIdx.x) & 63;
    const int wave = static_cast<int>(threadIdx.x) >> 6;
#pragma unroll
    for(int delta = 1; delta < 64; delta <<= 1)
    {
        const int previous = __shfl_up(value, delta, 64);
        if(lane >= delta)
            value += previous;
    }
    if(lane == 63)
        wave_totals[wave] = value;
    __syncthreads();

    if(wave == 0)
    {
        int total = lane < 4 ? wave_totals[lane] : 0;
#pragma unroll
        for(int delta = 1; delta < 64; delta <<= 1)
        {
            const int previous = __shfl_up(total, delta, 64);
            if(lane >= delta)
                total += previous;
        }
        if(lane < 4)
            wave_totals[lane] = total;
    }
    __syncthreads();

    if(wave > 0)
        value += wave_totals[wave - 1];
    return value;
}

// Decode specialization for at most one CTA of routed rows.  It avoids
// clearing the general kernel's full 16 x 129 chunk histogram.  Stable rank
// is computed directly from the shared route IDs; at <=256 routes this is
// cheaper than the fixed initialization cost of the general 4096-route path.
__global__ __launch_bounds__(kThreads) void iq2r_route_sort_tasks_small_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    const int32_t* __restrict__ expert_map,
    int routes,
    int expert_map_count,
    int expert_count,
    int expert_start,
    int expert_stride,
    int task_capacity,
    int task_rows,
    bool drop_nonlocal_tasks)
{
    __shared__ int counts[kMaxExperts + 1];
    __shared__ int offsets[kMaxExperts + 2];
    __shared__ int route_experts[kSmallRoutes];

    for(int bucket = threadIdx.x; bucket <= expert_count; bucket += blockDim.x)
        counts[bucket] = 0;
    __syncthreads();

    if(threadIdx.x < routes)
    {
        const int expert           = iq2r_local_expert(expert_ids[threadIdx.x],
                                             expert_map,
                                             expert_map_count,
                                             expert_start,
                                             expert_stride,
                                             expert_count);
        const int bucket           = expert >= 0 ? expert : expert_count;
        route_experts[threadIdx.x] = expert;
        atomicAdd(counts + bucket, 1);
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        offsets[0] = 0;
        for(int bucket = 0; bucket <= expert_count; ++bucket)
            offsets[bucket + 1] = offsets[bucket] + counts[bucket];
    }
    __syncthreads();

    if(threadIdx.x < routes)
    {
        const int expert = route_experts[threadIdx.x];
        const int bucket = expert >= 0 && expert < expert_count ? expert : expert_count;
        int local_rank   = 0;
        for(int previous = 0; previous < threadIdx.x; ++previous)
        {
            const int previous_expert = route_experts[previous];
            const int previous_bucket = previous_expert >= 0 && previous_expert < expert_count
                                            ? previous_expert
                                            : expert_count;
            local_rank += previous_bucket == bucket;
        }
        const int sorted_route          = offsets[bucket] + local_rank;
        sorted_expert_ids[sorted_route] = expert;
        gather_indices[sorted_route]    = threadIdx.x;
        scatter_indices[threadIdx.x]    = drop_nonlocal_tasks && expert < 0 ? -1 : sorted_route;
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        int task         = 0;
        int sorted_begin = 0;
        for(int expert = 0; expert < expert_count; ++expert)
        {
            const int count = counts[expert];
            for(int local = 0; local < count; local += task_rows)
            {
                if(task < task_capacity)
                {
                    tasks[task * kTaskColumns]     = sorted_begin + local;
                    tasks[task * kTaskColumns + 1] = min(task_rows, count - local);
                    tasks[task * kTaskColumns + 2] = expert;
                }
                ++task;
            }
            sorted_begin += count;
        }
        const int invalid_count = counts[expert_count];
        for(int local = 0; !drop_nonlocal_tasks && local < invalid_count; local += task_rows)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns]     = sorted_begin + local;
                tasks[task * kTaskColumns + 1] = min(task_rows, invalid_count - local);
                tasks[task * kTaskColumns + 2] = -1;
            }
            ++task;
        }
        task_count[0] = task <= task_capacity ? task : -1;
    }
}

// Plain GLM-5.3 TP retains all 256 routed experts on every rank and may append
// one fused shared expert. Keep one thread per routed expert and use wave scans
// for offsets/tasks so decode does not pay two serial 256/257-entry passes per
// MoE layer. Thread zero appends expert 256 when the fused row is present.
__global__ __launch_bounds__(kThreads) void iq2r_route_sort_tasks_glm_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    const int32_t* __restrict__ expert_map,
    int routes,
    int expert_map_count,
    int expert_count,
    int expert_start,
    int expert_stride,
    int task_capacity,
    int task_rows,
    bool drop_nonlocal_tasks)
{
    __shared__ int counts[kGlmExperts + 2];
    __shared__ int offsets[kGlmExperts + 2];
    __shared__ int cursors[kGlmExperts + 2];
    __shared__ int wave_totals[4];
    __shared__ int valid_task_count;

    const int expert = static_cast<int>(threadIdx.x);
    counts[expert]   = 0;
    if(expert == 0)
    {
        counts[kGlmExperts]     = 0;
        counts[kGlmExperts + 1] = 0;
    }
    __syncthreads();

    for(int route = expert; route < routes; route += kThreads)
    {
        const int local_expert = iq2r_local_expert(expert_ids[route],
                                                   expert_map,
                                                   expert_map_count,
                                                   expert_start,
                                                   expert_stride,
                                                   expert_count);
        atomicAdd(counts + (local_expert >= 0 ? local_expert : expert_count), 1);
    }
    __syncthreads();

    const int count_prefix = iq2r_block_inclusive_scan_256(counts[expert], wave_totals);
    offsets[expert]        = count_prefix - counts[expert];
    cursors[expert]        = offsets[expert];
    if(expert == kGlmExperts - 1)
    {
        offsets[kGlmExperts] = count_prefix;
        cursors[kGlmExperts] = count_prefix;
    }
    __syncthreads();

    if(expert == 0 && expert_count == kGlmExperts + 1)
    {
        offsets[kGlmExperts + 1] = offsets[kGlmExperts] + counts[kGlmExperts];
        cursors[kGlmExperts + 1] = offsets[kGlmExperts + 1];
    }
    __syncthreads();

    for(int route = expert; route < routes; route += kThreads)
    {
        const int local_expert          = iq2r_local_expert(expert_ids[route],
                                                   expert_map,
                                                   expert_map_count,
                                                   expert_start,
                                                   expert_stride,
                                                   expert_count);
        const int bucket                = local_expert >= 0 ? local_expert : expert_count;
        const int sorted_route          = atomicAdd(cursors + bucket, 1);
        sorted_expert_ids[sorted_route] = local_expert;
        gather_indices[sorted_route]    = route;
        scatter_indices[route] = drop_nonlocal_tasks && local_expert < 0 ? -1 : sorted_route;
    }
    __syncthreads();

    const int local_tasks = (counts[expert] + task_rows - 1) / task_rows;
    const int task_prefix = iq2r_block_inclusive_scan_256(local_tasks, wave_totals);
    const int task_begin  = task_prefix - local_tasks;
    int task              = task_begin;
    for(int local = 0; local < counts[expert]; local += task_rows, ++task)
    {
        if(task < task_capacity)
        {
            tasks[task * kTaskColumns]     = offsets[expert] + local;
            tasks[task * kTaskColumns + 1] = min(task_rows, counts[expert] - local);
            tasks[task * kTaskColumns + 2] = expert;
        }
    }
    if(expert == kGlmExperts - 1)
        valid_task_count = task_prefix;
    __syncthreads();

    if(expert == 0)
    {
        int total_tasks         = valid_task_count;
        if(expert_count == kGlmExperts + 1)
        {
            const int shared_count = counts[kGlmExperts];
            for(int local = 0; local < shared_count; local += task_rows)
            {
                if(total_tasks < task_capacity)
                {
                    tasks[total_tasks * kTaskColumns] = offsets[kGlmExperts] + local;
                    tasks[total_tasks * kTaskColumns + 1] =
                        min(task_rows, shared_count - local);
                    tasks[total_tasks * kTaskColumns + 2] = kGlmExperts;
                }
                ++total_tasks;
            }
        }
        const int invalid_count = counts[expert_count];
        for(int local = 0; !drop_nonlocal_tasks && local < invalid_count; local += task_rows)
        {
            if(total_tasks < task_capacity)
            {
                tasks[total_tasks * kTaskColumns]     = offsets[expert_count] + local;
                tasks[total_tasks * kTaskColumns + 1] = min(task_rows, invalid_count - local);
                tasks[total_tasks * kTaskColumns + 2] = -1;
            }
            ++total_tasks;
        }
        task_count[0] = total_tasks <= task_capacity ? total_tasks : -1;
    }
}

// GLM-5.3 reaches 131072 routed rows at ATOM's 16K-token, top-k=8 production
// prefill contract. A single CTA builds an expert-grouped permutation and
// homogeneous tasks without global atomics or temporary allocations. Row order
// within an expert is deliberately unspecified: every routed row is computed
// independently and scatter_indices restores the original top-k order before
// reduction, so stability has no semantic value here. Using one shared-memory
// cursor per expert makes this pass O(routes), avoiding the former stable-rank
// implementation's O(routes * 256) same-expert comparisons.
__global__ __launch_bounds__(kThreads) void iq2r_route_sort_tasks_kernel(
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    const int32_t* __restrict__ expert_map,
    int routes,
    int expert_map_count,
    int expert_count,
    int expert_start,
    int expert_stride,
    int task_capacity,
    int task_rows,
    bool drop_nonlocal_tasks)
{
    __shared__ int counts[kMaxExperts + 1];
    __shared__ int offsets[kMaxExperts + 2];
    __shared__ int cursors[kMaxExperts + 1];

    for(int bucket = threadIdx.x; bucket <= expert_count; bucket += blockDim.x)
        counts[bucket] = 0;
    __syncthreads();

    for(int route = threadIdx.x; route < routes; route += blockDim.x)
    {
        const int local_expert = iq2r_local_expert(expert_ids[route],
                                                   expert_map,
                                                   expert_map_count,
                                                   expert_start,
                                                   expert_stride,
                                                   expert_count);
        const int bucket       = local_expert >= 0 ? local_expert : expert_count;
        atomicAdd(counts + bucket, 1);
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        offsets[0] = 0;
        for(int bucket = 0; bucket <= expert_count; ++bucket)
            offsets[bucket + 1] = offsets[bucket] + counts[bucket];
    }
    __syncthreads();

    for(int bucket = threadIdx.x; bucket <= expert_count; bucket += blockDim.x)
        cursors[bucket] = offsets[bucket];
    __syncthreads();

    for(int route = threadIdx.x; route < routes; route += blockDim.x)
    {
        const int expert                = iq2r_local_expert(expert_ids[route],
                                             expert_map,
                                             expert_map_count,
                                             expert_start,
                                             expert_stride,
                                             expert_count);
        const int bucket                = expert >= 0 ? expert : expert_count;
        const int sorted_route          = atomicAdd(cursors + bucket, 1);
        sorted_expert_ids[sorted_route] = expert;
        gather_indices[sorted_route]    = route;
        scatter_indices[route]          = drop_nonlocal_tasks && expert < 0 ? -1 : sorted_route;
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        int task         = 0;
        int sorted_begin = 0;
        for(int expert = 0; expert < expert_count; ++expert)
        {
            const int count = counts[expert];
            for(int local = 0; local < count; local += task_rows)
            {
                if(task < task_capacity)
                {
                    tasks[task * kTaskColumns]     = sorted_begin + local;
                    tasks[task * kTaskColumns + 1] = min(task_rows, count - local);
                    tasks[task * kTaskColumns + 2] = expert;
                }
                ++task;
            }
            sorted_begin += count;
        }
        const int invalid_count = counts[expert_count];
        for(int local = 0; !drop_nonlocal_tasks && local < invalid_count; local += task_rows)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns]     = sorted_begin + local;
                tasks[task * kTaskColumns + 1] = min(task_rows, invalid_count - local);
                tasks[task * kTaskColumns + 2] = -1;
            }
            ++task;
        }
        task_count[0] = task <= task_capacity ? task : -1;
    }
}

__global__ void iq2r_route_gather_indexed_kernel(const __hip_bfloat16* __restrict__ input,
                                                 const int32_t* __restrict__ gather_indices,
                                                 __hip_bfloat16* __restrict__ output,
                                                 int64_t elements,
                                                 int hidden,
                                                 int input_stride,
                                                 int topk)
{
    for(int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const int sorted_route = static_cast<int>(index / hidden);
        const int column       = static_cast<int>(index % hidden);
        const int source_route = gather_indices[sorted_route];
        const int token        = source_route / topk;
        output[index]          = input[static_cast<int64_t>(token) * input_stride + column];
    }
}

// Each lane owns one complete 32-value MX block, matching AITER's canonical
// dynamic_per_group_scaled_quant implementation for group_size=32.  Folding
// the indexed gather into this pass avoids materializing and rereading a BF16
// [routes, hidden] tensor.
__global__ __launch_bounds__(kQuantThreads) void iq2r_route_gather_quant_kernel(
    const opus::bf16_t* __restrict__ input,
    const int32_t* __restrict__ gather_indices,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int64_t groups,
    int hidden,
    int input_stride,
    int groups_per_row,
    int topk,
    int scale_m_blocks,
    bool tiled_scales)
{
    const int64_t group_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(group_id >= groups)
        return;

    const int route             = static_cast<int>(group_id / groups_per_row);
    const int group             = static_cast<int>(group_id % groups_per_row);
    const int source_route      = gather_indices[route];
    const int token             = source_route / topk;
    const int64_t input_offset  = static_cast<int64_t>(token) * input_stride + group * kQuantGroup;
    const int64_t output_offset = static_cast<int64_t>(route) * hidden + group * kQuantGroup;

    using input_vector        = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector       = opus::vector_t<opus::fp8_t, kQuantGroup>;
    const input_vector values = *reinterpret_cast<const input_vector*>(input + input_offset);

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] = opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

    *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
    scales[iq2r_scale_offset(route, group, groups_per_row, scale_m_blocks, tiled_scales)] =
        block_scale.byte;
}

// Decode-specialized path for at most 16 routed rows. Production GPT-OSS c1-c4
// routes are overwhelmingly one row per expert, so sorting them spends more
// time initializing the general counting-sort scratch than it can recover via
// expert reuse. Build one-row tasks in original route order and quantize each
// source token once before broadcasting the packed row to all of its top-k
// routes. The previous route-major implementation repeated the BF16 loads,
// absolute-max reduction, scale conversion, and FP8 conversion top-k times.
constexpr int kDefaultQuantThreads       = 128;
constexpr int kMaxQuantThreads           = 256;
constexpr int kDirectTopKThreads         = kGptOssExperts;
constexpr int kGlmGroupedDecodeTopK      = 9;
constexpr int kGlmGroupedDecodeMaxTokens = 16;
constexpr int kGlmGroupedDecodeMaxRoutes = kGlmGroupedDecodeTopK * kGlmGroupedDecodeMaxTokens;

__global__ __launch_bounds__(kMaxQuantThreads) void iq2r_route_direct_gather_quant_kernel(
    const opus::bf16_t* __restrict__ input,
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int routes,
    int hidden,
    int input_stride,
    int groups_per_row,
    int topk,
    int expert_count,
    const int32_t* __restrict__ expert_map,
    int expert_map_count,
    int expert_start,
    int expert_stride,
    int scale_m_blocks,
    bool tiled_scales)
{
    const int token = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(threadIdx.x);
    if(token >= routes / topk || group >= groups_per_row)
        return;

    if(group < topk)
    {
        const int route                 = token * topk + group;
        const int valid_expert          = iq2r_local_expert(expert_ids[route],
                                                   expert_map,
                                                   expert_map_count,
                                                   expert_start,
                                                   expert_stride,
                                                   expert_count);
        sorted_expert_ids[route]        = valid_expert;
        gather_indices[route]           = route;
        scatter_indices[route]          = route;
        tasks[route * kTaskColumns]     = route;
        tasks[route * kTaskColumns + 1] = 1;
        tasks[route * kTaskColumns + 2] = valid_expert;
        if(token == 0 && group == 0)
            task_count[0] = routes;
    }

    const int column_begin     = group * kQuantGroup;
    const int64_t input_offset = static_cast<int64_t>(token) * input_stride + column_begin;

    using input_vector        = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector       = opus::vector_t<opus::fp8_t, kQuantGroup>;
    const input_vector values = *reinterpret_cast<const input_vector*>(input + input_offset);

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] = opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

    for(int route_in_token = 0; route_in_token < topk; ++route_in_token)
    {
        const int route             = token * topk + route_in_token;
        const int64_t output_offset = static_cast<int64_t>(route) * hidden + column_begin;
        *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
        scales[iq2r_scale_offset(route, group, groups_per_row, scale_m_blocks, tiled_scales)] =
            block_scale.byte;
    }
}

// Fused shared-expert GLM decode specialization.  One CTA owns each source
// token, computes stable expert-grouped destinations for its nine routes, and
// quantizes the 6144-wide BF16 row once before broadcasting it.  Block zero
// additionally emits the same 16-row grouped tasks as the regular GLM sorter.
// Keeping those grouped tasks is essential: replacing them with one-row tasks
// made the expert body about 29% slower in the matched M4 experiment.
__global__ __launch_bounds__(kMaxQuantThreads)
void iq2r_route_grouped_gather_quant_glm_kernel(
    const opus::bf16_t* __restrict__ input,
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int tokens,
    int hidden,
    int input_stride,
    int groups_per_row,
    int task_capacity,
    int scale_m_blocks,
    bool tiled_scales)
{
    constexpr int kExpertCount  = kGlmExperts + 1;
    constexpr int kInvalidBucket = kExpertCount;
    constexpr int kTaskRows      = 16;

    __shared__ int route_experts[kGlmGroupedDecodeMaxRoutes];
    __shared__ int token_sorted_routes[kGlmGroupedDecodeTopK];
    __shared__ int counts[kExpertCount + 1];
    __shared__ int offsets[kExpertCount + 1];
    __shared__ int wave_totals[4];
    __shared__ int routed_count;
    __shared__ int routed_task_count;

    const int thread = static_cast<int>(threadIdx.x);
    const int token  = static_cast<int>(blockIdx.x);
    const int routes = tokens * kGlmGroupedDecodeTopK;

    for(int route = thread; route < routes; route += kMaxQuantThreads)
    {
        const int expert = expert_ids[route];
        route_experts[route] = expert >= 0 && expert < kExpertCount ? expert : -1;
    }
    __syncthreads();

    if(thread < kGlmGroupedDecodeTopK)
    {
        const int route  = token * kGlmGroupedDecodeTopK + thread;
        const int expert = route_experts[route];
        const int bucket = expert >= 0 ? expert : kInvalidBucket;
        int sorted_route = 0;
        for(int previous = 0; previous < routes; ++previous)
        {
            const int previous_expert = route_experts[previous];
            const int previous_bucket =
                previous_expert >= 0 ? previous_expert : kInvalidBucket;
            sorted_route += previous_bucket < bucket ||
                            (previous_bucket == bucket && previous < route);
        }
        token_sorted_routes[thread]        = sorted_route;
        sorted_expert_ids[sorted_route] = expert;
        gather_indices[sorted_route]    = route;
        scatter_indices[route]          = sorted_route;
    }
    __syncthreads();

    if(token == 0)
    {
        counts[thread] = 0;
        if(thread == 0)
        {
            counts[kGlmExperts] = 0;
            counts[kInvalidBucket] = 0;
        }
        __syncthreads();

        for(int route = thread; route < routes; route += kMaxQuantThreads)
        {
            const int expert = route_experts[route];
            atomicAdd(counts + (expert >= 0 ? expert : kInvalidBucket), 1);
        }
        __syncthreads();

        const int count_prefix =
            iq2r_block_inclusive_scan_256(counts[thread], wave_totals);
        offsets[thread] = count_prefix - counts[thread];
        if(thread == kGlmExperts - 1)
            routed_count = count_prefix;
        __syncthreads();

        const int local_tasks = (counts[thread] + kTaskRows - 1) / kTaskRows;
        const int task_prefix =
            iq2r_block_inclusive_scan_256(local_tasks, wave_totals);
        int task = task_prefix - local_tasks;
        for(int local = 0; local < counts[thread]; local += kTaskRows, ++task)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns] = offsets[thread] + local;
                tasks[task * kTaskColumns + 1] =
                    min(kTaskRows, counts[thread] - local);
                tasks[task * kTaskColumns + 2] = thread;
            }
        }
        if(thread == kGlmExperts - 1)
            routed_task_count = task_prefix;
        __syncthreads();

        if(thread == 0)
        {
            int total_tasks = routed_task_count;
            int sorted_begin = routed_count;
            for(int local = 0; local < counts[kGlmExperts]; local += kTaskRows)
            {
                if(total_tasks < task_capacity)
                {
                    tasks[total_tasks * kTaskColumns] = sorted_begin + local;
                    tasks[total_tasks * kTaskColumns + 1] =
                        min(kTaskRows, counts[kGlmExperts] - local);
                    tasks[total_tasks * kTaskColumns + 2] = kGlmExperts;
                }
                ++total_tasks;
            }
            sorted_begin += counts[kGlmExperts];
            for(int local = 0; local < counts[kInvalidBucket]; local += kTaskRows)
            {
                if(total_tasks < task_capacity)
                {
                    tasks[total_tasks * kTaskColumns] = sorted_begin + local;
                    tasks[total_tasks * kTaskColumns + 1] =
                        min(kTaskRows, counts[kInvalidBucket] - local);
                    tasks[total_tasks * kTaskColumns + 2] = -1;
                }
                ++total_tasks;
            }
            task_count[0] = total_tasks <= task_capacity ? total_tasks : -1;
        }
        __syncthreads();
    }

    if(thread >= groups_per_row)
        return;

    const int column_begin = thread * kQuantGroup;
    const int64_t input_offset =
        static_cast<int64_t>(token) * input_stride + column_begin;
    using input_vector = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector = opus::vector_t<opus::fp8_t, kQuantGroup>;
    const input_vector values =
        *reinterpret_cast<const input_vector*>(input + input_offset);

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] =
            opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

#pragma unroll
    for(int rank = 0; rank < kGlmGroupedDecodeTopK; ++rank)
    {
        const int sorted_route = token_sorted_routes[rank];
        const int64_t output_offset =
            static_cast<int64_t>(sorted_route) * hidden + column_begin;
        *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
        scales[iq2r_scale_offset(
            sorted_route, thread, groups_per_row, scale_m_blocks, tiled_scales)] =
            block_scale.byte;
    }
}

__global__ __launch_bounds__(kMaxQuantThreads)
void iq2r_route_grouped_ballot_quant_glm_kernel(
    const opus::bf16_t* __restrict__ input,
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int tokens,
    int hidden,
    int input_stride,
    int groups_per_row,
    int task_capacity,
    int scale_m_blocks,
    bool tiled_scales)
{
    constexpr int kExpertCount  = kGlmExperts + 1;
    constexpr int kInvalidBucket = kExpertCount;
    constexpr int kTaskRows      = 16;

    __shared__ int route_experts[kGlmGroupedDecodeMaxRoutes];
    __shared__ int token_sorted_routes[kGlmGroupedDecodeTopK];
    __shared__ int counts[kExpertCount + 1];
    __shared__ int offsets[kExpertCount + 1];
    __shared__ int wave_totals[4];
    __shared__ int routed_count;
    __shared__ int routed_task_count;

    const int thread = static_cast<int>(threadIdx.x);
    const int token  = static_cast<int>(blockIdx.x);
    const int routes = tokens * kGlmGroupedDecodeTopK;

    for(int route = thread; route < routes; route += kMaxQuantThreads)
    {
        const int expert = expert_ids[route];
        route_experts[route] = expert >= 0 && expert < kExpertCount ? expert : -1;
    }
    __syncthreads();

    // Four waves rank the token's nine routes. Each ballot compares the
    // target with 64 independent predecessors instead of serial LDS reads.
    const int lane=thread%64,wave=thread/64;
    int previous_bucket[3];
#pragma unroll
    for(int chunk=0;chunk<3;++chunk)
    {
        const int previous=chunk*64+lane;
        const int previous_expert=previous<routes?route_experts[previous]:-1;
        previous_bucket[chunk]=previous_expert>=0?previous_expert:kInvalidBucket;
    }
    for(int rank=wave;rank<kGlmGroupedDecodeTopK;rank+=4)
    {
        const int route=token*kGlmGroupedDecodeTopK+rank;
        const int expert=route_experts[route];
        const int bucket=expert>=0?expert:kInvalidBucket;
        int sorted_route=0;
#pragma unroll
        for(int chunk=0;chunk<3;++chunk)
        {
            const int previous=chunk*64+lane;
            const bool precedes=previous<routes && (previous_bucket[chunk]<bucket ||
                (previous_bucket[chunk]==bucket && previous<route));
            sorted_route+=__popcll(__ballot(precedes));
        }
        if(lane==0)
        {
            token_sorted_routes[rank]=sorted_route;
            sorted_expert_ids[sorted_route]=expert;
            gather_indices[sorted_route]=route;
            scatter_indices[route]=sorted_route;
        }
    }
    __syncthreads();

    if(token == 0)
    {
        counts[thread] = 0;
        if(thread == 0)
        {
            counts[kGlmExperts] = 0;
            counts[kInvalidBucket] = 0;
        }
        __syncthreads();

        for(int route = thread; route < routes; route += kMaxQuantThreads)
        {
            const int expert = route_experts[route];
            atomicAdd(counts + (expert >= 0 ? expert : kInvalidBucket), 1);
        }
        __syncthreads();

        const int count_prefix =
            iq2r_block_inclusive_scan_256(counts[thread], wave_totals);
        offsets[thread] = count_prefix - counts[thread];
        if(thread == kGlmExperts - 1)
            routed_count = count_prefix;
        __syncthreads();

        const int local_tasks = (counts[thread] + kTaskRows - 1) / kTaskRows;
        const int task_prefix =
            iq2r_block_inclusive_scan_256(local_tasks, wave_totals);
        int task = task_prefix - local_tasks;
        for(int local = 0; local < counts[thread]; local += kTaskRows, ++task)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns] = offsets[thread] + local;
                tasks[task * kTaskColumns + 1] =
                    min(kTaskRows, counts[thread] - local);
                tasks[task * kTaskColumns + 2] = thread;
            }
        }
        if(thread == kGlmExperts - 1)
            routed_task_count = task_prefix;
        __syncthreads();

        if(thread == 0)
        {
            int total_tasks = routed_task_count;
            int sorted_begin = routed_count;
            for(int local = 0; local < counts[kGlmExperts]; local += kTaskRows)
            {
                if(total_tasks < task_capacity)
                {
                    tasks[total_tasks * kTaskColumns] = sorted_begin + local;
                    tasks[total_tasks * kTaskColumns + 1] =
                        min(kTaskRows, counts[kGlmExperts] - local);
                    tasks[total_tasks * kTaskColumns + 2] = kGlmExperts;
                }
                ++total_tasks;
            }
            sorted_begin += counts[kGlmExperts];
            for(int local = 0; local < counts[kInvalidBucket]; local += kTaskRows)
            {
                if(total_tasks < task_capacity)
                {
                    tasks[total_tasks * kTaskColumns] = sorted_begin + local;
                    tasks[total_tasks * kTaskColumns + 1] =
                        min(kTaskRows, counts[kInvalidBucket] - local);
                    tasks[total_tasks * kTaskColumns + 2] = -1;
                }
                ++total_tasks;
            }
            task_count[0] = total_tasks <= task_capacity ? total_tasks : -1;
        }
        __syncthreads();
    }

    if(thread >= groups_per_row)
        return;

    const int column_begin = thread * kQuantGroup;
    const int64_t input_offset =
        static_cast<int64_t>(token) * input_stride + column_begin;
    using input_vector = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector = opus::vector_t<opus::fp8_t, kQuantGroup>;
    const input_vector values =
        *reinterpret_cast<const input_vector*>(input + input_offset);

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] =
            opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

#pragma unroll
    for(int rank = 0; rank < kGlmGroupedDecodeTopK; ++rank)
    {
        const int sorted_route = token_sorted_routes[rank];
        const int64_t output_offset =
            static_cast<int64_t>(sorted_route) * hidden + column_begin;
        *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
        scales[iq2r_scale_offset(
            sorted_route, thread, groups_per_row, scale_m_blocks, tiled_scales)] =
            block_scale.byte;
    }
}

// EP decode variant for at most sixteen routes. One CTA builds compact tasks
// for the local expert range and quantizes each source token once. Non-local
// routes receive scatter index -1 and require no GEMM, SwiGLU, or reduction
// storage traffic.
__global__ __launch_bounds__(kMaxQuantThreads) void iq2r_route_direct_scatter_quant_kernel(
    const opus::bf16_t* __restrict__ input,
    const int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int routes,
    int hidden,
    int input_stride,
    int groups_per_row,
    int topk,
    int expert_count,
    const int32_t* __restrict__ expert_map,
    int expert_map_count,
    int expert_start,
    int expert_stride,
    int scale_m_blocks,
    bool tiled_scales)
{
    if(threadIdx.x == 0)
    {
        int task = 0;
        for(int route = 0; route < routes; ++route)
        {
            const int valid_expert   = iq2r_local_expert(expert_ids[route],
                                                       expert_map,
                                                       expert_map_count,
                                                       expert_start,
                                                       expert_stride,
                                                       expert_count);
            sorted_expert_ids[route] = valid_expert;
            gather_indices[route]    = route;
            scatter_indices[route]   = valid_expert >= 0 ? route : -1;
            if(valid_expert >= 0)
            {
                tasks[task * kTaskColumns]     = route;
                tasks[task * kTaskColumns + 1] = 1;
                tasks[task * kTaskColumns + 2] = valid_expert;
                ++task;
            }
        }
        task_count[0] = task;
    }

    const int groups = (routes / topk) * groups_per_row;
    for(int group_id = static_cast<int>(threadIdx.x); group_id < groups;
        group_id += static_cast<int>(blockDim.x))
    {
        const int token      = group_id / groups_per_row;
        const int group      = group_id % groups_per_row;
        bool has_local_route = false;
#pragma unroll
        for(int route_in_token = 0; route_in_token < 8; ++route_in_token)
        {
            if(route_in_token >= topk)
                break;
            const int route = token * topk + route_in_token;
            has_local_route |= iq2r_local_expert(expert_ids[route],
                                                 expert_map,
                                                 expert_map_count,
                                                 expert_start,
                                                 expert_stride,
                                                 expert_count) >= 0;
        }
        if(!has_local_route)
            continue;

        const int column_begin     = group * kQuantGroup;
        const int64_t input_offset = static_cast<int64_t>(token) * input_stride + column_begin;
        using input_vector         = opus::vector_t<opus::bf16_t, kQuantGroup>;
        using output_vector        = opus::vector_t<opus::fp8_t, kQuantGroup>;
        const input_vector values  = *reinterpret_cast<const input_vector*>(input + input_offset);

        float abs_max = 1.0e-10f;
#pragma unroll
        for(int element = 0; element < kQuantGroup; ++element)
            abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

        const auto block_scale =
            fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
        const float inverse_scale = 1.0f / block_scale.dq_scale;
        output_vector quantized;
#pragma unroll
        for(int element = 0; element < kQuantGroup; ++element)
            quantized[element] =
                opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

#pragma unroll
        for(int route_in_token = 0; route_in_token < 8; ++route_in_token)
        {
            if(route_in_token >= topk)
                break;
            const int route = token * topk + route_in_token;
            if(iq2r_local_expert(expert_ids[route],
                                 expert_map,
                                 expert_map_count,
                                 expert_start,
                                 expert_stride,
                                 expert_count) < 0)
                continue;
            const int64_t output_offset = static_cast<int64_t>(route) * hidden + column_begin;
            *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
            scales[iq2r_scale_offset(route, group, groups_per_row, scale_m_blocks, tiled_scales)] =
                block_scale.byte;
        }
    }
}

// GPT-OSS decode front end for one to four tokens.  A single CTA per token
// computes exact top-4 softmax routing, writes the one-row direct GEMM tasks,
// and quantizes the hidden row once before broadcasting it to the four routes.
// This removes the standalone top-k launch from the launch-bound decode path.
template <typename RouterType>
__global__ __launch_bounds__(kDirectTopKThreads) void iq2r_route_topk_direct_gather_quant_kernel(
    const opus::bf16_t* __restrict__ input,
    const RouterType* __restrict__ router_logits,
    const opus::bf16_t* __restrict__ router_bias,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int tokens,
    int hidden,
    int input_stride,
    int router_stride,
    int groups_per_row,
    bool renormalize,
    int scale_m_blocks,
    bool tiled_scales)
{
    constexpr int kTopK    = 4;
    constexpr int kExperts = 128;
    using kvp              = KeyValuePair<int, float>;

    const int token = static_cast<int>(blockIdx.x);
    const int lane  = static_cast<int>(threadIdx.x);
    if(token >= tokens)
        return;

    float candidate = iq2r_router_value(
        router_logits, router_bias, static_cast<int64_t>(token) * router_stride + lane, lane);
    kvp local = {lane, candidate};
    __shared__ int selected_ids[kTopK];
    __shared__ float selected_numerators[kTopK];
    __shared__ float first_max;

#pragma unroll
    for(int rank = 0; rank < kTopK; ++rank)
    {
        const kvp selected = block_reduce<kvp, ArgMax, kDirectTopKThreads, true>(local, ArgMax());
        if(lane == 0)
        {
            if(rank == 0)
                first_max = selected.value;
            selected_ids[rank]        = selected.key;
            selected_numerators[rank] = expf(selected.value - first_max);
        }
        if(lane == selected.key)
            local.value = -INFINITY;
        // Consecutive block_reduce calls share staging memory.
        __syncthreads();
    }

    float full_row_sum = 0.0f;
    if(!renormalize)
    {
        const float remainder = expf(local.value - first_max);
        full_row_sum = block_reduce<float, Sum, kDirectTopKThreads, true>(remainder, Sum());
        __syncthreads();
    }

    if(lane == 0)
    {
        float denominator = full_row_sum;
#pragma unroll
        for(int rank = 0; rank < kTopK; ++rank)
            denominator += selected_numerators[rank];
        const float inverse = denominator != 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
        for(int rank = 0; rank < kTopK; ++rank)
        {
            const int route                 = token * kTopK + rank;
            const int expert                = selected_ids[rank];
            topk_ids[route]                 = expert;
            topk_weights[route]             = selected_numerators[rank] * inverse;
            sorted_expert_ids[route]        = expert;
            gather_indices[route]           = route;
            scatter_indices[route]          = route;
            tasks[route * kTaskColumns]     = route;
            tasks[route * kTaskColumns + 1] = 1;
            tasks[route * kTaskColumns + 2] = expert;
        }
        if(token == 0)
            task_count[0] = tokens * kTopK;
    }

    if(lane >= groups_per_row)
        return;

    const int column_begin     = lane * kQuantGroup;
    const int64_t input_offset = static_cast<int64_t>(token) * input_stride + column_begin;
    using input_vector         = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector        = opus::vector_t<opus::fp8_t, kQuantGroup>;
    const input_vector values  = *reinterpret_cast<const input_vector*>(input + input_offset);

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] = opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

#pragma unroll
    for(int rank = 0; rank < kTopK; ++rank)
    {
        const int route             = token * kTopK + rank;
        const int64_t output_offset = static_cast<int64_t>(route) * hidden + column_begin;
        *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
        scales[iq2r_scale_offset(route, lane, groups_per_row, scale_m_blocks, tiled_scales)] =
            block_scale.byte;
    }
}

// GPT-OSS decode front end for five to sixteen tokens. Eight lanes cooperate
// on each router row, then the full CTA builds one stable expert-grouped route
// permutation and homogeneous GEMM tasks. One CTA is intentional: all routing
// metadata remains in shared memory and requires no temporary global scratch.
template <typename RouterType>
__global__ __launch_bounds__(kFusedSortThreads) void iq2r_route_topk_sort_tasks_kernel(
    const RouterType* __restrict__ router_logits,
    const opus::bf16_t* __restrict__ router_bias,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    int tokens,
    int router_stride,
    int task_capacity,
    int task_rows,
    bool renormalize)
{
    constexpr int kRouterLanes    = 32;
    constexpr int kExpertsPerLane = kGptOssExperts / kRouterLanes;
    using kvp                     = KeyValuePair<int, float>;

    __shared__ int counts[kGptOssExperts];
    __shared__ int offsets[kGptOssExperts + 1];
    __shared__ int cursors[kGptOssExperts];
    __shared__ int wave_sums[2];
    __shared__ int route_experts[kFusedSortRoutes];

    const int thread              = static_cast<int>(threadIdx.x);
    const int router_thread_count = tokens * kRouterLanes;
    if(thread < router_thread_count)
    {
        const int token = thread / kRouterLanes;
        const int lane  = thread % kRouterLanes;
        int selected_ids[kGptOssTopK];
        float selected_values[kGptOssTopK];

#pragma unroll
        for(int rank = 0; rank < kGptOssTopK; ++rank)
        {
            kvp local = {INT_MAX, -INFINITY};
#pragma unroll
            for(int item = 0; item < kExpertsPerLane; ++item)
            {
                const int expert     = lane + item * kRouterLanes;
                bool selected_before = false;
#pragma unroll
                for(int prior = 0; prior < rank; ++prior)
                    selected_before |= selected_ids[prior] == expert;
                const float value =
                    selected_before
                        ? -INFINITY
                        : iq2r_router_value(router_logits,
                                            router_bias,
                                            static_cast<int64_t>(token) * router_stride + expert,
                                            expert);
                local = ArgMax()(local, kvp{expert, value});
            }
            const kvp selected    = multithread_reduce<kvp, ArgMax>(local, ArgMax(), kRouterLanes);
            selected_ids[rank]    = selected.key;
            selected_values[rank] = selected.value;
        }

        float denominator = 0.0f;
        if(renormalize)
        {
#pragma unroll
            for(int rank = 0; rank < kGptOssTopK; ++rank)
                denominator += expf(selected_values[rank] - selected_values[0]);
        }
        else
        {
            float local_sum = 0.0f;
#pragma unroll
            for(int item = 0; item < kExpertsPerLane; ++item)
            {
                const int expert = lane + item * kRouterLanes;
                local_sum +=
                    expf(iq2r_router_value(router_logits,
                                           router_bias,
                                           static_cast<int64_t>(token) * router_stride + expert,
                                           expert) -
                         selected_values[0]);
            }
            denominator = multithread_reduce<float, Sum>(local_sum, Sum(), kRouterLanes);
        }

        if(lane == 0)
        {
            const float inverse = denominator != 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
            for(int rank = 0; rank < kGptOssTopK; ++rank)
            {
                const int route      = token * kGptOssTopK + rank;
                const int expert     = selected_ids[rank];
                topk_ids[route]      = expert;
                topk_weights[route]  = expf(selected_values[rank] - selected_values[0]) * inverse;
                route_experts[route] = expert;
            }
        }
    }
    __syncthreads();

    for(int expert = thread; expert < kGptOssExperts; expert += kFusedSortThreads)
        counts[expert] = 0;
    __syncthreads();

    const int routes = tokens * kGptOssTopK;
    if(thread < routes)
        atomicAdd(counts + route_experts[thread], 1);
    __syncthreads();

    int count_prefix = thread < kGptOssExperts ? counts[thread] : 0;
#pragma unroll
    for(int delta = 1; delta < 64; delta <<= 1)
    {
        const int previous = __shfl_up(count_prefix, delta, 64);
        if((thread & 63) >= delta)
            count_prefix += previous;
    }
    if((thread & 63) == 63 && thread < kGptOssExperts)
        wave_sums[thread >> 6] = count_prefix;
    __syncthreads();
    if(thread >= 64 && thread < kGptOssExperts)
        count_prefix += wave_sums[0];
    if(thread < kGptOssExperts)
    {
        offsets[thread] = count_prefix - counts[thread];
        if(thread == kGptOssExperts - 1)
            offsets[kGptOssExperts] = count_prefix;
    }
    __syncthreads();

    if(thread < kGptOssExperts)
        cursors[thread] = offsets[thread];
    __syncthreads();

    if(thread < routes)
    {
        const int expert                = route_experts[thread];
        const int sorted_route          = atomicAdd(cursors + expert, 1);
        sorted_expert_ids[sorted_route] = expert;
        gather_indices[sorted_route]    = thread;
        scatter_indices[thread]         = sorted_route;
    }
    __syncthreads();

    int task_prefix = thread < kGptOssExperts ? (counts[thread] + task_rows - 1) / task_rows : 0;
#pragma unroll
    for(int delta = 1; delta < 64; delta <<= 1)
    {
        const int previous = __shfl_up(task_prefix, delta, 64);
        if((thread & 63) >= delta)
            task_prefix += previous;
    }
    if((thread & 63) == 63 && thread < kGptOssExperts)
        wave_sums[thread >> 6] = task_prefix;
    __syncthreads();
    if(thread >= 64 && thread < kGptOssExperts)
        task_prefix += wave_sums[0];
    if(thread < kGptOssExperts)
    {
        const int count      = counts[thread];
        const int task_begin = task_prefix - (count + task_rows - 1) / task_rows;
        int task             = task_begin;
        for(int local = 0; local < count; local += task_rows, ++task)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns]     = offsets[thread] + local;
                tasks[task * kTaskColumns + 1] = min(task_rows, count - local);
                tasks[task * kTaskColumns + 2] = thread;
            }
        }
        if(thread == kGptOssExperts - 1)
            task_count[0] = task_prefix <= task_capacity ? task_prefix : -1;
    }
}

// Plain GLM-5.3 decode front end for one or two tokens. One 256-thread CTA
// computes biased-sigmoid top-8 routing for every token, maps global experts
// to the rank-local EP ownership, emits compact one-row tasks, and quantizes
// each hidden row once before broadcasting it only to local routes.
template <typename RouterType>
__global__ __launch_bounds__(kGlmExperts) void iq2r_route_glm_topk_direct_gather_quant_kernel(
    const opus::bf16_t* __restrict__ input,
    const RouterType* __restrict__ router_logits,
    const RouterType* __restrict__ correction_bias,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    const int32_t* __restrict__ expert_map,
    int expert_map_count,
    int expert_count,
    int expert_start,
    int expert_stride,
    int tokens,
    int hidden,
    int input_stride,
    int router_stride,
    int groups_per_row,
    float routed_scaling_factor,
    int scale_m_blocks,
    bool tiled_scales)
{
    using kvp        = KeyValuePair<int, float>;
    const int expert = static_cast<int>(threadIdx.x);
    __shared__ int selected_ids[2 * kGlmTopK];

    for(int token = 0; token < tokens; ++token)
    {
        const int64_t offset = static_cast<int64_t>(token) * router_stride + expert;
        const float sigmoid  = iq2r_router_sigmoid(router_logits, offset);
        kvp local            = {expert, sigmoid + static_cast<float>(correction_bias[expert])};
#pragma unroll
        for(int rank = 0; rank < kGlmTopK; ++rank)
        {
            const kvp selected = block_reduce<kvp, ArgMax, kGlmExperts, true>(local, ArgMax());
            if(expert == 0)
                selected_ids[token * kGlmTopK + rank] = selected.key;
            if(expert == selected.key)
                local.value = -INFINITY;
            __syncthreads();
        }
    }

    if(expert == 0)
    {
        int task = 0;
        for(int token = 0; token < tokens; ++token)
        {
            float denominator = 0.0f;
#pragma unroll
            for(int rank = 0; rank < kGlmTopK; ++rank)
            {
                const int selected = selected_ids[token * kGlmTopK + rank];
                denominator += iq2r_router_sigmoid(
                    router_logits, static_cast<int64_t>(token) * router_stride + selected);
            }
            const float scale = denominator != 0.0f ? routed_scaling_factor / denominator : 0.0f;
#pragma unroll
            for(int rank = 0; rank < kGlmTopK; ++rank)
            {
                const int route        = token * kGlmTopK + rank;
                const int selected     = selected_ids[route];
                const int local_expert = iq2r_local_expert(selected,
                                                           expert_map,
                                                           expert_map_count,
                                                           expert_start,
                                                           expert_stride,
                                                           expert_count);
                topk_ids[route]        = selected;
                topk_weights[route] =
                    iq2r_router_sigmoid(router_logits,
                                        static_cast<int64_t>(token) * router_stride + selected) *
                    scale;
                sorted_expert_ids[route] = local_expert;
                gather_indices[route]    = route;
                scatter_indices[route]   = local_expert >= 0 ? route : -1;
                if(local_expert >= 0)
                {
                    tasks[task * kTaskColumns]     = route;
                    tasks[task * kTaskColumns + 1] = 1;
                    tasks[task * kTaskColumns + 2] = local_expert;
                    ++task;
                }
            }
        }
        task_count[0] = task;
    }
    __syncthreads();

    const int quant_groups = tokens * groups_per_row;
    for(int group_id = expert; group_id < quant_groups; group_id += kGlmExperts)
    {
        const int token      = group_id / groups_per_row;
        const int group      = group_id % groups_per_row;
        bool has_local_route = false;
#pragma unroll
        for(int rank = 0; rank < kGlmTopK; ++rank)
            has_local_route |= scatter_indices[token * kGlmTopK + rank] >= 0;
        if(!has_local_route)
            continue;

        const int column_begin     = group * kQuantGroup;
        const int64_t input_offset = static_cast<int64_t>(token) * input_stride + column_begin;
        using input_vector         = opus::vector_t<opus::bf16_t, kQuantGroup>;
        using output_vector        = opus::vector_t<opus::fp8_t, kQuantGroup>;
        const input_vector values  = *reinterpret_cast<const input_vector*>(input + input_offset);

        float abs_max = 1.0e-10f;
#pragma unroll
        for(int element = 0; element < kQuantGroup; ++element)
            abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));
        const auto block_scale =
            fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
        const float inverse_scale = 1.0f / block_scale.dq_scale;
        output_vector quantized;
#pragma unroll
        for(int element = 0; element < kQuantGroup; ++element)
            quantized[element] =
                opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

#pragma unroll
        for(int rank = 0; rank < kGlmTopK; ++rank)
        {
            const int route = token * kGlmTopK + rank;
            if(scatter_indices[route] < 0)
                continue;
            const int64_t output_offset = static_cast<int64_t>(route) * hidden + column_begin;
            *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
            scales[iq2r_scale_offset(route, group, groups_per_row, scale_m_blocks, tiled_scales)] =
                block_scale.byte;
        }
    }
}

// Plain GLM-5.3 decode front end for three to sixteen tokens. Thirty-two lanes
// cooperate on each 256-expert router row. The CTA then maps the selected
// global experts to rank-local ownership and creates the grouped route tasks.
template <typename RouterType>
__global__ __launch_bounds__(kFusedSortThreads) void iq2r_route_glm_topk_sort_tasks_kernel(
    const RouterType* __restrict__ router_logits,
    const RouterType* __restrict__ correction_bias,
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_expert_ids,
    int32_t* __restrict__ gather_indices,
    int32_t* __restrict__ scatter_indices,
    int32_t* __restrict__ tasks,
    int32_t* __restrict__ task_count,
    const int32_t* __restrict__ expert_map,
    int expert_map_count,
    int expert_count,
    int expert_start,
    int expert_stride,
    int tokens,
    int router_stride,
    int task_capacity,
    int task_rows,
    float routed_scaling_factor,
    bool drop_nonlocal_routes)
{
    constexpr int kRouterLanes    = 32;
    constexpr int kExpertsPerLane = kGlmExperts / kRouterLanes;
    using kvp                     = KeyValuePair<int, float>;

    __shared__ int counts[kGlmExperts + 1];
    __shared__ int offsets[kGlmExperts + 2];
    __shared__ int cursors[kGlmExperts + 1];
    __shared__ int route_experts[kGlmFusedSortRoutes];
    __shared__ int route_local_experts[kGlmFusedSortRoutes];

    const int thread              = static_cast<int>(threadIdx.x);
    const int router_thread_count = tokens * kRouterLanes;
    if(thread < router_thread_count)
    {
        const int token = thread / kRouterLanes;
        const int lane  = thread % kRouterLanes;
        int selected_ids[kGlmTopK];

#pragma unroll
        for(int rank = 0; rank < kGlmTopK; ++rank)
        {
            kvp local = {INT_MAX, -INFINITY};
#pragma unroll
            for(int item = 0; item < kExpertsPerLane; ++item)
            {
                const int candidate_expert = lane + item * kRouterLanes;
                bool selected_before       = false;
#pragma unroll
                for(int prior = 0; prior < rank; ++prior)
                    selected_before |= selected_ids[prior] == candidate_expert;
                const int64_t offset =
                    static_cast<int64_t>(token) * router_stride + candidate_expert;
                const float selection_score =
                    selected_before ? -INFINITY
                                    : iq2r_router_sigmoid(router_logits, offset) +
                                          static_cast<float>(correction_bias[candidate_expert]);
                local = ArgMax()(local, kvp{candidate_expert, selection_score});
            }
            const kvp selected = multithread_reduce<kvp, ArgMax>(local, ArgMax(), kRouterLanes);
            selected_ids[rank] = selected.key;
        }

        float denominator = 0.0f;
#pragma unroll
        for(int rank = 0; rank < kGlmTopK; ++rank)
            denominator += iq2r_router_sigmoid(
                router_logits, static_cast<int64_t>(token) * router_stride + selected_ids[rank]);
        if(lane == 0)
        {
            const float scale = denominator != 0.0f ? routed_scaling_factor / denominator : 0.0f;
#pragma unroll
            for(int rank = 0; rank < kGlmTopK; ++rank)
            {
                const int route    = token * kGlmTopK + rank;
                const int selected = selected_ids[rank];
                topk_ids[route]    = selected;
                topk_weights[route] =
                    iq2r_router_sigmoid(router_logits,
                                        static_cast<int64_t>(token) * router_stride + selected) *
                    scale;
                route_experts[route] = selected;
            }
        }
    }
    __syncthreads();

    for(int bucket = thread; bucket <= expert_count; bucket += kFusedSortThreads)
        counts[bucket] = 0;
    __syncthreads();

    const int routes = tokens * kGlmTopK;
    if(thread < routes)
    {
        const int local_expert      = iq2r_local_expert(route_experts[thread],
                                                   expert_map,
                                                   expert_map_count,
                                                   expert_start,
                                                   expert_stride,
                                                   expert_count);
        route_local_experts[thread] = local_expert;
        atomicAdd(counts + (local_expert >= 0 ? local_expert : expert_count), 1);
    }
    __syncthreads();

    if(thread == 0)
    {
        offsets[0] = 0;
        for(int bucket = 0; bucket <= expert_count; ++bucket)
            offsets[bucket + 1] = offsets[bucket] + counts[bucket];
        int task         = 0;
        int sorted_begin = 0;
        for(int local_expert = 0; local_expert < expert_count; ++local_expert)
        {
            const int count = counts[local_expert];
            for(int local = 0; local < count; local += task_rows)
            {
                if(task < task_capacity)
                {
                    tasks[task * kTaskColumns]     = sorted_begin + local;
                    tasks[task * kTaskColumns + 1] = min(task_rows, count - local);
                    tasks[task * kTaskColumns + 2] = local_expert;
                }
                ++task;
            }
            sorted_begin += count;
        }
        const int invalid_count = counts[expert_count];
        for(int local = 0; !drop_nonlocal_routes && local < invalid_count; local += task_rows)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns]     = sorted_begin + local;
                tasks[task * kTaskColumns + 1] = min(task_rows, invalid_count - local);
                tasks[task * kTaskColumns + 2] = -1;
            }
            ++task;
        }
        task_count[0] = task <= task_capacity ? task : -1;
    }
    __syncthreads();

    for(int bucket = thread; bucket <= expert_count; bucket += kFusedSortThreads)
        cursors[bucket] = offsets[bucket];
    __syncthreads();

    if(thread < routes)
    {
        const int local_expert          = route_local_experts[thread];
        const int bucket                = local_expert >= 0 ? local_expert : expert_count;
        const int sorted_route          = atomicAdd(cursors + bucket, 1);
        sorted_expert_ids[sorted_route] = local_expert;
        gather_indices[sorted_route]    = thread;
        scatter_indices[thread] = drop_nonlocal_routes && local_expert < 0 ? -1 : sorted_route;
    }
}

// Quantize every source token once, then broadcast its packed row to all
// expert-grouped destinations. A CTA per token restores enough parallelism
// while avoiding top-k redundant BF16 reads, max reductions, and FP8
// conversions.
template <int TopK>
__global__ __launch_bounds__(kMaxQuantThreads) void iq2r_route_gather_quant_broadcast_kernel(
    const opus::bf16_t* __restrict__ input,
    const int32_t* __restrict__ scatter_indices,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    int tokens,
    int hidden,
    int input_stride,
    int groups_per_row,
    int scale_m_blocks,
    bool tiled_scales)
{
    const int token = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(threadIdx.x);
    if(token >= tokens || group >= groups_per_row)
        return;

    const int column_begin     = group * kQuantGroup;
    const int64_t input_offset = static_cast<int64_t>(token) * input_stride + column_begin;
    using input_vector         = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector        = opus::vector_t<opus::fp8_t, kQuantGroup>;
    const input_vector values  = *reinterpret_cast<const input_vector*>(input + input_offset);

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] = opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

#pragma unroll
    for(int rank = 0; rank < TopK; ++rank)
    {
        const int route        = token * TopK + rank;
        const int sorted_route = scatter_indices[route];
        if(sorted_route < 0)
            continue;
        const int64_t output_offset = static_cast<int64_t>(sorted_route) * hidden + column_begin;
        *reinterpret_cast<output_vector*>(output + output_offset)               = quantized;
        scales[iq2r_scale_offset(
            sorted_route, group, groups_per_row, scale_m_blocks, tiled_scales)] = block_scale.byte;
    }
}

__global__ void iq2r_swiglu_kernel(const __hip_bfloat16* __restrict__ gate_up,
                                   __hip_bfloat16* __restrict__ output,
                                   int64_t elements,
                                   int intermediate,
                                   float limit,
                                   float alpha,
                                   float up_offset)
{
    for(int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const int64_t row            = index / intermediate;
        const int column             = static_cast<int>(index % intermediate);
        const int64_t gate_up_offset = row * (2 * intermediate);
        const float unclamped_gate   = __bfloat162float(gate_up[gate_up_offset + 2 * column]);
        const float unclamped_up     = __bfloat162float(gate_up[gate_up_offset + 2 * column + 1]);
        const bool clamp             = limit > 0.0f;
        const float gate             = clamp && unclamped_gate > limit ? limit : unclamped_gate;
        const float up               = clamp && unclamped_up < -limit
                                           ? -limit
                                           : (clamp && unclamped_up > limit ? limit : unclamped_up);
        const float swish            = gate / (1.0f + __expf(-alpha * gate));
        output[index]                = __float2bfloat16(swish * (up + up_offset));
    }
}

// Apply the exact GPT-OSS clipped SwiGLU and immediately quantize the rounded
// BF16 activation to canonical row-major MXFP8/E8M0 blocks.  The optional BF16
// output exists for focused validation; production passes nullptr and avoids
// both the intermediate write and the following read.
__global__ __launch_bounds__(kQuantThreads) void iq2r_swiglu_quant_kernel(
    const __hip_bfloat16* __restrict__ gate_up,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    __hip_bfloat16* __restrict__ activated,
    int64_t groups,
    int intermediate,
    int groups_per_row,
    int scale_m_blocks,
    bool tiled_scales,
    float limit,
    float alpha,
    float up_offset)
{
    const int64_t group_id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(group_id >= groups)
        return;

    const int64_t row            = group_id / groups_per_row;
    const int group              = static_cast<int>(group_id % groups_per_row);
    const int column_begin       = group * kQuantGroup;
    const int64_t gate_up_offset = row * (2 * intermediate);
    const int64_t output_offset  = row * intermediate + column_begin;
    using activated_vector       = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector          = opus::vector_t<opus::fp8_t, kQuantGroup>;
    activated_vector values;

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
    {
        const int column             = column_begin + element;
        const float unclamped_gate   = __bfloat162float(gate_up[gate_up_offset + 2 * column]);
        const float unclamped_up     = __bfloat162float(gate_up[gate_up_offset + 2 * column + 1]);
        const bool clamp             = limit > 0.0f;
        const float gate             = clamp && unclamped_gate > limit ? limit : unclamped_gate;
        const float up               = clamp && unclamped_up < -limit
                                           ? -limit
                                           : (clamp && unclamped_up > limit ? limit : unclamped_up);
        const float swish            = gate / (1.0f + __expf(-alpha * gate));
        const __hip_bfloat16 rounded = __float2bfloat16(swish * (up + up_offset));
        values[element]              = __builtin_bit_cast(opus::bf16_t, rounded);
        abs_max                      = fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
    }

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] = opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

    *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
    scales[iq2r_scale_offset(
        static_cast<int>(row), group, groups_per_row, scale_m_blocks, tiled_scales)] =
        block_scale.byte;
    if(activated != nullptr)
    {
        *reinterpret_cast<activated_vector*>(activated + output_offset) = values;
    }
}

// Low-row GPT-OSS specialization. Four neighboring lanes cooperate on one
// 32-value MXFP8 block while each lane computes eight interleaved gate/up
// pairs. This exposes substantially more parallelism than assigning an entire
// quantization block to one thread, while preserving the BF16 rounding point
// before scale selection and FP8 conversion.
__global__ __launch_bounds__(kQuantThreads) void iq2r_swiglu_quant_parallel8_kernel(
    const __hip_bfloat16* __restrict__ gate_up,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    __hip_bfloat16* __restrict__ activated,
    int rows,
    int intermediate,
    int groups_per_row,
    int blocks_per_row,
    int scale_m_blocks,
    bool tiled_scales,
    float limit,
    float alpha,
    float up_offset)
{
    const int row          = static_cast<int>(blockIdx.x) / blocks_per_row;
    const int block_in_row = static_cast<int>(blockIdx.x) % blocks_per_row;
    const int column_begin =
        (block_in_row * blockDim.x + static_cast<int>(threadIdx.x)) * kSwiGLUQuantElementsPerThread;
    const bool active            = row < rows && column_begin < intermediate;
    const int64_t gate_up_offset = static_cast<int64_t>(row) * 2 * intermediate;
    const int64_t output_offset  = static_cast<int64_t>(row) * intermediate + column_begin;
    using activated_vector       = opus::vector_t<opus::bf16_t, kSwiGLUQuantElementsPerThread>;
    using output_vector          = opus::vector_t<opus::fp8_t, kSwiGLUQuantElementsPerThread>;
    activated_vector values;

    float abs_max = 1.0e-10f;
    if(active)
    {
#pragma unroll
        for(int element = 0; element < kSwiGLUQuantElementsPerThread; ++element)
        {
            const int column           = column_begin + element;
            const float unclamped_gate = __bfloat162float(gate_up[gate_up_offset + 2 * column]);
            const float unclamped_up   = __bfloat162float(gate_up[gate_up_offset + 2 * column + 1]);
            const bool clamp           = limit > 0.0f;
            const float gate           = clamp && unclamped_gate > limit ? limit : unclamped_gate;
            const float up             = clamp && unclamped_up < -limit
                                             ? -limit
                                             : (clamp && unclamped_up > limit ? limit : unclamped_up);
            const float swish          = gate / (1.0f + __expf(-alpha * gate));
            const __hip_bfloat16 rounded = __float2bfloat16(swish * (up + up_offset));
            values[element]              = __builtin_bit_cast(opus::bf16_t, rounded);
            abs_max                      = fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
        }
    }

    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 1));
    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
    if(!active)
        return;

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kSwiGLUQuantElementsPerThread; ++element)
        quantized[element] = opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

    *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
    if(activated != nullptr)
        *reinterpret_cast<activated_vector*>(activated + output_offset) = values;
    if((threadIdx.x & 3) == 0)
    {
        const int group = column_begin / kQuantGroup;
        scales[iq2r_scale_offset(row, group, groups_per_row, scale_m_blocks, tiled_scales)] =
            block_scale.byte;
    }
}

// Expert-parallel specialization. The route sorter records non-local routes
// with scatter index -1. Quantize only the local destinations for each token,
// avoiding activation work and writes for the other EP ranks.
template <int TopK>
__global__ __launch_bounds__(kQuantThreads) void iq2r_swiglu_quant_scatter_kernel(
    const __hip_bfloat16* __restrict__ gate_up,
    const int32_t* __restrict__ scatter_indices,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    __hip_bfloat16* __restrict__ activated,
    int tokens,
    int intermediate,
    int groups_per_row,
    int blocks_per_row,
    int scale_m_blocks,
    bool tiled_scales,
    float limit,
    float alpha,
    float up_offset)
{
    const int token        = static_cast<int>(blockIdx.x) / blocks_per_row;
    const int block_in_row = static_cast<int>(blockIdx.x) % blocks_per_row;
    const int column_begin =
        (block_in_row * blockDim.x + static_cast<int>(threadIdx.x)) * kSwiGLUQuantElementsPerThread;
    const bool active_column = token < tokens && column_begin < intermediate;

#pragma unroll
    for(int route_in_token = 0; route_in_token < TopK; ++route_in_token)
    {
        const int original_route = token * TopK + route_in_token;
        const int row            = scatter_indices[original_route];
        if(row < 0)
            continue;

        const int64_t gate_up_offset = static_cast<int64_t>(row) * 2 * intermediate;
        const int64_t output_offset  = static_cast<int64_t>(row) * intermediate + column_begin;
        using activated_vector       = opus::vector_t<opus::bf16_t, kSwiGLUQuantElementsPerThread>;
        using output_vector          = opus::vector_t<opus::fp8_t, kSwiGLUQuantElementsPerThread>;
        activated_vector values;

        float abs_max = 1.0e-10f;
        if(active_column)
        {
#pragma unroll
            for(int element = 0; element < kSwiGLUQuantElementsPerThread; ++element)
            {
                const int column           = column_begin + element;
                const float unclamped_gate = __bfloat162float(gate_up[gate_up_offset + 2 * column]);
                const float unclamped_up =
                    __bfloat162float(gate_up[gate_up_offset + 2 * column + 1]);
                const bool clamp  = limit > 0.0f;
                const float gate  = clamp && unclamped_gate > limit ? limit : unclamped_gate;
                const float up    = clamp && unclamped_up < -limit
                                        ? -limit
                                        : (clamp && unclamped_up > limit ? limit : unclamped_up);
                const float swish = gate / (1.0f + __expf(-alpha * gate));
                const __hip_bfloat16 rounded = __float2bfloat16(swish * (up + up_offset));
                values[element]              = __builtin_bit_cast(opus::bf16_t, rounded);
                abs_max                      = fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
            }
        }

        abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 1));
        abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
        if(!active_column)
            continue;

        const auto block_scale =
            fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(abs_max);
        const float inverse_scale = 1.0f / block_scale.dq_scale;
        output_vector quantized;
#pragma unroll
        for(int element = 0; element < kSwiGLUQuantElementsPerThread; ++element)
            quantized[element] =
                opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

        *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
        if(activated != nullptr)
            *reinterpret_cast<activated_vector*>(activated + output_offset) = values;
        if((threadIdx.x & 3) == 0)
        {
            const int group = column_begin / kQuantGroup;
            scales[iq2r_scale_offset(row, group, groups_per_row, scale_m_blocks, tiled_scales)] =
                block_scale.byte;
        }
    }
}

template <bool AddShared>
__global__ void iq2r_route_reduce_indexed_kernel(const __hip_bfloat16* __restrict__ route_output,
                                                 const float* __restrict__ route_weights,
                                                 const int32_t* __restrict__ scatter_indices,
                                                 const __hip_bfloat16* __restrict__ shared_output,
                                                 __hip_bfloat16* __restrict__ output,
                                                 int64_t elements,
                                                 int hidden,
                                                 int topk)
{
    for(int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const int64_t token = index / hidden;
        const int column    = static_cast<int>(index % hidden);
        float value         = 0.0f;
        for(int route = 0; route < topk; ++route)
        {
            const int64_t original_route = token * topk + route;
            const int sorted_route       = scatter_indices[original_route];
            if(sorted_route >= 0)
                value =
                    fmaf(__bfloat162float(
                             route_output[static_cast<int64_t>(sorted_route) * hidden + column]),
                         route_weights[original_route],
                         value);
        }
        const __hip_bfloat16 routed = __float2bfloat16(value);
        if constexpr(AddShared)
            output[index] = __float2bfloat16(__bfloat162float(routed) +
                                             __bfloat162float(shared_output[index]));
        else
            output[index] = routed;
    }
}

// Plain GLM-5.3 uses hidden size 6144. The routed-only form has top-k 8;
// fusing its always-on shared expert produces top-k 9. One workgroup owns one
// token's 256-column tile, so the token/tile mapping uses only a compile-time
// 32-bit divide by 24 instead of the generic per-element 64-bit quotient and
// remainder. Lanes 0-7 load route metadata once per wave and broadcast it to
// the other lanes while the route loop is fully unrolled.
template <int ItemsPerThread, int TopK, bool AddShared>
__global__ __launch_bounds__(256 / ItemsPerThread) void iq2r_route_reduce_glm6144_kernel(
    const __hip_bfloat16* __restrict__ route_output,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ scatter_indices,
    const __hip_bfloat16* __restrict__ shared_output,
    __hip_bfloat16* __restrict__ output,
    int tokens)
{
    static_assert(ItemsPerThread == 1 || ItemsPerThread == 2 || ItemsPerThread == 4);
    constexpr int kHidden          = 6144;
    constexpr int kColumnsPerBlock = 256;
    constexpr int kTilesPerToken   = kHidden / kColumnsPerBlock;
    const int block_index          = static_cast<int>(blockIdx.x);
    const int token                = block_index / kTilesPerToken;
    if(token >= tokens)
        return;
    const int tile = block_index - token * kTilesPerToken;
    const int column_begin =
        tile * kColumnsPerBlock + static_cast<int>(threadIdx.x) * ItemsPerThread;
    const int lane            = static_cast<int>(threadIdx.x) & 63;
    const int metadata_offset = token * TopK + lane;
    const int lane_route      = lane < TopK ? scatter_indices[metadata_offset] : -1;
    const float lane_weight   = lane < TopK ? route_weights[metadata_offset] : 0.0f;

    int sorted_routes[TopK];
    float weights[TopK];
#pragma unroll
    for(int route = 0; route < TopK; ++route)
    {
        sorted_routes[route] = __shfl(lane_route, route);
        weights[route]       = __shfl(lane_weight, route);
    }

    float values[ItemsPerThread] = {};
#pragma unroll
    for(int route = 0; route < TopK; ++route)
    {
        const int sorted_route = sorted_routes[route];
        if(sorted_route >= 0)
        {
            const int route_offset = sorted_route * kHidden + column_begin;
#pragma unroll
            for(int item = 0; item < ItemsPerThread; ++item)
                values[item] = fmaf(__bfloat162float(route_output[route_offset + item]),
                                    weights[route],
                                    values[item]);
        }
    }

    const int output_offset = token * kHidden + column_begin;
#pragma unroll
    for(int item = 0; item < ItemsPerThread; ++item)
    {
        const int output_index       = output_offset + item;
        const __hip_bfloat16 routed = __float2bfloat16(values[item]);
        if constexpr(AddShared)
            output[output_index] = __float2bfloat16(__bfloat162float(routed) +
                                                    __bfloat162float(shared_output[output_index]));
        else
            output[output_index] = routed;
    }
}

template <int BlockSize, int ItemsPerThread>
__global__ __launch_bounds__(BlockSize) void iq2r_route_reduce_add_rmsnorm_indexed_kernel(
    const __hip_bfloat16* __restrict__ route_output,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ scatter_indices,
    const __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ norm_weight,
    __hip_bfloat16* __restrict__ output,
    __hip_bfloat16* __restrict__ residual_out,
    int hidden,
    int topk,
    float epsilon)
{
    const int token = blockIdx.x;
    float values[ItemsPerThread];
    float square_sum = 0.0f;

#pragma unroll
    for(int item = 0; item < ItemsPerThread; ++item)
    {
        const int column = threadIdx.x + item * BlockSize;
        float combined   = 0.0f;
        if(column < hidden)
        {
            float route_value = 0.0f;
#pragma unroll
            for(int route = 0; route < kGptOssTopK; ++route)
            {
                if(route < topk)
                {
                    const int64_t original_route = static_cast<int64_t>(token) * topk + route;
                    const int sorted_route       = scatter_indices[original_route];
                    if(sorted_route >= 0)
                        route_value = fmaf(
                            __bfloat162float(
                                route_output[static_cast<int64_t>(sorted_route) * hidden + column]),
                            route_weights[original_route],
                            route_value);
                }
            }

            // Preserve the unfused boundary: route_reduce writes BF16, then
            // add_rmsnorm reloads that rounded value before adding residual.
            const __hip_bfloat16 rounded_route = __float2bfloat16(route_value);
            combined                           = __bfloat162float(rounded_route) +
                       __bfloat162float(residual[static_cast<int64_t>(token) * hidden + column]);
            residual_out[static_cast<int64_t>(token) * hidden + column] =
                __float2bfloat16(combined);
            square_sum = fmaf(combined, combined, square_sum);
        }
        values[item] = combined;
    }

    const float row_square_sum = block_reduce<float, Sum, BlockSize, true>(square_sum, Sum());
    const float inverse_rms    = rsqrtf(row_square_sum / static_cast<float>(hidden) + epsilon);

#pragma unroll
    for(int item = 0; item < ItemsPerThread; ++item)
    {
        const int column = threadIdx.x + item * BlockSize;
        if(column < hidden)
        {
            const float normalized =
                values[item] * inverse_rms * __bfloat162float(norm_weight[column]);
            output[static_cast<int64_t>(token) * hidden + column] = __float2bfloat16(normalized);
        }
    }
}

int launch_blocks(int64_t elements)
{
    return static_cast<int>((elements + kThreads - 1) / kThreads);
}

} // namespace

void iq2r_route_sort_tasks_out(const aiter_tensor_t& expert_ids,
                               aiter_tensor_t& sorted_expert_ids,
                               aiter_tensor_t& gather_indices,
                               aiter_tensor_t& scatter_indices,
                               aiter_tensor_t& tasks,
                               aiter_tensor_t& task_count,
                               std::optional<aiter_tensor_t> expert_map,
                               int64_t expert_count,
                               int64_t expert_start,
                               int64_t expert_stride,
                               int64_t task_rows,
                               bool drop_nonlocal_tasks)
{
    AITER_CHECK(expert_ids.is_gpu() && sorted_expert_ids.is_gpu() && gather_indices.is_gpu() &&
                    scatter_indices.is_gpu() && tasks.is_gpu() && task_count.is_gpu(),
                "IQ2R route sorting requires GPU tensors");
    const int device = expert_ids.device_id;
    AITER_CHECK(sorted_expert_ids.device_id == device && gather_indices.device_id == device &&
                    scatter_indices.device_id == device && tasks.device_id == device &&
                    task_count.device_id == device,
                "IQ2R route sorting tensors must share a GPU");
    AITER_CHECK(expert_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    tasks.dtype() == AITER_DTYPE_i32 && task_count.dtype() == AITER_DTYPE_i32,
                "IQ2R route sorting tensors must be int32");
    AITER_CHECK(
        expert_ids.dim() == 1 && expert_ids.numel() > 0 && expert_ids.numel() <= kMaxRoutes &&
            sorted_expert_ids.numel() == expert_ids.numel() &&
            gather_indices.numel() == expert_ids.numel() &&
            scatter_indices.numel() == expert_ids.numel() && tasks.dim() == 2 &&
            tasks.size(1) == kTaskColumns && task_count.dim() == 1 && task_count.size(0) == 1,
        "IQ2R route sorting shape mismatch");
    AITER_CHECK(expert_ids.is_contiguous() && sorted_expert_ids.is_contiguous() &&
                    gather_indices.is_contiguous() && scatter_indices.is_contiguous() &&
                    tasks.is_contiguous() && task_count.is_contiguous(),
                "IQ2R route sorting tensors must be contiguous");
    const int32_t* expert_map_ptr = nullptr;
    int64_t expert_map_count      = 0;
    if(expert_map.has_value())
    {
        AITER_CHECK(expert_map->is_gpu() && expert_map->device_id == device &&
                        expert_map->dtype() == AITER_DTYPE_i32 && expert_map->dim() == 1 &&
                        expert_map->numel() > 0 && expert_map->numel() <= kMaxExperts &&
                        expert_map->is_contiguous(),
                    "IQ2R expert_map must be contiguous int32 [1..512] on the route GPU");
        expert_map_ptr   = static_cast<const int32_t*>(expert_map->data_ptr());
        expert_map_count = expert_map->numel();
    }
    AITER_CHECK(expert_count > 0 && expert_count <= kMaxExperts,
                "IQ2R supports at most 512 experts");
    AITER_CHECK(expert_start >= 0 && expert_stride > 0 &&
                    expert_start + (expert_count - 1) * expert_stride < kMaxExperts,
                "IQ2R strided local expert range must fit within 512 global experts");
    AITER_CHECK(task_rows == 16 || task_rows == 32 || task_rows == 64 || task_rows == 128 ||
                    task_rows == 256,
                "IQ2R task_rows must be 16, 32, 64, 128, or 256");
    const int64_t routes = expert_ids.numel();
    const int64_t required_capacity =
        (routes + task_rows - 1) / task_rows + std::min<int64_t>(routes, expert_count + 1);
    AITER_CHECK(tasks.size(0) >= required_capacity, "IQ2R task capacity is too small");

    if(routes <= kSmallRoutes)
        hipLaunchKernelGGL(iq2r_route_sort_tasks_small_kernel,
                           dim3(1),
                           dim3(kThreads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const int32_t*>(expert_ids.data_ptr()),
                           static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
                           static_cast<int32_t*>(gather_indices.data_ptr()),
                           static_cast<int32_t*>(scatter_indices.data_ptr()),
                           static_cast<int32_t*>(tasks.data_ptr()),
                           static_cast<int32_t*>(task_count.data_ptr()),
                           expert_map_ptr,
                           static_cast<int>(routes),
                           static_cast<int>(expert_map_count),
                           static_cast<int>(expert_count),
                           static_cast<int>(expert_start),
                           static_cast<int>(expert_stride),
                           static_cast<int>(tasks.size(0)),
                           static_cast<int>(task_rows),
                           drop_nonlocal_tasks);
    else if(expert_count == kGlmExperts || expert_count == kGlmExperts + 1)
        hipLaunchKernelGGL(iq2r_route_sort_tasks_glm_kernel,
                           dim3(1),
                           dim3(kThreads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const int32_t*>(expert_ids.data_ptr()),
                           static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
                           static_cast<int32_t*>(gather_indices.data_ptr()),
                           static_cast<int32_t*>(scatter_indices.data_ptr()),
                           static_cast<int32_t*>(tasks.data_ptr()),
                           static_cast<int32_t*>(task_count.data_ptr()),
                           expert_map_ptr,
                           static_cast<int>(routes),
                           static_cast<int>(expert_map_count),
                           static_cast<int>(expert_count),
                           static_cast<int>(expert_start),
                           static_cast<int>(expert_stride),
                           static_cast<int>(tasks.size(0)),
                           static_cast<int>(task_rows),
                           drop_nonlocal_tasks);
    else
        hipLaunchKernelGGL(iq2r_route_sort_tasks_kernel,
                           dim3(1),
                           dim3(kThreads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const int32_t*>(expert_ids.data_ptr()),
                           static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
                           static_cast<int32_t*>(gather_indices.data_ptr()),
                           static_cast<int32_t*>(scatter_indices.data_ptr()),
                           static_cast<int32_t*>(tasks.data_ptr()),
                           static_cast<int32_t*>(task_count.data_ptr()),
                           expert_map_ptr,
                           static_cast<int>(routes),
                           static_cast<int>(expert_map_count),
                           static_cast<int>(expert_count),
                           static_cast<int>(expert_start),
                           static_cast<int>(expert_stride),
                           static_cast<int>(tasks.size(0)),
                           static_cast<int>(task_rows),
                           drop_nonlocal_tasks);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_gather_indexed_out(const aiter_tensor_t& input,
                                   const aiter_tensor_t& gather_indices,
                                   aiter_tensor_t& output,
                                   int64_t topk)
{
    AITER_CHECK(input.is_gpu() && gather_indices.is_gpu() && output.is_gpu() &&
                    input.device_id == gather_indices.device_id &&
                    input.device_id == output.device_id,
                "IQ2R route gather tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 && output.dtype() == AITER_DTYPE_bf16 &&
                    gather_indices.dtype() == AITER_DTYPE_i32,
                "IQ2R route gather expects BF16 data and int32 indices");
    AITER_CHECK(input.dim() == 2 && output.dim() == 2 && gather_indices.dim() == 1 && topk > 0 &&
                    output.size(0) == input.size(0) * topk && output.size(1) == input.size(1) &&
                    gather_indices.numel() == output.size(0),
                "IQ2R route gather shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    gather_indices.is_contiguous() && output.is_contiguous(),
                "IQ2R route gather input must have contiguous columns and "
                "non-overlapping rows; indices/output must be contiguous");
    const int64_t elements = output.numel();
    hipLaunchKernelGGL(iq2r_route_gather_indexed_kernel,
                       dim3(launch_blocks(elements)),
                       dim3(kThreads),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const __hip_bfloat16*>(input.data_ptr()),
                       static_cast<const int32_t*>(gather_indices.data_ptr()),
                       static_cast<__hip_bfloat16*>(output.data_ptr()),
                       elements,
                       static_cast<int>(input.size(1)),
                       static_cast<int>(input.stride(0)),
                       static_cast<int>(topk));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_gather_quant_out(const aiter_tensor_t& input,
                                 const aiter_tensor_t& gather_indices,
                                 aiter_tensor_t& output,
                                 aiter_tensor_t& scales,
                                 int64_t topk)
{
    AITER_CHECK(input.is_gpu() && gather_indices.is_gpu() && output.is_gpu() && scales.is_gpu(),
                "IQ2R fused gather/quant requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(gather_indices.device_id == device && output.device_id == device &&
                    scales.device_id == device,
                "IQ2R fused gather/quant tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 && gather_indices.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 && scales.dtype() == AITER_DTYPE_u8,
                "IQ2R fused gather/quant dtype mismatch");
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    AITER_CHECK(groups_per_row <= kMaxQuantThreads,
                "IQ2R broadcast gather hidden size exceeds one-CTA quantization capacity");
    AITER_CHECK(input.dim() == 2 && gather_indices.dim() == 1 && output.dim() == 2 && topk > 0 &&
                    output.size(0) == input.size(0) * topk &&
                    gather_indices.numel() == output.size(0) && output.size(1) == input.size(1) &&
                    output.size(1) % kQuantGroup == 0 &&
                    iq2r_valid_scale_shape(scales, output.size(0), groups_per_row),
                "IQ2R fused gather/quant shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    gather_indices.is_contiguous() && output.is_contiguous() &&
                    scales.is_contiguous(),
                "IQ2R fused gather/quant input must have contiguous columns and "
                "non-overlapping rows; outputs must be contiguous");

    const bool tiled_scales  = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((output.size(0) + 15) / 16);
    const int64_t groups     = output.size(0) * groups_per_row;
    hipLaunchKernelGGL(iq2r_route_gather_quant_kernel,
                       dim3((groups + kQuantThreads - 1) / kQuantThreads),
                       dim3(kQuantThreads),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::bf16_t*>(input.data_ptr()),
                       static_cast<const int32_t*>(gather_indices.data_ptr()),
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(scales.data_ptr()),
                       groups,
                       static_cast<int>(input.size(1)),
                       static_cast<int>(input.stride(0)),
                       static_cast<int>(groups_per_row),
                       static_cast<int>(topk),
                       scale_m_blocks,
                       tiled_scales);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_scatter_quant_out(const aiter_tensor_t& input,
                                  const aiter_tensor_t& scatter_indices,
                                  aiter_tensor_t& output,
                                  aiter_tensor_t& scales,
                                  int64_t topk)
{
    AITER_CHECK(input.is_gpu() && scatter_indices.is_gpu() && output.is_gpu() && scales.is_gpu(),
                "IQ2R fused scatter/quant requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(scatter_indices.device_id == device && output.device_id == device &&
                    scales.device_id == device,
                "IQ2R fused scatter/quant tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 && scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 && scales.dtype() == AITER_DTYPE_u8,
                "IQ2R fused scatter/quant dtype mismatch");
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    AITER_CHECK(groups_per_row <= kMaxQuantThreads,
                "IQ2R scatter/quant hidden size exceeds one-CTA quantization capacity");
    AITER_CHECK(input.dim() == 2 && scatter_indices.dim() == 1 && output.dim() == 2 &&
                    (topk == 4 || topk == 8 || topk == 9) &&
                    output.size(0) == input.size(0) * topk &&
                    scatter_indices.numel() == output.size(0) && output.size(1) == input.size(1) &&
                    output.size(1) % kQuantGroup == 0 &&
                    iq2r_valid_scale_shape(scales, output.size(0), groups_per_row),
                "IQ2R fused scatter/quant shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    scatter_indices.is_contiguous() && output.is_contiguous() &&
                    scales.is_contiguous(),
                "IQ2R fused scatter/quant input must have contiguous columns and "
                "non-overlapping rows; outputs must be contiguous");

    const bool tiled_scales  = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((output.size(0) + 15) / 16);
    const int quant_threads =
        groups_per_row > kDefaultQuantThreads ? kMaxQuantThreads : kDefaultQuantThreads;
#define IQ2R_LAUNCH_SCATTER_QUANT(TOPK)                                         \
    hipLaunchKernelGGL((iq2r_route_gather_quant_broadcast_kernel<TOPK>),        \
                       dim3(static_cast<unsigned int>(input.size(0))),          \
                       dim3(quant_threads),                                     \
                       0,                                                       \
                       getCurrentHIPStream(),                                   \
                       static_cast<const opus::bf16_t*>(input.data_ptr()),      \
                       static_cast<const int32_t*>(scatter_indices.data_ptr()), \
                       static_cast<opus::fp8_t*>(output.data_ptr()),            \
                       static_cast<uint8_t*>(scales.data_ptr()),                \
                       static_cast<int>(input.size(0)),                         \
                       static_cast<int>(input.size(1)),                         \
                       static_cast<int>(input.stride(0)),                       \
                       static_cast<int>(groups_per_row),                        \
                       scale_m_blocks,                                          \
                       tiled_scales)
    if(topk == 4)
        IQ2R_LAUNCH_SCATTER_QUANT(4);
    else if(topk == 8)
        IQ2R_LAUNCH_SCATTER_QUANT(8);
    else
        IQ2R_LAUNCH_SCATTER_QUANT(9);
#undef IQ2R_LAUNCH_SCATTER_QUANT
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_direct_gather_quant_out(const aiter_tensor_t& input,
                                        const aiter_tensor_t& expert_ids,
                                        aiter_tensor_t& sorted_expert_ids,
                                        aiter_tensor_t& gather_indices,
                                        aiter_tensor_t& scatter_indices,
                                        aiter_tensor_t& tasks,
                                        aiter_tensor_t& task_count,
                                        aiter_tensor_t& output,
                                        aiter_tensor_t& scales,
                                        std::optional<aiter_tensor_t> expert_map,
                                        int64_t topk,
                                        int64_t expert_count,
                                        int64_t expert_start,
                                        int64_t expert_stride,
                                        bool drop_nonlocal_routes)
{
    AITER_CHECK(input.is_gpu() && expert_ids.is_gpu() && sorted_expert_ids.is_gpu() &&
                    gather_indices.is_gpu() && scatter_indices.is_gpu() && tasks.is_gpu() &&
                    task_count.is_gpu() && output.is_gpu() && scales.is_gpu(),
                "IQ2R direct routing requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(expert_ids.device_id == device && sorted_expert_ids.device_id == device &&
                    gather_indices.device_id == device && scatter_indices.device_id == device &&
                    tasks.device_id == device && task_count.device_id == device &&
                    output.device_id == device && scales.device_id == device,
                "IQ2R direct routing tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 && expert_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    tasks.dtype() == AITER_DTYPE_i32 && task_count.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 && scales.dtype() == AITER_DTYPE_u8,
                "IQ2R direct routing tensor dtypes are invalid");
    const int64_t routes = expert_ids.numel();
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    const bool glm53_grouped_decode = !drop_nonlocal_routes && !expert_map.has_value() &&
                                      topk == kGlmGroupedDecodeTopK &&
                                      expert_count == kGlmExperts + 1 && routes > 16 &&
                                      routes <= kGlmGroupedDecodeMaxRoutes &&
                                      input.size(1) == 6144 && expert_start == 0 &&
                                      expert_stride == 1 && scales.dim() == 2;
    AITER_CHECK(((routes > 0 && routes <= 16) || glm53_grouped_decode) && topk > 0 &&
                    routes % topk == 0,
                "IQ2R direct routing requires 1..16 complete top-k rows or the "
                "fused GLM-5.3 grouped decode shape");
    AITER_CHECK(groups_per_row <= kMaxQuantThreads,
                "IQ2R direct routing hidden size exceeds one-CTA quantization capacity");
    AITER_CHECK(
        input.dim() == 2 && input.size(0) * topk == routes && input.size(1) % kQuantGroup == 0 &&
            output.dim() == 2 && output.size(0) == routes && output.size(1) == input.size(1) &&
            iq2r_valid_scale_shape(scales, routes, groups_per_row) &&
            sorted_expert_ids.numel() == routes && gather_indices.numel() == routes &&
            scatter_indices.numel() == routes && tasks.dim() == 2 && tasks.size(0) >= routes &&
            tasks.size(1) == kTaskColumns && task_count.dim() == 1 && task_count.size(0) == 1,
        "IQ2R direct routing shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    expert_ids.is_contiguous() && sorted_expert_ids.is_contiguous() &&
                    gather_indices.is_contiguous() && scatter_indices.is_contiguous() &&
                    tasks.is_contiguous() && task_count.is_contiguous() && output.is_contiguous() &&
                    scales.is_contiguous(),
                "IQ2R direct routing tensors must be contiguous");
    AITER_CHECK(expert_count > 0 && expert_count <= kMaxExperts,
                "IQ2R direct routing supports at most 512 experts");
    const int32_t* expert_map_ptr = nullptr;
    int64_t expert_map_count      = 0;
    if(expert_map.has_value())
    {
        AITER_CHECK(expert_map->is_gpu() && expert_map->device_id == device &&
                        expert_map->dtype() == AITER_DTYPE_i32 && expert_map->dim() == 1 &&
                        expert_map->numel() > 0 && expert_map->numel() <= kMaxExperts &&
                        expert_map->is_contiguous(),
                    "IQ2R expert_map must be contiguous int32 [1..512] on the route GPU");
        expert_map_ptr   = static_cast<const int32_t*>(expert_map->data_ptr());
        expert_map_count = expert_map->numel();
    }
    AITER_CHECK(expert_start >= 0 && expert_stride > 0 &&
                    expert_start + (expert_count - 1) * expert_stride < kMaxExperts,
                "IQ2R strided local expert range must fit within 512 global experts");

    const bool tiled_scales  = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((routes + 15) / 16);
    const int quant_threads =
        groups_per_row > kDefaultQuantThreads ? kMaxQuantThreads : kDefaultQuantThreads;
    if(glm53_grouped_decode)
    {
        const char* ballot_setting=std::getenv("IQ2R_GLM53_BALLOT_ROUTER");
        const bool ballot=ballot_setting && std::strcmp(ballot_setting,"1" )==0 && input.size(0)>=4;
        if(ballot)
        {
            static thread_local bool audited[17]={};
            const int tokens=static_cast<int>(input.size(0));
            const char* audit=std::getenv("ATOM_IQ2R_AUDIT");
            if(audit && std::strcmp(audit,"1")==0 && !audited[tokens])
            {
                audited[tokens]=true;
                std::fprintf(stderr,"IQ2R_BALLOT_ROUTER device=%d tokens=%d kernel=iq2r_route_grouped_ballot_quant_glm_kernel\n",input.device_id,tokens);
            }
        hipLaunchKernelGGL(iq2r_route_grouped_ballot_quant_glm_kernel,
                           dim3(static_cast<uint32_t>(input.size(0))),
                           dim3(quant_threads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::bf16_t*>(input.data_ptr()),
                           static_cast<const int32_t*>(expert_ids.data_ptr()),
                           static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
                           static_cast<int32_t*>(gather_indices.data_ptr()),
                           static_cast<int32_t*>(scatter_indices.data_ptr()),
                           static_cast<int32_t*>(tasks.data_ptr()),
                           static_cast<int32_t*>(task_count.data_ptr()),
                           static_cast<opus::fp8_t*>(output.data_ptr()),
                           static_cast<uint8_t*>(scales.data_ptr()),
                           static_cast<int>(input.size(0)),
                           static_cast<int>(input.size(1)),
                           static_cast<int>(input.stride(0)),
                           static_cast<int>(groups_per_row),
                           static_cast<int>(tasks.size(0)),
                           scale_m_blocks,
                           tiled_scales);
        }
        else
        {
        hipLaunchKernelGGL(iq2r_route_grouped_gather_quant_glm_kernel,
                           dim3(static_cast<uint32_t>(input.size(0))),
                           dim3(quant_threads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::bf16_t*>(input.data_ptr()),
                           static_cast<const int32_t*>(expert_ids.data_ptr()),
                           static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
                           static_cast<int32_t*>(gather_indices.data_ptr()),
                           static_cast<int32_t*>(scatter_indices.data_ptr()),
                           static_cast<int32_t*>(tasks.data_ptr()),
                           static_cast<int32_t*>(task_count.data_ptr()),
                           static_cast<opus::fp8_t*>(output.data_ptr()),
                           static_cast<uint8_t*>(scales.data_ptr()),
                           static_cast<int>(input.size(0)),
                           static_cast<int>(input.size(1)),
                           static_cast<int>(input.stride(0)),
                           static_cast<int>(groups_per_row),
                           static_cast<int>(tasks.size(0)),
                           scale_m_blocks,
                           tiled_scales);
        }
    }
    else if(drop_nonlocal_routes)
    {
        hipLaunchKernelGGL(iq2r_route_direct_scatter_quant_kernel,
                           dim3(1),
                           dim3(quant_threads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::bf16_t*>(input.data_ptr()),
                           static_cast<const int32_t*>(expert_ids.data_ptr()),
                           static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
                           static_cast<int32_t*>(gather_indices.data_ptr()),
                           static_cast<int32_t*>(scatter_indices.data_ptr()),
                           static_cast<int32_t*>(tasks.data_ptr()),
                           static_cast<int32_t*>(task_count.data_ptr()),
                           static_cast<opus::fp8_t*>(output.data_ptr()),
                           static_cast<uint8_t*>(scales.data_ptr()),
                           static_cast<int>(routes),
                           static_cast<int>(input.size(1)),
                           static_cast<int>(input.stride(0)),
                           static_cast<int>(groups_per_row),
                           static_cast<int>(topk),
                           static_cast<int>(expert_count),
                           expert_map_ptr,
                           static_cast<int>(expert_map_count),
                           static_cast<int>(expert_start),
                           static_cast<int>(expert_stride),
                           scale_m_blocks,
                           tiled_scales);
    }
    else
    {
        hipLaunchKernelGGL(iq2r_route_direct_gather_quant_kernel,
                           dim3(static_cast<uint32_t>(input.size(0))),
                           dim3(quant_threads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const opus::bf16_t*>(input.data_ptr()),
                           static_cast<const int32_t*>(expert_ids.data_ptr()),
                           static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
                           static_cast<int32_t*>(gather_indices.data_ptr()),
                           static_cast<int32_t*>(scatter_indices.data_ptr()),
                           static_cast<int32_t*>(tasks.data_ptr()),
                           static_cast<int32_t*>(task_count.data_ptr()),
                           static_cast<opus::fp8_t*>(output.data_ptr()),
                           static_cast<uint8_t*>(scales.data_ptr()),
                           static_cast<int>(routes),
                           static_cast<int>(input.size(1)),
                           static_cast<int>(input.stride(0)),
                           static_cast<int>(groups_per_row),
                           static_cast<int>(topk),
                           static_cast<int>(expert_count),
                           expert_map_ptr,
                           static_cast<int>(expert_map_count),
                           static_cast<int>(expert_start),
                           static_cast<int>(expert_stride),
                           scale_m_blocks,
                           tiled_scales);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_topk_direct_gather_quant_out(const aiter_tensor_t& input,
                                             const aiter_tensor_t& router_logits,
                                             aiter_tensor_t& topk_weights,
                                             aiter_tensor_t& topk_ids,
                                             aiter_tensor_t& sorted_expert_ids,
                                             aiter_tensor_t& gather_indices,
                                             aiter_tensor_t& scatter_indices,
                                             aiter_tensor_t& tasks,
                                             aiter_tensor_t& task_count,
                                             aiter_tensor_t& output,
                                             aiter_tensor_t& scales,
                                             bool renormalize,
                                             std::optional<aiter_tensor_t> router_bias,
                                             bool biased_sigmoid,
                                             double routed_scaling_factor,
                                             std::optional<aiter_tensor_t> expert_map,
                                             int64_t expert_count,
                                             int64_t expert_start,
                                             int64_t expert_stride,
                                             bool drop_nonlocal_routes)
{
    AITER_CHECK(input.is_gpu() && router_logits.is_gpu() && topk_weights.is_gpu() &&
                    topk_ids.is_gpu() && sorted_expert_ids.is_gpu() && gather_indices.is_gpu() &&
                    scatter_indices.is_gpu() && tasks.is_gpu() && task_count.is_gpu() &&
                    output.is_gpu() && scales.is_gpu(),
                "fused IQ2R routing requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(router_logits.device_id == device && topk_weights.device_id == device &&
                    topk_ids.device_id == device && sorted_expert_ids.device_id == device &&
                    gather_indices.device_id == device && scatter_indices.device_id == device &&
                    tasks.device_id == device && task_count.device_id == device &&
                    output.device_id == device && scales.device_id == device,
                "fused IQ2R routing tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 &&
                    (router_logits.dtype() == AITER_DTYPE_bf16 ||
                     router_logits.dtype() == AITER_DTYPE_fp32) &&
                    topk_weights.dtype() == AITER_DTYPE_fp32 &&
                    topk_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    tasks.dtype() == AITER_DTYPE_i32 && task_count.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 && scales.dtype() == AITER_DTYPE_u8,
                "fused IQ2R routing tensor dtypes are invalid");
    const int global_experts = biased_sigmoid ? kGlmExperts : kGptOssExperts;
    const int topk           = biased_sigmoid ? kGlmTopK : kGptOssTopK;
    const int max_tokens     = kSmallRoutes / topk;
    AITER_CHECK(input.dim() == 2 && input.size(0) > 0 && input.size(0) <= max_tokens &&
                    input.size(1) % kQuantGroup == 0 && router_logits.dim() == 2 &&
                    router_logits.size(0) == input.size(0) &&
                    router_logits.size(1) == global_experts && topk_weights.dim() == 2 &&
                    topk_weights.size(0) == input.size(0) && topk_weights.size(1) == topk &&
                    topk_ids.dim() == 2 && topk_ids.size(0) == input.size(0) &&
                    topk_ids.size(1) == topk,
                "fused IQ2R routing input shape mismatch");
    const int routes             = static_cast<int>(input.size(0) * topk);
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    AITER_CHECK(sorted_expert_ids.numel() == routes && gather_indices.numel() == routes &&
                    scatter_indices.numel() == routes && tasks.dim() == 2 &&
                    tasks.size(0) >= routes && tasks.size(1) == kTaskColumns &&
                    task_count.dim() == 1 && task_count.size(0) == 1 && output.dim() == 2 &&
                    output.size(0) == routes && output.size(1) == input.size(1) &&
                    iq2r_valid_scale_shape(scales, routes, groups_per_row),
                "fused IQ2R routing output shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    router_logits.stride(1) == 1 && router_logits.stride(0) >= global_experts &&
                    topk_weights.is_contiguous() && topk_ids.is_contiguous() &&
                    sorted_expert_ids.is_contiguous() && gather_indices.is_contiguous() &&
                    scatter_indices.is_contiguous() && tasks.is_contiguous() &&
                    task_count.is_contiguous() && output.is_contiguous() && scales.is_contiguous(),
                "fused IQ2R routing tensors must be contiguous");
    AITER_CHECK(expert_count > 0 && expert_count <= global_experts && expert_start >= 0 &&
                    expert_stride > 0 &&
                    expert_start + (expert_count - 1) * expert_stride < global_experts,
                "fused IQ2R router local expert range is invalid");
    const int32_t* expert_map_ptr = nullptr;
    int64_t expert_map_count      = 0;
    if(expert_map.has_value())
    {
        AITER_CHECK(expert_map->is_gpu() && expert_map->device_id == device &&
                        expert_map->dtype() == AITER_DTYPE_i32 && expert_map->dim() == 1 &&
                        expert_map->numel() >= global_experts && expert_map->is_contiguous(),
                    "fused IQ2R expert_map must cover every global expert");
        expert_map_ptr   = static_cast<const int32_t*>(expert_map->data_ptr());
        expert_map_count = expert_map->numel();
    }
    AITER_CHECK(!biased_sigmoid || renormalize,
                "the fused GLM IQ2R router requires renormalization");
    AITER_CHECK(biased_sigmoid || (routed_scaling_factor == 1.0 && expert_map_ptr == nullptr &&
                                   expert_count == kGptOssExperts && expert_start == 0 &&
                                   expert_stride == 1 && !drop_nonlocal_routes),
                "the fused softmax IQ2R router supports only unsharded GPT-OSS");

    const opus::bf16_t* gpt_router_bias_ptr = nullptr;
    const void* glm_correction_bias_ptr     = nullptr;
    if(router_bias.has_value())
    {
        const auto& bias = router_bias.value();
        AITER_CHECK(bias.is_gpu() && bias.device_id == device && bias.dim() == 1 &&
                        bias.size(0) == global_experts && bias.is_contiguous(),
                    "fused IQ2R router bias must cover every global expert");
        if(biased_sigmoid)
        {
            AITER_CHECK(bias.dtype() == router_logits.dtype(),
                        "GLM correction bias dtype must match router logits");
            glm_correction_bias_ptr = bias.data_ptr();
        }
        else
        {
            AITER_CHECK(bias.dtype() == AITER_DTYPE_bf16, "GPT-OSS router bias must be BF16");
            gpt_router_bias_ptr = static_cast<const opus::bf16_t*>(bias.data_ptr());
        }
    }
    AITER_CHECK(!biased_sigmoid || glm_correction_bias_ptr != nullptr,
                "the fused GLM IQ2R router requires correction bias");

    HipDeviceGuard device_guard(device);
    const bool tiled_scales  = scales.dim() == 4;
    const int scale_m_blocks = (routes + 15) / 16;
    if(biased_sigmoid)
    {
#define IQ2R_LAUNCH_GLM_DIRECT(ROUTER_TYPE)                                           \
    hipLaunchKernelGGL((iq2r_route_glm_topk_direct_gather_quant_kernel<ROUTER_TYPE>), \
                       dim3(1),                                                       \
                       dim3(kGlmExperts),                                             \
                       0,                                                             \
                       getCurrentHIPStream(),                                         \
                       static_cast<const opus::bf16_t*>(input.data_ptr()),            \
                       static_cast<const ROUTER_TYPE*>(router_logits.data_ptr()),     \
                       static_cast<const ROUTER_TYPE*>(glm_correction_bias_ptr),      \
                       static_cast<float*>(topk_weights.data_ptr()),                  \
                       static_cast<int32_t*>(topk_ids.data_ptr()),                    \
                       static_cast<int32_t*>(sorted_expert_ids.data_ptr()),           \
                       static_cast<int32_t*>(gather_indices.data_ptr()),              \
                       static_cast<int32_t*>(scatter_indices.data_ptr()),             \
                       static_cast<int32_t*>(tasks.data_ptr()),                       \
                       static_cast<int32_t*>(task_count.data_ptr()),                  \
                       static_cast<opus::fp8_t*>(output.data_ptr()),                  \
                       static_cast<uint8_t*>(scales.data_ptr()),                      \
                       expert_map_ptr,                                                \
                       static_cast<int>(expert_map_count),                            \
                       static_cast<int>(expert_count),                                \
                       static_cast<int>(expert_start),                                \
                       static_cast<int>(expert_stride),                               \
                       static_cast<int>(input.size(0)),                               \
                       static_cast<int>(input.size(1)),                               \
                       static_cast<int>(input.stride(0)),                             \
                       static_cast<int>(router_logits.stride(0)),                     \
                       static_cast<int>(groups_per_row),                              \
                       static_cast<float>(routed_scaling_factor),                     \
                       scale_m_blocks,                                                \
                       tiled_scales)
        if(router_logits.dtype() == AITER_DTYPE_bf16)
            IQ2R_LAUNCH_GLM_DIRECT(opus::bf16_t);
        else
            IQ2R_LAUNCH_GLM_DIRECT(float);
#undef IQ2R_LAUNCH_GLM_DIRECT
    }
    else
    {
#define IQ2R_LAUNCH_GPT_DIRECT(ROUTER_TYPE)                                       \
    hipLaunchKernelGGL((iq2r_route_topk_direct_gather_quant_kernel<ROUTER_TYPE>), \
                       dim3(static_cast<unsigned int>(input.size(0))),            \
                       dim3(kDirectTopKThreads),                                  \
                       0,                                                         \
                       getCurrentHIPStream(),                                     \
                       static_cast<const opus::bf16_t*>(input.data_ptr()),        \
                       static_cast<const ROUTER_TYPE*>(router_logits.data_ptr()), \
                       gpt_router_bias_ptr,                                       \
                       static_cast<float*>(topk_weights.data_ptr()),              \
                       static_cast<int32_t*>(topk_ids.data_ptr()),                \
                       static_cast<int32_t*>(sorted_expert_ids.data_ptr()),       \
                       static_cast<int32_t*>(gather_indices.data_ptr()),          \
                       static_cast<int32_t*>(scatter_indices.data_ptr()),         \
                       static_cast<int32_t*>(tasks.data_ptr()),                   \
                       static_cast<int32_t*>(task_count.data_ptr()),              \
                       static_cast<opus::fp8_t*>(output.data_ptr()),              \
                       static_cast<uint8_t*>(scales.data_ptr()),                  \
                       static_cast<int>(input.size(0)),                           \
                       static_cast<int>(input.size(1)),                           \
                       static_cast<int>(input.stride(0)),                         \
                       static_cast<int>(router_logits.stride(0)),                 \
                       static_cast<int>(groups_per_row),                          \
                       renormalize,                                               \
                       scale_m_blocks,                                            \
                       tiled_scales)
        if(router_logits.dtype() == AITER_DTYPE_bf16)
            IQ2R_LAUNCH_GPT_DIRECT(opus::bf16_t);
        else
            IQ2R_LAUNCH_GPT_DIRECT(float);
#undef IQ2R_LAUNCH_GPT_DIRECT
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_topk_sort_gather_quant_out(const aiter_tensor_t& input,
                                           const aiter_tensor_t& router_logits,
                                           aiter_tensor_t& topk_weights,
                                           aiter_tensor_t& topk_ids,
                                           aiter_tensor_t& sorted_expert_ids,
                                           aiter_tensor_t& gather_indices,
                                           aiter_tensor_t& scatter_indices,
                                           aiter_tensor_t& tasks,
                                           aiter_tensor_t& task_count,
                                           aiter_tensor_t& output,
                                           aiter_tensor_t& scales,
                                           int64_t task_rows,
                                           bool renormalize,
                                           std::optional<aiter_tensor_t> router_bias,
                                           bool biased_sigmoid,
                                           double routed_scaling_factor,
                                           std::optional<aiter_tensor_t> expert_map,
                                           int64_t expert_count,
                                           int64_t expert_start,
                                           int64_t expert_stride,
                                           bool drop_nonlocal_routes)
{
    AITER_CHECK(input.is_gpu() && router_logits.is_gpu() && topk_weights.is_gpu() &&
                    topk_ids.is_gpu() && sorted_expert_ids.is_gpu() && gather_indices.is_gpu() &&
                    scatter_indices.is_gpu() && tasks.is_gpu() && task_count.is_gpu() &&
                    output.is_gpu() && scales.is_gpu(),
                "fused sorted IQ2R routing requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(router_logits.device_id == device && topk_weights.device_id == device &&
                    topk_ids.device_id == device && sorted_expert_ids.device_id == device &&
                    gather_indices.device_id == device && scatter_indices.device_id == device &&
                    tasks.device_id == device && task_count.device_id == device &&
                    output.device_id == device && scales.device_id == device,
                "fused sorted IQ2R routing tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 &&
                    (router_logits.dtype() == AITER_DTYPE_bf16 ||
                     router_logits.dtype() == AITER_DTYPE_fp32) &&
                    topk_weights.dtype() == AITER_DTYPE_fp32 &&
                    topk_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    tasks.dtype() == AITER_DTYPE_i32 && task_count.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 && scales.dtype() == AITER_DTYPE_u8,
                "fused sorted IQ2R routing tensor dtypes are invalid");
    const int global_experts = biased_sigmoid ? kGlmExperts : kGptOssExperts;
    const int topk           = biased_sigmoid ? kGlmTopK : kGptOssTopK;
    AITER_CHECK(input.dim() == 2 && input.size(0) > 0 && input.size(0) <= kFusedSortTokens &&
                    input.size(1) % kQuantGroup == 0 && router_logits.dim() == 2 &&
                    router_logits.size(0) == input.size(0) &&
                    router_logits.size(1) == global_experts && topk_weights.dim() == 2 &&
                    topk_weights.size(0) == input.size(0) && topk_weights.size(1) == topk &&
                    topk_ids.dim() == 2 && topk_ids.size(0) == input.size(0) &&
                    topk_ids.size(1) == topk,
                "fused sorted IQ2R routing input shape mismatch");
    const int64_t routes         = input.size(0) * topk;
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    AITER_CHECK(task_rows == 16 || task_rows == 32 || task_rows == 64 || task_rows == 128 ||
                    task_rows == 256,
                "IQ2R task_rows must be 16, 32, 64, 128, or 256");
    const int64_t required_capacity =
        (routes + task_rows - 1) / task_rows + std::min<int64_t>(routes, expert_count);
    AITER_CHECK(sorted_expert_ids.numel() == routes && gather_indices.numel() == routes &&
                    scatter_indices.numel() == routes && tasks.dim() == 2 &&
                    tasks.size(0) >= required_capacity && tasks.size(1) == kTaskColumns &&
                    task_count.dim() == 1 && task_count.size(0) == 1 && output.dim() == 2 &&
                    output.size(0) == routes && output.size(1) == input.size(1) &&
                    iq2r_valid_scale_shape(scales, routes, groups_per_row),
                "fused sorted IQ2R routing output shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    router_logits.stride(1) == 1 && router_logits.stride(0) >= global_experts &&
                    topk_weights.is_contiguous() && topk_ids.is_contiguous() &&
                    sorted_expert_ids.is_contiguous() && gather_indices.is_contiguous() &&
                    scatter_indices.is_contiguous() && tasks.is_contiguous() &&
                    task_count.is_contiguous() && output.is_contiguous() && scales.is_contiguous(),
                "fused sorted IQ2R routing tensors must be contiguous");
    AITER_CHECK(expert_count > 0 && expert_count <= global_experts && expert_start >= 0 &&
                    expert_stride > 0 &&
                    expert_start + (expert_count - 1) * expert_stride < global_experts,
                "fused sorted IQ2R router local expert range is invalid");
    const int32_t* expert_map_ptr = nullptr;
    int64_t expert_map_count      = 0;
    if(expert_map.has_value())
    {
        AITER_CHECK(expert_map->is_gpu() && expert_map->device_id == device &&
                        expert_map->dtype() == AITER_DTYPE_i32 && expert_map->dim() == 1 &&
                        expert_map->numel() >= global_experts && expert_map->is_contiguous(),
                    "fused sorted IQ2R expert_map must cover every global expert");
        expert_map_ptr   = static_cast<const int32_t*>(expert_map->data_ptr());
        expert_map_count = expert_map->numel();
    }
    AITER_CHECK(!biased_sigmoid || renormalize,
                "the fused GLM IQ2R router requires renormalization");
    AITER_CHECK(biased_sigmoid || (routed_scaling_factor == 1.0 && expert_map_ptr == nullptr &&
                                   expert_count == kGptOssExperts && expert_start == 0 &&
                                   expert_stride == 1 && !drop_nonlocal_routes),
                "the fused softmax IQ2R router supports only unsharded GPT-OSS");

    const opus::bf16_t* gpt_router_bias_ptr = nullptr;
    const void* glm_correction_bias_ptr     = nullptr;
    if(router_bias.has_value())
    {
        const auto& bias = router_bias.value();
        AITER_CHECK(bias.is_gpu() && bias.device_id == device && bias.dim() == 1 &&
                        bias.size(0) == global_experts && bias.is_contiguous(),
                    "fused sorted IQ2R router bias must cover every global expert");
        if(biased_sigmoid)
        {
            AITER_CHECK(bias.dtype() == router_logits.dtype(),
                        "GLM correction bias dtype must match router logits");
            glm_correction_bias_ptr = bias.data_ptr();
        }
        else
        {
            AITER_CHECK(bias.dtype() == AITER_DTYPE_bf16, "GPT-OSS router bias must be BF16");
            gpt_router_bias_ptr = static_cast<const opus::bf16_t*>(bias.data_ptr());
        }
    }
    AITER_CHECK(!biased_sigmoid || glm_correction_bias_ptr != nullptr,
                "the fused GLM IQ2R router requires correction bias");

    HipDeviceGuard device_guard(device);
    const bool tiled_scales  = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((routes + 15) / 16);
    if(biased_sigmoid)
    {
#define IQ2R_LAUNCH_GLM_SORT(ROUTER_TYPE)                                         \
    hipLaunchKernelGGL((iq2r_route_glm_topk_sort_tasks_kernel<ROUTER_TYPE>),      \
                       dim3(1),                                                   \
                       dim3(kFusedSortThreads),                                   \
                       0,                                                         \
                       getCurrentHIPStream(),                                     \
                       static_cast<const ROUTER_TYPE*>(router_logits.data_ptr()), \
                       static_cast<const ROUTER_TYPE*>(glm_correction_bias_ptr),  \
                       static_cast<float*>(topk_weights.data_ptr()),              \
                       static_cast<int32_t*>(topk_ids.data_ptr()),                \
                       static_cast<int32_t*>(sorted_expert_ids.data_ptr()),       \
                       static_cast<int32_t*>(gather_indices.data_ptr()),          \
                       static_cast<int32_t*>(scatter_indices.data_ptr()),         \
                       static_cast<int32_t*>(tasks.data_ptr()),                   \
                       static_cast<int32_t*>(task_count.data_ptr()),              \
                       expert_map_ptr,                                            \
                       static_cast<int>(expert_map_count),                        \
                       static_cast<int>(expert_count),                            \
                       static_cast<int>(expert_start),                            \
                       static_cast<int>(expert_stride),                           \
                       static_cast<int>(input.size(0)),                           \
                       static_cast<int>(router_logits.stride(0)),                 \
                       static_cast<int>(tasks.size(0)),                           \
                       static_cast<int>(task_rows),                               \
                       static_cast<float>(routed_scaling_factor),                 \
                       drop_nonlocal_routes)
        if(router_logits.dtype() == AITER_DTYPE_bf16)
            IQ2R_LAUNCH_GLM_SORT(opus::bf16_t);
        else
            IQ2R_LAUNCH_GLM_SORT(float);
#undef IQ2R_LAUNCH_GLM_SORT
    }
    else
    {
#define IQ2R_LAUNCH_GPT_SORT(ROUTER_TYPE)                                         \
    hipLaunchKernelGGL((iq2r_route_topk_sort_tasks_kernel<ROUTER_TYPE>),          \
                       dim3(1),                                                   \
                       dim3(kFusedSortThreads),                                   \
                       0,                                                         \
                       getCurrentHIPStream(),                                     \
                       static_cast<const ROUTER_TYPE*>(router_logits.data_ptr()), \
                       gpt_router_bias_ptr,                                       \
                       static_cast<float*>(topk_weights.data_ptr()),              \
                       static_cast<int32_t*>(topk_ids.data_ptr()),                \
                       static_cast<int32_t*>(sorted_expert_ids.data_ptr()),       \
                       static_cast<int32_t*>(gather_indices.data_ptr()),          \
                       static_cast<int32_t*>(scatter_indices.data_ptr()),         \
                       static_cast<int32_t*>(tasks.data_ptr()),                   \
                       static_cast<int32_t*>(task_count.data_ptr()),              \
                       static_cast<int>(input.size(0)),                           \
                       static_cast<int>(router_logits.stride(0)),                 \
                       static_cast<int>(tasks.size(0)),                           \
                       static_cast<int>(task_rows),                               \
                       renormalize)
        if(router_logits.dtype() == AITER_DTYPE_bf16)
            IQ2R_LAUNCH_GPT_SORT(opus::bf16_t);
        else
            IQ2R_LAUNCH_GPT_SORT(float);
#undef IQ2R_LAUNCH_GPT_SORT
    }
    HIP_CALL_LAUNCH(hipGetLastError());

    const int quant_threads =
        groups_per_row > kDefaultQuantThreads ? kMaxQuantThreads : kDefaultQuantThreads;
#define IQ2R_LAUNCH_ROUTER_QUANT(TOPK)                                          \
    hipLaunchKernelGGL((iq2r_route_gather_quant_broadcast_kernel<TOPK>),        \
                       dim3(static_cast<unsigned int>(input.size(0))),          \
                       dim3(quant_threads),                                     \
                       0,                                                       \
                       getCurrentHIPStream(),                                   \
                       static_cast<const opus::bf16_t*>(input.data_ptr()),      \
                       static_cast<const int32_t*>(scatter_indices.data_ptr()), \
                       static_cast<opus::fp8_t*>(output.data_ptr()),            \
                       static_cast<uint8_t*>(scales.data_ptr()),                \
                       static_cast<int>(input.size(0)),                         \
                       static_cast<int>(input.size(1)),                         \
                       static_cast<int>(input.stride(0)),                       \
                       static_cast<int>(groups_per_row),                        \
                       scale_m_blocks,                                          \
                       tiled_scales)
    if(biased_sigmoid)
        IQ2R_LAUNCH_ROUTER_QUANT(kGlmTopK);
    else
        IQ2R_LAUNCH_ROUTER_QUANT(kGptOssTopK);
#undef IQ2R_LAUNCH_ROUTER_QUANT
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_swiglu_out(const aiter_tensor_t& gate_up,
                     aiter_tensor_t& output,
                     double limit,
                     double alpha,
                     double up_offset)
{
    AITER_CHECK(gate_up.is_gpu() && output.is_gpu() && gate_up.device_id == output.device_id,
                "IQ2R SwiGLU tensors must share a GPU");
    AITER_CHECK(gate_up.dtype() == AITER_DTYPE_bf16 && output.dtype() == AITER_DTYPE_bf16,
                "IQ2R SwiGLU tensors must be BF16");
    AITER_CHECK(gate_up.dim() == 2 && output.dim() == 2 && gate_up.size(0) == output.size(0) &&
                    gate_up.size(1) == output.size(1) * 2,
                "IQ2R SwiGLU shape mismatch");
    AITER_CHECK(gate_up.is_contiguous() && output.is_contiguous(),
                "IQ2R SwiGLU tensors must be contiguous");
    const int64_t elements = output.numel();
    hipLaunchKernelGGL(iq2r_swiglu_kernel,
                       dim3(launch_blocks(elements)),
                       dim3(kThreads),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const __hip_bfloat16*>(gate_up.data_ptr()),
                       static_cast<__hip_bfloat16*>(output.data_ptr()),
                       elements,
                       static_cast<int>(output.size(1)),
                       static_cast<float>(limit),
                       static_cast<float>(alpha),
                       static_cast<float>(up_offset));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_swiglu_quant_out(const aiter_tensor_t& gate_up,
                           aiter_tensor_t& output,
                           aiter_tensor_t& scales,
                           std::optional<aiter_tensor_t> activated,
                           double limit,
                           double alpha,
                           double up_offset)
{
    AITER_CHECK(gate_up.is_gpu() && output.is_gpu() && scales.is_gpu(),
                "IQ2R fused SwiGLU/quant requires GPU tensors");
    const int device = gate_up.device_id;
    AITER_CHECK(output.device_id == device && scales.device_id == device,
                "IQ2R fused SwiGLU/quant tensors must share a GPU");
    AITER_CHECK(gate_up.dtype() == AITER_DTYPE_bf16 && output.dtype() == AITER_DTYPE_fp8 &&
                    scales.dtype() == AITER_DTYPE_u8,
                "IQ2R fused SwiGLU/quant dtype mismatch");
    const int64_t groups_per_row = output.size(1) / kQuantGroup;
    AITER_CHECK(gate_up.dim() == 2 && output.dim() == 2 && gate_up.size(0) == output.size(0) &&
                    gate_up.size(1) == output.size(1) * 2 && output.size(1) % kQuantGroup == 0 &&
                    iq2r_valid_scale_shape(scales, output.size(0), groups_per_row),
                "IQ2R fused SwiGLU/quant shape mismatch");
    AITER_CHECK(gate_up.is_contiguous() && output.is_contiguous() && scales.is_contiguous(),
                "IQ2R fused SwiGLU/quant tensors must be contiguous");

    __hip_bfloat16* activated_ptr = nullptr;
    if(activated.has_value())
    {
        AITER_CHECK(activated->is_gpu() && activated->device_id == device &&
                        activated->dtype() == AITER_DTYPE_bf16 && activated->dim() == 2 &&
                        activated->size(0) == output.size(0) &&
                        activated->size(1) == output.size(1) && activated->is_contiguous(),
                    "IQ2R optional activated output must be contiguous BF16 and "
                    "match the quantized output shape");
        activated_ptr = static_cast<__hip_bfloat16*>(activated->data_ptr());
    }

    const bool tiled_scales  = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((output.size(0) + 15) / 16);
    const char* family       = std::getenv("IQ2R_SWIGLU_QUANT_FAMILY");
    const bool use_parallel8 =
        family == nullptr || family[0] == '\0' || std::strcmp(family, "parallel8") == 0;
    AITER_CHECK(use_parallel8 || std::strcmp(family, "group32") == 0,
                "IQ2R_SWIGLU_QUANT_FAMILY must be parallel8 or group32");
    if(use_parallel8)
    {
        const int blocks_per_row =
            (static_cast<int>(output.size(1)) + kQuantThreads * kSwiGLUQuantElementsPerThread - 1) /
            (kQuantThreads * kSwiGLUQuantElementsPerThread);
        hipLaunchKernelGGL(iq2r_swiglu_quant_parallel8_kernel,
                           dim3(static_cast<uint32_t>(output.size(0)) * blocks_per_row),
                           dim3(kQuantThreads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const __hip_bfloat16*>(gate_up.data_ptr()),
                           static_cast<opus::fp8_t*>(output.data_ptr()),
                           static_cast<uint8_t*>(scales.data_ptr()),
                           activated_ptr,
                           static_cast<int>(output.size(0)),
                           static_cast<int>(output.size(1)),
                           static_cast<int>(groups_per_row),
                           blocks_per_row,
                           scale_m_blocks,
                           tiled_scales,
                           static_cast<float>(limit),
                           static_cast<float>(alpha),
                           static_cast<float>(up_offset));
    }
    else
    {
        const int64_t groups = output.size(0) * groups_per_row;
        hipLaunchKernelGGL(iq2r_swiglu_quant_kernel,
                           dim3((groups + kQuantThreads - 1) / kQuantThreads),
                           dim3(kQuantThreads),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const __hip_bfloat16*>(gate_up.data_ptr()),
                           static_cast<opus::fp8_t*>(output.data_ptr()),
                           static_cast<uint8_t*>(scales.data_ptr()),
                           activated_ptr,
                           groups,
                           static_cast<int>(output.size(1)),
                           static_cast<int>(groups_per_row),
                           scale_m_blocks,
                           tiled_scales,
                           static_cast<float>(limit),
                           static_cast<float>(alpha),
                           static_cast<float>(up_offset));
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_swiglu_quant_scatter_out(const aiter_tensor_t& gate_up,
                                   const aiter_tensor_t& scatter_indices,
                                   aiter_tensor_t& output,
                                   aiter_tensor_t& scales,
                                   int64_t topk,
                                   std::optional<aiter_tensor_t> activated,
                                   double limit,
                                   double alpha,
                                   double up_offset)
{
    AITER_CHECK(gate_up.is_gpu() && scatter_indices.is_gpu() && output.is_gpu() && scales.is_gpu(),
                "IQ2R indexed fused SwiGLU/quant requires GPU tensors");
    const int device = gate_up.device_id;
    AITER_CHECK(scatter_indices.device_id == device && output.device_id == device &&
                    scales.device_id == device,
                "IQ2R indexed fused SwiGLU/quant tensors must share a GPU");
    AITER_CHECK(gate_up.dtype() == AITER_DTYPE_bf16 && scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 && scales.dtype() == AITER_DTYPE_u8,
                "IQ2R indexed fused SwiGLU/quant dtype mismatch");
    const int64_t groups_per_row = output.size(1) / kQuantGroup;
    AITER_CHECK(gate_up.dim() == 2 && scatter_indices.dim() == 1 && output.dim() == 2 &&
                    (topk == 4 || topk == 8) && scatter_indices.numel() == gate_up.size(0) &&
                    scatter_indices.numel() % topk == 0 && gate_up.size(0) == output.size(0) &&
                    gate_up.size(1) == output.size(1) * 2 && output.size(1) % kQuantGroup == 0 &&
                    iq2r_valid_scale_shape(scales, output.size(0), groups_per_row),
                "IQ2R indexed fused SwiGLU/quant shape mismatch");
    AITER_CHECK(gate_up.is_contiguous() && scatter_indices.is_contiguous() &&
                    output.is_contiguous() && scales.is_contiguous(),
                "IQ2R indexed fused SwiGLU/quant tensors must be contiguous");

    __hip_bfloat16* activated_ptr = nullptr;
    if(activated.has_value())
    {
        AITER_CHECK(activated->is_gpu() && activated->device_id == device &&
                        activated->dtype() == AITER_DTYPE_bf16 && activated->dim() == 2 &&
                        activated->size(0) == output.size(0) &&
                        activated->size(1) == output.size(1) && activated->is_contiguous(),
                    "IQ2R indexed activated output must match the quantized output");
        activated_ptr = static_cast<__hip_bfloat16*>(activated->data_ptr());
    }

    const bool tiled_scales  = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((output.size(0) + 15) / 16);
    const int blocks_per_row =
        (static_cast<int>(output.size(1)) + kQuantThreads * kSwiGLUQuantElementsPerThread - 1) /
        (kQuantThreads * kSwiGLUQuantElementsPerThread);
    const int tokens = static_cast<int>(scatter_indices.numel() / topk);
#define IQ2R_LAUNCH_SWIGLU_SCATTER(TOPK)                                        \
    hipLaunchKernelGGL((iq2r_swiglu_quant_scatter_kernel<TOPK>),                \
                       dim3(static_cast<uint32_t>(tokens * blocks_per_row)),    \
                       dim3(kQuantThreads),                                     \
                       0,                                                       \
                       getCurrentHIPStream(),                                   \
                       static_cast<const __hip_bfloat16*>(gate_up.data_ptr()),  \
                       static_cast<const int32_t*>(scatter_indices.data_ptr()), \
                       static_cast<opus::fp8_t*>(output.data_ptr()),            \
                       static_cast<uint8_t*>(scales.data_ptr()),                \
                       activated_ptr,                                           \
                       tokens,                                                  \
                       static_cast<int>(output.size(1)),                        \
                       static_cast<int>(groups_per_row),                        \
                       blocks_per_row,                                          \
                       scale_m_blocks,                                          \
                       tiled_scales,                                            \
                       static_cast<float>(limit),                               \
                       static_cast<float>(alpha),                               \
                       static_cast<float>(up_offset))
    if(topk == 4)
        IQ2R_LAUNCH_SWIGLU_SCATTER(4);
    else
        IQ2R_LAUNCH_SWIGLU_SCATTER(8);
#undef IQ2R_LAUNCH_SWIGLU_SCATTER
    HIP_CALL_LAUNCH(hipGetLastError());
}

static void iq2r_route_reduce_indexed_out_impl(const aiter_tensor_t& route_output,
                                               const aiter_tensor_t& route_weights,
                                               const aiter_tensor_t& scatter_indices,
                                               const aiter_tensor_t* shared_output,
                                               aiter_tensor_t& output,
                                               int64_t topk)
{
    AITER_CHECK(route_output.is_gpu() && route_weights.is_gpu() && scatter_indices.is_gpu() &&
                    output.is_gpu(),
                "IQ2R route reduction requires GPU tensors");
    const int device = route_output.device_id;
    AITER_CHECK(route_weights.device_id == device && scatter_indices.device_id == device &&
                    output.device_id == device,
                "IQ2R route reduction tensors must share a GPU");
    AITER_CHECK(
        route_output.dtype() == AITER_DTYPE_bf16 && route_weights.dtype() == AITER_DTYPE_fp32 &&
            scatter_indices.dtype() == AITER_DTYPE_i32 && output.dtype() == AITER_DTYPE_bf16,
        "IQ2R route reduction dtype mismatch");
    AITER_CHECK(
        route_output.dim() == 2 && route_weights.dim() == 2 && scatter_indices.dim() == 1 &&
            output.dim() == 2 && topk > 0 && route_output.size(0) == route_weights.size(0) * topk &&
            scatter_indices.numel() == route_output.size(0) &&
            route_output.size(1) == output.size(1) && output.size(0) == route_weights.size(0),
        "IQ2R route reduction shape mismatch");
    AITER_CHECK(route_output.is_contiguous() && route_weights.is_contiguous() &&
                    scatter_indices.is_contiguous() && output.is_contiguous(),
                "IQ2R route reduction tensors must be contiguous");
    if(shared_output != nullptr)
    {
        AITER_CHECK(shared_output->is_gpu() && shared_output->device_id == device,
                    "IQ2R shared output must be on the route-output GPU");
        AITER_CHECK(shared_output->dtype() == AITER_DTYPE_bf16 && shared_output->dim() == 2 &&
                        shared_output->size(0) == output.size(0) &&
                        shared_output->size(1) == output.size(1) && shared_output->is_contiguous(),
                    "IQ2R shared output must be contiguous BF16 with the output shape");
    }
    const int64_t elements = output.numel();
    const int tokens       = static_cast<int>(output.size(0));
    const int hidden       = static_cast<int>(output.size(1));
    const auto* shared_ptr = shared_output == nullptr
                                 ? nullptr
                                 : static_cast<const __hip_bfloat16*>(shared_output->data_ptr());
    const char* family     = std::getenv("IQ2R_ROUTE_REDUCE_FAMILY");
    const bool supported_glm = hidden == 6144 && (topk == 8 || topk == 9);
    const bool automatic_glm =
        (family == nullptr || family[0] == '\0') && supported_glm;
    if(automatic_glm ||
       (family != nullptr && family[0] != '\0' && std::strcmp(family, "generic") != 0))
    {
        AITER_CHECK(supported_glm,
                    "specialized IQ2R route reduction requires hidden=6144 and topk=8 or 9");
#define IQ2R_LAUNCH_GLM_REDUCE(ITEMS, TOPK, ADD_SHARED)                              \
    hipLaunchKernelGGL((iq2r_route_reduce_glm6144_kernel<ITEMS, TOPK, ADD_SHARED>),  \
                       dim3(static_cast<uint32_t>(tokens * 24)),                    \
                       dim3(256 / ITEMS),                                           \
                       0,                                                           \
                       getCurrentHIPStream(),                                       \
                       static_cast<const __hip_bfloat16*>(route_output.data_ptr()), \
                       static_cast<const float*>(route_weights.data_ptr()),         \
                       static_cast<const int32_t*>(scatter_indices.data_ptr()),     \
                       shared_ptr,                                                  \
                       static_cast<__hip_bfloat16*>(output.data_ptr()),             \
                       tokens)
#define IQ2R_DISPATCH_GLM_REDUCE(ITEMS, ADD_SHARED)       \
    do                                                     \
    {                                                      \
        if(topk == 8)                                      \
            IQ2R_LAUNCH_GLM_REDUCE(ITEMS, 8, ADD_SHARED); \
        else                                               \
            IQ2R_LAUNCH_GLM_REDUCE(ITEMS, 9, ADD_SHARED); \
    } while(false)
        if(!automatic_glm && std::strcmp(family, "glm1") == 0)
        {
            if(shared_output != nullptr)
                IQ2R_DISPATCH_GLM_REDUCE(1, true);
            else
                IQ2R_DISPATCH_GLM_REDUCE(1, false);
        }
        else if(!automatic_glm && std::strcmp(family, "glm2") == 0)
        {
            if(shared_output != nullptr)
                IQ2R_DISPATCH_GLM_REDUCE(2, true);
            else
                IQ2R_DISPATCH_GLM_REDUCE(2, false);
        }
        else if(automatic_glm || std::strcmp(family, "glm4") == 0)
        {
            if(shared_output != nullptr)
                IQ2R_DISPATCH_GLM_REDUCE(4, true);
            else
                IQ2R_DISPATCH_GLM_REDUCE(4, false);
        }
        else
            AITER_CHECK(false, "IQ2R route reduction family must be generic, glm1, glm2, or glm4");
#undef IQ2R_DISPATCH_GLM_REDUCE
#undef IQ2R_LAUNCH_GLM_REDUCE
        HIP_CALL_LAUNCH(hipGetLastError());
        return;
    }
#define IQ2R_LAUNCH_GENERIC_REDUCE(ADD_SHARED)                                      \
    hipLaunchKernelGGL((iq2r_route_reduce_indexed_kernel<ADD_SHARED>),              \
                       dim3(launch_blocks(elements)),                               \
                       dim3(kThreads),                                              \
                       0,                                                           \
                       getCurrentHIPStream(),                                       \
                       static_cast<const __hip_bfloat16*>(route_output.data_ptr()), \
                       static_cast<const float*>(route_weights.data_ptr()),         \
                       static_cast<const int32_t*>(scatter_indices.data_ptr()),     \
                       shared_ptr,                                                  \
                       static_cast<__hip_bfloat16*>(output.data_ptr()),             \
                       elements,                                                    \
                       hidden,                                                      \
                       static_cast<int>(topk))
    if(shared_output != nullptr)
        IQ2R_LAUNCH_GENERIC_REDUCE(true);
    else
        IQ2R_LAUNCH_GENERIC_REDUCE(false);
#undef IQ2R_LAUNCH_GENERIC_REDUCE
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_reduce_indexed_out(const aiter_tensor_t& route_output,
                                   const aiter_tensor_t& route_weights,
                                   const aiter_tensor_t& scatter_indices,
                                   aiter_tensor_t& output,
                                   int64_t topk)
{
    iq2r_route_reduce_indexed_out_impl(
        route_output, route_weights, scatter_indices, nullptr, output, topk);
}

void iq2r_route_reduce_add_indexed_out(const aiter_tensor_t& route_output,
                                       const aiter_tensor_t& route_weights,
                                       const aiter_tensor_t& scatter_indices,
                                       const aiter_tensor_t& shared_output,
                                       aiter_tensor_t& output,
                                       int64_t topk)
{
    iq2r_route_reduce_indexed_out_impl(
        route_output, route_weights, scatter_indices, &shared_output, output, topk);
}

void iq2r_route_reduce_add_rmsnorm_indexed_out(const aiter_tensor_t& route_output,
                                               const aiter_tensor_t& route_weights,
                                               const aiter_tensor_t& scatter_indices,
                                               const aiter_tensor_t& residual,
                                               const aiter_tensor_t& norm_weight,
                                               aiter_tensor_t& output,
                                               aiter_tensor_t& residual_out,
                                               int64_t topk,
                                               double epsilon,
                                               int64_t block_size)
{
    AITER_CHECK(route_output.is_gpu() && route_weights.is_gpu() && scatter_indices.is_gpu() &&
                    residual.is_gpu() && norm_weight.is_gpu() && output.is_gpu() &&
                    residual_out.is_gpu(),
                "IQ2R fused route reduction/RMSNorm requires GPU tensors");
    const int device = route_output.device_id;
    AITER_CHECK(route_weights.device_id == device && scatter_indices.device_id == device &&
                    residual.device_id == device && norm_weight.device_id == device &&
                    output.device_id == device && residual_out.device_id == device,
                "IQ2R fused route reduction/RMSNorm tensors must share a GPU");
    AITER_CHECK(
        route_output.dtype() == AITER_DTYPE_bf16 && route_weights.dtype() == AITER_DTYPE_fp32 &&
            scatter_indices.dtype() == AITER_DTYPE_i32 && residual.dtype() == AITER_DTYPE_bf16 &&
            norm_weight.dtype() == AITER_DTYPE_bf16 && output.dtype() == AITER_DTYPE_bf16 &&
            residual_out.dtype() == AITER_DTYPE_bf16,
        "IQ2R fused route reduction/RMSNorm dtype mismatch");
    AITER_CHECK(
        route_output.dim() == 2 && route_weights.dim() == 2 && scatter_indices.dim() == 1 &&
            residual.dim() == 2 && norm_weight.dim() == 1 && output.dim() == 2 &&
            residual_out.dim() == 2 && topk == kGptOssTopK &&
            route_output.size(0) == route_weights.size(0) * topk &&
            scatter_indices.numel() == route_output.size(0) &&
            route_output.size(1) == residual.size(1) && residual.size(0) == route_weights.size(0) &&
            output.size(0) == residual.size(0) && output.size(1) == residual.size(1) &&
            residual_out.size(0) == residual.size(0) && residual_out.size(1) == residual.size(1) &&
            norm_weight.numel() == residual.size(1),
        "IQ2R fused route reduction/RMSNorm shape mismatch");
    AITER_CHECK(route_output.size(1) == 2880,
                "IQ2R fused route reduction/RMSNorm currently requires GPT-OSS hidden=2880");
    AITER_CHECK(route_output.is_contiguous() && route_weights.is_contiguous() &&
                    scatter_indices.is_contiguous() && residual.is_contiguous() &&
                    norm_weight.is_contiguous() && output.is_contiguous() &&
                    residual_out.is_contiguous(),
                "IQ2R fused route reduction/RMSNorm tensors must be contiguous");

    const int tokens      = static_cast<int>(route_weights.size(0));
    const int hidden      = static_cast<int>(route_output.size(1));
    const float epsilon_f = static_cast<float>(epsilon);
    if(block_size == 256)
        hipLaunchKernelGGL((iq2r_route_reduce_add_rmsnorm_indexed_kernel<256, 12>),
                           dim3(tokens),
                           dim3(256),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const __hip_bfloat16*>(route_output.data_ptr()),
                           static_cast<const float*>(route_weights.data_ptr()),
                           static_cast<const int32_t*>(scatter_indices.data_ptr()),
                           static_cast<const __hip_bfloat16*>(residual.data_ptr()),
                           static_cast<const __hip_bfloat16*>(norm_weight.data_ptr()),
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           static_cast<__hip_bfloat16*>(residual_out.data_ptr()),
                           hidden,
                           static_cast<int>(topk),
                           epsilon_f);
    else if(block_size == 512)
        hipLaunchKernelGGL((iq2r_route_reduce_add_rmsnorm_indexed_kernel<512, 6>),
                           dim3(tokens),
                           dim3(512),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const __hip_bfloat16*>(route_output.data_ptr()),
                           static_cast<const float*>(route_weights.data_ptr()),
                           static_cast<const int32_t*>(scatter_indices.data_ptr()),
                           static_cast<const __hip_bfloat16*>(residual.data_ptr()),
                           static_cast<const __hip_bfloat16*>(norm_weight.data_ptr()),
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           static_cast<__hip_bfloat16*>(residual_out.data_ptr()),
                           hidden,
                           static_cast<int>(topk),
                           epsilon_f);
    else
    {
        AITER_CHECK(block_size == 1024,
                    "IQ2R fused route reduction/RMSNorm block_size must be 256, 512, or 1024");
        hipLaunchKernelGGL((iq2r_route_reduce_add_rmsnorm_indexed_kernel<1024, 3>),
                           dim3(tokens),
                           dim3(1024),
                           0,
                           getCurrentHIPStream(),
                           static_cast<const __hip_bfloat16*>(route_output.data_ptr()),
                           static_cast<const float*>(route_weights.data_ptr()),
                           static_cast<const int32_t*>(scatter_indices.data_ptr()),
                           static_cast<const __hip_bfloat16*>(residual.data_ptr()),
                           static_cast<const __hip_bfloat16*>(norm_weight.data_ptr()),
                           static_cast<__hip_bfloat16*>(output.data_ptr()),
                           static_cast<__hip_bfloat16*>(residual_out.data_ptr()),
                           hidden,
                           static_cast<int>(topk),
                           epsilon_f);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter
