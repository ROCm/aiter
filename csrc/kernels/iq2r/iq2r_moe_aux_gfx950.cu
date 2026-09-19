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
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <type_traits>

namespace aiter {
namespace {

constexpr int kThreads       = 256;
constexpr int kQuantThreads  = 64;
constexpr int kSwiGLUQuantElementsPerThread = 8;
constexpr int kTaskColumns   = 3;
constexpr int kMaxExperts    = 128;
constexpr int kMaxRoutes     = 65536;
constexpr int kSmallRoutes   = 16;
constexpr int kQuantGroup    = 32;
constexpr int kGptOssTopK    = 4;
constexpr int kFusedSortTokens = 16;
constexpr int kFusedSortRoutes = kFusedSortTokens * kGptOssTopK;
constexpr int kFusedSortThreads = 512;
constexpr float kSwigluAlpha = 1.702f;
constexpr float kSwigluLimit = 7.0f;
constexpr float kUpOffset    = 1.0f;

template <typename RouterType>
__device__ __forceinline__ float iq2r_router_value(
    const RouterType* __restrict__ router_logits,
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

bool iq2r_valid_scale_shape(
    const aiter_tensor_t& scales, int64_t rows, int64_t groups_per_row)
{
    return (scales.dim() == 2 && scales.size(0) == rows &&
            scales.size(1) == groups_per_row) ||
           (scales.dim() == 4 && scales.size(0) == (groups_per_row + 3) / 4 &&
            scales.size(1) >= (rows + 15) / 16 && scales.size(2) == 4 &&
            scales.size(3) == 16);
}

__device__ __forceinline__ int64_t iq2r_scale_offset(
    int row,
    int group,
    int groups_per_row,
    int m_blocks,
    bool tiled_scales)
{
    if(tiled_scales)
        return (static_cast<int64_t>(group / 4) * m_blocks + row / 16) * 64 +
               (group % 4) * 16 + row % 16;
    return static_cast<int64_t>(row) * groups_per_row + group;
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
    int routes,
    int expert_count,
    int task_capacity,
    int task_rows)
{
    __shared__ int counts[kMaxExperts + 1];
    __shared__ int offsets[kMaxExperts + 2];
    __shared__ int route_experts[kSmallRoutes];

    for(int bucket = threadIdx.x; bucket <= expert_count; bucket += blockDim.x)
        counts[bucket] = 0;
    __syncthreads();

    if(threadIdx.x < routes)
    {
        const int expert = expert_ids[threadIdx.x];
        const int bucket = expert >= 0 && expert < expert_count ? expert
                                                                : expert_count;
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
        const int bucket = expert >= 0 && expert < expert_count ? expert
                                                                : expert_count;
        int local_rank = 0;
        for(int previous = 0; previous < threadIdx.x; ++previous)
        {
            const int previous_expert = route_experts[previous];
            const int previous_bucket =
                previous_expert >= 0 && previous_expert < expert_count
                    ? previous_expert
                    : expert_count;
            local_rank += previous_bucket == bucket;
        }
        const int sorted_route = offsets[bucket] + local_rank;
        sorted_expert_ids[sorted_route] = expert;
        gather_indices[sorted_route] = threadIdx.x;
        scatter_indices[threadIdx.x] = sorted_route;
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        int task = 0;
        int sorted_begin = 0;
        for(int expert = 0; expert < expert_count; ++expert)
        {
            const int count = counts[expert];
            for(int local = 0; local < count; local += task_rows)
            {
                if(task < task_capacity)
                {
                    tasks[task * kTaskColumns] = sorted_begin + local;
                    tasks[task * kTaskColumns + 1] = min(task_rows, count - local);
                    tasks[task * kTaskColumns + 2] = expert;
                }
                ++task;
            }
            sorted_begin += count;
        }
        const int invalid_count = counts[expert_count];
        for(int local = 0; local < invalid_count; local += task_rows)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns] = sorted_begin + local;
                tasks[task * kTaskColumns + 1] = min(task_rows, invalid_count - local);
                tasks[task * kTaskColumns + 2] = -1;
            }
            ++task;
        }
        task_count[0] = task <= task_capacity ? task : -1;
    }
}

// GPT-OSS has only 128 experts and at most 65536 routed rows in the production
// prefill contract. A single CTA builds an expert-grouped permutation and homogeneous
// tasks without global atomics or temporary allocations. Row order within an
// expert is deliberately unspecified: every routed row is computed
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
    int routes,
    int expert_count,
    int task_capacity,
    int task_rows)
{
    __shared__ int counts[kMaxExperts + 1];
    __shared__ int offsets[kMaxExperts + 2];
    __shared__ int cursors[kMaxExperts + 1];

    for(int bucket = threadIdx.x; bucket <= expert_count; bucket += blockDim.x)
        counts[bucket] = 0;
    __syncthreads();

    for(int route = threadIdx.x; route < routes; route += blockDim.x)
    {
        const int expert = expert_ids[route];
        const int bucket = expert >= 0 && expert < expert_count ? expert
                                                                : expert_count;
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
        const int expert = expert_ids[route];
        const int bucket = expert >= 0 && expert < expert_count ? expert
                                                                : expert_count;
        const int sorted_route = atomicAdd(cursors + bucket, 1);
        sorted_expert_ids[sorted_route] = expert;
        gather_indices[sorted_route] = route;
        scatter_indices[route] = sorted_route;
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
        int task = 0;
        int sorted_begin = 0;
        for(int expert = 0; expert < expert_count; ++expert)
        {
            const int count = counts[expert];
            for(int local = 0; local < count; local += task_rows)
            {
                if(task < task_capacity)
                {
                    tasks[task * kTaskColumns] = sorted_begin + local;
                    tasks[task * kTaskColumns + 1] = min(task_rows, count - local);
                    tasks[task * kTaskColumns + 2] = expert;
                }
                ++task;
            }
            sorted_begin += count;
        }

        // Preserve a deterministic permutation even for malformed route IDs.
        // The corresponding task is marked expert=-1, so the GEMM safely skips
        // it. Production routing is required to supply IDs in [0,E).
        const int invalid_count = counts[expert_count];
        for(int local = 0; local < invalid_count; local += task_rows)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns] = sorted_begin + local;
                tasks[task * kTaskColumns + 1] = min(task_rows, invalid_count - local);
                tasks[task * kTaskColumns + 2] = -1;
            }
            ++task;
        }
        task_count[0] = task <= task_capacity ? task : -1;
    }
}

__global__ void iq2r_route_gather_indexed_kernel(
    const __hip_bfloat16* __restrict__ input,
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
        const int column = static_cast<int>(index % hidden);
        const int source_route = gather_indices[sorted_route];
        const int token = source_route / topk;
        output[index] =
            input[static_cast<int64_t>(token) * input_stride + column];
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
    const int64_t group_id =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(group_id >= groups)
        return;

    const int route = static_cast<int>(group_id / groups_per_row);
    const int group = static_cast<int>(group_id % groups_per_row);
    const int source_route = gather_indices[route];
    const int token = source_route / topk;
    const int64_t input_offset =
        static_cast<int64_t>(token) * input_stride + group * kQuantGroup;
    const int64_t output_offset =
        static_cast<int64_t>(route) * hidden + group * kQuantGroup;

    using input_vector = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector = opus::vector_t<opus::fp8_t, kQuantGroup>;
    const input_vector values =
        *reinterpret_cast<const input_vector*>(input + input_offset);

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        abs_max = fmaxf(abs_max, fabsf(static_cast<float>(values[element])));

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(
            abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] =
            opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

    *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
    scales[iq2r_scale_offset(
        route, group, groups_per_row, scale_m_blocks, tiled_scales)] =
        block_scale.byte;
}

// Decode-specialized path for at most 16 routed rows. Production GPT-OSS c1-c4
// routes are overwhelmingly one row per expert, so sorting them spends more
// time initializing the general counting-sort scratch than it can recover via
// expert reuse. Build one-row tasks in original route order and quantize each
// source token once before broadcasting the packed row to all of its top-k
// routes. The previous route-major implementation repeated the BF16 loads,
// absolute-max reduction, scale conversion, and FP8 conversion top-k times.
constexpr int kDirectQuantThreads = 128;

__global__ __launch_bounds__(kDirectQuantThreads)
void iq2r_route_direct_gather_quant_kernel(
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
    int scale_m_blocks,
    bool tiled_scales)
{
    const int token = static_cast<int>(blockIdx.x);
    const int group = static_cast<int>(threadIdx.x);
    if(token >= routes / topk || group >= groups_per_row)
        return;

    if(group < topk)
    {
        const int route = token * topk + group;
        const int expert = expert_ids[route];
        const int valid_expert = expert >= 0 && expert < expert_count ? expert : -1;
        sorted_expert_ids[route] = expert;
        gather_indices[route] = route;
        scatter_indices[route] = route;
        tasks[route * kTaskColumns] = route;
        tasks[route * kTaskColumns + 1] = 1;
        tasks[route * kTaskColumns + 2] = valid_expert;
        if(token == 0 && group == 0)
            task_count[0] = routes;
    }

    const int column_begin = group * kQuantGroup;
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
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(
            abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] =
            opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

    for(int route_in_token = 0; route_in_token < topk; ++route_in_token)
    {
        const int route = token * topk + route_in_token;
        const int64_t output_offset =
            static_cast<int64_t>(route) * hidden + column_begin;
        *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
        scales[iq2r_scale_offset(
            route, group, groups_per_row, scale_m_blocks, tiled_scales)] =
            block_scale.byte;
    }
}

// GPT-OSS decode front end for one to four tokens.  A single CTA per token
// computes exact top-4 softmax routing, writes the one-row direct GEMM tasks,
// and quantizes the hidden row once before broadcasting it to the four routes.
// This removes the standalone top-k launch from the launch-bound decode path.
template <typename RouterType>
__global__ __launch_bounds__(kDirectQuantThreads)
void iq2r_route_topk_direct_gather_quant_kernel(
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
    constexpr int kTopK = 4;
    constexpr int kExperts = 128;
    using kvp = KeyValuePair<int, float>;

    const int token = static_cast<int>(blockIdx.x);
    const int lane = static_cast<int>(threadIdx.x);
    if(token >= tokens)
        return;

    float candidate = iq2r_router_value(
        router_logits,
        router_bias,
        static_cast<int64_t>(token) * router_stride + lane,
        lane);
    kvp local = {lane, candidate};
    __shared__ int selected_ids[kTopK];
    __shared__ float selected_numerators[kTopK];
    __shared__ float first_max;

#pragma unroll
    for(int rank = 0; rank < kTopK; ++rank)
    {
        const kvp selected =
            block_reduce<kvp, ArgMax, kDirectQuantThreads, true>(local, ArgMax());
        if(lane == 0)
        {
            if(rank == 0)
                first_max = selected.value;
            selected_ids[rank] = selected.key;
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
        full_row_sum = block_reduce<float, Sum, kDirectQuantThreads, true>(
            remainder, Sum());
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
            const int route = token * kTopK + rank;
            const int expert = selected_ids[rank];
            topk_ids[route] = expert;
            topk_weights[route] = selected_numerators[rank] * inverse;
            sorted_expert_ids[route] = expert;
            gather_indices[route] = route;
            scatter_indices[route] = route;
            tasks[route * kTaskColumns] = route;
            tasks[route * kTaskColumns + 1] = 1;
            tasks[route * kTaskColumns + 2] = expert;
        }
        if(token == 0)
            task_count[0] = tokens * kTopK;
    }

    if(lane >= groups_per_row)
        return;

    const int column_begin = lane * kQuantGroup;
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
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(
            abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] =
            opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

#pragma unroll
    for(int rank = 0; rank < kTopK; ++rank)
    {
        const int route = token * kTopK + rank;
        const int64_t output_offset =
            static_cast<int64_t>(route) * hidden + column_begin;
        *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
        scales[iq2r_scale_offset(
            route, lane, groups_per_row, scale_m_blocks, tiled_scales)] =
            block_scale.byte;
    }
}

// GPT-OSS decode front end for five to sixteen tokens. Eight lanes cooperate
// on each router row, then the full CTA builds one stable expert-grouped route
// permutation and homogeneous GEMM tasks. One CTA is intentional: all routing
// metadata remains in shared memory and requires no temporary global scratch.
template <typename RouterType>
__global__ __launch_bounds__(kFusedSortThreads)
void iq2r_route_topk_sort_tasks_kernel(
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
    constexpr int kRouterLanes = 32;
    constexpr int kExpertsPerLane = kMaxExperts / kRouterLanes;
    using kvp = KeyValuePair<int, float>;

    __shared__ int counts[kMaxExperts];
    __shared__ int offsets[kMaxExperts + 1];
    __shared__ int cursors[kMaxExperts];
    __shared__ int wave_sums[2];
    __shared__ int route_experts[kFusedSortRoutes];

    const int thread = static_cast<int>(threadIdx.x);
    const int router_thread_count = tokens * kRouterLanes;
    if(thread < router_thread_count)
    {
        const int token = thread / kRouterLanes;
        const int lane = thread % kRouterLanes;
        int selected_ids[kGptOssTopK];
        float selected_values[kGptOssTopK];

#pragma unroll
        for(int rank = 0; rank < kGptOssTopK; ++rank)
        {
            kvp local = {INT_MAX, -INFINITY};
#pragma unroll
            for(int item = 0; item < kExpertsPerLane; ++item)
            {
                const int expert = lane + item * kRouterLanes;
                bool selected_before = false;
#pragma unroll
                for(int prior = 0; prior < rank; ++prior)
                    selected_before |= selected_ids[prior] == expert;
                const float value =
                    selected_before
                        ? -INFINITY
                        : iq2r_router_value(
                              router_logits,
                              router_bias,
                              static_cast<int64_t>(token) * router_stride + expert,
                              expert);
                local = ArgMax()(local, kvp{expert, value});
            }
            const kvp selected =
                multithread_reduce<kvp, ArgMax>(local, ArgMax(), kRouterLanes);
            selected_ids[rank] = selected.key;
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
                local_sum += expf(iq2r_router_value(
                                      router_logits,
                                      router_bias,
                                      static_cast<int64_t>(token) * router_stride +
                                          expert,
                                      expert) -
                                  selected_values[0]);
            }
            denominator = multithread_reduce<float, Sum>(
                local_sum, Sum(), kRouterLanes);
        }

        if(lane == 0)
        {
            const float inverse = denominator != 0.0f ? 1.0f / denominator : 0.0f;
#pragma unroll
            for(int rank = 0; rank < kGptOssTopK; ++rank)
            {
                const int route = token * kGptOssTopK + rank;
                const int expert = selected_ids[rank];
                topk_ids[route] = expert;
                topk_weights[route] =
                    expf(selected_values[rank] - selected_values[0]) * inverse;
                route_experts[route] = expert;
            }
        }
    }
    __syncthreads();

    for(int expert = thread; expert < kMaxExperts; expert += kFusedSortThreads)
        counts[expert] = 0;
    __syncthreads();

    const int routes = tokens * kGptOssTopK;
    if(thread < routes)
        atomicAdd(counts + route_experts[thread], 1);
    __syncthreads();

    int count_prefix = thread < kMaxExperts ? counts[thread] : 0;
#pragma unroll
    for(int delta = 1; delta < 64; delta <<= 1)
    {
        const int previous = __shfl_up(count_prefix, delta, 64);
        if((thread & 63) >= delta)
            count_prefix += previous;
    }
    if((thread & 63) == 63 && thread < kMaxExperts)
        wave_sums[thread >> 6] = count_prefix;
    __syncthreads();
    if(thread >= 64 && thread < kMaxExperts)
        count_prefix += wave_sums[0];
    if(thread < kMaxExperts)
    {
        offsets[thread] = count_prefix - counts[thread];
        if(thread == kMaxExperts - 1)
            offsets[kMaxExperts] = count_prefix;
    }
    __syncthreads();

    if(thread < kMaxExperts)
        cursors[thread] = offsets[thread];
    __syncthreads();

    if(thread < routes)
    {
        const int expert = route_experts[thread];
        const int sorted_route = atomicAdd(cursors + expert, 1);
        sorted_expert_ids[sorted_route] = expert;
        gather_indices[sorted_route] = thread;
        scatter_indices[thread] = sorted_route;
    }
    __syncthreads();

    int task_prefix =
        thread < kMaxExperts ? (counts[thread] + task_rows - 1) / task_rows : 0;
#pragma unroll
    for(int delta = 1; delta < 64; delta <<= 1)
    {
        const int previous = __shfl_up(task_prefix, delta, 64);
        if((thread & 63) >= delta)
            task_prefix += previous;
    }
    if((thread & 63) == 63 && thread < kMaxExperts)
        wave_sums[thread >> 6] = task_prefix;
    __syncthreads();
    if(thread >= 64 && thread < kMaxExperts)
        task_prefix += wave_sums[0];
    if(thread < kMaxExperts)
    {
        const int count = counts[thread];
        const int task_begin =
            task_prefix - (count + task_rows - 1) / task_rows;
        int task = task_begin;
        for(int local = 0; local < count; local += task_rows, ++task)
        {
            if(task < task_capacity)
            {
                tasks[task * kTaskColumns] = offsets[thread] + local;
                tasks[task * kTaskColumns + 1] = min(task_rows, count - local);
                tasks[task * kTaskColumns + 2] = thread;
            }
        }
        if(thread == kMaxExperts - 1)
            task_count[0] = task_prefix <= task_capacity ? task_prefix : -1;
    }
}

// Quantize every source token once, then broadcast its packed row to the four
// expert-grouped destinations. A CTA per token restores enough parallelism for
// the 2880-column GPT-OSS hidden state while retaining the 4x reduction in BF16
// reads, max reductions, and FP8 conversions.
__global__ __launch_bounds__(kDirectQuantThreads)
void iq2r_route_gather_quant_broadcast_kernel(
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

    const int column_begin = group * kQuantGroup;
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
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(
            abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] =
            opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

#pragma unroll
    for(int rank = 0; rank < kGptOssTopK; ++rank)
    {
        const int route = token * kGptOssTopK + rank;
        const int sorted_route = scatter_indices[route];
        const int64_t output_offset =
            static_cast<int64_t>(sorted_route) * hidden + column_begin;
        *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
        scales[iq2r_scale_offset(
            sorted_route,
            group,
            groups_per_row,
            scale_m_blocks,
            tiled_scales)] =
            block_scale.byte;
    }
}

__global__ void iq2r_swiglu_kernel(const __hip_bfloat16* __restrict__ gate_up,
                                    __hip_bfloat16* __restrict__ output,
                                    int64_t elements,
                                    int intermediate)
{
    for(int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < elements;
        index += static_cast<int64_t>(gridDim.x) * blockDim.x)
    {
        const int64_t row = index / intermediate;
        const int column = static_cast<int>(index % intermediate);
        const int64_t gate_up_offset = row * (2 * intermediate);
        const float unclamped_gate =
            __bfloat162float(gate_up[gate_up_offset + 2 * column]);
        const float unclamped_up =
            __bfloat162float(gate_up[gate_up_offset + 2 * column + 1]);
        const float gate = unclamped_gate > kSwigluLimit
                               ? kSwigluLimit
                               : unclamped_gate;
        const float up = unclamped_up < -kSwigluLimit
                             ? -kSwigluLimit
                             : (unclamped_up > kSwigluLimit
                                    ? kSwigluLimit
                                    : unclamped_up);
        const float swish = gate / (1.0f + __expf(-kSwigluAlpha * gate));
        output[index] = __float2bfloat16(swish * (up + kUpOffset));
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
    bool tiled_scales)
{
    const int64_t group_id =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(group_id >= groups)
        return;

    const int64_t row = group_id / groups_per_row;
    const int group = static_cast<int>(group_id % groups_per_row);
    const int column_begin = group * kQuantGroup;
    const int64_t gate_up_offset = row * (2 * intermediate);
    const int64_t output_offset = row * intermediate + column_begin;
    using activated_vector = opus::vector_t<opus::bf16_t, kQuantGroup>;
    using output_vector = opus::vector_t<opus::fp8_t, kQuantGroup>;
    activated_vector values;

    float abs_max = 1.0e-10f;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
    {
        const int column = column_begin + element;
        const float unclamped_gate =
            __bfloat162float(gate_up[gate_up_offset + 2 * column]);
        const float unclamped_up =
            __bfloat162float(gate_up[gate_up_offset + 2 * column + 1]);
        const float gate =
            unclamped_gate > kSwigluLimit ? kSwigluLimit : unclamped_gate;
        const float up = unclamped_up < -kSwigluLimit
                             ? -kSwigluLimit
                             : (unclamped_up > kSwigluLimit ? kSwigluLimit
                                                            : unclamped_up);
        const float swish = gate / (1.0f + __expf(-kSwigluAlpha * gate));
        const __hip_bfloat16 rounded =
            __float2bfloat16(swish * (up + kUpOffset));
        values[element] = __builtin_bit_cast(opus::bf16_t, rounded);
        abs_max = fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
    }

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(
            abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kQuantGroup; ++element)
        quantized[element] =
            opus::fp32_to_fp8(static_cast<float>(values[element]) * inverse_scale);

    *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
    scales[iq2r_scale_offset(
        static_cast<int>(row),
        group,
        groups_per_row,
        scale_m_blocks,
        tiled_scales)] =
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
__global__ __launch_bounds__(kQuantThreads)
void iq2r_swiglu_quant_parallel8_kernel(
    const __hip_bfloat16* __restrict__ gate_up,
    opus::fp8_t* __restrict__ output,
    uint8_t* __restrict__ scales,
    __hip_bfloat16* __restrict__ activated,
    int rows,
    int intermediate,
    int groups_per_row,
    int blocks_per_row,
    int scale_m_blocks,
    bool tiled_scales)
{
    const int row = static_cast<int>(blockIdx.x) / blocks_per_row;
    const int block_in_row = static_cast<int>(blockIdx.x) % blocks_per_row;
    const int column_begin =
        (block_in_row * blockDim.x + static_cast<int>(threadIdx.x)) *
        kSwiGLUQuantElementsPerThread;
    const bool active = row < rows && column_begin < intermediate;
    const int64_t gate_up_offset = static_cast<int64_t>(row) * 2 * intermediate;
    const int64_t output_offset =
        static_cast<int64_t>(row) * intermediate + column_begin;
    using activated_vector =
        opus::vector_t<opus::bf16_t, kSwiGLUQuantElementsPerThread>;
    using output_vector =
        opus::vector_t<opus::fp8_t, kSwiGLUQuantElementsPerThread>;
    activated_vector values;

    float abs_max = 1.0e-10f;
    if(active)
    {
#pragma unroll
        for(int element = 0; element < kSwiGLUQuantElementsPerThread; ++element)
        {
            const int column = column_begin + element;
            const float unclamped_gate =
                __bfloat162float(gate_up[gate_up_offset + 2 * column]);
            const float unclamped_up =
                __bfloat162float(gate_up[gate_up_offset + 2 * column + 1]);
            const float gate =
                unclamped_gate > kSwigluLimit ? kSwigluLimit : unclamped_gate;
            const float up = unclamped_up < -kSwigluLimit
                                 ? -kSwigluLimit
                                 : (unclamped_up > kSwigluLimit ? kSwigluLimit
                                                                : unclamped_up);
            const float swish = gate / (1.0f + __expf(-kSwigluAlpha * gate));
            const __hip_bfloat16 rounded =
                __float2bfloat16(swish * (up + kUpOffset));
            values[element] = __builtin_bit_cast(opus::bf16_t, rounded);
            abs_max = fmaxf(abs_max, fabsf(__bfloat162float(rounded)));
        }
    }

    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 1));
    abs_max = fmaxf(abs_max, __shfl_xor(abs_max, 2));
    if(!active)
        return;

    const auto block_scale =
        fp_f32_to_e8m0_block_scale<kDefaultMxScaleRoundMode, MxDtype::FP8_E4M3>(
            abs_max);
    const float inverse_scale = 1.0f / block_scale.dq_scale;
    output_vector quantized;
#pragma unroll
    for(int element = 0; element < kSwiGLUQuantElementsPerThread; ++element)
        quantized[element] = opus::fp32_to_fp8(
            static_cast<float>(values[element]) * inverse_scale);

    *reinterpret_cast<output_vector*>(output + output_offset) = quantized;
    if(activated != nullptr)
        *reinterpret_cast<activated_vector*>(activated + output_offset) = values;
    if((threadIdx.x & 3) == 0)
    {
        const int group = column_begin / kQuantGroup;
        scales[iq2r_scale_offset(
            row, group, groups_per_row, scale_m_blocks, tiled_scales)] =
            block_scale.byte;
    }
}

__global__ void iq2r_route_reduce_indexed_kernel(
    const __hip_bfloat16* __restrict__ route_output,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ scatter_indices,
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
        const int column = static_cast<int>(index % hidden);
        float value = 0.0f;
        for(int route = 0; route < topk; ++route)
        {
            const int64_t original_route = token * topk + route;
            const int sorted_route = scatter_indices[original_route];
            value = fmaf(
                __bfloat162float(
                    route_output[static_cast<int64_t>(sorted_route) * hidden + column]),
                route_weights[original_route],
                value);
        }
        output[index] = __float2bfloat16(value);
    }
}

template <int BlockSize, int ItemsPerThread>
__global__ __launch_bounds__(BlockSize)
void iq2r_route_reduce_add_rmsnorm_indexed_kernel(
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
        float combined = 0.0f;
        if(column < hidden)
        {
            float route_value = 0.0f;
#pragma unroll
            for(int route = 0; route < kGptOssTopK; ++route)
            {
                if(route < topk)
                {
                    const int64_t original_route =
                        static_cast<int64_t>(token) * topk + route;
                    const int sorted_route = scatter_indices[original_route];
                    route_value = fmaf(
                        __bfloat162float(route_output[
                            static_cast<int64_t>(sorted_route) * hidden + column]),
                        route_weights[original_route],
                        route_value);
                }
            }

            // Preserve the unfused boundary: route_reduce writes BF16, then
            // add_rmsnorm reloads that rounded value before adding residual.
            const __hip_bfloat16 rounded_route = __float2bfloat16(route_value);
            combined = __bfloat162float(rounded_route) +
                       __bfloat162float(
                           residual[static_cast<int64_t>(token) * hidden + column]);
            residual_out[static_cast<int64_t>(token) * hidden + column] =
                __float2bfloat16(combined);
            square_sum = fmaf(combined, combined, square_sum);
        }
        values[item] = combined;
    }

    const float row_square_sum =
        block_reduce<float, Sum, BlockSize, true>(square_sum, Sum());
    const float inverse_rms =
        rsqrtf(row_square_sum / static_cast<float>(hidden) + epsilon);

#pragma unroll
    for(int item = 0; item < ItemsPerThread; ++item)
    {
        const int column = threadIdx.x + item * BlockSize;
        if(column < hidden)
        {
            const float normalized =
                values[item] * inverse_rms * __bfloat162float(norm_weight[column]);
            output[static_cast<int64_t>(token) * hidden + column] =
                __float2bfloat16(normalized);
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
                               int64_t expert_count,
                               int64_t task_rows)
{
    AITER_CHECK(expert_ids.is_gpu() && sorted_expert_ids.is_gpu() &&
                    gather_indices.is_gpu() && scatter_indices.is_gpu() &&
                    tasks.is_gpu() && task_count.is_gpu(),
                "IQ2R route sorting requires GPU tensors");
    const int device = expert_ids.device_id;
    AITER_CHECK(sorted_expert_ids.device_id == device &&
                    gather_indices.device_id == device &&
                    scatter_indices.device_id == device && tasks.device_id == device &&
                    task_count.device_id == device,
                "IQ2R route sorting tensors must share a GPU");
    AITER_CHECK(expert_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    tasks.dtype() == AITER_DTYPE_i32 &&
                    task_count.dtype() == AITER_DTYPE_i32,
                "IQ2R route sorting tensors must be int32");
    AITER_CHECK(expert_ids.dim() == 1 && expert_ids.numel() > 0 &&
                    expert_ids.numel() <= kMaxRoutes &&
                    sorted_expert_ids.numel() == expert_ids.numel() &&
                    gather_indices.numel() == expert_ids.numel() &&
                    scatter_indices.numel() == expert_ids.numel() &&
                    tasks.dim() == 2 && tasks.size(1) == kTaskColumns &&
                    task_count.dim() == 1 && task_count.size(0) == 1,
                "IQ2R route sorting shape mismatch");
    AITER_CHECK(expert_ids.is_contiguous() && sorted_expert_ids.is_contiguous() &&
                    gather_indices.is_contiguous() && scatter_indices.is_contiguous() &&
                    tasks.is_contiguous() && task_count.is_contiguous(),
                "IQ2R route sorting tensors must be contiguous");
    AITER_CHECK(expert_count > 0 && expert_count <= kMaxExperts,
                "IQ2R supports at most 128 experts");
    AITER_CHECK(task_rows == 16 || task_rows == 32 || task_rows == 64 ||
                    task_rows == 128 ||
                    task_rows == 256,
                "IQ2R task_rows must be 16, 32, 64, 128, or 256");
    const int64_t routes = expert_ids.numel();
    const int64_t required_capacity =
        (routes + task_rows - 1) / task_rows +
        std::min<int64_t>(routes, expert_count + 1);
    AITER_CHECK(tasks.size(0) >= required_capacity,
                "IQ2R task capacity is too small");

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
                           static_cast<int>(routes),
                           static_cast<int>(expert_count),
                           static_cast<int>(tasks.size(0)),
                           static_cast<int>(task_rows));
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
                           static_cast<int>(routes),
                           static_cast<int>(expert_count),
                           static_cast<int>(tasks.size(0)),
                           static_cast<int>(task_rows));
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
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 &&
                    output.dtype() == AITER_DTYPE_bf16 &&
                    gather_indices.dtype() == AITER_DTYPE_i32,
                "IQ2R route gather expects BF16 data and int32 indices");
    AITER_CHECK(input.dim() == 2 && output.dim() == 2 &&
                    gather_indices.dim() == 1 && topk > 0 &&
                    output.size(0) == input.size(0) * topk &&
                    output.size(1) == input.size(1) &&
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
    AITER_CHECK(input.is_gpu() && gather_indices.is_gpu() && output.is_gpu() &&
                    scales.is_gpu(),
                "IQ2R fused gather/quant requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(gather_indices.device_id == device && output.device_id == device &&
                    scales.device_id == device,
                "IQ2R fused gather/quant tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 &&
                    scales.dtype() == AITER_DTYPE_u8,
                "IQ2R fused gather/quant dtype mismatch");
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    AITER_CHECK(input.dim() == 2 && gather_indices.dim() == 1 &&
                    output.dim() == 2 && topk > 0 &&
                    output.size(0) == input.size(0) * topk &&
                    gather_indices.numel() == output.size(0) &&
                    output.size(1) == input.size(1) &&
                    output.size(1) % kQuantGroup == 0 &&
                    iq2r_valid_scale_shape(
                        scales, output.size(0), groups_per_row),
                "IQ2R fused gather/quant shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    gather_indices.is_contiguous() && output.is_contiguous() &&
                    scales.is_contiguous(),
                "IQ2R fused gather/quant input must have contiguous columns and "
                "non-overlapping rows; outputs must be contiguous");

    const bool tiled_scales = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((output.size(0) + 15) / 16);
    const int64_t groups = output.size(0) * groups_per_row;
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

void iq2r_route_direct_gather_quant_out(const aiter_tensor_t& input,
                                        const aiter_tensor_t& expert_ids,
                                        aiter_tensor_t& sorted_expert_ids,
                                        aiter_tensor_t& gather_indices,
                                        aiter_tensor_t& scatter_indices,
                                        aiter_tensor_t& tasks,
                                        aiter_tensor_t& task_count,
                                        aiter_tensor_t& output,
                                        aiter_tensor_t& scales,
                                        int64_t topk,
                                        int64_t expert_count)
{
    AITER_CHECK(input.is_gpu() && expert_ids.is_gpu() &&
                    sorted_expert_ids.is_gpu() && gather_indices.is_gpu() &&
                    scatter_indices.is_gpu() && tasks.is_gpu() && task_count.is_gpu() &&
                    output.is_gpu() && scales.is_gpu(),
                "IQ2R direct routing requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(expert_ids.device_id == device &&
                    sorted_expert_ids.device_id == device &&
                    gather_indices.device_id == device &&
                    scatter_indices.device_id == device && tasks.device_id == device &&
                    task_count.device_id == device && output.device_id == device &&
                    scales.device_id == device,
                "IQ2R direct routing tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 &&
                    expert_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    tasks.dtype() == AITER_DTYPE_i32 &&
                    task_count.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 &&
                    scales.dtype() == AITER_DTYPE_u8,
                "IQ2R direct routing tensor dtypes are invalid");
    const int64_t routes = expert_ids.numel();
    AITER_CHECK(routes > 0 && routes <= 16 && topk > 0 && routes % topk == 0,
                "IQ2R direct routing requires 1..16 complete top-k route rows");
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    AITER_CHECK(input.dim() == 2 && input.size(0) * topk == routes &&
                    input.size(1) % kQuantGroup == 0 &&
                    output.dim() == 2 && output.size(0) == routes &&
                    output.size(1) == input.size(1) &&
                    iq2r_valid_scale_shape(scales, routes, groups_per_row) &&
                    sorted_expert_ids.numel() == routes &&
                    gather_indices.numel() == routes &&
                    scatter_indices.numel() == routes && tasks.dim() == 2 &&
                    tasks.size(0) >= routes && tasks.size(1) == kTaskColumns &&
                    task_count.dim() == 1 && task_count.size(0) == 1,
                "IQ2R direct routing shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    expert_ids.is_contiguous() && sorted_expert_ids.is_contiguous() &&
                    gather_indices.is_contiguous() && scatter_indices.is_contiguous() &&
                    tasks.is_contiguous() && task_count.is_contiguous() &&
                    output.is_contiguous() && scales.is_contiguous(),
                "IQ2R direct routing tensors must be contiguous");
    AITER_CHECK(expert_count > 0 && expert_count <= kMaxExperts,
                "IQ2R direct routing supports at most 128 experts");

    const bool tiled_scales = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((routes + 15) / 16);
    hipLaunchKernelGGL(iq2r_route_direct_gather_quant_kernel,
                       dim3(static_cast<uint32_t>(input.size(0))),
                       dim3(kDirectQuantThreads),
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
                       scale_m_blocks,
                       tiled_scales);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_topk_direct_gather_quant_out(
    const aiter_tensor_t& input,
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
    std::optional<aiter_tensor_t> router_bias)
{
    AITER_CHECK(input.is_gpu() && router_logits.is_gpu() &&
                    topk_weights.is_gpu() && topk_ids.is_gpu() &&
                    sorted_expert_ids.is_gpu() && gather_indices.is_gpu() &&
                    scatter_indices.is_gpu() && tasks.is_gpu() &&
                    task_count.is_gpu() && output.is_gpu() && scales.is_gpu(),
                "fused IQ2R routing requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(router_logits.device_id == device &&
                    topk_weights.device_id == device && topk_ids.device_id == device &&
                    sorted_expert_ids.device_id == device &&
                    gather_indices.device_id == device &&
                    scatter_indices.device_id == device && tasks.device_id == device &&
                    task_count.device_id == device && output.device_id == device &&
                    scales.device_id == device,
                "fused IQ2R routing tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 &&
                    (router_logits.dtype() == AITER_DTYPE_bf16 ||
                     router_logits.dtype() == AITER_DTYPE_fp32) &&
                    topk_weights.dtype() == AITER_DTYPE_fp32 &&
                    topk_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    tasks.dtype() == AITER_DTYPE_i32 &&
                    task_count.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 &&
                    scales.dtype() == AITER_DTYPE_u8,
                "fused IQ2R routing tensor dtypes are invalid");
    AITER_CHECK(input.dim() == 2 && input.size(0) > 0 && input.size(0) <= 4 &&
                    input.size(1) % kQuantGroup == 0 &&
                    router_logits.dim() == 2 &&
                    router_logits.size(0) == input.size(0) &&
                    router_logits.size(1) == kMaxExperts &&
                    topk_weights.dim() == 2 && topk_weights.size(0) == input.size(0) &&
                    topk_weights.size(1) == 4 && topk_ids.dim() == 2 &&
                    topk_ids.size(0) == input.size(0) && topk_ids.size(1) == 4,
                "fused IQ2R routing input shape mismatch");
    const int routes = static_cast<int>(input.size(0) * 4);
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    AITER_CHECK(sorted_expert_ids.numel() == routes &&
                    gather_indices.numel() == routes &&
                    scatter_indices.numel() == routes && tasks.dim() == 2 &&
                    tasks.size(0) >= routes && tasks.size(1) == kTaskColumns &&
                    task_count.dim() == 1 && task_count.size(0) == 1 &&
                    output.dim() == 2 && output.size(0) == routes &&
                    output.size(1) == input.size(1) &&
                    iq2r_valid_scale_shape(scales, routes, groups_per_row),
                "fused IQ2R routing output shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    router_logits.stride(1) == 1 &&
                    router_logits.stride(0) >= kMaxExperts &&
                    topk_weights.is_contiguous() && topk_ids.is_contiguous() &&
                    sorted_expert_ids.is_contiguous() &&
                    gather_indices.is_contiguous() && scatter_indices.is_contiguous() &&
                    tasks.is_contiguous() && task_count.is_contiguous() &&
                    output.is_contiguous() && scales.is_contiguous(),
                "fused IQ2R routing tensors must be contiguous");
    const opus::bf16_t* router_bias_ptr = nullptr;
    if(router_bias.has_value())
    {
        const auto& bias = router_bias.value();
        AITER_CHECK(bias.is_gpu() && bias.device_id == device &&
                        bias.dtype() == AITER_DTYPE_bf16 && bias.dim() == 1 &&
                        bias.size(0) == kMaxExperts && bias.is_contiguous(),
                    "fused IQ2R router bias must be contiguous BF16 [128] on the same GPU");
        router_bias_ptr = static_cast<const opus::bf16_t*>(bias.data_ptr());
    }

    HipDeviceGuard device_guard(device);
    const dim3 grid(static_cast<unsigned int>(input.size(0)));
    const dim3 block(kDirectQuantThreads);
    const bool tiled_scales = scales.dim() == 4;
    const int scale_m_blocks = (routes + 15) / 16;
    if(router_logits.dtype() == AITER_DTYPE_bf16)
        hipLaunchKernelGGL(
            (iq2r_route_topk_direct_gather_quant_kernel<opus::bf16_t>),
            grid,
            block,
            0,
            getCurrentHIPStream(),
            static_cast<const opus::bf16_t*>(input.data_ptr()),
            static_cast<const opus::bf16_t*>(router_logits.data_ptr()),
            router_bias_ptr,
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int32_t*>(topk_ids.data_ptr()),
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
            static_cast<int>(router_logits.stride(0)),
            static_cast<int>(groups_per_row),
            renormalize,
            scale_m_blocks,
            tiled_scales);
    else
        hipLaunchKernelGGL(
            (iq2r_route_topk_direct_gather_quant_kernel<float>),
            grid,
            block,
            0,
            getCurrentHIPStream(),
            static_cast<const opus::bf16_t*>(input.data_ptr()),
            static_cast<const float*>(router_logits.data_ptr()),
            router_bias_ptr,
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int32_t*>(topk_ids.data_ptr()),
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
            static_cast<int>(router_logits.stride(0)),
            static_cast<int>(groups_per_row),
            renormalize,
            scale_m_blocks,
            tiled_scales);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_topk_sort_gather_quant_out(
    const aiter_tensor_t& input,
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
    std::optional<aiter_tensor_t> router_bias)
{
    AITER_CHECK(input.is_gpu() && router_logits.is_gpu() &&
                    topk_weights.is_gpu() && topk_ids.is_gpu() &&
                    sorted_expert_ids.is_gpu() && gather_indices.is_gpu() &&
                    scatter_indices.is_gpu() && tasks.is_gpu() &&
                    task_count.is_gpu() && output.is_gpu() && scales.is_gpu(),
                "fused sorted IQ2R routing requires GPU tensors");
    const int device = input.device_id;
    AITER_CHECK(router_logits.device_id == device &&
                    topk_weights.device_id == device && topk_ids.device_id == device &&
                    sorted_expert_ids.device_id == device &&
                    gather_indices.device_id == device &&
                    scatter_indices.device_id == device && tasks.device_id == device &&
                    task_count.device_id == device && output.device_id == device &&
                    scales.device_id == device,
                "fused sorted IQ2R routing tensors must share a GPU");
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 &&
                    (router_logits.dtype() == AITER_DTYPE_bf16 ||
                     router_logits.dtype() == AITER_DTYPE_fp32) &&
                    topk_weights.dtype() == AITER_DTYPE_fp32 &&
                    topk_ids.dtype() == AITER_DTYPE_i32 &&
                    sorted_expert_ids.dtype() == AITER_DTYPE_i32 &&
                    gather_indices.dtype() == AITER_DTYPE_i32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    tasks.dtype() == AITER_DTYPE_i32 &&
                    task_count.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_fp8 &&
                    scales.dtype() == AITER_DTYPE_u8,
                "fused sorted IQ2R routing tensor dtypes are invalid");
    AITER_CHECK(input.dim() == 2 && input.size(0) > 0 &&
                    input.size(0) <= kFusedSortTokens &&
                    input.size(1) % kQuantGroup == 0 &&
                    router_logits.dim() == 2 &&
                    router_logits.size(0) == input.size(0) &&
                    router_logits.size(1) == kMaxExperts &&
                    topk_weights.dim() == 2 && topk_weights.size(0) == input.size(0) &&
                    topk_weights.size(1) == kGptOssTopK && topk_ids.dim() == 2 &&
                    topk_ids.size(0) == input.size(0) &&
                    topk_ids.size(1) == kGptOssTopK,
                "fused sorted IQ2R routing input shape mismatch");
    const int64_t routes = input.size(0) * kGptOssTopK;
    const int64_t groups_per_row = input.size(1) / kQuantGroup;
    AITER_CHECK(task_rows == 16 || task_rows == 32 || task_rows == 64 ||
                    task_rows == 128 ||
                    task_rows == 256,
                "IQ2R task_rows must be 16, 32, 64, 128, or 256");
    const int64_t required_capacity =
        (routes + task_rows - 1) / task_rows +
        std::min<int64_t>(routes, kMaxExperts);
    AITER_CHECK(sorted_expert_ids.numel() == routes &&
                    gather_indices.numel() == routes &&
                    scatter_indices.numel() == routes && tasks.dim() == 2 &&
                    tasks.size(0) >= required_capacity &&
                    tasks.size(1) == kTaskColumns && task_count.dim() == 1 &&
                    task_count.size(0) == 1 && output.dim() == 2 &&
                    output.size(0) == routes && output.size(1) == input.size(1) &&
                    iq2r_valid_scale_shape(scales, routes, groups_per_row),
                "fused sorted IQ2R routing output shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    router_logits.stride(1) == 1 &&
                    router_logits.stride(0) >= kMaxExperts &&
                    topk_weights.is_contiguous() && topk_ids.is_contiguous() &&
                    sorted_expert_ids.is_contiguous() &&
                    gather_indices.is_contiguous() && scatter_indices.is_contiguous() &&
                    tasks.is_contiguous() && task_count.is_contiguous() &&
                    output.is_contiguous() && scales.is_contiguous(),
                "fused sorted IQ2R routing tensors must be contiguous");
    const opus::bf16_t* router_bias_ptr = nullptr;
    if(router_bias.has_value())
    {
        const auto& bias = router_bias.value();
        AITER_CHECK(bias.is_gpu() && bias.device_id == device &&
                        bias.dtype() == AITER_DTYPE_bf16 && bias.dim() == 1 &&
                        bias.size(0) == kMaxExperts && bias.is_contiguous(),
                    "fused sorted IQ2R router bias must be contiguous BF16 [128] on the same GPU");
        router_bias_ptr = static_cast<const opus::bf16_t*>(bias.data_ptr());
    }

    HipDeviceGuard device_guard(device);
    const dim3 grid(1);
    const dim3 block(kFusedSortThreads);
    const bool tiled_scales = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((routes + 15) / 16);
    if(router_logits.dtype() == AITER_DTYPE_bf16)
        hipLaunchKernelGGL(
            (iq2r_route_topk_sort_tasks_kernel<opus::bf16_t>),
            grid,
            block,
            0,
            getCurrentHIPStream(),
            static_cast<const opus::bf16_t*>(router_logits.data_ptr()),
            router_bias_ptr,
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int32_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
            static_cast<int32_t*>(gather_indices.data_ptr()),
            static_cast<int32_t*>(scatter_indices.data_ptr()),
            static_cast<int32_t*>(tasks.data_ptr()),
            static_cast<int32_t*>(task_count.data_ptr()),
            static_cast<int>(input.size(0)),
            static_cast<int>(router_logits.stride(0)),
            static_cast<int>(tasks.size(0)),
            static_cast<int>(task_rows),
            renormalize);
    else
        hipLaunchKernelGGL(
            (iq2r_route_topk_sort_tasks_kernel<float>),
            grid,
            block,
            0,
            getCurrentHIPStream(),
            static_cast<const float*>(router_logits.data_ptr()),
            router_bias_ptr,
            static_cast<float*>(topk_weights.data_ptr()),
            static_cast<int32_t*>(topk_ids.data_ptr()),
            static_cast<int32_t*>(sorted_expert_ids.data_ptr()),
            static_cast<int32_t*>(gather_indices.data_ptr()),
            static_cast<int32_t*>(scatter_indices.data_ptr()),
            static_cast<int32_t*>(tasks.data_ptr()),
            static_cast<int32_t*>(task_count.data_ptr()),
            static_cast<int>(input.size(0)),
            static_cast<int>(router_logits.stride(0)),
            static_cast<int>(tasks.size(0)),
            static_cast<int>(task_rows),
            renormalize);
    HIP_CALL_LAUNCH(hipGetLastError());

    hipLaunchKernelGGL(iq2r_route_gather_quant_broadcast_kernel,
                       dim3(static_cast<unsigned int>(input.size(0))),
                       dim3(kDirectQuantThreads),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const opus::bf16_t*>(input.data_ptr()),
                       static_cast<const int32_t*>(scatter_indices.data_ptr()),
                       static_cast<opus::fp8_t*>(output.data_ptr()),
                       static_cast<uint8_t*>(scales.data_ptr()),
                       static_cast<int>(input.size(0)),
                       static_cast<int>(input.size(1)),
                       static_cast<int>(input.stride(0)),
                       static_cast<int>(groups_per_row),
                       scale_m_blocks,
                       tiled_scales);
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_swiglu_out(const aiter_tensor_t& gate_up, aiter_tensor_t& output)
{
    AITER_CHECK(gate_up.is_gpu() && output.is_gpu() &&
                    gate_up.device_id == output.device_id,
                "IQ2R SwiGLU tensors must share a GPU");
    AITER_CHECK(gate_up.dtype() == AITER_DTYPE_bf16 &&
                    output.dtype() == AITER_DTYPE_bf16,
                "IQ2R SwiGLU tensors must be BF16");
    AITER_CHECK(gate_up.dim() == 2 && output.dim() == 2 &&
                    gate_up.size(0) == output.size(0) &&
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
                       static_cast<int>(output.size(1)));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_swiglu_quant_out(const aiter_tensor_t& gate_up,
                           aiter_tensor_t& output,
                           aiter_tensor_t& scales,
                           std::optional<aiter_tensor_t> activated)
{
    AITER_CHECK(gate_up.is_gpu() && output.is_gpu() && scales.is_gpu(),
                "IQ2R fused SwiGLU/quant requires GPU tensors");
    const int device = gate_up.device_id;
    AITER_CHECK(output.device_id == device && scales.device_id == device,
                "IQ2R fused SwiGLU/quant tensors must share a GPU");
    AITER_CHECK(gate_up.dtype() == AITER_DTYPE_bf16 &&
                    output.dtype() == AITER_DTYPE_fp8 &&
                    scales.dtype() == AITER_DTYPE_u8,
                "IQ2R fused SwiGLU/quant dtype mismatch");
    const int64_t groups_per_row = output.size(1) / kQuantGroup;
    AITER_CHECK(gate_up.dim() == 2 && output.dim() == 2 &&
                    gate_up.size(0) == output.size(0) &&
                    gate_up.size(1) == output.size(1) * 2 &&
                    output.size(1) % kQuantGroup == 0 &&
                    iq2r_valid_scale_shape(
                        scales, output.size(0), groups_per_row),
                "IQ2R fused SwiGLU/quant shape mismatch");
    AITER_CHECK(gate_up.is_contiguous() && output.is_contiguous() &&
                    scales.is_contiguous(),
                "IQ2R fused SwiGLU/quant tensors must be contiguous");

    __hip_bfloat16* activated_ptr = nullptr;
    if(activated.has_value())
    {
        AITER_CHECK(activated->is_gpu() && activated->device_id == device &&
                        activated->dtype() == AITER_DTYPE_bf16 &&
                        activated->dim() == 2 &&
                        activated->size(0) == output.size(0) &&
                        activated->size(1) == output.size(1) &&
                        activated->is_contiguous(),
                    "IQ2R optional activated output must be contiguous BF16 and "
                    "match the quantized output shape");
        activated_ptr = static_cast<__hip_bfloat16*>(activated->data_ptr());
    }

    const bool tiled_scales = scales.dim() == 4;
    const int scale_m_blocks = static_cast<int>((output.size(0) + 15) / 16);
    const char* family = std::getenv("IQ2R_SWIGLU_QUANT_FAMILY");
    const bool use_parallel8 =
        family == nullptr || family[0] == '\0' ||
        std::strcmp(family, "parallel8") == 0;
    AITER_CHECK(use_parallel8 || std::strcmp(family, "group32") == 0,
                "IQ2R_SWIGLU_QUANT_FAMILY must be parallel8 or group32");
    if(use_parallel8)
    {
        const int blocks_per_row =
            (static_cast<int>(output.size(1)) +
             kQuantThreads * kSwiGLUQuantElementsPerThread - 1) /
            (kQuantThreads * kSwiGLUQuantElementsPerThread);
        hipLaunchKernelGGL(
            iq2r_swiglu_quant_parallel8_kernel,
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
            tiled_scales);
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
                           tiled_scales);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_reduce_indexed_out(const aiter_tensor_t& route_output,
                                   const aiter_tensor_t& route_weights,
                                   const aiter_tensor_t& scatter_indices,
                                   aiter_tensor_t& output,
                                   int64_t topk)
{
    AITER_CHECK(route_output.is_gpu() && route_weights.is_gpu() &&
                    scatter_indices.is_gpu() && output.is_gpu(),
                "IQ2R route reduction requires GPU tensors");
    const int device = route_output.device_id;
    AITER_CHECK(route_weights.device_id == device &&
                    scatter_indices.device_id == device && output.device_id == device,
                "IQ2R route reduction tensors must share a GPU");
    AITER_CHECK(route_output.dtype() == AITER_DTYPE_bf16 &&
                    route_weights.dtype() == AITER_DTYPE_fp32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    output.dtype() == AITER_DTYPE_bf16,
                "IQ2R route reduction dtype mismatch");
    AITER_CHECK(route_output.dim() == 2 && route_weights.dim() == 2 &&
                    scatter_indices.dim() == 1 && output.dim() == 2 && topk > 0 &&
                    route_output.size(0) == route_weights.size(0) * topk &&
                    scatter_indices.numel() == route_output.size(0) &&
                    route_output.size(1) == output.size(1) &&
                    output.size(0) == route_weights.size(0),
                "IQ2R route reduction shape mismatch");
    AITER_CHECK(route_output.is_contiguous() && route_weights.is_contiguous() &&
                    scatter_indices.is_contiguous() && output.is_contiguous(),
                "IQ2R route reduction tensors must be contiguous");
    const int64_t elements = output.numel();
    hipLaunchKernelGGL(iq2r_route_reduce_indexed_kernel,
                       dim3(launch_blocks(elements)),
                       dim3(kThreads),
                       0,
                       getCurrentHIPStream(),
                       static_cast<const __hip_bfloat16*>(route_output.data_ptr()),
                       static_cast<const float*>(route_weights.data_ptr()),
                       static_cast<const int32_t*>(scatter_indices.data_ptr()),
                       static_cast<__hip_bfloat16*>(output.data_ptr()),
                       elements,
                       static_cast<int>(output.size(1)),
                       static_cast<int>(topk));
    HIP_CALL_LAUNCH(hipGetLastError());
}

void iq2r_route_reduce_add_rmsnorm_indexed_out(
    const aiter_tensor_t& route_output,
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
    AITER_CHECK(route_output.is_gpu() && route_weights.is_gpu() &&
                    scatter_indices.is_gpu() && residual.is_gpu() &&
                    norm_weight.is_gpu() && output.is_gpu() && residual_out.is_gpu(),
                "IQ2R fused route reduction/RMSNorm requires GPU tensors");
    const int device = route_output.device_id;
    AITER_CHECK(route_weights.device_id == device &&
                    scatter_indices.device_id == device && residual.device_id == device &&
                    norm_weight.device_id == device && output.device_id == device &&
                    residual_out.device_id == device,
                "IQ2R fused route reduction/RMSNorm tensors must share a GPU");
    AITER_CHECK(route_output.dtype() == AITER_DTYPE_bf16 &&
                    route_weights.dtype() == AITER_DTYPE_fp32 &&
                    scatter_indices.dtype() == AITER_DTYPE_i32 &&
                    residual.dtype() == AITER_DTYPE_bf16 &&
                    norm_weight.dtype() == AITER_DTYPE_bf16 &&
                    output.dtype() == AITER_DTYPE_bf16 &&
                    residual_out.dtype() == AITER_DTYPE_bf16,
                "IQ2R fused route reduction/RMSNorm dtype mismatch");
    AITER_CHECK(route_output.dim() == 2 && route_weights.dim() == 2 &&
                    scatter_indices.dim() == 1 && residual.dim() == 2 &&
                    norm_weight.dim() == 1 && output.dim() == 2 &&
                    residual_out.dim() == 2 && topk == kGptOssTopK &&
                    route_output.size(0) == route_weights.size(0) * topk &&
                    scatter_indices.numel() == route_output.size(0) &&
                    route_output.size(1) == residual.size(1) &&
                    residual.size(0) == route_weights.size(0) &&
                    output.size(0) == residual.size(0) &&
                    output.size(1) == residual.size(1) &&
                    residual_out.size(0) == residual.size(0) &&
                    residual_out.size(1) == residual.size(1) &&
                    norm_weight.numel() == residual.size(1),
                "IQ2R fused route reduction/RMSNorm shape mismatch");
    AITER_CHECK(route_output.size(1) == 2880,
                "IQ2R fused route reduction/RMSNorm currently requires GPT-OSS hidden=2880");
    AITER_CHECK(route_output.is_contiguous() && route_weights.is_contiguous() &&
                    scatter_indices.is_contiguous() && residual.is_contiguous() &&
                    norm_weight.is_contiguous() && output.is_contiguous() &&
                    residual_out.is_contiguous(),
                "IQ2R fused route reduction/RMSNorm tensors must be contiguous");

    const int tokens = static_cast<int>(route_weights.size(0));
    const int hidden = static_cast<int>(route_output.size(1));
    const float epsilon_f = static_cast<float>(epsilon);
    if(block_size == 256)
        hipLaunchKernelGGL(
            (iq2r_route_reduce_add_rmsnorm_indexed_kernel<256, 12>),
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
        hipLaunchKernelGGL(
            (iq2r_route_reduce_add_rmsnorm_indexed_kernel<512, 6>),
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
        hipLaunchKernelGGL(
            (iq2r_route_reduce_add_rmsnorm_indexed_kernel<1024, 3>),
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
