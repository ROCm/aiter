// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_hip_common.h"
#include "aiter_opus_plus.h"
#include "aiter_stream.h"
#include "iq2r.h"
#include "mx_quant_utils.h"

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>

namespace aiter {
namespace {

constexpr int kThreads       = 256;
constexpr int kQuantThreads  = 64;
constexpr int kTaskColumns   = 3;
constexpr int kMaxExperts    = 128;
constexpr int kMaxRoutes     = 4096;
constexpr int kSmallRoutes   = 16;
constexpr int kQuantGroup    = 32;
constexpr float kSwigluAlpha = 1.702f;
constexpr float kSwigluLimit = 7.0f;
constexpr float kUpOffset    = 1.0f;

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

// GPT-OSS has only 128 experts and at most 4096 routed rows in the initial
// contract. A single CTA builds an expert-grouped permutation and homogeneous
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
    int topk)
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
    scales[group_id] = block_scale.byte;
}

// Decode-specialized path for at most 16 routed rows. Production GPT-OSS c1-c4
// routes are overwhelmingly one row per expert, so sorting them spends more
// time initializing the general counting-sort scratch than it can recover via
// expert reuse. Build one-row tasks while gathering/quantizing in original
// route order, eliminating the standalone routing launch entirely.
__global__ __launch_bounds__(kQuantThreads)
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
    int64_t groups,
    int routes,
    int hidden,
    int input_stride,
    int groups_per_row,
    int topk,
    int expert_count)
{
    const int64_t group_id =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(group_id >= groups)
        return;

    const int route = static_cast<int>(group_id / groups_per_row);
    const int group = static_cast<int>(group_id % groups_per_row);
    if(group == 0)
    {
        const int expert = expert_ids[route];
        const int valid_expert = expert >= 0 && expert < expert_count ? expert : -1;
        sorted_expert_ids[route] = expert;
        gather_indices[route] = route;
        scatter_indices[route] = route;
        tasks[route * kTaskColumns] = route;
        tasks[route * kTaskColumns + 1] = 1;
        tasks[route * kTaskColumns + 2] = valid_expert;
        if(route == 0)
            task_count[0] = routes;
    }

    const int token = route / topk;
    const int column_begin = group * kQuantGroup;
    const int64_t input_offset =
        static_cast<int64_t>(token) * input_stride + column_begin;
    const int64_t output_offset =
        static_cast<int64_t>(route) * hidden + column_begin;

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
    scales[group_id] = block_scale.byte;
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
    int groups_per_row)
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
    scales[group_id] = block_scale.byte;
    if(activated != nullptr)
    {
        *reinterpret_cast<activated_vector*>(activated + output_offset) = values;
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
    AITER_CHECK(task_rows == 16 || task_rows == 32 || task_rows == 64,
                "IQ2R task_rows must be 16, 32, or 64");
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
    AITER_CHECK(input.dim() == 2 && gather_indices.dim() == 1 &&
                    output.dim() == 2 && scales.dim() == 2 && topk > 0 &&
                    output.size(0) == input.size(0) * topk &&
                    gather_indices.numel() == output.size(0) &&
                    output.size(1) == input.size(1) &&
                    output.size(1) % kQuantGroup == 0 &&
                    scales.size(0) == output.size(0) &&
                    scales.size(1) == output.size(1) / kQuantGroup,
                "IQ2R fused gather/quant shape mismatch");
    AITER_CHECK(input.stride(1) == 1 && input.stride(0) >= input.size(1) &&
                    gather_indices.is_contiguous() && output.is_contiguous() &&
                    scales.is_contiguous(),
                "IQ2R fused gather/quant input must have contiguous columns and "
                "non-overlapping rows; outputs must be contiguous");

    const int64_t groups = scales.numel();
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
                       static_cast<int>(scales.size(1)),
                       static_cast<int>(topk));
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
    AITER_CHECK(input.dim() == 2 && input.size(0) * topk == routes &&
                    input.size(1) % kQuantGroup == 0 &&
                    output.dim() == 2 && output.size(0) == routes &&
                    output.size(1) == input.size(1) &&
                    scales.dim() == 2 && scales.size(0) == routes &&
                    scales.size(1) == input.size(1) / kQuantGroup &&
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

    const int groups_per_row = static_cast<int>(input.size(1) / kQuantGroup);
    const int64_t groups = routes * groups_per_row;
    hipLaunchKernelGGL(iq2r_route_direct_gather_quant_kernel,
                       dim3((groups + kQuantThreads - 1) / kQuantThreads),
                       dim3(kQuantThreads),
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
                       groups,
                       static_cast<int>(routes),
                       static_cast<int>(input.size(1)),
                       static_cast<int>(input.stride(0)),
                       groups_per_row,
                       static_cast<int>(topk),
                       static_cast<int>(expert_count));
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
    AITER_CHECK(gate_up.dim() == 2 && output.dim() == 2 && scales.dim() == 2 &&
                    gate_up.size(0) == output.size(0) &&
                    gate_up.size(1) == output.size(1) * 2 &&
                    output.size(1) % kQuantGroup == 0 &&
                    scales.size(0) == output.size(0) &&
                    scales.size(1) == output.size(1) / kQuantGroup,
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

    const int64_t groups = scales.numel();
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
                       static_cast<int>(scales.size(1)));
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

} // namespace aiter
