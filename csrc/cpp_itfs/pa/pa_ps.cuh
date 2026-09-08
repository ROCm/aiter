// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace aiter {

constexpr float kPaPsReduceLog2E = 1.4426950408889634f;
constexpr int kPaPsReduceWarpSize = 64;

constexpr int pa_ps_next_power_of_2(int value)
{
    int pow2 = 1;
    while(pow2 < value)
    {
        pow2 <<= 1;
    }
    return pow2;
}

template <typename T>
__device__ __forceinline__ float pa_ps_to_float(T value)
{
    if constexpr(std::is_same_v<T, float>)
    {
        return value;
    }
    else if constexpr(std::is_same_v<T, _Float16>)
    {
        return static_cast<float>(value);
    }
    else if constexpr(std::is_same_v<T, __hip_bfloat16>)
    {
        return __bfloat162float(value);
    }
}

template <typename T>
__device__ __forceinline__ T pa_ps_from_float(float value)
{
    if constexpr(std::is_same_v<T, float>)
    {
        return value;
    }
    else if constexpr(std::is_same_v<T, _Float16>)
    {
        return static_cast<_Float16>(value);
    }
    else if constexpr(std::is_same_v<T, __hip_bfloat16>)
    {
        return __float2bfloat16(value);
    }
}

__device__ __forceinline__ __amdgpu_buffer_rsrc_t pa_ps_make_buffer_rsrc(const void* ptr)
{
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(ptr), 0, 0xffffffff, 0x27000);
}

template <typename T>
__device__ __forceinline__ T pa_ps_buffer_load(__amdgpu_buffer_rsrc_t rsrc, int byte_offset)
{
    if constexpr(std::is_same_v<T, float>)
    {
        return __builtin_bit_cast(float, __builtin_amdgcn_raw_buffer_load_b32(rsrc, byte_offset, 0, 0));
    }
    else if constexpr(std::is_same_v<T, _Float16>)
    {
        return __builtin_bit_cast(_Float16, __builtin_amdgcn_raw_buffer_load_b16(rsrc, byte_offset, 0, 0));
    }
    else if constexpr(std::is_same_v<T, __hip_bfloat16>)
    {
        return __builtin_bit_cast(__hip_bfloat16,
                                  __builtin_amdgcn_raw_buffer_load_b16(rsrc, byte_offset, 0, 0));
    }
}

template <typename T>
__device__ __forceinline__ void pa_ps_buffer_store(__amdgpu_buffer_rsrc_t rsrc,
                                                   int byte_offset,
                                                   T value)
{
    if constexpr(std::is_same_v<T, float>)
    {
        __builtin_amdgcn_raw_buffer_store_b32(
            __builtin_bit_cast(uint32_t, value), rsrc, byte_offset, 0, 0);
    }
    else if constexpr(std::is_same_v<T, _Float16>)
    {
        __builtin_amdgcn_raw_buffer_store_b16(
            __builtin_bit_cast(uint16_t, value), rsrc, byte_offset, 0, 0);
    }
    else if constexpr(std::is_same_v<T, __hip_bfloat16>)
    {
        __builtin_amdgcn_raw_buffer_store_b16(
            __builtin_bit_cast(uint16_t, value), rsrc, byte_offset, 0, 0);
    }
}

template <int OFFSET>
__device__ __forceinline__ float pa_ps_ds_swizzle_xor(float value)
{
    const int value_bits = __builtin_bit_cast(int, value);
    const int swizzled_bits =
        __builtin_amdgcn_ds_swizzle(value_bits, (OFFSET << 10) | 0x1f);
    return __builtin_bit_cast(float, swizzled_bits);
}

template <int REDUCE_WIDTH>
__device__ __forceinline__ float pa_ps_wave_reduce_max(float value)
{
    if constexpr(REDUCE_WIDTH > 32)
    {
        value = fmaxf(value, pa_ps_ds_swizzle_xor<32>(value));
    }
    if constexpr(REDUCE_WIDTH > 16)
    {
        value = fmaxf(value, pa_ps_ds_swizzle_xor<16>(value));
    }
    if constexpr(REDUCE_WIDTH > 8)
    {
        value = fmaxf(value, pa_ps_ds_swizzle_xor<8>(value));
    }
    if constexpr(REDUCE_WIDTH > 4)
    {
        value = fmaxf(value, pa_ps_ds_swizzle_xor<4>(value));
    }
    if constexpr(REDUCE_WIDTH > 2)
    {
        value = fmaxf(value, pa_ps_ds_swizzle_xor<2>(value));
    }
    if constexpr(REDUCE_WIDTH > 1)
    {
        value = fmaxf(value, pa_ps_ds_swizzle_xor<1>(value));
    }
    return value;
}

template <int REDUCE_WIDTH>
__device__ __forceinline__ float pa_ps_wave_reduce_sum(float value)
{
    if constexpr(REDUCE_WIDTH > 32)
    {
        value += pa_ps_ds_swizzle_xor<32>(value);
    }
    if constexpr(REDUCE_WIDTH > 16)
    {
        value += pa_ps_ds_swizzle_xor<16>(value);
    }
    if constexpr(REDUCE_WIDTH > 8)
    {
        value += pa_ps_ds_swizzle_xor<8>(value);
    }
    if constexpr(REDUCE_WIDTH > 4)
    {
        value += pa_ps_ds_swizzle_xor<4>(value);
    }
    if constexpr(REDUCE_WIDTH > 2)
    {
        value += pa_ps_ds_swizzle_xor<2>(value);
    }
    if constexpr(REDUCE_WIDTH > 1)
    {
        value += pa_ps_ds_swizzle_xor<1>(value);
    }
    return value;
}

__device__ __forceinline__ float pa_ps_broadcast_weight(float weight_local, int src_lane)
{
    const int weight_bits = __builtin_bit_cast(int, weight_local);
    const int bcast_bits = __builtin_amdgcn_ds_bpermute(src_lane << 2, weight_bits);
    return __builtin_bit_cast(float, bcast_bits);
}

template <typename output_t,
          typename logits_t,
          typename sink_t,
          bool USE_SINKS,
          int HEAD_SIZE,
          int QUERY_GROUP_SIZE,
          int CONTEXT_PARTITION_NUM>
__global__ __launch_bounds__(HEAD_SIZE) void pa_decode_ps_reduce_hip_kernel(
    output_t* __restrict__ output_ptr,
    const float* __restrict__ exp_sums_ptr,
    const float* __restrict__ max_logits_ptr,
    const logits_t* __restrict__ logits_ptr,
    const sink_t* __restrict__ sink_token_ptr,
    const int stride_output_bs,
    const int stride_output_len,
    const int stride_output_kv_head,
    const int stride_output_group_size,
    const int stride_exp_sums_seq,
    const int stride_exp_sums_head,
    const int stride_exp_sums_part,
    const int stride_logits_seq,
    const int stride_logits_head,
    const int stride_logits_part,
    const int stride_logits_group)
{
    static_assert(HEAD_SIZE > 0 && HEAD_SIZE <= 1024,
                  "pa_decode_ps_reduce_hip_kernel requires 0 < HEAD_SIZE <= 1024");
    static_assert(QUERY_GROUP_SIZE > 0, "query_group_size must be positive");
    static_assert(CONTEXT_PARTITION_NUM > 0 && CONTEXT_PARTITION_NUM <= kPaPsReduceWarpSize,
                  "optimized PS reduce supports 1..64 partition slots");
    constexpr int reduce_width = pa_ps_next_power_of_2(CONTEXT_PARTITION_NUM);

    const int tid        = static_cast<int>(threadIdx.x);
    const int batch_idx  = static_cast<int>(blockIdx.x);
    const int kv_head_idx = static_cast<int>(blockIdx.y);
    const int eqgs_idx   = static_cast<int>(blockIdx.z);
    const int query_idx  = eqgs_idx / QUERY_GROUP_SIZE;
    const int group_idx  = eqgs_idx - query_idx * QUERY_GROUP_SIZE;
    const int lane       = tid & (kPaPsReduceWarpSize - 1);

    const auto out_rsrc = pa_ps_make_buffer_rsrc(output_ptr);
    const auto es_rsrc = pa_ps_make_buffer_rsrc(exp_sums_ptr);
    const auto ml_rsrc = pa_ps_make_buffer_rsrc(max_logits_ptr);
    const auto logits_rsrc = pa_ps_make_buffer_rsrc(logits_ptr);

    const bool lane_in_range = lane < CONTEXT_PARTITION_NUM;
    const int part_idx_local = lane_in_range ? lane : 0;
    const int part_offset    = batch_idx * stride_exp_sums_seq +
                            kv_head_idx * stride_exp_sums_head +
                            part_idx_local * stride_exp_sums_part + eqgs_idx;
    float part_sum = 0.0f;
    float part_max = -INFINITY;
    if(lane_in_range)
    {
        const int part_byte_offset = part_offset * static_cast<int>(sizeof(float));
        part_sum = pa_ps_buffer_load<float>(es_rsrc, part_byte_offset);
        part_max = pa_ps_buffer_load<float>(ml_rsrc, part_byte_offset);
    }

    const float global_max = pa_ps_wave_reduce_max<reduce_width>(part_max);
    const bool has_finite_global_max = global_max > -FLT_MAX;
    const float safe_global_max = has_finite_global_max ? global_max : 0.0f;
    const bool valid_part = lane_in_range && part_max > -FLT_MAX;
    const float part_scale =
        valid_part ? exp2f((part_max - safe_global_max) * kPaPsReduceLog2E) : 0.0f;
    const float scaled_sum = part_sum * part_scale;

    float global_exp_sum = pa_ps_wave_reduce_sum<reduce_width>(scaled_sum);
    if constexpr(USE_SINKS)
    {
        if(sink_token_ptr != nullptr && has_finite_global_max)
        {
            const int sink_offset = kv_head_idx * QUERY_GROUP_SIZE + group_idx;
            const float sink_value = pa_ps_to_float<sink_t>(sink_token_ptr[sink_offset]);
            global_exp_sum += exp2f((sink_value - safe_global_max) * kPaPsReduceLog2E);
        }
    }

    const float safe_global_exp_sum = (global_exp_sum > 0.0f) ? global_exp_sum : 1.0f;
    const float weight_local = scaled_sum / safe_global_exp_sum;

    float acc = 0.0f;
#pragma unroll
    for(int part_idx = 0; part_idx < CONTEXT_PARTITION_NUM; ++part_idx)
    {
        const float weight = pa_ps_broadcast_weight(weight_local, part_idx);
        const int logits_offset = batch_idx * stride_logits_seq +
                                  kv_head_idx * stride_logits_head +
                                  part_idx * stride_logits_part +
                                  eqgs_idx * stride_logits_group + tid;
        const auto part_logits = pa_ps_buffer_load<logits_t>(
            logits_rsrc, logits_offset * static_cast<int>(sizeof(logits_t)));
        acc = fmaf(pa_ps_to_float<logits_t>(part_logits), weight, acc);
    }

    const int output_offset = batch_idx * stride_output_bs +
                              query_idx * stride_output_len +
                              kv_head_idx * stride_output_kv_head +
                              group_idx * stride_output_group_size + tid;
    pa_ps_buffer_store<output_t>(
        out_rsrc, output_offset * static_cast<int>(sizeof(output_t)), pa_ps_from_float<output_t>(acc));
}

} // namespace aiter
