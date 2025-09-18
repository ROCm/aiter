// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "custom_all_reduce.cuh"
#include "mla.h"


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
    int32_t        kv_granularity;
    bool           is_causal;
};

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

CK_TILE_HOST_DEVICE int32_t cal_packed_causal_kv_len(
    const int32_t qo_len,
    const int32_t kv_len,
    const int32_t qo_tile_idx,
    const int32_t packed_qo_tile_len,
    const int32_t num_qo_tiles,
    const int32_t num_heads,
    const bool    is_causal)
{
    int result = kv_len;

    if (is_causal && (qo_tile_idx < num_qo_tiles))
    {
        const int kv_len_init = kv_len - qo_len;
        const int kv_len_slop = ck_tile::integer_divide_ceil((qo_tile_idx + 1) * packed_qo_tile_len, num_heads);
        result = ck_tile::min(kv_len_init + kv_len_slop, kv_len);
    }

    return result;
}
