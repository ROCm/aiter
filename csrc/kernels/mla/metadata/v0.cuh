// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <torch/python.h>
#include <ATen/hip/HIPContext.h>
#include "aiter_hip_common.h"

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

    const int32_t num_loops = ck_tile::integer_divide_ceil(num_batches, kWarpSize);
    for (int32_t i = 0; i < num_loops; ++i)
    {
        const int32_t bid = threadIdx.x + i * kWarpSize;
        int32_t splits = 0;

        if (bid < num_batches)
        {
            const int32_t seqlen = p_seqlens[bid + 1] - p_seqlens[bid];
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
            const int32_t remote = ck_tile::warp_shuffle_up(scan, offset);
            scan += (threadIdx.x >= offset) ? remote : 0;
        }

        const int32_t global_scan = scan + base_scan;

        if (bid < num_batches)
        {
            p_num_kv_splits[bid + 1] = global_scan;
        }

        // update base_scan
        base_scan = ck_tile::warp_shuffle(global_scan, kWarpSize - 1);
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
