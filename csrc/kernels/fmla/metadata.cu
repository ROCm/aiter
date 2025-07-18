// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <limits>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "dispatch_utils.h"


__device__ constexpr int32_t get_warp_size()
{
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
    return 64;
#else
    return 32;
#endif
}

__device__ float get_overhead(
    const int32_t num_cu,
    const int32_t batch_size,
    const int32_t seqlen,
    const int32_t num_splits)
{
    constexpr float kSplitOverhead = 84.1f;

    const float bs_ratio = float(batch_size * num_splits) /
                           float((batch_size * num_splits + num_cu - 1) / num_cu) * float(num_cu);
    const float sq_ratio = float(seqlen) / float(seqlen + kSplitOverhead * num_splits);
    const float overhead = bs_ratio * sq_ratio;

    return overhead;
}

__launch_bounds__(get_warp_size())
__global__ void kn_get_mla_metadata_v0(
    int32_t*       p_num_kv_splits,
    int32_t*       p_max_num_splits,
    const int32_t* p_seqlens,
    const int32_t  num_cu,
    const int32_t  batch_size,
    const int32_t  num_heads_per_head_k,
    const int32_t  num_heads_k)
{
    constexpr int32_t kMaxSplits = 16;
    constexpr int32_t kWarpSize  = get_warp_size();

    int32_t base_scan  = 0;
    int32_t max_splits = 1;

    const int32_t num_loops = (batch_size + kWarpSize - 1) / kWarpSize;
    for (int32_t i = 0; i < num_loops; ++i)
    {
        const int32_t seqlen_idx = threadIdx.x + i * kWarpSize;
        int32_t splits = 0;

        if (seqlen_idx < batch_size)
        {
            const int32_t seqlen = p_seqlens[seqlen_idx + 1] - p_seqlens[seqlen_idx];
            float min_overhead   = std::numeric_limits<float>::max();
            #pragma unroll
            for (int32_t test_splits = 1; test_splits <= kMaxSplits; ++test_splits)
            {
                const float overhead = get_overhead(num_cu, batch_size, seqlen, test_splits);
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

        if (seqlen_idx < batch_size)
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
//   [0] num_kv_splits: (batch_size + 1), dtype torch.int32.
//   [1] max_num_splits: (1), dtype torch.int32.
//
std::vector<torch::Tensor> get_mla_metadata_v0(
    const torch::Tensor& cum_seqlens,           // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k)
{
    TORCH_CHECK(cum_seqlens.stride(0) == 1, __func__, ": cum_seqlens should be continuous!");
    TORCH_CHECK(cum_seqlens.scalar_type() == at::ScalarType::Int, __func__, ": cum_seqlens's element type should be int!");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(cum_seqlens));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t batch_size = cum_seqlens.size(0) - 1;

    // declare outputs
    auto num_kv_splits = torch::empty({batch_size + 1}, cum_seqlens.options());
    auto max_num_splits = torch::empty({1}, cum_seqlens.options());

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity
    kn_get_mla_metadata_v0<<<grid, num_thr, 0, stream>>>(
        num_kv_splits.data_ptr<int32_t>(),
        max_num_splits.data_ptr<int32_t>(),
        cum_seqlens.data_ptr<int32_t>(),
        dev_prop.multiProcessorCount,
        batch_size,
        num_heads_per_head_k,
        num_heads_k);

    return {num_kv_splits, max_num_splits};
}