/*
 * Copyright © Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2025, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

#include <ATen/ATen.h>

#include "aiter_hip_common.h"
#include "dispatch_utils.h"
#include "hip_compat.h"

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

namespace vllm {

namespace {
__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row, int32_t col)
{
    // don't worry about overflow because num_experts is relatively small
    return row * total_col + col;
}
} // namespace

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids,
                                            int32_t* sorted_token_ids,
                                            int32_t* expert_ids,
                                            int32_t* token_nums,
                                            int32_t* total_tokens_post_pad,
                                            int32_t num_experts,
                                            int32_t block_size,
                                            size_t numel)
{
    const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
    const size_t start_idx         = threadIdx.x * tokens_per_thread;

    extern __shared__ int32_t shared_mem[];

    // Optimized shared memory layout using only O(num_experts):
    // expert_token_counts[num_experts] - total tokens per expert
    // cumsum[num_experts + 1] - prefix sum for block offsets
    // write_positions[num_experts] - current write position per expert (for atomic writes)
    int32_t* expert_token_counts = shared_mem;
    int32_t* cumsum = shared_mem + num_experts;
    int32_t* write_positions = cumsum + (num_experts + 1);

    // Initialize expert token counts
    if(threadIdx.x < num_experts)
    {
        expert_token_counts[threadIdx.x] = 0;
    }
    __syncthreads();

    /**
     * Pass 1: Count tokens per expert using atomic operations
     * Each thread processes its shard and atomically increments expert counts
     */
    for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
    {
        int32_t expert_id = topk_ids[i];
        atomicAdd(&expert_token_counts[expert_id], 1);
    }

    __syncthreads();

    /**
     * Pass 2: Compute cumsum and initialize write positions
     * Thread 0 computes the prefix sum and initializes atomic write counters
     */
    if(threadIdx.x == 0)
    {
        cumsum[0] = 0;
        for(int i = 0; i < num_experts; ++i)
        {
            int32_t num_blocks = CEILDIV(expert_token_counts[i], block_size);
            cumsum[i + 1] = cumsum[i] + num_blocks;
            write_positions[i] = cumsum[i] * block_size; // Initialize write position
        }
        *total_tokens_post_pad = cumsum[num_experts] * block_size;
    }

    __syncthreads();

    /**
     * Pass 3: Write expert metadata
     * Each thread handles one expert (if threadIdx.x < num_experts)
     */
    if(threadIdx.x < num_experts)
    {
        int32_t num_tokens = expert_token_counts[threadIdx.x];
        for(int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i++)
        {
            expert_ids[i] = threadIdx.x;
            token_nums[i] = num_tokens;
            num_tokens -= block_size;
        }
    }

    __syncthreads();

    /**
     * Pass 4: Assign tokens to output positions
     * Each thread processes its shard and uses atomic operations to get write positions
     */
    for(int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i)
    {
        int32_t expert_id = topk_ids[i];
        // Atomically get the next write position for this expert
        int32_t rank_post_pad = atomicAdd(&write_positions[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}
} // namespace vllm

namespace aiter {

void moe_align_block_size(torch::Tensor topk_ids,
                          int64_t num_experts,
                          int64_t block_size,
                          torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor token_nums,
                          torch::Tensor num_tokens_post_pad)
{
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(topk_ids));
    const hipStream_t stream = at::hip::getCurrentHIPStream();
    VLLM_DISPATCH_INTEGRAL_TYPES(topk_ids.scalar_type(), "moe_align_block_size_kernel", [&] {
        // Optimized shared memory: O(num_experts) instead of O(num_experts^2)
        // expert_token_counts[num_experts] + cumsum[num_experts + 1] + write_positions[num_experts]
        const int32_t shared_mem = (3 * num_experts + 1) * sizeof(int32_t);

        // set dynamic shared mem
        auto kernel = vllm::moe_align_block_size_kernel<scalar_t>;
        HIP_CALL(
            VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize((void*)kernel, shared_mem));
        kernel<<<1, num_experts, shared_mem, stream>>>(topk_ids.data_ptr<scalar_t>(),
                                                       sorted_token_ids.data_ptr<int32_t>(),
                                                       experts_ids.data_ptr<int32_t>(),
                                                       token_nums.data_ptr<int32_t>(),
                                                       num_tokens_post_pad.data_ptr<int32_t>(),
                                                       num_experts,
                                                       block_size,
                                                       topk_ids.numel());
    });
}

} // namespace aiter
