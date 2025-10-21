// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "paged_mqa_logits_metadata.h"

#include <c10/cuda/CUDAGuard.h>

template <typename T>
static constexpr T align(const T& a, const T& b) {
    return (a + b - 1) / b * b;
}

template <int32_t kAlignedBatchSize, int32_t SPLIT_KV, int32_t kNumCUs>
__global__ __launch_bounds__(64, 1)
void paged_mqa_logits_metadata_impl(const int32_t batch_size, const int32_t* context_lens, int32_t* schedule_metadata) {
    static_assert(kAlignedBatchSize % 64 == 0);
    const int64_t lane_idx = threadIdx.x;

    int64_t num_segs[kAlignedBatchSize / 64];
#pragma unroll
    for (int64_t k = 0; k < kAlignedBatchSize / 64; ++ k) {
        const int64_t& context_len = (k * 64 + lane_idx < batch_size ? context_lens[k * 64 + lane_idx] : 0);
        num_segs[k] = (context_len + SPLIT_KV - 1) / SPLIT_KV;
    }

    __shared__ int64_t prefix_sum[kAlignedBatchSize];
    int64_t sum = 0;
#pragma unroll
    for (int64_t k = 0; k < kAlignedBatchSize / 64; ++ k) {
        int64_t x = num_segs[k];
#pragma unroll
        for (int64_t offset = 1; offset < 64; offset <<= 1) {
            const int64_t& y = __shfl_up(x, offset);
            x += (lane_idx >= offset ? y : 0);
        }
        x += sum;
        prefix_sum[k * 64 + lane_idx] = x;
        sum = __shfl(x, 63);
    }

    const int64_t& q = sum / kNumCUs, r = sum % kNumCUs;
    for (int64_t sm_idx = lane_idx; sm_idx <= kNumCUs; sm_idx += 64) {
        int64_t seg_starts = sm_idx * q + min(sm_idx, r);
        int64_t q_idx = 0;
        while (q_idx < batch_size and prefix_sum[q_idx] <= seg_starts)
            ++ q_idx;
        const int32_t& kv_split_idx = (q_idx == 0 ? seg_starts : seg_starts - prefix_sum[q_idx - 1]);
        __syncthreads();

        schedule_metadata[sm_idx * 2] = q_idx;
        schedule_metadata[sm_idx * 2 + 1] = kv_split_idx;
    }
}

void paged_mqa_logits_metadata(const torch::Tensor& context_lens,
                                      const torch::Tensor& schedule_metadata,
                                      const int& batch_size,
                                      const int& block_kv,
                                      const int& num_cu) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(schedule_metadata));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // cudaDeviceProp_t prop;
    // cudaGetDeviceProperties(&prop, 0);
    //
    constexpr int num_math_warpgroups = 4;
    constexpr int num_threads = 64;
    const int aligned_batch_size = align(batch_size, 64);
    const int split_kv = block_kv * num_math_warpgroups;

    // Calculate shared memory size
    const int smem_size = aligned_batch_size * static_cast<int>(sizeof(int));

    // Launch
    paged_mqa_logits_metadata_impl<64, 64 * 4, 80><<<1, num_threads, smem_size, stream>>>(
        batch_size,
        context_lens.data_ptr<int32_t>(),
        schedule_metadata.data_ptr<int32_t>()
    );
}
