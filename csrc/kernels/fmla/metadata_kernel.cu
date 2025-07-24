
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <limits>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "dispatch_utils.h"

#define ROUND(a, b) ((a + b / 2) / b)
#define CEIL(a, b)  ((a + b - 1) / b)

__device__ constexpr int32_t get_warp_size()
{
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
    return 64;
#else
    return 32;
#endif
}

__launch_bounds__(64, 2)
__global__ void kn_get_mla_metadata(
    const int32_t* kv_indptr,
    int32_t*       num_kv_splits,
    int32_t*       batch_split_table,
    int32_t*       split_table,
    const int32_t  batch_size,
    const int32_t  cu_num,
    const int32_t  fixed_blocked_len
    )
{
    const int32_t tidx = threadIdx.x;

    constexpr int32_t max_batch_size = 200;
    constexpr int32_t max_cu_num = 480;

    constexpr int32_t max_local_arr_size = 200;
    __shared__ int32_t kv_seq_les[max_batch_size];
    __shared__ int32_t num_kv_splits_shard[max_batch_size + 1];
    int32_t template_data_local[max_local_arr_size];
    __shared__ int32_t batch_split_table_shared[max_cu_num];
    __shared__ int32_t split_table_shared[max_cu_num];

    int64_t total_kv_pad = 0;
    if (tidx < batch_size) 
    {
        int32_t kv_len = kv_indptr[tidx + 1] - kv_indptr[tidx];
        kv_seq_les[tidx] = kv_len;
        total_kv_pad = static_cast<int64_t>(CEIL(kv_len, 16) * 16);
    }

    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        int64_t tmp = __shfl_xor(total_kv_pad, stride);
        total_kv_pad += tmp;
    }
    __syncthreads();


    int32_t split_size_pad      = static_cast<int32_t>(CEIL(total_kv_pad, cu_num) + fixed_blocked_len);
    int32_t split_size_pad_half = split_size_pad / 2;

    if (tidx < batch_size) 
    {
        int32_t num_kv_splits_cur_batch = CEIL(kv_seq_les[tidx], split_size_pad);
        num_kv_splits_shard[tidx] = num_kv_splits_cur_batch;
    }

    __syncthreads();

    if (tidx == 0)
    {
        int32_t split_size_pad_copy = split_size_pad_half;
        // int32_t split_shift = 1;
        // while (split_size_pad_copy >>= 1)
        // {
        //     split_shift++;
        // }

        template_data_local[0] = 0;
        for (int i = 1; i < batch_size + 1; ++i)
        {
            template_data_local[i] = template_data_local[i - 1] + num_kv_splits_shard[i - 1];
        }

        int32_t fix_size = template_data_local[batch_size] - cu_num;
        int32_t sign = (fix_size > 0) ? 1 : -1;
        int32_t fixed_size = 0;

        int32_t double_split_size_pad = 2 * split_size_pad;
        if (fix_size > 0)
        {
            for (int i = 1; i < batch_size + 1; ++i)
            {
                int32_t cur_seq_len = kv_seq_les[i - 1];
                // if ((cur_seq_len > split_size_pad) && ((cur_seq_len >> split_shift) <= split_size_pad_half))
                if ((fix_size != fixed_size) && (cur_seq_len > split_size_pad) && ((cur_seq_len % split_size_pad) <= split_size_pad_half))
                {
                    fixed_size += sign;
                }
                template_data_local[i] -= fixed_size;
            }
        }
        else if (fix_size < 0)
        {
            for (int i = 1; i < batch_size + 1; ++i)
            {
                int32_t cur_seq_len = kv_seq_les[i - 1];
                // if ((cur_seq_len > 3 * split_size_pad) && ((cur_seq_len >> split_shift) > split_size_pad_half))
                if ((fix_size != fixed_size) && (cur_seq_len > 3 * split_size_pad) && ((cur_seq_len % split_size_pad) > split_size_pad_half))
                {
                    fixed_size += sign;
                }
                template_data_local[i] -= fixed_size;
            }
        }
        int32_t end_dim = batch_size;
        int32_t fixed_gap = template_data_local[batch_size] - cu_num;

        while (fixed_gap != 0)
        {
            template_data_local[end_dim] -= fixed_gap;
            if (kv_seq_les[end_dim - 1] > double_split_size_pad)
            {
                fixed_gap -= sign;
            }
            end_dim -= 1;
        }

        for (int i = 0; i < batch_size + 1; ++i)
            num_kv_splits_shard[i] = template_data_local[i];

        __syncthreads();
        //TODO: maybe move to cpu but how?
        int split_idx = 0;
        int b_idx = 0;
        for (int i = 0; i < cu_num; ++i)
        {
            if (i < template_data_local[b_idx + 1])
            {
                batch_split_table_shared[i] = b_idx;
                split_table_shared[i] = split_idx;
            }
            else
            {
                split_idx = 0;
                b_idx +=  1;
                batch_split_table_shared[i] = b_idx;
                split_table_shared[i] = split_idx;

            }
            split_idx += 1;
        }
    }

    __syncthreads();


    for (int i = tidx; i < batch_size + 1; i += blockDim.x)
    {
        num_kv_splits[i] = num_kv_splits_shard[i];
    }

    for (int i = tidx; i < cu_num; i += blockDim.x)
    {
        batch_split_table[i] = batch_split_table_shared[i];
        split_table[i] = split_table_shared[i];
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
std::vector<torch::Tensor> get_mla_metadata_impl(
    const torch::Tensor& kv_indptr,            // [batch size + 1]
    torch::Tensor&       num_kv_splits_indptr, // [batch size + 1]
    torch::Tensor&       batch_split_table,    // [max_cu_num]
    torch::Tensor&       split_table)          // [max_cu_num]
{
    TORCH_CHECK(kv_indptr.scalar_type() == at::ScalarType::Int, __func__, ": kv_indptr's element type should be int!");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_indptr));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t batch_size = kv_indptr.size(0) - 1;
    const int32_t cu_num =
        ROUND(batch_size, 16) * dev_prop.multiProcessorCount;
    auto opt = kv_indptr.options();

    // declare outputs
    auto num_kv_splits = num_kv_splits_indptr.data_ptr() ?
        num_kv_splits_indptr :
        torch::empty({batch_size + 1}, opt);

    auto batch_split_table_ptr = batch_split_table.data_ptr() ?
        batch_split_table :
        torch::empty({cu_num}, opt);

    auto split_table_ptr = split_table.data_ptr() ?
        split_table :
        torch::empty({cu_num}, opt);

    constexpr int32_t fixed_blocked_len = 80;

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity
    kn_get_mla_metadata<<<grid, num_thr, 0, stream>>>(
        kv_indptr.data_ptr<int32_t>(),
        num_kv_splits.data_ptr<int32_t>(),
        batch_split_table_ptr.data_ptr<int32_t>(),
        split_table_ptr.data_ptr<int32_t>(),
        batch_size,
        cu_num,
        fixed_blocked_len);

    return {num_kv_splits_indptr, batch_split_table, split_table};
}

