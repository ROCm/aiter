// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <queue>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include "aiter_hip_common.h"
#include "custom_all_reduce.cuh"
#include "mla.h"

// ======================================================= flash mla style metadata ==================================================

// =====================================================================================================================
// Definitions and helper structures
//

template <int32_t kSizeD_,
          int32_t kSizeDV_,
          int32_t kBlockM_,
          int32_t kBlockN_,
          int32_t kNumWarps_>
struct FlashMlaKernelTrait
{
    static constexpr int32_t kSizeD                  = kSizeD_;    // hidden dimension size of query and key
    static constexpr int32_t kSizeDV                 = kSizeDV_;   // hidden dimension size of value
    static constexpr int32_t kNumWarps               = kNumWarps_;
    static constexpr int32_t kNumThreads             = kNumWarps * warpSize;
    static constexpr int32_t kNumWarpsSoftmax        = 4;
    static constexpr int32_t kNumThreadsSoftmax      = kNumWarpsSoftmax * warpSize;
    static constexpr int32_t kBlockM                 = kBlockM_;
    static constexpr int32_t kBlockN                 = kBlockN_;
    static constexpr int32_t kFixedOverheadNumBlocks = 16;
    static constexpr int32_t kMaxBatchSize           = 4096;

    static_assert(kSizeD % 64 == 0);
    static_assert(kSizeDV % 64 == 0);
    static_assert(kSizeD >= kSizeDV);
};

// using FlashMlaKernelTraitsInstance = FlashMlaKernelTrait<576, 512, 64, 64, 4>;
using FlashMlaKernelTraitsInstance = FlashMlaKernelTrait<192, 128, 64, 16, 4>;

union TileSchedulerMetaData
{
    struct Core
    {
        int32_t batch_idx;
        int32_t partial_idx;
        int32_t begin_seqlen_q_idx;
        int32_t end_seqlen_q_idx;
        int32_t begin_seqlen_kv_idx;
        int32_t end_seqlen_kv_idx;
        int32_t end_seqlen_kv_idx_to_end;
    };
    uint32_t data[8];
};
constexpr size_t TileSchedulerMetaDataSizeInDw = sizeof(TileSchedulerMetaData) / sizeof(int32_t);


union FmlaTileSchedulerMetaData
{
    struct Core
    {
        int32_t begin_batch_idx;
        int32_t begin_seqlen_idx;
        int32_t end_batch_idx;
        int32_t end_seqlen_idx;
        int32_t begin_n_split_idx;
    };
    uint32_t data[8];
};

struct FlashMlaFwdParams
{
    int32_t* __restrict__ p_cu_seqlens_k;
    int32_t* __restrict__ p_block_table;
    int32_t* __restrict__ p_work_indptr;
    int32_t* __restrict__ p_tile_scheduler_metadata;
    
    void* __restrict__ p_query;
    void* __restrict__ p_key;
    void* __restrict__ p_value;
    void* __restrict__ p_output;
    void* __restrict__ p_softmax_lse;
    void* __restrict__ p_softmax_lseaccum;
    void* __restrict__ p_output_accum;

    int32_t size_b;
    int32_t size_s;
    int32_t size_h;
    int32_t hq_hk_ratio;
    int32_t num_groups;
    int32_t num_cu_parts;
    int64_t block_table_batch_stride;
    int32_t page_block_size;
    float   scale_softmax;
    float   scale_softmax_log2;
    bool    is_causal;

    // Use int64_t if there is int32 overflow case. For now, just use int32 to save sgpr and prevent using
    // spill table.
    using index_t = int32_t;

    index_t stride_b_q;     // stride in batch of query
    index_t stride_s_q;     //    ... in sequence ...
    index_t stride_h_q;     //    ... in head ...
    index_t stride_b_k;     // stride in batch of key
    index_t stride_s_k;     //    ... in sequence ...
    index_t stride_h_k;     //    ... in head ...
    index_t stride_b_v;     // stride in batch of value
    index_t stride_s_v;     //    ... in sequence ...
    index_t stride_h_v;     //    ... in head ...
    index_t stride_b_o;     // stride in batch of output
    index_t stride_s_o;     //    ... in sequence ...
    index_t stride_h_o;     //    ... in head ...
};

// =====================================================================================================================
// Kernel Entries
//


__inline__ __device__ void transfer_metadata_into_work_info(
    int32_t& info_idx,
    int32_t& partial_idx,
    FmlaTileSchedulerMetaData::Core& fmla_metadata,

    int32_t* p_reduce_indptr,
    int32_t* p_reduce_final_map,
    int32_t* p_reduce_partial_map,

    const int32_t* p_seqlens_k,
    const int32_t* p_seqlens_kv_indptr,
    const int32_t* p_seqlens_qo_indptr,
    int32_t* p_tile_scheduler_metadata)
{
    int32_t info_size = fmla_metadata.end_batch_idx - fmla_metadata.begin_batch_idx + 1;
    int32_t qo_len = p_seqlens_qo_indptr[1] - p_seqlens_qo_indptr[0];

    for (int i = 0; i < info_size; ++i)
    {
        int32_t batch_idx = fmla_metadata.begin_batch_idx + i;
        int32_t seqlen_k  = p_seqlens_k[batch_idx];
        TileSchedulerMetaData::Core metadata;
        metadata.batch_idx = batch_idx;
        metadata.begin_seqlen_q_idx = batch_idx * qo_len;
        metadata.end_seqlen_q_idx   = (batch_idx + 1) * qo_len;

        metadata.begin_seqlen_kv_idx = i == 0 ? fmla_metadata.begin_seqlen_idx : 0;
        metadata.end_seqlen_kv_idx = i == info_size - 1 ? fmla_metadata.end_seqlen_idx : seqlen_k;

        metadata.end_seqlen_kv_idx_to_end = seqlen_k - metadata.end_seqlen_kv_idx;


        metadata.partial_idx = metadata.end_seqlen_kv_idx == seqlen_k && metadata.begin_seqlen_kv_idx == 0 ? -1 : partial_idx;

        if (metadata.end_seqlen_kv_idx == seqlen_k && metadata.begin_seqlen_kv_idx == 0)
        {
            p_reduce_indptr[batch_idx + 1] = p_reduce_indptr[batch_idx];
            p_reduce_final_map[batch_idx * 2] = -2;
            p_reduce_final_map[batch_idx * 2 + 1] = -1;
        }
        else
        {
            int reduce_cur_idx = metadata.partial_idx / qo_len;
            p_reduce_indptr[batch_idx + 1] = reduce_cur_idx + 1;
            p_reduce_final_map[batch_idx * 2]     = p_seqlens_qo_indptr[batch_idx];
            p_reduce_final_map[batch_idx * 2 + 1] = p_seqlens_qo_indptr[batch_idx + 1];
            p_reduce_partial_map[reduce_cur_idx] = metadata.partial_idx;
            partial_idx += qo_len;
        }
        metadata.begin_seqlen_kv_idx += p_seqlens_kv_indptr[batch_idx];
        metadata.end_seqlen_kv_idx += p_seqlens_kv_indptr[batch_idx];

        *reinterpret_cast<TileSchedulerMetaData::Core*>(
            p_tile_scheduler_metadata + (i + info_idx)* TileSchedulerMetaDataSizeInDw) = metadata;

    }

    info_idx += info_size;
}

template <typename Traits>
__global__ void kn_get_mla_metadata(
    uint64_t*      p_work_metadata,
    int32_t*       p_tile_scheduler_metadata,
    int32_t*       p_work_indptr,
    int32_t*       p_reduce_indptr,
    int32_t*       p_reduce_final_map,
    int32_t*       p_reduce_partial_map,
    const int32_t* p_seqlens_qo_indptr,
    const int32_t* p_seqlens_kv_indptr,
    // int32_t*       p_tile_scheduler_metadata_ori,
    const int32_t  max_reduce_size,
    const int32_t  batch_size,
    const int32_t  num_cu_parts)
{
    __shared__ int lds_num_blocks[Traits::kMaxBatchSize];
    __shared__ int lds_num_splits[Traits::kMaxBatchSize];
    __shared__ int lds_seqlens_k[Traits::kMaxBatchSize];

    int32_t sum_blocks = 0;
    for (int32_t i = threadIdx.x; i < batch_size; i += warpSize)
    {
        int32_t seqlen_k = p_seqlens_kv_indptr[i + 1] - p_seqlens_kv_indptr[i];
        const int32_t num_blocks = ck_tile::integer_divide_ceil(seqlen_k, Traits::kBlockN);
        sum_blocks += num_blocks;
        lds_num_blocks[i] = num_blocks;
        lds_seqlens_k[i]  = seqlen_k;
    }
    __syncthreads();

    for (int32_t offset = 32; offset > 0; offset >>= 1)
    {
        sum_blocks += __shfl_xor(sum_blocks, offset);
    }
    __syncthreads();

    sum_blocks += batch_size * Traits::kFixedOverheadNumBlocks;

    if (threadIdx.x == 0)
    {
        // expected payload handled by each cu part.
        const int32_t payload = ck_tile::integer_divide_ceil(sum_blocks, num_cu_parts) +
                                Traits::kFixedOverheadNumBlocks;

        int32_t curr_batch = 0;         // batch ID of the batch which is under review
        int32_t curr_block = 0;         // #blocks handled by previous cu part(s)
        int32_t curr_n_split_idx = 0;   // #cu parts used to handle current batch
        int32_t cum_num_splits = 0;
        p_reduce_indptr[0] = 0;

        lds_num_splits[0] = 0;

        int info_idx = 0;
        int partial_idx = 0;
        for (int32_t i = 0; i < num_cu_parts; ++i)
        {
            FmlaTileSchedulerMetaData::Core metadata;
            metadata.begin_batch_idx   = curr_batch;
            metadata.begin_seqlen_idx  = curr_block * Traits::kBlockN;
            metadata.begin_n_split_idx = curr_n_split_idx;

            int remain_payload = payload;
            while (curr_batch < batch_size)
            {
                const int32_t num_blocks = lds_num_blocks[curr_batch];
                const int32_t curr_remain_blocks = num_blocks - curr_block;

                // If current cu part is able to handle this batch of seqences
                if (remain_payload >= (curr_remain_blocks + Traits::kFixedOverheadNumBlocks))
                {
                    cum_num_splits += (curr_n_split_idx + 1);
                    remain_payload -= (curr_remain_blocks + Traits::kFixedOverheadNumBlocks);
                    ++curr_batch;
                    curr_block = 0;
                    curr_n_split_idx = 0;
                }
                else
                {
                    if (remain_payload > Traits::kFixedOverheadNumBlocks)
                    {
                        curr_block += (remain_payload - Traits::kFixedOverheadNumBlocks);
                        ++curr_n_split_idx;
                    }
                    break;
                }
            }

            metadata.end_batch_idx  = curr_block > 0 ? curr_batch : (curr_batch - 1);
            metadata.end_seqlen_idx = curr_block > 0 ? curr_block * Traits::kBlockN : lds_seqlens_k[curr_batch - 1];

            transfer_metadata_into_work_info(
                info_idx,
                partial_idx,
                metadata,

                p_reduce_indptr,
                p_reduce_final_map,
                p_reduce_partial_map,

                lds_seqlens_k,
				p_seqlens_kv_indptr,
				p_seqlens_qo_indptr,
                p_tile_scheduler_metadata
            );

            lds_num_splits[i + 1] = info_idx;
            // *reinterpret_cast<FmlaTileSchedulerMetaData::Core*>(
            //     p_tile_scheduler_metadata_ori + i * TileSchedulerMetaDataSizeInDw) = metadata;
        }

        p_work_metadata[0] = (uint64_t)(p_work_indptr);
        p_work_metadata[1] = (uint64_t)(p_tile_scheduler_metadata);

        int32_t reduce_indptr_tail = p_reduce_indptr[batch_size];

        for (int32_t i = batch_size; i < max_reduce_size; ++i)
        {
            p_reduce_indptr[i] = reduce_indptr_tail;
        }
    }

    for (int32_t i = threadIdx.x; i <= num_cu_parts; i += warpSize)
    {
        p_work_indptr[i] = lds_num_splits[i];
    }
}





// =====================================================================================================================
// Dispatches
//

template <typename Traits>
void dispatch_get_mla_metadata(
    uint64_t*      p_work_metadata,
    int32_t*       p_tile_scheduler_metadata,
    int32_t*       p_work_indptr,
    int32_t*       p_reduce_indptr,
    int32_t*       p_reduce_final_map,
    int32_t*       p_reduce_partial_map,
    const int32_t* p_seqlens_qo_indptr,
    const int32_t* p_seqlens_kv_indptr,
    // int32_t*       p_tile_scheduler_metadata_ori,
    const int32_t  max_reduce_size,
    const int32_t  batch_size,
    const int32_t  num_cu_parts)
{
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const uint32_t grid  = 1;
    const uint32_t block = warpSize;

    kn_get_mla_metadata<Traits><<<grid, block, 0, stream>>>(
        p_work_metadata,
        p_tile_scheduler_metadata,
        p_work_indptr,
        p_reduce_indptr,
        p_reduce_final_map,
        p_reduce_partial_map,
        p_seqlens_qo_indptr,
        p_seqlens_kv_indptr,
        // p_tile_scheduler_metadata_ori,
        max_reduce_size,
        batch_size,
        num_cu_parts);
}

template <typename Traits, typename scalar_t>
void dispatch_fmla_fwd_splictkv_combine(
    const FlashMlaFwdParams& params)
{

}

#define DISPATCH_TYPES(TYPE, NAME, ...)                 \
    switch ((TYPE))                                     \
    {                                                   \
        case at::ScalarType::BFloat16:                  \
        {                                               \
            using scalar_t = at::BFloat16;              \
            __VA_ARGS__;                                \
            break;                                      \
        }                                               \
        case at::ScalarType::Half:                      \
        {                                               \
            using scalar_t = at::Half;                  \
            __VA_ARGS__;                                \
            break;                                      \
        }                                               \
        default:                                        \
            TORCH_CHECK(false, NAME " does't support ", \
                        toString((TYPE)), ".");         \
    }

void get_mla_metadata_v1_2_device(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,         // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    torch::Tensor& work_meta_data,
    torch::Tensor& work_info_set_tsr,
    torch::Tensor& work_indptr_tsr,
    torch::Tensor& reduce_indptr_tsr,
    torch::Tensor& reduce_final_map_tsr,
    torch::Tensor& reduce_partial_map_tsr)
{
    using Traits = FlashMlaKernelTraitsInstance;

    const torch::TensorOptions tensor_options = seqlens_kv_indptr.options();
    const int32_t batch_size = seqlens_kv_indptr.size(0) - 1;
    const int32_t max_reduce_size = reduce_indptr_tsr.size(0);
    assert(batch_size <= Traits::kMaxBatchSize);

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    hipGetDevice(&dev);
    hipGetDeviceProperties(&dev_prop, dev);
    // const int32_t cu_count = dev_prop.multiProcessorCount;
    const int32_t cu_count = dev_prop.multiProcessorCount;
    const int32_t cu_parts = cu_count / num_heads_k /
                             ck_tile::integer_divide_ceil(num_heads_per_head_k, Traits::kBlockM);

    dispatch_get_mla_metadata<Traits>(
        work_meta_data.data_ptr<uint64_t>(),
        work_info_set_tsr.data_ptr<int32_t>(),
        work_indptr_tsr.data_ptr<int32_t>(),
        reduce_indptr_tsr.data_ptr<int32_t>(),
        reduce_final_map_tsr.data_ptr<int32_t>(),
        reduce_partial_map_tsr.data_ptr<int32_t>(),
        seqlens_qo_indptr.data_ptr<int32_t>(),
        seqlens_kv_indptr.data_ptr<int32_t>(),
        max_reduce_size,
        batch_size,
        cu_parts);
}
