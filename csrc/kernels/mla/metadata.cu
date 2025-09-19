// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "v0.cuh"
#include "v1_1_device.cuh"
#include "v1_1_host.cuh"
#include "v1_2_device.cuh"

// ===================================================================================================================
// MLA Metadata V0
// ===================================================================================================================

//
// Get per batch kv split count for ASM MLA without persistent thread
// group support.
//
// Returns
//   [0] num_kv_splits:  (num_batches + 1), dtype torch.int32.
//   [1] max_num_splits: (1), dtype torch.int32.
//
std::vector<torch::Tensor> get_mla_metadata_v0(
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k)
{
    TORCH_CHECK(seqlens_kv_indptr.stride(0) == 1,
                __func__, ": seqlens_kv_indptr should be continuous!");
    TORCH_CHECK(seqlens_kv_indptr.scalar_type() == at::ScalarType::Int,
                __func__, ": seqlens_kv_indptr's element type should be int!");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(seqlens_kv_indptr));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    hipDevice_t dev;
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDevice(&dev));
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));

    const int32_t num_batches = seqlens_kv_indptr.size(0) - 1;

    // declare outputs
    auto num_kv_splits = torch::empty({num_batches + 1}, seqlens_kv_indptr.options());
    auto max_num_splits = torch::empty({1}, seqlens_kv_indptr.options());

    // launch kernel
    const dim3 grid = dim3(1, 1, 1);
    const int32_t num_thr = dev_prop.warpSize; // only use 1 warp for simplicity
    kn_get_mla_metadata_v0<<<grid, num_thr, 0, stream>>>(
        num_kv_splits.data_ptr<int32_t>(),
        max_num_splits.data_ptr<int32_t>(),
        seqlens_kv_indptr.data_ptr<int32_t>(),
        dev_prop.multiProcessorCount,
        num_batches,
        num_heads_per_head_k,
        num_heads_k);

    return {num_kv_splits, max_num_splits};
}

// ===================================================================================================================
// MLA Metadata V1
// ===================================================================================================================

//
// Persistent thread group solution which take variable query/output lengths into consideration as well.
//
// Returns
//   [0] work_metadata_ptrs  (2)                 Two 64-bits pointers point to the 1st element of work_indptr and
//                                               work_info.
//   [1] work_info           (#work, 8)
//   [1.0] bs_index:         (#work),            The index of batch handled by each work.
//   [1.1] partial_index:    (#work),            The index of tile in output buffer when splits. -1 means no split.
//   [1.2] q_start:          (#work),            The global index in seq where q/o starts. Use global index here can
//                                               reduce memory access count in kernel.
//   [1.3] q_end:            (#work),            The global index in seq where q/o ends (not included).
//   [1.4] kv_start:         (#work),            The global index in seq where k/v starts.
//   [1.5] kv_end:           (#work),            The global index in seq where k/v ends (not included).
//   [1.6] pad               (#work, 2),         Pad to 8 DWs.
//   [2] work_indptr:        (#cu_part + 1),     The IDs of work handled by each cu_part.
//   [3] reduce_indptr:      (sum(qo_seqlen_blk_count) + 1),
//                                               The IDs in reduce_partial_map indicates the tiles should be merged
//                                               together.
//   [4] reduce_final_map:   (sum(qo_seqlen_blk_count)),
//                                               The final output location of each group of tiles.
//   [5] reduce_partial_map: (#partial_tiles),   The locations in partial buffer of partial tiles waiting for being
//                                               reduced.
//
void get_mla_metadata_v1(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    torch::Tensor&       work_metadata_ptrs,
    torch::Tensor&       work_info_set,
    torch::Tensor&       work_indptr,
    torch::Tensor&       reduce_indptr,
    torch::Tensor&       reduce_final_map,
    torch::Tensor&       reduce_partial_map,
    std::optional<std::map<std::string, int32_t>> split_params)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(seqlens_kv_indptr));

    int32_t kv_granularity = 16;
    int32_t max_seqlen_qo  = -1;
    int32_t uni_seqlen_qo  = -1;
    bool    fast_mode      = false;

    if (split_params.has_value())
    {
        if (split_params->find("kv_granularity") != split_params->end())
        {
            kv_granularity = split_params->find("kv_granularity")->second;
        }

        if (split_params->find("max_seqlen_qo") != split_params->end())
        {
            max_seqlen_qo = split_params->find("max_seqlen_qo")->second;
        }

        if (split_params->find("uni_seqlen_qo") != split_params->end())
        {
            uni_seqlen_qo = split_params->find("uni_seqlen_qo")->second;
            max_seqlen_qo = (uni_seqlen_qo > 0) ? uni_seqlen_qo : max_seqlen_qo;
        }

        if (split_params->find("fast_mode") != split_params->end())
        {
            fast_mode = split_params->find("fast_mode")->second != 0;
        }
    }

    if (fast_mode)
    {
        get_mla_metadata_v1_2_device(
            seqlens_qo_indptr,
            seqlens_kv_indptr,
            num_heads_per_head_k,
            num_heads_k,
            is_causal,
            kv_granularity,
            max_seqlen_qo,
            uni_seqlen_qo,
            work_metadata_ptrs,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map);
    }
    else
    {
        // This default settings is for our ASM MLA decode kernel. This kernel supports num_heads=16 and qo size from 1
        // to 4 without support to split qo for each workgroup. This means that kPackedQoLenPerWg should be 4*16=64 to
        // prevent spliting in any case supported by it.
        //                                PackedQoLenPerWg, MaxClusterSize
        using Traits  = MlaMetadataTraits<128,              1>;

        get_mla_metadata_v1_1_device<Traits>(
            seqlens_qo_indptr,
            seqlens_kv_indptr,
            num_heads_per_head_k,
            num_heads_k,
            is_causal,
            false,
            kv_granularity,
            max_seqlen_qo,
            work_metadata_ptrs,
            work_info_set,
            work_indptr,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map);
    }
}

std::vector<torch::Tensor> get_mla_metadata_v1_no_redundant(
    const torch::Tensor& seqlens_qo_indptr,     // [batch size + 1]
    const torch::Tensor& seqlens_kv_indptr,     // [batch size + 1]
    const int32_t        num_heads_per_head_k,
    const int32_t        num_heads_k,
    const bool           is_causal,
    const int32_t        kv_granularity)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(seqlens_kv_indptr));

    // This default settings is for our ASM MLA decode kernel. This kernel supports num_heads=16 and qo size from 1 to 4
    // without support to split qo for each workgroup. This means that kPackedQoLenPerWg should be 4*16=64 to prevent
    // spliting in any case supported by it.
    //                                PackedQoLenPerWg, MaxClusterSize
    using Traits  = MlaMetadataTraits<64,               1>;

    return get_mla_metadata_v1_1_host<Traits>(
        seqlens_qo_indptr,
        seqlens_kv_indptr,
        num_heads_per_head_k,
        num_heads_k,
        is_causal,
        kv_granularity,
        true);
}
