// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

struct __attribute__((packed)) TopKDecodeKernelArgs
{
    void* ptr_workspace;
    void* ptr_logits;
    void* ptr_rowStarts;
    void* ptr_rowEnds;
    void* ptr_indices;
    void* ptr_values;
    int32_t stride0;
    int32_t stride1;
};

template <typename T, typename IdxT, int kNumThreadsPerBlock = 1024>
int64_t invokePrefillTopKLastDimWorkspaceSize(int32_t numRows, int32_t topkValue)
{
    return topkValue * (sizeof(T) + sizeof(IdxT)) * numRows;
}

void top_k_per_row_prefill_fast(const torch::Tensor& logits,
                                const torch::Tensor& rowStarts,
                                const torch::Tensor& rowEnds,
                                torch::Tensor& indices,
                                std::optional<torch::Tensor> values,
                                int64_t numRows,
                                int64_t stride0,
                                int64_t stride1)
{
    // Compute workspace size and allocate workspace tensor
    const auto numColumns             = logits.size(1);
    int64_t workspace_size   = invokePrefillTopKLastDimWorkspaceSize<float, int32_t>(numRows, 2048);
    auto options            = torch::TensorOptions().dtype(torch::kUInt8).device(logits.device());
    torch::Tensor workspace = torch::empty({workspace_size}, options);
    
    TopKDecodeKernelArgs args;
    size_t arg_size = sizeof(args);
    
    args.ptr_workspace  = static_cast<void*>(workspace.data_ptr<uint8_t>());
    args.ptr_logits     = logits.data_ptr<float>();
    args.ptr_rowStarts  = rowStarts.data_ptr<int>();
    args.ptr_rowEnds    = rowEnds.data_ptr<int>();
    args.ptr_indices    = indices.data_ptr<int>();
    args.ptr_values     = nullptr;
    args.stride0        = static_cast<int32_t>(stride0);
    args.stride1        = static_cast<int32_t>(stride1);

    // Load the compiled assembly kernel
    static AiterAsmKernel impl_topk_decode(
        "_ZN5aiter11PrefillTopKL10topKPerRowILi1024ELi2048ELi2048ELi512EEEvPvPKfPKiS6_PiPfii",
        "/topk_per_row_prefill/asm_top_k_per_row_prefill.co");

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(logits));
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // Launch kernel configuration
    constexpr int kNumThreadsPerBlock = 1024;
    uint64_t gdx = numRows;
    
    TORCH_CHECK(gdx >> 31 == 0, "numRows too large: ", numRows);
    
    impl_topk_decode.launch_kernel({&args,
                                    &arg_size,
                                    static_cast<int>(gdx),  // gdx: one block per row
                                    1,                      // gdy
                                    1,                      // gdz
                                    kNumThreadsPerBlock,    // bdx: 1024 threads
                                    1,                      // bdy
                                    1,                      // bdz
                                    stream});
}

