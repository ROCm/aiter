// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "py_itfs_common.h"
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

struct __attribute__((packed)) TopKDecodeKernelArgs
{
    void* ptr_logits;
    void* ptr_seqLens;
    void* ptr_outIndices;
    int32_t stride0;
    int32_t stride1;
    int32_t next_n;
};

void top_k_per_row_decode_fast(const torch::Tensor& logits,
                               int64_t next_n,
                               const torch::Tensor& seqLens,
                               torch::Tensor& indices,
                               int64_t numRows,
                               int64_t stride0,
                               int64_t stride1)
{
    TopKDecodeKernelArgs args;
    size_t arg_size = sizeof(args);
    
    args.ptr_logits     = logits.data_ptr<float>();
    args.ptr_seqLens    = seqLens.data_ptr<int>();
    args.ptr_outIndices = indices.data_ptr<int>();
    args.stride0        = static_cast<int32_t>(stride0);
    args.stride1        = static_cast<int32_t>(stride1);
    args.next_n         = static_cast<int32_t>(next_n);

    // Load the compiled assembly kernel
    // The mangled name: _ZN5aiter10DecodeTopKL19topk_per_row_decodeILi1024ELb0ELi4EEEvPKfPKiPiiii
    // corresponds to: aiter::DecodeTopK::topk_per_row_decode<1024, false, 4>
    static AiterAsmKernel impl_topk_decode(
        "_ZN5aiter10DecodeTopKL19topk_per_row_decodeILi1024ELb0ELi4EEEvPKfPKiPiiii",
        "/topk_per_row_decode/asm_top_k_per_row_decode.co");

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

