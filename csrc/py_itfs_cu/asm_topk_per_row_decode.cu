// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_hip_common.h"
#include "py_itfs_common.h"

struct __attribute__((packed)) TopKDecodeKernelArgs
{
    void* ptr_logits;
    void* ptr_seqLens;
    void* ptr_outIndices;
    int32_t stride0;
    int32_t stride1;
    int32_t next_n;
};

AITER_C_ITFS void top_k_per_row_decode_fast(
    aiter_tensor_t* logits,
    int64_t next_n,
    aiter_tensor_t* seqLens,
    aiter_tensor_t* indices,
    int64_t numRows,
    int64_t stride0,
    int64_t stride1,
    hipStream_t stream)
{
    const HipDeviceGuard device_guard(logits->device_id);

    TopKDecodeKernelArgs args;
    size_t arg_size = sizeof(args);

    args.ptr_logits     = logits->data_ptr();
    args.ptr_seqLens    = seqLens->data_ptr();
    args.ptr_outIndices = indices->data_ptr();
    args.stride0        = static_cast<int32_t>(stride0);
    args.stride1        = static_cast<int32_t>(stride1);
    args.next_n         = static_cast<int32_t>(next_n);

    static AiterAsmKernel impl_topk_decode(
        "_ZN5aiter10DecodeTopKL19topk_per_row_decodeILi1024ELb0ELi4EEEvPKfPKiPiiii",
        "/topk_per_row_decode/asm_top_k_per_row_decode.co");

    constexpr int kNumThreadsPerBlock = 1024;
    AITER_CHECK(numRows >> 31 == 0, "numRows too large: ", numRows);

    impl_topk_decode.launch_kernel({&args,
                                    &arg_size,
                                    static_cast<int>(numRows),
                                    1, 1,
                                    kNumThreadsPerBlock,
                                    1, 1,
                                    stream});
}
