#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free C-ABI for the CK batched bf16 GEMM.
//
// POD args only: the caller owns every buffer (the inputs *and* the output Y)
// and supplies the stream/device. This entry point allocates nothing -- it is
// the standalone, libtorch-free counterpart of the pybind module, mirroring the
// libmha_{fwd,bwd} torch-free model.
#include <hip/hip_runtime.h>

struct batched_gemm_bf16_args
{
    // Device pointers, all bf16.
    //   a (XQ): [B, M, K] row-major
    //   b (WQ): [B, N, K]  -- column-major GEMM operand, i.e. E = A * W^T
    //   e (Y) : [B, M, N] row-major (output, caller-allocated)
    //   bias  : currently unused by this kernel; kept for API parity.
    const void* a_ptr;
    const void* b_ptr;
    void*       e_ptr;
    const void* bias_ptr;

    int B;
    int M;
    int N;
    int K;
    int kbatch;    // split-K batch count (1 == no split-K)
    int device_id; // HIP device the buffers live on

    hipStream_t stream; // nullptr == default stream
};

#ifdef __cplusplus
extern "C" {
#endif

// Runs the batched bf16 GEMM described by *args on args->stream, selecting a CK
// kernel instance via shape heuristics. The output is written into args->e_ptr.
// Returns 0 on success, non-zero if an error was caught (exceptions are not
// allowed to cross this C boundary).
//
// visibility("default") is required: the library is built with
// -fvisibility=hidden + --gc-sections, so an unmarked entry point would be
// hidden and garbage-collected out of the .so.
__attribute__((visibility("default"))) int
aiter_batched_gemm_bf16(const batched_gemm_bf16_args* args);

#ifdef __cplusplus
}
#endif
