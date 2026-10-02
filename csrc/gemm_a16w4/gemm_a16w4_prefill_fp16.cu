// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a16w4_launch.h"
#include "gfx1201/gemm_a16w4_prefill_fp16_gfx1201.cuh"

namespace aiter {
namespace a16w4 {

bool prefill_fp16_supported(int M, int N, int K, const char** why)
{
    if(M % prefill_fp16::kBlockM)
    {
        if(why)
            *why = "M % 128 != 0";
        return false;
    }
    if(N % prefill_fp16::kBlockN)
    {
        if(why)
            *why = "N % 512 != 0";
        return false;
    }
    if(K % prefill_fp16::kBlockK)
    {
        if(why)
            *why = "K % 64 != 0";
        return false;
    }
    if(K % kGroupSize)
    {
        if(why)
            *why = "K % 128 != 0";
        return false;
    }
    const int ks = K / prefill_fp16::kBlockK;
    if(ks < 2 || (ks % 2))
    {
        if(why)
            *why = "K/BK must be even and >= 2";
        return false;
    }
    return true;
}

void launch_prefill_fp16(const void* A,
                         const void* W_q,
                         const void* scales,
                         const void* zeros_biased,
                         void* C,
                         int M,
                         int N,
                         int K,
                         hipStream_t stream)
{
    dim3 grid(M / prefill_fp16::kBlockM, N / prefill_fp16::kBlockN);
    dim3 block(prefill_fp16::kThreads);
    hipLaunchKernelGGL(prefill_fp16::gemm_a16w4_prefill_fp16_kernel,
                       grid,
                       block,
                       0,
                       stream,
                       reinterpret_cast<const fp16_raw_t*>(A),
                       reinterpret_cast<const unsigned int*>(W_q),
                       reinterpret_cast<const fp16_raw_t*>(scales),
                       reinterpret_cast<const fp16_raw_t*>(zeros_biased),
                       reinterpret_cast<fp16_raw_t*>(C),
                       M,
                       N,
                       K);
}

} // namespace a16w4
} // namespace aiter
