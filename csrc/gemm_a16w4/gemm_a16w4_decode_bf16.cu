// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a16w4_launch.h"
#include "gfx1201/gemm_a16w4_decode_bf16_gfx1201.cuh"

namespace aiter {
namespace a16w4 {

namespace {
// Threads per block of the split-K reduce. Not tuned: the reduce is launch
// latency bound (constant ~3 us across SPLIT_K = 2/4/8 at M=1 N=5120), not
// bandwidth bound.
constexpr int kReduceBlock = 256;
} // namespace

bool decode_bf16_supported(int M, int N, int K, const char** why)
{
    (void)M; // the epilogue predicates rows, so ragged M is fine here
    if(N % decode_bf16::kBlockN)
    {
        if(why)
            *why = "N % 128 != 0";
        return false;
    }
    if(K % decode_bf16::kBlockK)
    {
        if(why)
            *why = "K % 64 != 0";
        return false;
    }
    if(K % (decode_bf16::kSplitK * kGroupSize))
    {
        if(why)
            *why = "K % (SPLIT_K*128) != 0: a split would start mid quant-group";
        return false;
    }
    const int ks = K / decode_bf16::kSplitK / decode_bf16::kBlockK;
    if(ks < 2)
    {
        if(why)
            *why = "K/SPLIT_K/BK < 2: pipeline needs 2 K-steps";
        return false;
    }
    if(ks % 2)
    {
        if(why)
            *why = "K/SPLIT_K/BK is odd: main loop is unrolled x2";
        return false;
    }
    return true;
}

int64_t decode_bf16_workspace_elems(int64_t M, int64_t N)
{
    return (int64_t)decode_bf16::kSplitK * M * N;
}

void launch_decode_bf16(const void* A,
                        const void* W_q,
                        const void* scales,
                        const void* zeros,
                        float* workspace,
                        void* C,
                        int M,
                        int N,
                        int K,
                        hipStream_t stream)
{
    dim3 grid((M + decode_bf16::kBlockM - 1) / decode_bf16::kBlockM,
              N / decode_bf16::kBlockN,
              decode_bf16::kSplitK);
    dim3 block(decode_bf16::kThreads);
    hipLaunchKernelGGL(decode_bf16::gemm_a16w4_decode_bf16_kernel,
                       grid,
                       block,
                       0,
                       stream,
                       reinterpret_cast<const bf16_raw_t*>(A),
                       reinterpret_cast<const unsigned int*>(W_q),
                       reinterpret_cast<const bf16_raw_t*>(scales),
                       reinterpret_cast<const bf16_raw_t*>(zeros),
                       workspace,
                       M,
                       N,
                       K);

    const int MN = M * N;
    hipLaunchKernelGGL(decode_bf16::gemm_a16w4_decode_bf16_reduce_kernel,
                       dim3((MN + kReduceBlock - 1) / kReduceBlock),
                       dim3(kReduceBlock),
                       0,
                       stream,
                       workspace,
                       reinterpret_cast<bf16_raw_t*>(C),
                       MN);
}

} // namespace a16w4
} // namespace aiter
