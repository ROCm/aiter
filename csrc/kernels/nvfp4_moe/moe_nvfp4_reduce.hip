// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "moe_nvfp4_common.h"

// ---------------------------------------------------------------- top-k reduce
// Stage 2 kernels with Cfg::reduce store the weighted output row of each
// (token, slot) to an f32 workspace ws [T * topk, H] (no zeroing, no atomics).
// This sums the topk rows of each token in slot order and writes out [T, H]
// (bf16 or f16, round to nearest even) once, so the result is deterministic.
// One lane per 8 columns: two 16 B loads per row, one 16 B store.
constexpr int REDUCE_THREADS = 256;

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));

template <bool BF16>
__global__ __launch_bounds__(REDUCE_THREADS) void nvfp4_moe_reduce(
    const float* ws, v4i* out, int T, int H, int topk)
{
    const int vec    = blockIdx.x * REDUCE_THREADS + threadIdx.x;
    const int h_vecs = H / 8;
    if(vec >= T * h_vecs)
        return;
    const int t    = vec / h_vecs;
    const v4f* src = (const v4f*)(ws + (size_t)t * topk * H + 8 * (vec - t * h_vecs)); // row (t, 0)
    v4f lo = src[0], hi = src[1];
    for(int k = 1; k < topk; k++)
    {
        lo += src[k * H / 4];
        hi += src[k * H / 4 + 1];
    }
    const v8f s = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
    if constexpr(BF16)
        out[vec] = __builtin_bit_cast(v4i, __builtin_convertvector(s, v8bf));
    else
        out[vec] = __builtin_bit_cast(v4i, __builtin_convertvector(s, v8h));
}

extern "C" int moe_nvfp4_kernels_reduce(
    const float* ws, void* out, int T, int H, int topk, int bf16, hipStream_t stream)
{
    const dim3 grid((T * (H / 8) + REDUCE_THREADS - 1) / REDUCE_THREADS);
    if(bf16)
        hipLaunchKernelGGL(nvfp4_moe_reduce<true>,
                           grid,
                           dim3(REDUCE_THREADS),
                           0,
                           stream,
                           ws,
                           (v4i*)out,
                           T,
                           H,
                           topk);
    else
        hipLaunchKernelGGL(nvfp4_moe_reduce<false>,
                           grid,
                           dim3(REDUCE_THREADS),
                           0,
                           stream,
                           ws,
                           (v4i*)out,
                           T,
                           H,
                           topk);
    return (int)hipGetLastError();
}
