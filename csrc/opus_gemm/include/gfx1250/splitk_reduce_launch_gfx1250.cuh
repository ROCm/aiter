// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 split-K reduce HOST launch dispatcher.
//
// Maps the runtime split_k (chosen by the GEMM tuner) to the matching
// COMPILE-TIME reduce instantiation (SPLIT_K_ template arg) so the reduce's
// split-K accumulation loop is fully unrolled (triton MAX_KSPLIT idiom). The
// generated per-kid launcher calls opus_splitk_reduce_launch_gfx1250<...>()
// with a single clean call (no split_k switch in the fragile codegen template).
//
// The reduce kernel definition lives in splitk_reduce_gfx1250.cuh and is
// explicitly instantiated (per SPLIT_K value + D_WS) in the dedicated
// splitk_reduce_gfx1250.device.cu TU; here we only forward-declare it and issue
// the <<<>>> launch, so this header is host-only.
#pragma once
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)

#include "opus_gemm_traits_a16w16_gfx1250.cuh"
#include <hip/hip_runtime.h>

// Forward declaration of the reduce kernel (definition in splitk_reduce_gfx1250.cuh;
// explicit instantiations in splitk_reduce_gfx1250.device.cu). Must match the
// kernel's full template signature (8 params incl. SPLIT_K_ and D_WS_).
template <int VEC_,
          int BLOCK_,
          typename D_OUT,
          bool HAS_BIAS_,
          typename D_BIAS_,
          bool HAS_OOB_,
          int SPLIT_K_,
          typename D_WS_>
__global__ void splitk_reduce_kernel_gfx1250(const void* ws_ptr,
                                             D_OUT* c_out,
                                             int split_k,
                                             int M,
                                             int N,
                                             int batch,
                                             int padded_M,
                                             int padded_N,
                                             const D_BIAS_* bias,
                                             int stride_bias_batch);

// Launch the reduce, dispatching the runtime split_k to a compile-time
// SPLIT_K_ instance (1..16 fully-unrolled; other values fall back to the
// SPLIT_K_=0 runtime-`split_k` loop). D_WS is the workspace (partial-sum)
// dtype the main GEMM wrote (bf16 by default on gfx1250).
template <int VEC,
          int BLOCK,
          typename D_OUT,
          bool HAS_BIAS,
          typename D_BIAS,
          bool HAS_OOB,
          typename D_WS>
inline void opus_splitk_reduce_launch_gfx1250(dim3 grid,
                                              dim3 block,
                                              hipStream_t stream,
                                              const void* ws_ptr,
                                              D_OUT* c_out,
                                              int split_k,
                                              int M,
                                              int N,
                                              int batch,
                                              int padded_M,
                                              int padded_N,
                                              const D_BIAS* bias,
                                              int stride_bias_batch)
{
#define OPUS_RDK(SK)                                                                     \
    splitk_reduce_kernel_gfx1250<VEC, BLOCK, D_OUT, HAS_BIAS, D_BIAS, HAS_OOB, SK, D_WS> \
        <<<grid, block, 0, stream>>>(                                                    \
            ws_ptr, c_out, split_k, M, N, batch, padded_M, padded_N, bias, stride_bias_batch)
    switch(split_k)
    {
    case 1: OPUS_RDK(1); break;
    case 2: OPUS_RDK(2); break;
    case 3: OPUS_RDK(3); break;
    case 4: OPUS_RDK(4); break;
    case 5: OPUS_RDK(5); break;
    case 6: OPUS_RDK(6); break;
    case 7: OPUS_RDK(7); break;
    case 8: OPUS_RDK(8); break;
    case 9: OPUS_RDK(9); break;
    case 10: OPUS_RDK(10); break;
    case 11: OPUS_RDK(11); break;
    case 12: OPUS_RDK(12); break;
    case 13: OPUS_RDK(13); break;
    case 14: OPUS_RDK(14); break;
    case 15: OPUS_RDK(15); break;
    case 16: OPUS_RDK(16); break;
    default: OPUS_RDK(0); break; // runtime-`split_k` loop fallback
    }
#undef OPUS_RDK
}

#endif // host-only
