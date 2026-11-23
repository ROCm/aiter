// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// The Gluon-order kernels of ar_k5_gfx1250.cuh. Built with IEEE fp32 denormals
// (-fno-gpu-flush-denormals-to-zero, optCompilerConfig.json), as Triton builds the Gluon
// all-reduce kernels whose bits they keep.
#include "ar_k5_gfx1250.cuh"

namespace aiter {
namespace k5ar {

void launch_gluon(const Args& a, bool fp8_out, hipStream_t stream)
{
    const dim3 grid(2 * a.rows), block(kThreads);
    if(fp8_out)
        k5_ar_rms_kernel<false, true><<<grid, block, 0, stream>>>(a);
    else
        k5_ar_rms_kernel<false, false><<<grid, block, 0, stream>>>(a);
}

} // namespace k5ar
} // namespace aiter
