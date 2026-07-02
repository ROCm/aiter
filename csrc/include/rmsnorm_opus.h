// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm host-side launch helpers. Fully torch-free / HIP-runtime-free:
// only opus/hip_minimal.hpp (host launch) + opus.hpp (device, via the kernel
// header). Tensors cross the ctypes boundary as raw pointers + dims, so nothing
// pulls <hip/hip_runtime.h> or aiter_tensor.h.
#pragma once
#include "opus/hip_minimal.hpp" // dim3, hipStream_t, hipLaunchKernelGGL (both passes)
#include "opus/rmsnorm_opus_kernel.hpp"

namespace aiter {
namespace rmsnorm_opus {

// Fewer threads when there are many rows (matches csrc/kernels/rmsnorm_kernels.cu).
inline int pick_block(int rows, int hidden)
{
    const int max_block = (rows < 256) ? 1024 : 256;
    int b               = hidden < max_block ? hidden : max_block;
    return b < 1 ? 1 : b;
}

inline bool aligned16(const void* p) { return (reinterpret_cast<size_t>(p) % 16) == 0; }

template <typename scalar_t>
inline void launch_rms(void* out,
                       const void* in,
                       const void* weight,
                       float epsilon,
                       int rows,
                       int hidden,
                       hipStream_t stream)
{
    const int block = pick_block(rows, hidden);
    const bool vec8 = (hidden % 8 == 0) && aligned16(out) && aligned16(in) && aligned16(weight);
    if(vec8)
        hipLaunchKernelGGL((rmsnorm2d_fwd_kernel<scalar_t, 8>),
                           dim3(rows),
                           dim3(block),
                           0,
                           stream,
                           out,
                           in,
                           weight,
                           epsilon,
                           rows,
                           hidden);
    else
        hipLaunchKernelGGL((rmsnorm2d_fwd_kernel<scalar_t, 1>),
                           dim3(rows),
                           dim3(block),
                           0,
                           stream,
                           out,
                           in,
                           weight,
                           epsilon,
                           rows,
                           hidden);
}

template <typename scalar_t>
inline void launch_fused_add(void* inout,
                             void* residual,
                             const void* weight,
                             float epsilon,
                             int rows,
                             int hidden,
                             hipStream_t stream)
{
    const int block = pick_block(rows, hidden);
    const bool vec8 =
        (hidden % 8 == 0) && aligned16(inout) && aligned16(residual) && aligned16(weight);
    if(vec8)
        hipLaunchKernelGGL((fused_add_rmsnorm2d_fwd_kernel<scalar_t, 8>),
                           dim3(rows),
                           dim3(block),
                           0,
                           stream,
                           inout,
                           residual,
                           weight,
                           epsilon,
                           rows,
                           hidden);
    else
        hipLaunchKernelGGL((fused_add_rmsnorm2d_fwd_kernel<scalar_t, 1>),
                           dim3(rows),
                           dim3(block),
                           0,
                           stream,
                           inout,
                           residual,
                           weight,
                           epsilon,
                           rows,
                           hidden);
}

} // namespace rmsnorm_opus
} // namespace aiter
