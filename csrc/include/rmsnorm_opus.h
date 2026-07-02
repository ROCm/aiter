// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// OPUS RMSNorm host-side launch helpers for the plain (non-quant, non
// model-sensitive) path. Torch-free / pybind-free: tensors cross as the POD
// aiter_tensor_t; the extern "C" entrypoints live in rmsnorm_opus_kernels.cu.
#pragma once
#include "aiter_hip_common.h" // AITER_CHECK, HipDeviceGuard
#include "aiter_tensor.h"     // aiter_tensor_t (torch-free POD)
#include "opus/rmsnorm_opus_kernel.hpp"

#include <cstdint>
#include <hip/hip_runtime.h>

namespace aiter {
namespace rmsnorm_opus {

// Fewer threads when there are many rows (matches csrc/kernels/rmsnorm_kernels.cu).
inline int pick_block(int rows, int hidden)
{
    const int max_block = (rows < 256) ? 1024 : 256;
    int b               = hidden < max_block ? hidden : max_block;
    return b < 1 ? 1 : b;
}

inline bool aligned16(const void* p) { return (reinterpret_cast<std::uintptr_t>(p) % 16) == 0; }

inline void check_common(const aiter_tensor_t& input, const aiter_tensor_t& weight)
{
    AITER_CHECK(input.dtype() == AITER_DTYPE_bf16 || input.dtype() == AITER_DTYPE_fp16,
                "rms_norm_opus: only bf16/fp16 supported, got dtype ",
                AiterDtype_to_str(input.dtype()));
    AITER_CHECK(weight.dtype() == input.dtype(), "rms_norm_opus: weight dtype must match input");
    AITER_CHECK(input.is_contiguous(), "rms_norm_opus: input must be contiguous");
    AITER_CHECK(input.dim() >= 1, "rms_norm_opus: input needs >= 1 dim");
    AITER_CHECK(weight.size(-1) == input.size(-1),
                "rms_norm_opus: weight length must equal hidden size");
}

template <typename scalar_t>
inline void launch_rms(aiter_tensor_t& out,
                       aiter_tensor_t& input,
                       aiter_tensor_t& weight,
                       float epsilon,
                       int rows,
                       int hidden,
                       hipStream_t stream)
{
    const int block = pick_block(rows, hidden);
    const bool vec8 = (hidden % 8 == 0) && aligned16(input.data_ptr()) &&
                      aligned16(out.data_ptr()) && aligned16(weight.data_ptr());
    if(vec8)
        hipLaunchKernelGGL((rmsnorm2d_fwd_kernel<scalar_t, 8>),
                           dim3(rows),
                           dim3(block),
                           0,
                           stream,
                           out.data_ptr(),
                           input.data_ptr(),
                           weight.data_ptr(),
                           epsilon,
                           rows,
                           hidden);
    else
        hipLaunchKernelGGL((rmsnorm2d_fwd_kernel<scalar_t, 1>),
                           dim3(rows),
                           dim3(block),
                           0,
                           stream,
                           out.data_ptr(),
                           input.data_ptr(),
                           weight.data_ptr(),
                           epsilon,
                           rows,
                           hidden);
}

template <typename scalar_t>
inline void launch_fused_add(aiter_tensor_t& input,
                             aiter_tensor_t& residual,
                             aiter_tensor_t& weight,
                             float epsilon,
                             int rows,
                             int hidden,
                             hipStream_t stream)
{
    const int block = pick_block(rows, hidden);
    const bool vec8 = (hidden % 8 == 0) && aligned16(input.data_ptr()) &&
                      aligned16(residual.data_ptr()) && aligned16(weight.data_ptr());
    if(vec8)
        hipLaunchKernelGGL((fused_add_rmsnorm2d_fwd_kernel<scalar_t, 8>),
                           dim3(rows),
                           dim3(block),
                           0,
                           stream,
                           input.data_ptr(),
                           residual.data_ptr(),
                           weight.data_ptr(),
                           epsilon,
                           rows,
                           hidden);
    else
        hipLaunchKernelGGL((fused_add_rmsnorm2d_fwd_kernel<scalar_t, 1>),
                           dim3(rows),
                           dim3(block),
                           0,
                           stream,
                           input.data_ptr(),
                           residual.data_ptr(),
                           weight.data_ptr(),
                           epsilon,
                           rows,
                           hidden);
}

} // namespace rmsnorm_opus
} // namespace aiter
