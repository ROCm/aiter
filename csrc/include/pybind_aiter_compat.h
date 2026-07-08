#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Utility for pybind wrappers: shallow-converts a torch::Tensor to the
// aiter_tensor_t POD struct used by the torch-free kernel layer.  The caller
// retains ownership of the underlying storage; aiter_tensor_t holds a raw
// pointer and must not outlive the source tensor.
#include "aiter_tensor.h"
#include <torch/extension.h>

namespace aiter_pybind {

static inline aiter_tensor_t make_aiter_tensor(const torch::Tensor& t)
{
    TORCH_CHECK(t.is_cuda(), "aiter: tensor must be on a CUDA device");
    TORCH_CHECK(t.dim() <= 8, "aiter: tensor rank exceeds aiter_tensor_t maximum of 8");
    aiter_tensor_t at{};
    at.ptr      = t.data_ptr();
    at.numel_   = static_cast<size_t>(t.numel());
    at.ndim     = t.dim();
    for(int i = 0; i < t.dim(); ++i)
    {
        at.shape[i]   = t.size(i);
        at.strides[i] = t.stride(i);
    }
    at.device_id = static_cast<int>(t.device().index());
    switch(t.scalar_type())
    {
    case at::ScalarType::Float:    at.dtype_ = AITER_DTYPE_fp32;  break;
    case at::ScalarType::Half:     at.dtype_ = AITER_DTYPE_fp16;  break;
    case at::ScalarType::BFloat16: at.dtype_ = AITER_DTYPE_bf16;  break;
    case at::ScalarType::Byte:     at.dtype_ = AITER_DTYPE_u8;    break;
    case at::ScalarType::Char:     at.dtype_ = AITER_DTYPE_i8;    break;
    default:
        // fp4x2 / fp8 variants have no at::ScalarType; callers pass opaque
        // uint8-typed buffers whose semantic dtype is implicit in the op.
        at.dtype_ = AITER_DTYPE_fp4x2;
        break;
    }
    return at;
}

} // namespace aiter_pybind
