#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <hip/hip_runtime.h>
#include <string>

namespace aiter {

// Library entry points: the caller owns the stream.
__attribute__((visibility("default"))) aiter_tensor_t&
gemm_a4w4_blockscale(aiter_tensor_t& XQ,
                     aiter_tensor_t& WQ,
                     aiter_tensor_t& x_scale,
                     aiter_tensor_t& w_scale,
                     aiter_tensor_t& Y,
                     int splitK,
                     hipStream_t stream,
                     std::string kernelName = "");

__attribute__((visibility("default"))) aiter_tensor_t&
gemm_a4w4_blockscale_tune(aiter_tensor_t& XQ,
                          aiter_tensor_t& WQ,
                          aiter_tensor_t& x_scale,
                          aiter_tensor_t& w_scale,
                          aiter_tensor_t& Y,
                          int kernelId,
                          int splitK,
                          hipStream_t stream);

namespace torch_itfs {

// Pybind-facing wrappers. Torch-free: @compile_ops(..., develop=True) hands these
// aiter_tensor_t and pushes the caller's stream through _set_current_hip_stream(),
// which getCurrentHIPStream() picks up. They write Y in place; the Python wrapper
// in aiter/ops/gemm_op_a4w4.py returns it so the public API still yields a Tensor.
void gemm_a4w4_blockscale(aiter_tensor_t& XQ,
                          aiter_tensor_t& WQ,
                          aiter_tensor_t& x_scale,
                          aiter_tensor_t& w_scale,
                          aiter_tensor_t& Y,
                          int splitK,
                          std::string kernelName = "");

void gemm_a4w4_blockscale_tune(aiter_tensor_t& XQ,
                               aiter_tensor_t& WQ,
                               aiter_tensor_t& x_scale,
                               aiter_tensor_t& w_scale,
                               aiter_tensor_t& Y,
                               int kernelId,
                               int splitK);

} // namespace torch_itfs
} // namespace aiter
