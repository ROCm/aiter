#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <hip/hip_runtime.h>
#include <string>

namespace aiter {

__attribute__((visibility("default")))
aiter_tensor_t& gemm_a4w4_blockscale(aiter_tensor_t& XQ,
                                      aiter_tensor_t& WQ,
                                      aiter_tensor_t& x_scale,
                                      aiter_tensor_t& w_scale,
                                      aiter_tensor_t& Y,
                                      int splitK,
                                      hipStream_t stream,
                                      std::string kernelName = "");

__attribute__((visibility("default")))
aiter_tensor_t& gemm_a4w4_blockscale_tune(aiter_tensor_t& XQ,
                                           aiter_tensor_t& WQ,
                                           aiter_tensor_t& x_scale,
                                           aiter_tensor_t& w_scale,
                                           aiter_tensor_t& Y,
                                           int kernelId,
                                           int splitK,
                                           hipStream_t stream);

} // namespace aiter
