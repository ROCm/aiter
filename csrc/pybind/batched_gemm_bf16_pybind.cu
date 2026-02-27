// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "batched_gemm_bf16.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    BATCHED_GEMM_BF16_PYBIND;
}
