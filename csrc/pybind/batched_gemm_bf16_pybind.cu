// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_dispatch_utils.h"
#include "rocm_ops.hpp"
#include "batched_gemm_bf16.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    BATCHED_GEMM_BF16_PYBIND;
    TUNED_CU_FALLBACK_PYBIND;
}
