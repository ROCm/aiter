// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_dispatch_utils.h"
#include "rocm_ops.hpp"
#include "gemm_a8w8.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    GEMM_A8W8_PYBIND;
    TUNED_CU_FALLBACK_PYBIND;
}
