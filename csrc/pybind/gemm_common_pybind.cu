// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "gemm_common.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_COMMON_PYBIND;
}