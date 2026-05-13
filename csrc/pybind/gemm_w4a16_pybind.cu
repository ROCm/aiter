// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "gemm_w4a16.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_W4A16_PYBIND;
}
