// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "asm_gemm_a4w4.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_A4W4_ASM_PYBIND;
}