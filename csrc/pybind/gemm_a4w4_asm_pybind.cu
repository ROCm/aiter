// SPDX-License-Identifier: MIT
<<<<<<< HEAD
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
=======
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
>>>>>>> origin/main
#include "rocm_ops.hpp"
#include "asm_gemm_a4w4.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_A4W4_ASM_PYBIND;
}