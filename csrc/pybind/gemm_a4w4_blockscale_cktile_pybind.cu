// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "gemm_a4w4_blockscale_cktile.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_A4W4_BLOCKSCALE_CKTILE_PYBIND;
}
