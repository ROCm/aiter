// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "ck_tile_gemm_a8w8_blockscale.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    CK_TILE_GEMM_A8W8_BLOCKSCALE_PYBIND;
}
