// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "gemm_a8w8_blockscale_bpreshuffle.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_TUNE_PYBIND;
}
