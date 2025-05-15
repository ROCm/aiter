// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "gemm_a8w8_blockscale_wpreshuffle.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_A8W8_BLOCKSCALE_WPRESHUFFLE_PYBIND;
}
