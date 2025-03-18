// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "gemm_a8w8.h"

using namespace aiter;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GEMM_A8W8_PYBIND;
}
