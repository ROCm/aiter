// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "gfx1201_gemm_a8w8.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    GFX1201_GEMM_A8W8_PYBIND;
}
