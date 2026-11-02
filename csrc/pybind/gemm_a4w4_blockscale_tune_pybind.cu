// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// Torch-free TU; see gemm_a4w4_blockscale_pybind.cu for the rationale.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "gemm_a4w4_blockscale.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    GEMM_A4W4_BLOCKSCALE_TUNE_PYBIND;
}
