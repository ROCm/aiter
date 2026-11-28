// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_a8w8_bpreshuffle_cktile.h"
#include "gemm_dispatch_utils.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    GEMM_A8W8_BPRESHUFFLE_CKTILE_PYBIND;
    TUNED_CU_FALLBACK_PYBIND;
}
