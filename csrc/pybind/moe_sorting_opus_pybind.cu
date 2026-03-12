// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "moe_sorting_opus.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MOE_SORTING_OPUS_PYBIND;
}
