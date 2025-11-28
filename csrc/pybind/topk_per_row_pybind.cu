// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "topk_per_row.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TOP_K_PER_ROW_PYBIND;
}
