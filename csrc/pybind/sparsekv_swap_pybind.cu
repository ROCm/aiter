// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "sparsekv_swap.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    SPARSEKV_SWAP_PYBIND;
}
