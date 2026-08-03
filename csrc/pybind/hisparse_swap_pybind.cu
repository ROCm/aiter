// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "hisparse_swap.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    HISPARSE_SWAP_PYBIND;
}
