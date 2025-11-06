// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "mla.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MLA_REDUCE_PYBIND;
}