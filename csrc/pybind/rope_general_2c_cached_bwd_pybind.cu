// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "rope.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    ROPE_2C_CACHED_BWD_PYBIND;
}
