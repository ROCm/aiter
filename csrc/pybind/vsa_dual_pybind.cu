// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "vsa_dual.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    VSA_DUAL_PYBIND;
}
