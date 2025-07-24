// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "fmla.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    FLASH_MLA_FWD_PYBIND;
}
