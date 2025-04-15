// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include "rocm_ops.hpp"
#include "aiter_enum.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_ENUM_PYBIND;
}
