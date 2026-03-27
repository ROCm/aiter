// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rocm_ops.hpp"
#include "aiter_tensor.h"

PYBIND11_MODULE(module_aiter_tensor, m)
{
    AITER_TENSOR_PYBIND;
}
