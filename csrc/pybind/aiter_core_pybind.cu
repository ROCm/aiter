// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_tensor.h"
#include "aiter_stream.h"

PYBIND11_MODULE(module_aiter_core, m)
{
    AITER_CORE_PYBIND;
}
