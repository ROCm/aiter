// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
// MHC Layer pybind11 bindings for aiter

#include "rocm_ops.hpp"
#include "mhc_layer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MHC_LAYER_PYBIND;
}
