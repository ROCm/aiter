// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "mhc_layer.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MHC_LAYER_PYBIND;
}
