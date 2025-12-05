// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "causal_conv1d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    CAUSAL_CONV1D_UPDATE_PYBIND;
}

