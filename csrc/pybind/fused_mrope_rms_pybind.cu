// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "fused_mrope_rms.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    FUSED_MROPE_RMS_PYBIND;
}
