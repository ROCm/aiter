// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "fused_split_gdr_update.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    FUSED_SPLIT_GDR_UPDATE_PYBIND;
}
