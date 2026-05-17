// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "pa_gqa_v5.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    PA_GQA_V5_PYBIND;
}
