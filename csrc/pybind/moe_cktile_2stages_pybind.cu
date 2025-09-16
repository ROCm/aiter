// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "moe_cktile2stages.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      MOE_CKTILE_2STAGES_PYBIND;
}
