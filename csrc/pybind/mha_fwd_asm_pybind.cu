// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "torch/mha_v3_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MHA_FWD_ASM_PYBIND;
}