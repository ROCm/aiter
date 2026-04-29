// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "torch/fmha_fwd_f16.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    FMHA_FWD_F16_ASM_PYBIND;
}
