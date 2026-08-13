// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "mha_native.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MHA_FWD_NATIVE_SPLITKV_PYBIND;
}
