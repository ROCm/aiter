// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "unified_attention_ck.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    UNIFIED_ATTENTION_CK_PYBIND;
}
