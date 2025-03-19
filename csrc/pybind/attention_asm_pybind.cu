// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "attention_asm.h"

using namespace aiter;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    ATTENTION_ASM_PYBIND;
}
