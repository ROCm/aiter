/* SPDX-License-Identifier: MIT
   Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
*/
#include "moe_op.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // Register enums with exception handling to support both cases:
    // 1. Same internal version (pybind11 == PyTorch): enums already registered, catch exception
    // 2. Different internal versions: enums need to be registered locally
    try {
        AITER_ENUM_PYBIND;
    } catch (const std::runtime_error&) {
        // Enums already registered (same internal version case)
        // This is expected and safe to ignore
    }
    MOE_OP_PYBIND;
}