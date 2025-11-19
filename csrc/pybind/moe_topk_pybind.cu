/* SPDX-License-Identifier: MIT
   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
*/
#include "moe_op.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MOE_TOPK_PYBIND;
}