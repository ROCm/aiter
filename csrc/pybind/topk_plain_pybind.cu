/* SPDX-License-Identifier: MIT
   Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
*/
#include "topk_plain.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    TOPK_PLAIN_PYBIND;
}
