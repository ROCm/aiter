// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "pos_encoding.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    POS_ENCODING_PYBIND;
}