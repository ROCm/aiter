// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "split_gdr_decode_hip.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    SPLIT_GDR_DECODE_HIP_PYBIND;
}
