// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "moe_mxfp4_aux.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    MXFP4_MOE_AUX_PYBIND;
}
