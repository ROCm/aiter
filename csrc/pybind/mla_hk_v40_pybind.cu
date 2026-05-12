// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "mla_hk.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    MLA_HK_V40_PYBIND;
}
