// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "m3_tp1_prefill.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    M3_TP1_NORM_PYBIND;
}
