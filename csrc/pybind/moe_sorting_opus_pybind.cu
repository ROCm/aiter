// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "moe_sorting_opus.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    MOE_SORTING_OPUS_PYBIND;
}
