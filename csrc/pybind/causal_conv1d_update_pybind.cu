// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "causal_conv1d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    CAUSAL_CONV1D_UPDATE_PYBIND;
}
