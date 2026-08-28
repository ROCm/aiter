// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    QUANT_PYBIND;
}
