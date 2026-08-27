// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "inverse_rope_group_quant.h"
#include "rocm_ops.hpp"
#include "aiter_stream.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    INVERSE_ROPE_GROUP_QUANT_PYBIND;
}
