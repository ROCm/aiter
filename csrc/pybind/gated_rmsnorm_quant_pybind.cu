// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "rocm_ops.hpp"
#include "gated_rmsnorm_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    GATED_RMSNORM_QUANT_PYBIND
}
