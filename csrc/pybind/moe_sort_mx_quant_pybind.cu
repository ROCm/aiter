// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "moe_sort_mx_quant.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(module_moe_sort_mx_quant, m)
{
    AITER_SET_STREAM_PYBIND
    MOE_SORT_MX_QUANT_PYBIND;
}
