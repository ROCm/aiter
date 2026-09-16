// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "fused_moe_router.h"
#include "rocm_ops.hpp"
#include "aiter_stream.h"

PYBIND11_MODULE(module_fused_moe_router, m)
{
    AITER_SET_STREAM_PYBIND
    FUSED_MOE_ROUTER_PYBIND;
}
