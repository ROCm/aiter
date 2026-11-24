// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "topk_per_row.h"
#include "aiter_stream.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    TOP_K_PER_ROW_SAMPLED_PYBIND;
}
