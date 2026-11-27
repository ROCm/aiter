// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "rocm_ops.hpp"
#include "splitk_reduce_qk_rmsnorm.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    SPLITK_REDUCE_QK_RMSNORM_PYBIND;
}
