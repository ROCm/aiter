// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "minimax_m3_fused_qknorm_rope_cache_shuffle.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    MINIMAX_M3_FUSED_QKNORM_ROPE_CACHE_SHUFFLE_PYBIND;
}
