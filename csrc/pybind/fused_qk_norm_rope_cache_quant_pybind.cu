// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "fused_qk_norm_rope_cache_quant.h"
#include "rocm_ops.hpp"
#include "aiter_stream.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    FUSED_QKNORM_ROPE_CACHE_QUANT_PYBIND;
}
