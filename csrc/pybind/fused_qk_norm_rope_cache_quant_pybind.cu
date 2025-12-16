// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "fused_qk_norm_rope_cache_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    FUSED_QKNORM_ROPE_CACHE_QUANT_PYBIND;
}