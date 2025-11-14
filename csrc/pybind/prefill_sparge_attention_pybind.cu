// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "prefill_sparge_attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    PREFILL_SPARGE_ATTENTION_PYBIND;
}
