// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "mha_prefill_splitk_metadata.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MHA_PREFILL_SPLITK_METADATA_PYBIND;
}
