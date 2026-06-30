// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "opus_gdn_chunk_prepare.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    GDN_CHUNK_PREPARE_PYBIND;
}
