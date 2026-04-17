// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "chunk_gated_delta_rule_fwd_h.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    CHUNK_GDR_FWD_H_PYBIND;
}
