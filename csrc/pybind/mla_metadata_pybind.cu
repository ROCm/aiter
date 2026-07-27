// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "mla_metadata.h"
#include "rocm_ops.hpp"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { MLA_METADATA_PYBIND; }
