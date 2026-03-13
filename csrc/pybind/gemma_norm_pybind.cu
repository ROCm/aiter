// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "gemma_norm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  GEMMA_NORM_PYBIND;
}
