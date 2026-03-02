// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Gemma norm pybind. Kernel implementation and launch in csrc/kernels/gemma_norm_kernels.cu;
// API declared in csrc/include/gemma_norm.h (same call chain as rmsnorm).
#include "rocm_ops.hpp"
#include "gemma_norm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  GEMMA_NORM_PYBIND;
}
