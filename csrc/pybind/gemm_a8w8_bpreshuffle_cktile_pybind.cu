// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_a8w8_bpreshuffle_cktile.h"
#include "rocm_ops.hpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { GEMM_A8W8_BPRESHUFFLE_CKTILE_PYBIND; }
