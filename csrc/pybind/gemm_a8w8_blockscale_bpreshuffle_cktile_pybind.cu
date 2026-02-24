// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_a8w8_blockscale_bpreshuffle_cktile.h"
#include "gemm_a8w8_blockscale_cktile.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_CKTILE_PYBIND; }
