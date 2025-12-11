// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "asm_flatmm_a8w8_blockscale.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { FLATMM_A8W8_BLOCKSCALE_ASM_PYBIND; }
