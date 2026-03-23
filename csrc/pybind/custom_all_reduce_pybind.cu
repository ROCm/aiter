// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "custom_all_reduce.h"
#include "communication_asm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      CUSTOM_ALL_REDUCE_PYBIND;
}