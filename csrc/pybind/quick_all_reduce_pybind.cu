// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "quick_all_reduce.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      QUICK_ALL_REDUCE_PYBIND;
}