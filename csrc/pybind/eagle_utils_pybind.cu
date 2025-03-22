// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "eagle_utils.h"
#include "rocm_ops.hpp"

using namespace aiter;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  EAGLE_UTILS_PYBIND;
}