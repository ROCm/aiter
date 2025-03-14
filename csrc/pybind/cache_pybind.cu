// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "cache.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      CACHE_PYBIND;
}
