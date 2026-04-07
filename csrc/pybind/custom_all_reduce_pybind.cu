// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "custom_all_reduce.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      CUSTOM_ALL_REDUCE_PYBIND;
}