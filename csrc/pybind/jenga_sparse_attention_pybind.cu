// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "jenga_sparse_attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    JENGA_SPARSE_ATTENTION_PYBIND;
}
