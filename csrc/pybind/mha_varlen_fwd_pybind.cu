// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "mha_varlen_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    MHA_VARLEN_FWD_PYBIND;
}