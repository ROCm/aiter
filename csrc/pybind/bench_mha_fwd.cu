// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "bench_mha_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    BENCH_MHA_FWD_PYBIND;
}