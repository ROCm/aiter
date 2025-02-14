// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_sorting.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_sorting_fwd", &moe_sorting_fwd);
}
