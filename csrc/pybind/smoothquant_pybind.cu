// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "smoothquant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("smoothquant_fwd", &smoothquant_fwd);
    m.def("moe_smoothquant_fwd", &moe_smoothquant_fwd);
}
