// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "asm_mi300_a8w8_blockscale.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mi300_a8w8_blockscale_asm", &mi300_a8w8_blockscale_asm, "mi300_a8w8_blockscale_asm", 
        py::arg("A"),
        py::arg("B"),
        py::arg("a_scale"),
        py::arg("b_scale"),
        py::arg("Out"),
        py::arg("bias"));
}
