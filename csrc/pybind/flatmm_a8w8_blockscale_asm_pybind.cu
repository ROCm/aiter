// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "asm_flatmm_a8w8_blockscale.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flatmm_a8w8_blockscale_asm", &flatmm_a8w8_blockscale_asm, "flatmm_a8w8_blockscale_asm", 
        py::arg("XQ"),
        py::arg("WQ"),
        py::arg("x_scale"),
        py::arg("w_scale"),
        py::arg("Out"));
}
