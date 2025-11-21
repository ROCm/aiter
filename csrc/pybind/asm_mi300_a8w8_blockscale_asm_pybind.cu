// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "asm_mi300_a8w8_blockscale.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("a8w8_blockscale_bpreshuffle_asm", &a8w8_blockscale_bpreshuffle_asm, "a8w8_blockscale_bpreshuffle_asm", 
        py::arg("A"),
        py::arg("B"),
        py::arg("a_scale"),
        py::arg("b_scale"),
        py::arg("Out"),
        py::arg("bias"));
}
