// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_a8w8_blockscale.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_a8w8_blockscale", &gemm_a8w8_blockscale, "fp8 blockscale gemm", py::arg("XQ"), py::arg("WQ"),
          py::arg("x_scale"), py::arg("w_scale"), py::arg("Out"));
}
