// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_a8w8_bpreshuffle_cktile.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_a8w8_bpreshuffle_cktile",
          &gemm_a8w8_bpreshuffle_cktile,
          "gemm_a8w8_bpreshuffle_cktile",
          py::arg("XQ"),
          py::arg("WQ"),
          py::arg("x_scale"),
          py::arg("w_scale"),
          py::arg("Out"));
}
