// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "gemm_a8w8_bpreshuffle_cktile.h"
// #include "rocm_ops.hpp"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_a8w8_bpreshuffle_cktile_tune",
          &gemm_a8w8_bpreshuffle_cktile_tune,
          "gemm_a8w8_bpreshuffle_cktile_tune",
          py::arg("XQ"),
          py::arg("WQ"),
          py::arg("x_scale"),
          py::arg("w_scale"),
          py::arg("Out"),
          py::arg("kernelId") = 0,
          py::arg("splitK")   = 0);
}
