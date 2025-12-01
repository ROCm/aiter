// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include "asm_a8w8_blockscale_bpreshuffle.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("gemm_a8w8_blockscale_bpreshuffle_asm",
          &gemm_a8w8_blockscale_bpreshuffle_asm,
          "FP8 blockscale BpreShuffle GEMM assembly implementation",
          py::arg("A"),
          py::arg("B"),
          py::arg("out"),
          py::arg("A_scale"),
          py::arg("B_scale"),
          py::arg("bias") = py::none(),
          py::arg("splitK") = py::none(),
          py::arg("kernelName") = py::none(),
          py::arg("bpreshuffle") = true);
}
