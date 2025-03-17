// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "batched_gemm_bf16.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("batched_gemm_bf16", &batched_gemm_bf16, "batched_gemm_bf16", py::arg("XQ"), py::arg("WQ"),
           py::arg("Out"), py::arg("bias") = std::nullopt, py::arg("splitK") = 0);
}
