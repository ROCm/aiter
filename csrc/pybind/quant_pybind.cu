// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("static_scaled_fp8_quant", &static_scaled_fp8_quant);
    m.def("dynamic_scaled_fp8_quant", &dynamic_scaled_fp8_quant);
    m.def("dynamic_per_token_scaled_fp8_quant", &dynamic_per_token_scaled_fp8_quant,
          py::arg("out"), py::arg("input"),
          py::arg("scales"), py::arg("scale_ub") = std::nullopt);
}
