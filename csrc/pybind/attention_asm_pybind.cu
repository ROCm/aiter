// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "attention_asm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pa_fwd_asm", &pa_fwd, "pa_fwd",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("block_tables"),
          py::arg("context_lens"),
          py::arg("max_num_blocks"),
          py::arg("K_QScale") = std::nullopt,
          py::arg("V_QScale") = std::nullopt,
          py::arg("out_") = std::nullopt,
          py::arg("high_precision") = 1);
}
