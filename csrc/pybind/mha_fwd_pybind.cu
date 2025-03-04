// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "mha_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mha_fwd", &mha_fwd,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("dropout_p"),
          py::arg("softmax_scale"),
          py::arg("is_causal"),
          py::arg("window_size_left"),
          py::arg("window_size_right"),
          py::arg("return_softmax_lse"),
          py::arg("return_dropout_randval"),
          py::arg("out") = std::nullopt,
          py::arg("bias") = std::nullopt,
          py::arg("alibi_slopes") = std::nullopt,
          py::arg("gen") = std::nullopt);
}