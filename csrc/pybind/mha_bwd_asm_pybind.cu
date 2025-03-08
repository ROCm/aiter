// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_v3_bwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fmha_v3_bwd", &fmha_v3_bwd,
          py::arg("dout"),
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("out"),
          py::arg("softmax_lse"),
          py::arg("dropout_p"),
          py::arg("softmax_scale"),
          py::arg("is_causal"),
          py::arg("window_size_left"),
          py::arg("window_size_right"),
          py::arg("deterministic"),
          py::arg("is_v3_atomic_fp32"),
          py::arg("how_v3_bf16_cvt"),
          py::arg("dq") = std::nullopt,
          py::arg("dk") = std::nullopt,
          py::arg("dv") = std::nullopt,
          py::arg("alibi_slopes") = std::nullopt,
          py::arg("rng_state") = std::nullopt,
          py::arg("gen") = std::nullopt);
}