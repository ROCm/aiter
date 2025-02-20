// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "mha_varlen_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mha_varlen_fwd", &mha_varlen_fwd,
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("cu_seqlens_q"),
          py::arg("cu_seqlens_k"),
          py::arg("max_seqlen_q"),
          py::arg("max_seqlen_k"),
          py::arg("dropout_p"),
          py::arg("softmax_scale"),
          py::arg("zero_tensors"),
          py::arg("is_causal"),
          py::arg("window_size_left"),
          py::arg("window_size_right"),
          py::arg("return_softmax_lse"),
          py::arg("return_dropout_randval"),
          py::arg("out") = std::nullopt,
          py::arg("block_table") = std::nullopt,
          py::arg("alibi_slopes") = std::nullopt,
          py::arg("gen") = std::nullopt);
}