// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "mha_fwd.h"
#include "mha_varlen_fwd.h"
#include "mha_bwd.h"
#include "mha_varlen_bwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mha_varlen_bwd", &mha_varlen_bwd,
          py::arg("dout"),
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("out"),
          py::arg("softmax_lse"),
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
          py::arg("deterministic"),
          py::arg("dq") = std::nullopt,
          py::arg("dk") = std::nullopt,
          py::arg("dv") = std::nullopt,
          py::arg("alibi_slopes") = std::nullopt,
          py::arg("rng_state") = std::nullopt,
          py::arg("gen") = std::nullopt);
}