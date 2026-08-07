// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "torch/mha_v4_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fmha_v4_fwd",
          &aiter::torch_itfs::fmha_v4_fwd,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("q_descale"),
          py::arg("k_descale"),
          py::arg("v_descale"),
          py::arg("out"),
          py::arg("q_format"),
          py::arg("k_format"),
          py::arg("v_format"),
          py::arg("q_scale_mode"),
          py::arg("k_scale_mode"),
          py::arg("v_scale_mode"),
          py::arg("softmax_scale"));
}