// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "torch/mha_v4_fwd.h"
#include "torch/mha_v4_quant.h"

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
    m.def("rotate_activation_mxfp6_quant",
          &aiter::torch_itfs::rotate_activation_mxfp6_quant,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"),
          py::arg("multiplier"));
    m.def("rotate_activation_mxfp4_quant",
          &aiter::torch_itfs::rotate_activation_mxfp4_quant,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"),
          py::arg("multiplier"));
    m.def("rotate_activation_mxfp4_quant_k",
          &aiter::torch_itfs::rotate_activation_mxfp4_quant_k,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"));
}
