// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "mhc.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    MHC_PYBIND;
    m.def("mhc_post_pre_gram_gfx950",
          &aiter::mhc_post_pre_gram_gfx950,
          py::arg("post"),
          py::arg("comb"),
          py::arg("out"),
          py::arg("next_residual"),
          py::arg("partials"),
          py::arg("layer_input"),
          py::arg("residual_in"),
          py::arg("post_mix"),
          py::arg("comb_mix"),
          py::arg("fn"),
          py::arg("hc_scale"),
          py::arg("hc_base"),
          py::arg("weight"),
          py::arg("rms_eps"),
          py::arg("pre_eps"),
          py::arg("sink_eps"),
          py::arg("norm_eps"),
          py::arg("post_mult"),
          py::arg("parts") = 4);
}
