// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "mhc.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    MHC_PYBIND;
    m.def("mhc_post_project_gfx950",
          &aiter::mhc_post_project_gfx950,
          py::arg("projection"),
          py::arg("sq"),
          py::arg("next_residual"),
          py::arg("x"),
          py::arg("residual"),
          py::arg("packed_fn"),
          py::arg("post_mix"),
          py::arg("comb_mix"));
    m.def("mhc_finalize_gfx950",
          &aiter::mhc_finalize_gfx950,
          py::arg("post"),
          py::arg("comb"),
          py::arg("out"),
          py::arg("projection"),
          py::arg("sq"),
          py::arg("scale"),
          py::arg("base"),
          py::arg("residual"),
          py::arg("weight"),
          py::arg("rms_eps"),
          py::arg("pre_eps"),
          py::arg("sink_eps"),
          py::arg("norm_eps"),
          py::arg("post_mult"),
          py::arg("repeats"));
}
