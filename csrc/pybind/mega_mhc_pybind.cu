// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "mega_mhc.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("mega_mhc_bf16",
          &aiter::mega_mhc_bf16,
          py::arg("residual_out"),
          py::arg("post_out"),
          py::arg("comb_out"),
          py::arg("layer_input"),
          py::arg("next_pre"),
          py::arg("residual"),
          py::arg("fn_packed"),
          py::arg("hc_scale"),
          py::arg("hc_base"),
          py::arg("pre_mix"),
          py::arg("sublayer_out"),
          py::arg("post_mix"),
          py::arg("comb_mix"),
          py::arg("norm_weight"),
          py::arg("partials"),
          py::arg("counters"),
          py::arg("rms_eps"),
          py::arg("hc_pre_eps"),
          py::arg("hc_sinkhorn_eps"),
          py::arg("hc_post_mult"),
          py::arg("norm_eps"),
          py::arg("sinkhorn_repeat"),
          py::arg("split_k"),
          py::arg("has_post"),
          py::arg("identity_pre"));
}
