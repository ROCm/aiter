// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "fused_dsv41_mxfp8.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("fused_dsv41_mxfp8_fwd",
          &fused_dsv41_mxfp8_fwd,
          py::arg("q"),
          py::arg("pool"),
          py::arg("prefix"),
          py::arg("prefix_indptr"),
          py::arg("cos"),
          py::arg("sin"),
          py::arg("positions"),
          py::arg("out"),
          py::arg("out_scale"),
          py::arg("max_logits"),
          py::arg("lse"),
          py::arg("extend_kv"),
          py::arg("extend"),
          py::arg("extend_indptr"),
          py::arg("attn_sink"),
          py::arg("debug_out"),
          py::arg("softmax_scale"),
          py::arg("q_is_rotated"),
          py::arg("cache_scales") = py::none(),
          py::arg("cache_page_size") = 0,
          py::arg("cache_num_rows") = 0,
          py::arg("cache_data_page_stride") = 0,
          py::arg("cache_scale_page_stride") = 0);
}
