// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "ar_k5_gfx1250.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    m.def("_set_current_hip_stream",
          [](int64_t stream_ptr) { aiter::setCurrentHIPStream((hipStream_t)stream_ptr); },
          py::arg("stream_ptr"));
    m.def("k5_ar_buffer_size", &aiter::k5_ar_buffer_size,
          py::arg("world_size"), py::arg("max_rows"), py::arg("hidden"));
    m.def("init_k5_ar", &aiter::init_k5_ar,
          py::arg("rank"), py::arg("inbox_ptrs"), py::arg("max_rows"), py::arg("hidden"));
    m.def("k5_ar_rms", &aiter::k5_ar_rms,
          py::arg("ctx"), py::arg("inp"), py::arg("res_inp"), py::arg("w"),
          py::arg("out"), py::arg("res_out"), py::arg("scale"), py::arg("rows"),
          py::arg("hidden"), py::arg("eps"), py::arg("gemma_norm"), py::arg("lamport_order"),
          py::arg("fp8_out"), py::arg("maxit"));
    m.def("k5_ar_rms_tail", &aiter::k5_ar_rms_tail,
          py::arg("ctx"), py::arg("res_inp"), py::arg("w"), py::arg("out"),
          py::arg("res_out"), py::arg("scale"), py::arg("rows"), py::arg("hidden"),
          py::arg("eps"), py::arg("gemma_norm"), py::arg("lamport_order"), py::arg("fp8_out"),
          py::arg("maxit"), py::arg("tail_rows"), py::arg("tail_w"), py::arg("tail_g"),
          py::arg("tail_add"), py::arg("tail_scale"));
    m.def("k5_ar_errors", &aiter::k5_ar_errors, py::arg("ctx"));
}
