// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "ar_2shot_rms_gfx1250.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    m.def("_set_current_hip_stream",
          [](int64_t stream_ptr) { aiter::setCurrentHIPStream((hipStream_t)stream_ptr); },
          py::arg("stream_ptr"));
    m.def("ar2s_rms_buffer_size", &aiter::ar2s_rms_buffer_size,
          py::arg("world_size"), py::arg("max_bytes"));
    m.def("ar2s_rms_flags_size", &aiter::ar2s_rms_flags_size);
    m.def("init_ar2s_rms", &aiter::init_ar2s_rms,
          py::arg("rank"), py::arg("data_ptrs"), py::arg("flag_ptrs"),
          py::arg("max_bytes"), py::arg("grid"));
    m.def("ar2s_rms_max_rows", &aiter::ar2s_rms_max_rows, py::arg("ctx"), py::arg("hidden"));
    m.def("ar2s_rms", &aiter::ar2s_rms,
          py::arg("ctx"), py::arg("inp"), py::arg("res_inp"), py::arg("w"),
          py::arg("out"), py::arg("res_out"), py::arg("rows"), py::arg("hidden"),
          py::arg("eps"), py::arg("gemma_norm"));
    m.def("ar2s_rms_ex", &aiter::ar2s_rms_ex,
          py::arg("ctx"), py::arg("inp"), py::arg("res_inp"), py::arg("w"),
          py::arg("out"), py::arg("res_out"), py::arg("out_q"), py::arg("out_s"),
          py::arg("rows"), py::arg("hidden"), py::arg("eps"), py::arg("gemma_norm"),
          py::arg("fp8_out"), py::arg("sp_residual"), py::arg("aux_residual"),
          py::arg("grid") = 0);
}
