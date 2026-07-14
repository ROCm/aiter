// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "custom_all_reduce_gfx1250.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_set_current_hip_stream",
          [](int64_t stream_ptr) { aiter::setCurrentHIPStream((hipStream_t)stream_ptr); },
          py::arg("stream_ptr"));
    m.def("init_custom_ar", &aiter::init_custom_ar,
          py::arg("meta_ptr"), py::arg("rank_data_ptr"), py::arg("rank_data_sz"),
          py::arg("all_meta_ptrs"), py::arg("rank"),
          py::arg("fully_connected"));
    m.def("all_reduce", &aiter::all_reduce,
          py::arg("_fa"), py::arg("inp"), py::arg("out"),
          py::arg("use_new"), py::arg("open_fp8_quant"),
          py::arg("reg_inp_ptr"), py::arg("reg_inp_bytes"));
    m.def("reduce_scatter", &aiter::reduce_scatter,
          py::arg("_fa"), py::arg("inp"), py::arg("out"),
          py::arg("m"), py::arg("n"), py::arg("k"),
          py::arg("split_dim"),
          py::arg("reg_ptr"), py::arg("reg_bytes"));
    m.def("all_gather", &aiter::all_gather,
          py::arg("_fa"), py::arg("inp"), py::arg("out"),
          py::arg("dim"), py::arg("reg_inp_ptr"), py::arg("reg_inp_bytes"));
    m.def("p2p_bw_test", &aiter::p2p_bw_test,
          py::arg("_fa"), py::arg("inp"), py::arg("out"),
          py::arg("unroll"), py::arg("threads"), py::arg("blocks"),
          py::arg("reg_inp_ptr"), py::arg("reg_inp_bytes"));
    m.def("dispose", &aiter::dispose, py::arg("_fa"));
    m.def("meta_size", &aiter::meta_size);
    m.def("register_input_buffer", &aiter::register_input_buffer,
          py::arg("_fa"), py::arg("self_ptr"), py::arg("all_ptrs"));
    m.def("register_output_buffer", &aiter::register_output_buffer,
          py::arg("_fa"), py::arg("self_ptr"), py::arg("all_ptrs"));
    m.def("get_graph_buffer_count", &aiter::get_graph_buffer_count, py::arg("_fa"));
    m.def("get_graph_buffer_ptrs", &aiter::get_graph_buffer_ptrs,
          py::arg("_fa"), py::arg("ptrs_out"));
    m.def("register_graph_buffers", &aiter::register_graph_buffers,
          py::arg("_fa"), py::arg("ptrs_per_rank"));
    m.def("start_sync_latency", &aiter::start_sync_latency,
          py::arg("_fa"), py::arg("blocks"));
    m.def("end_sync_latency", &aiter::end_sync_latency,
          py::arg("_fa"), py::arg("blocks"));
    m.def("two_sync_latency", &aiter::two_sync_latency,
          py::arg("_fa"), py::arg("blocks"));
}
