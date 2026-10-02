// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Self-contained pybind (does not include rocm_ops.hpp / aiter_tensor.h, which
// would pull ck_tile into the gfx1250 device compile).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "aiter_stream.h"
#include "custom_all_reduce_ifoe.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "_set_current_hip_stream",
        [](int64_t stream_ptr) { aiter::setCurrentHIPStream((hipStream_t)stream_ptr); },
        py::arg("stream_ptr"));
    m.def("ifoe_alloc_fabric",
          &aiter::ifoe_alloc_fabric,
          py::arg("bytes"),
          py::arg("handle_out_ptr"));
    m.def("ifoe_import_fabric",
          &aiter::ifoe_import_fabric,
          py::arg("handle_ptr"),
          py::arg("bytes"));
    m.def("ifoe_init",
          &aiter::ifoe_init,
          py::arg("rank"),
          py::arg("world"),
          py::arg("self_input_ptr"),
          py::arg("self_signal_ptr"),
          py::arg("self_bf_ptr"),
          py::arg("peer_input_ptrs"),
          py::arg("peer_signal_ptrs"),
          py::arg("peer_bf_ptrs"));
    m.def("ifoe_all_reduce",
          &aiter::ifoe_all_reduce,
          py::arg("ctx"),
          py::arg("inp_ptr"),
          py::arg("out_ptr"),
          py::arg("numel"),
          py::arg("elt_size"),
          py::arg("mode"),
          py::arg("unroll"),
          py::arg("blocks"));
    m.def("ifoe_meta_size", &aiter::ifoe_meta_size);
    m.def("ifoe_dispose", &aiter::ifoe_dispose, py::arg("ctx"));
}
