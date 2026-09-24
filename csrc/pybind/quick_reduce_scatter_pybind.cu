// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"

namespace aiter {
void qr_reduce_scatter(int64_t handle, const aiter_tensor_t& input,
                      const aiter_tensor_t& output, bool cast_bf2half);
}

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("qr_reduce_scatter", &aiter::qr_reduce_scatter,
          py::arg("handle"), py::arg("input"), py::arg("output"),
          py::arg("cast_bf2half") = true);
}
