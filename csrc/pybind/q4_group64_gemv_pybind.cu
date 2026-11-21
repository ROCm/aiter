// SPDX-License-Identifier: MIT

#include "aiter_stream.h"
#include "q4_group64_gemv.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("q4_group64_gemv_out",
          &aiter::q4_group64_gemv_out,
          "q4_group64_gemv_out(x, packed_weight, out, mapping)",
          py::arg("x"),
          py::arg("packed_weight"),
          py::arg("out"),
          py::arg("mapping"));
}
