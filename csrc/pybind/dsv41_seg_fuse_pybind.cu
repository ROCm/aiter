// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "dsv41_seg_fuse.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("silu_mul_mxfp8",
          &aiter::silu_mul_mxfp8,
          "Clamped SiLU-mul rounded to BF16, then group-32 MXFP8 quant, one launch.",
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"),
          py::arg("limit") = 0.0f);
}
