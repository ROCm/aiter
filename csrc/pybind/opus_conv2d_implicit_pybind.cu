// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// pybind glue for opus_conv2d_implicit. Host-only TU.
#ifndef __HIP_DEVICE_COMPILE__

#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "opus_conv2d_implicit.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND

    m.def("opus_conv2d_implicit",
          &opus_conv2d_implicit,
          "Conv2D forward via implicit GEMM fused into opus asm pipeline (gfx942)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("output"),
          py::arg("N_batch"),
          py::arg("C"),
          py::arg("K"),
          py::arg("Hi"),
          py::arg("Wi"),
          py::arg("R"),
          py::arg("S"),
          py::arg("pad_h"),
          py::arg("pad_w"),
          py::arg("stride_h"),
          py::arg("stride_w"),
          py::arg("dil_h"),
          py::arg("dil_w"),
          py::arg("group"));
}

#endif // !__HIP_DEVICE_COMPILE__
