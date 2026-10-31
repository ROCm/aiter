// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Register the OPUS launch interfaces on the host pass only.
#ifndef __HIP_DEVICE_COMPILE__

#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "opus_bmm.h"
#include "opus_gemm.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    OPUS_GEMM_A16W16_LAUNCH_PYBIND;
    OPUS_GEMM_A8W8_LAUNCH_PYBIND;
    OPUS_GEMM_A8W8_BLOCKSCALE_LAUNCH_PYBIND;
    OPUS_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_LAUNCH_PYBIND;
    m.def("opus_gemm_a8w8_blockscale_bpreshuffle_workspace_launch",
          &opus_gemm_a8w8_blockscale_bpreshuffle_workspace_launch,
          "opus_gemm_a8w8_blockscale_bpreshuffle_workspace_launch",
          py::arg("XQ"),
          py::arg("WQ"),
          py::arg("x_scale"),
          py::arg("w_scale"),
          py::arg("Y"),
          py::arg("kid"),
          py::arg("workspace"));
    OPUS_GEMM_A8W8_MXSCALE_BMM_LAUNCH_PYBIND;
}

#endif // !__HIP_DEVICE_COMPILE__
