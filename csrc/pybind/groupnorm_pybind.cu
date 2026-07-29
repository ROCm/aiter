// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_stream.h"
#include "rocm_ops.hpp"
#include "../include/groupnorm.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("_groupnorm_run",
          &aiter::groupnorm_run,
          py::arg("y"),
          py::arg("workspace"),
          py::arg("x"),
          py::arg("num_groups"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("epsilon"),
          "Group Normalization (caller-allocated output and workspace)");
}
