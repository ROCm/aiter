// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_stream.h"
#include "rocm_ops.hpp"
#include "vsa_sparse_attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("vsa_sparse_attention_fwd",
          &vsa_sparse_attention_fwd,
          "vsa_sparse_attention_fwd(q, k, v, block_lut, block_counts, out)",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("block_lut"),
          py::arg("block_counts"),
          py::arg("out"));
}
