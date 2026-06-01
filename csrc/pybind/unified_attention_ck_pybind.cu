// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "unified_attention_ck.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("unified_attention_fwd",
          &unified_attention_fwd,
          "unified_attention_fwd",
          py::arg("output"),
          py::arg("query"),
          py::arg("key_cache"),
          py::arg("value_cache"),
          py::arg("block_tables"),
          py::arg("seq_lens"),
          py::arg("query_start_len"),
          py::arg("mask_type"),
          py::arg("scale_s"),
          py::arg("scale"),
          py::arg("scale_k"),
          py::arg("scale_v"),
          py::arg("scale_out"),
          py::arg("cache_ptr_int32_overflow_possible") = false,
          py::arg("num_splits")                        = 1,
          py::arg("o_acc_workspace")                   = std::nullopt,
          py::arg("lse_acc_workspace")                 = std::nullopt,
          py::arg("q_descale")                         = 1.0f,
          py::arg("k_descale")                         = 1.0f,
          py::arg("v_descale")                         = 1.0f,
          py::arg("max_seqlen_q_override")             = 0,
          py::arg("window_size_left")                  = -1,
          py::arg("window_size_right")                 = -1);
}
