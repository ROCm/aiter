// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

#include "opus_gdn_wu_prefill.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("_opus_gdn_wu_prefill_fwd",
          &opus_gdn_wu_prefill_fwd,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("g"),
          py::arg("beta"),
          py::arg("o"),
          py::arg("scale"),
          py::arg("initial_state"),
          py::arg("final_state"),
          py::arg("cu_seqlens"),
          py::arg("chunk_indices"),
          py::arg("chunk_offsets"),
          py::arg("varlen_max_chunks"),
          py::arg("has_initial_state"),
          py::arg("output_final_state"),
          py::arg("BT"),
          py::arg("BV"),
          py::arg("num_warps"),
          py::arg("k1_algo"),
          py::arg("k2_mode"),
          py::arg("use_env_overrides") = true);
}
