// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "mha_prefill_splitk_stage1_opus.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mha_prefill_splitk_stage1_opus",
          &mha_prefill_splitk_stage1_opus,
          "mha_prefill_splitk_stage1_opus",
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("kv_indptr"),
          py::arg("kv_page_indices"),
          py::arg("page_size"),
          py::arg("work_indptr"),
          py::arg("work_info_set"),
          py::arg("softmax_scale"),
          py::arg("split_o"),
          py::arg("split_lse"),
          py::arg("debug_qk_scores") = std::nullopt);
}
