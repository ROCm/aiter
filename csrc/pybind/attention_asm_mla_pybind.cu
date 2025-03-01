// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "attention_asm_mla.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mla_stage1_asm_fwd", &mla_stage1_asm_fwd, "mla_stage1_asm_fwd",
          py::arg("Q"),
          py::arg("KV"),
          py::arg("kv_indptr"),
          py::arg("kv_page_indices"),
          py::arg("kv_last_page_lens"),
          py::arg("softmax_scale"),
          py::arg("splitData"),
          py::arg("splitLse"));
}
