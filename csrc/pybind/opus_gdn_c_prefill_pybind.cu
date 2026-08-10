// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

#include "opus_gdn_c_prefill.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "_opus_gdn_c_prefill_fwd",
        &opus_gdn_c_prefill_fwd,
        "Dense gfx942 C-input GDN prefill forward",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("g"),
        py::arg("beta"),
        py::arg("o"),
        py::arg("scale"),
        py::arg("initial_state"),
        py::arg("final_state"),
        py::arg("has_initial_state"),
        py::arg("output_final_state"),
        py::arg("c_mode"),
        py::arg("use_env_overrides") = true);
}
