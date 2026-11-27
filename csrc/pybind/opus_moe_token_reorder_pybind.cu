// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "opus_moe_permute.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("opus_moe_permute_plan", &opus_moe_permute_plan, py::arg("expert_ids"), py::arg("counts"),
          py::arg("reverse_sorted"), py::arg("sorted_token_ids"), py::arg("grouped_layout"), py::arg("group_rows"),
          py::arg("num_valid_ids"));
    m.def("opus_moe_permute_fwd", &opus_moe_permute_fwd, py::arg("x"), py::arg("sorted_token_ids"), py::arg("x_perm"));
    m.def("opus_moe_permute_bwd", &opus_moe_permute_bwd, py::arg("d_perm"), py::arg("reverse_sorted"), py::arg("dx"));
    m.def("opus_moe_unpermute_fwd", &opus_moe_unpermute_fwd, py::arg("y_perm"), py::arg("reverse_sorted"),
          py::arg("weights"), py::arg("out"));
    m.def("opus_moe_unpermute_bwd", &opus_moe_unpermute_bwd, py::arg("d_out"), py::arg("y_perm"), py::arg("weights"),
          py::arg("sorted_token_ids"), py::arg("d_perm"), py::arg("d_weights"));
}
