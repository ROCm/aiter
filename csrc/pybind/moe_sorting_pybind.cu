// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_sorting.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_sorting_fwd", &moe_sorting_fwd,
          py::arg("topk_ids"), py::arg("topk_weights"), 
          py::arg("sorted_token_ids"), py::arg("sorted_weights"), 
          py::arg("sorted_expert_ids"), py::arg("num_valid_ids"), 
          py::arg("moe_buf"), py::arg("num_experts"), 
          py::arg("unit_size"), py::arg("local_expert_mask")= std::nullopt);
}
