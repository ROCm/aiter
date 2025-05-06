// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "asm_moe_stage2_blockscale.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("moe_stage2_blockscale_asm", &moe_stage2_blockscale_asm, "moe_stage2_blockscale_asm", 
        py::arg("inter_states"),      // [M, Top, K]
        py::arg("w1"),      // [E, N*2, K] -> [N/128, K*128]
        py::arg("w2"), // [E , N, K]
        py::arg("sorted_token_ids"),
        py::arg("sorted_expert_ids"),
        py::arg("sorted_weights"),
        py::arg("num_valid_ids"),
        py::arg("out"),    // [M, N]
        py::arg("topk"),
        py::arg("w2_scale"),
        py::arg("a2_scale"));
}
