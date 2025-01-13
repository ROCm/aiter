/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: attention_asm_pybind.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-12-09 12:48:53
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-12-09 13:08:42
 * @Description: This is description.
 */

#include "attention_asm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pa_fwd_asm", &pa_fwd, "pa_fwd",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("block_tables"),
          py::arg("context_lens"),
          py::arg("max_num_blocks"),
          py::arg("K_QScale") = std::nullopt,
          py::arg("V_QScale") = std::nullopt,
          py::arg("out_") = std::nullopt);
}
