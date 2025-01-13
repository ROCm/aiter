/*
 * Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
 *
 * @Script: attention_pybind.cu
 * @Author: valarLip
 * @Email: lingpeng.jin@amd.com
 * @Create At: 2024-12-04 21:32:40
 * @Last Modified By: valarLip
 * @Last Modified At: 2024-12-05 15:01:58
 * @Description: This is description.
 */

#include "attention_ck.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pa_fwd_naive", &pa_fwd_naive, "pa_fwd_naive",
          py::arg("Q"),
          py::arg("K"),
          py::arg("V"),
          py::arg("block_tables"),
          py::arg("context_lens"),
          py::arg("k_dequant_scales"),
          py::arg("v_dequant_scales"),
          py::arg("max_seq_len"),
          py::arg("num_kv_heads"),
          py::arg("scale_s"),
          py::arg("scale_k"),
          py::arg("scale_v"),
          py::arg("block_size"),
          py::arg("quant_algo"),
          py::arg("out_") = std::nullopt);
}
