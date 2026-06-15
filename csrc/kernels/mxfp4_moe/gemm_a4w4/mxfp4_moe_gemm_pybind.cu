// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "mxfp4_moe_gemm.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mxfp4_moe_gemm1_a4w4",
          &mxfp4_moe_gemm1_a4w4_kernel,
          py::arg("cumsum_tensor"),
          py::arg("a_quant"),
          py::arg("a_scale_sorted_shuffled"),
          py::arg("w12_shuffled_quant"),
          py::arg("w12_shuffled_scale"),
          py::arg("sorted_expert_ids"),
          py::arg("m_indices"),
          py::arg("inter_sorted_quant"),
          py::arg("inter_sorted_shuffled_scale"),
          py::arg("hidden_states"),
          py::arg("kernelName"));
    m.def("mxfp4_moe_gemm2_a4w4",
          &mxfp4_moe_gemm2_a4w4_kernel,
          py::arg("cumsum_tensor"),
          py::arg("inter_sorted_quant"),
          py::arg("inter_sorted_shuffled_scale"),
          py::arg("w3_shuffled_quant"),
          py::arg("w3_shuffled_scale"),
          py::arg("sorted_token_ids"),
          py::arg("sorted_expert_ids"),
          py::arg("sorted_weights"),
          py::arg("flat_out"),
          py::arg("M_logical"),
          py::arg("max_sorted"),
          py::arg("kernelName"));
    m.def("mxfp4_moe_gemm2_a4w4_mxfp4out",
          &mxfp4_moe_gemm2_a4w4_mxfp4out_kernel,
          py::arg("cumsum_tensor"),
          py::arg("inter_sorted_quant"),
          py::arg("inter_sorted_shuffled_scale"),
          py::arg("w3_shuffled_quant"),
          py::arg("w3_shuffled_scale"),
          py::arg("sorted_expert_ids"),
          py::arg("flat_out_q"),
          py::arg("flat_out_scale"),
          py::arg("NE"),
          py::arg("D_HIDDEN"),
          py::arg("D_INTER"),
          py::arg("max_sorted"));
}
