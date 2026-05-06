// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "rocm_ops.hpp"
#include "fused_qk_rmsnorm_group_quant.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_qk_rmsnorm_group_quant",
          &aiter::fused_qk_rmsnorm_group_quant,
          py::arg("q_out_quantized")          = std::nullopt,
          py::arg("q_out_scale")              = std::nullopt,
          py::arg("q")                        = std::nullopt,
          py::arg("q_weight")                 = std::nullopt,
          py::arg("q_epsilon")                = std::nullopt,
          py::arg("q_out_unquantized")        = std::nullopt,
          py::arg("k_out")                    = std::nullopt,
          py::arg("q_res_out")                = std::nullopt,
          py::arg("k")                        = std::nullopt,
          py::arg("k_weight")                 = std::nullopt,
          py::arg("k_epsilon")                = std::nullopt,
          py::arg("q_residual")               = std::nullopt,
          py::arg("group_size")               = 128,
          py::arg("transpose_scale")          = false,
          py::arg("gemma_norm")              = false);
}
