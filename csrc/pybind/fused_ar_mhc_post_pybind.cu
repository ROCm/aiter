// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>
#include "aiter_stream.h"
#include "fused_ar_mhc_post.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("fused_allreduce_mhc_post_only",
          &aiter::fused_allreduce_mhc_post_only,
          py::arg("_fa"),
          py::arg("inp"),
          py::arg("next_residual"),
          py::arg("residual_in"),
          py::arg("post_layer_mix"),
          py::arg("comb_res_mix"),
          py::arg("use_new") = true,
          py::arg("open_fp8_quant") = false,
          py::arg("reg_ptr") = static_cast<int64_t>(0),
          py::arg("reg_bytes") = static_cast<int64_t>(0));
    m.def("fused_allreduce_mhc_post_one_stage",
          &aiter::fused_allreduce_mhc_post_one_stage,
          py::arg("_fa"),
          py::arg("inp"),
          py::arg("next_residual"),
          py::arg("residual_in"),
          py::arg("post_layer_mix"),
          py::arg("comb_res_mix"),
          py::arg("use_new") = true,
          py::arg("open_fp8_quant") = false,
          py::arg("reg_ptr") = static_cast<int64_t>(0),
          py::arg("reg_bytes") = static_cast<int64_t>(0));
    m.def("fused_allreduce_mhc_post_split",
          &aiter::fused_allreduce_mhc_post_split,
          py::arg("_fa"),
          py::arg("inp"),
          py::arg("next_residual"),
          py::arg("residual_in"),
          py::arg("post_layer_mix"),
          py::arg("comb_res_mix"),
          py::arg("use_new") = true,
          py::arg("open_fp8_quant") = false,
          py::arg("reg_ptr") = static_cast<int64_t>(0),
          py::arg("reg_bytes") = static_cast<int64_t>(0));
}
