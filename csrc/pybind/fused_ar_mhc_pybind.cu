// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>
#include "aiter_stream.h"
#include "fused_ar_mhc_rmsnorm.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND;
    m.def("fused_allreduce_mhc_fused_post_pre_rmsnorm",
          &aiter::fused_allreduce_mhc_fused_post_pre_rmsnorm,
          py::arg("_fa"),
          py::arg("inp"),
          py::arg("layer_input"),
          py::arg("residual_in"),
          py::arg("post_layer_mix"),
          py::arg("comb_res_mix"),
          py::arg("fn"),
          py::arg("hc_scale"),
          py::arg("hc_base"),
          py::arg("norm_weight"),
          py::arg("gemm_out"),
          py::arg("gemm_out_sqrsum"),
          py::arg("next_residual"),
          py::arg("post_mix"),
          py::arg("comb_mix"),
          py::arg("layer_input_out"),
          py::arg("rms_eps") = 1e-6f,
          py::arg("hc_pre_eps") = 1e-6f,
          py::arg("hc_sinkhorn_eps") = 1e-6f,
          py::arg("norm_eps") = 1e-6f,
          py::arg("hc_post_mult_value") = 1.0f,
          py::arg("sinkhorn_repeat") = 20,
          py::arg("tile_m") = 16,
          py::arg("tile_n") = 32,
          py::arg("tile_k") = 32,
          py::arg("pre_tile_k") = 64,
          py::arg("post_store_nt") = -1,
          py::arg("use_large_m") = false,
          py::arg("use_ar_mhc_full_fusion") = false,
          py::arg("use_ar_mhc_post_epilogue") = false,
          py::arg("use_large_m_post_epilogue") = false,
          py::arg("use_new") = true,
          py::arg("open_fp8_quant") = false,
          py::arg("reg_ptr") = static_cast<int64_t>(0),
          py::arg("reg_bytes") = static_cast<int64_t>(0));
}
