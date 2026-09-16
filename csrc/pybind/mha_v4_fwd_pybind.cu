// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Torch (at::Tensor) TU. The torch-free rotate_activation_* quant ops that used
// to be defined here now live in mha_v4_quant_pybind.cu / module_mha_v4_quant --
// they take aiter_tensor_t, which only resolves inside module_aiter_core's ABI
// family, and this module links libtorch and so belongs to the other one.
// rocm_ops.hpp supplies pybind11 + `namespace py`; it coexists with
// <torch/extension.h> (see moe_topk_ck_pybind.cu).
#include "rocm_ops.hpp"
#include "torch/mha_v4_fwd.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    m.def("fmha_v4_fwd",
          &aiter::torch_itfs::fmha_v4_fwd,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("q_descale"),
          py::arg("k_descale"),
          py::arg("v_descale"),
          py::arg("out"),
          py::arg("q_format"),
          py::arg("k_format"),
          py::arg("v_format"),
          py::arg("v_pack"),
          py::arg("q_scale_mode"),
          py::arg("k_scale_mode"),
          py::arg("v_scale_mode"),
          py::arg("softmax_scale"));
    m.def("fmha_v4_fwd_sparse",
          &aiter::torch_itfs::fmha_v4_fwd_sparse,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("q_descale"),
          py::arg("k_descale"),
          py::arg("v_descale"),
          py::arg("out"),
          py::arg("q_format"),
          py::arg("k_format"),
          py::arg("v_format"),
          py::arg("v_pack"),
          py::arg("q_scale_mode"),
          py::arg("k_scale_mode"),
          py::arg("v_scale_mode"),
          py::arg("softmax_scale"),
          py::arg("kv_block_indices"),
          py::arg("lut_start"),
          py::arg("lut_count"));
    m.def("mha_v4_sparse_work_table",
          &aiter::torch_itfs::mha_v4_sparse_work_table,
          py::arg("lut_count"),
          py::arg("batch"),
          py::arg("nhead"),
          py::arg("q_tiles"));
}
