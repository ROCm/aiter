// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

// Torch-free TU. These ops used to ride along in module_fmha_v4_fwd, whose other
// half (fmha_v4_fwd) is still at::Tensor; that made the module link libtorch and
// therefore pick up torch's PYBIND11_INTERNALS_ID, while the aiter_tensor_t these
// signatures take is registered by module_aiter_core in the torch-free ABI family.
// The two IDs never match, so the develop=True marshalling could not hand an
// aiter_tensor_t across. Splitting them puts these ops in the same family as core.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "mha_v4_quant.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    // Required by the develop=True ops: lets the Python marshalling push the
    // current HIP stream into this TU.
    AITER_SET_STREAM_PYBIND
    m.def("rotate_activation_hd128",
          &aiter::torch_itfs::rotate_activation_hd128,
          py::arg("out"),
          py::arg("input"));
    m.def("rotate_activation_mxfp8_quant",
          &aiter::torch_itfs::rotate_activation_mxfp8_quant,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"),
          py::arg("multiplier"));
    m.def("rotate_activation_mxfp6_quant",
          &aiter::torch_itfs::rotate_activation_mxfp6_quant,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"),
          py::arg("multiplier"));
    m.def("rotate_activation_mxfp6_quant_k",
          &aiter::torch_itfs::rotate_activation_mxfp6_quant_k,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"));
    m.def("_quantize_v_mxfp6_fp6_p_hip",
          &aiter::torch_itfs::quantize_v_mxfp6_fp6_p,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"));
    m.def("rotate_activation_mxfp4_quant",
          &aiter::torch_itfs::rotate_activation_mxfp4_quant,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"),
          py::arg("multiplier"));
    m.def("rotate_activation_mxfp4_quant_k",
          &aiter::torch_itfs::rotate_activation_mxfp4_quant_k,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"));
    m.def("_quantize_v_mxfp4_fp6_p_hip",
          &aiter::torch_itfs::quantize_v_mxfp4_fp6_p,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"));
    m.def("_quantize_v_mxfp4_hip",
          &aiter::torch_itfs::quantize_v_mxfp4,
          py::arg("out"),
          py::arg("scale"),
          py::arg("input"));
}
