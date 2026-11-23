/* SPDX-License-Identifier: MIT
   Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
*/
#include "moe_router_topk_fused.h"
#include "rocm_ops.hpp"
#include "aiter_stream.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("moe_router_topk_fused",
          &aiter::moe_router_topk_fused,
          py::arg("topk_weights"),
          py::arg("topk_ids"),
          py::arg("x"),
          py::arg("w"),
          py::arg("correction_bias"),
          py::arg("workspace"),
          py::arg("semaphore"),
          py::arg("split_k"),
          py::arg("waves"),
          py::arg("chunk"),
          py::arg("need_renorm"),
          py::arg("routed_scaling_factor"),
          py::arg("num_valid_rows") = std::nullopt,
          py::arg("logits_out")     = std::nullopt,
          "Fused MoE router GEMM + biased sigmoid top-k gating.");
}
