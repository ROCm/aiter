// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "pa_sparse_prefill_fp8_opus_gfx1250.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("pa_sparse_prefill_fp8_opus_gfx1250_fwd", &pa_sparse_prefill_fp8_opus_gfx1250_fwd,
          py::arg("q_nope"), py::arg("q_rope"), py::arg("unified_kv_nope"), py::arg("unified_kv_rope"),
          py::arg("kv_indices_prefix"), py::arg("kv_indptr_prefix"), py::arg("kv_nope"), py::arg("kv_rope"),
          py::arg("kv_indices_extend"), py::arg("kv_indptr_extend"), py::arg("attn_sink"), py::arg("out"),
          py::arg("softmax_scale"));
}
