// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>
#include "torch/mha_batch_prefill_asm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mha_batch_prefill_asm",
          &aiter::torch_itfs::mha_batch_prefill_asm,
          py::arg("q"),
          py::arg("k"),
          py::arg("v"),
          py::arg("cu_seqlens_q"),
          py::arg("kv_indptr"),
          py::arg("kv_page_indices"),
          py::arg("seqlens_kvcache"),
          py::arg("out"),
          py::arg("q_descale_per_token"),
          py::arg("k_descale_per_token"),
          py::arg("v_descale_per_head"),
          py::arg("batch"),
          py::arg("num_heads"),
          py::arg("num_heads_k"),
          py::arg("head_size_q"),
          py::arg("head_size_v"),
          py::arg("page_block_size"),
          py::arg("num_total_pages"),
          py::arg("max_seqlen_q"),
          py::arg("softmax_scale"),
          py::arg("p_scale") = std::nullopt);
}
