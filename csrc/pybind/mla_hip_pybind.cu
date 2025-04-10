// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

torch::Tensor mla_decode_fwd_hip(torch::Tensor &Q,    //   [batch_size, num_heads, kv_lora_rank + qk_rope_head_dim]
    torch::Tensor &K,                       //   [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    torch::Tensor &kv_indptr,               //   [batch_size+1]
    torch::Tensor &kv_page_indices,         //   [num_page_used]
    torch::Tensor &kv_last_page_lens,       //   [batch_size]
    int head_size_v,
    float softmax_scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mla_decode_fwd_hip", &mla_decode_fwd_hip, "mla_decode_fwd_hip",
        py::arg("Q"),
        py::arg("KV"),
        py::arg("kv_indptr"),
        py::arg("kv_page_indices"),
        py::arg("kv_last_page_lens"),
        py::arg("head_size_v"),
        py::arg("softmax_scale"));
}
