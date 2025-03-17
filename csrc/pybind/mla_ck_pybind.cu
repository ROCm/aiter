// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

void mla_decode_fwd_ck(torch::Tensor &Q,    //   [batch_size, num_heads, kv_lora_rank + qk_rope_head_dim]
    torch::Tensor &K,                       //   [num_page * page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    std::optional<torch::Tensor> &v_,        //   [num_page * page_size, num_kv_heads, v_head_dim]
    std::optional<torch::Tensor> &out_,        //   [batch_size, num_heads, v_head_dim]
    int head_size_v,
    torch::Tensor &kv_indptr,               //   [batch_size+1]
    torch::Tensor &kv_page_indices,         //   [num_page_used]
    torch::Tensor &kv_last_page_lens,       //   [batch_size]
    float softmax_scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mla_decode_fwd_ck", &mla_decode_fwd_ck, "mla_decode_fwd_ck",
        py::arg("Q"),
        py::arg("K"),
        py::arg("v_") = std::nullopt,
        py::arg("out_") = std::nullopt,
        py::arg("head_size_v"),
        py::arg("kv_indptr"),
        py::arg("kv_page_indices"),
        py::arg("kv_last_page_lens"),
        py::arg("softmax_scale"));
}
