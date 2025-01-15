// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include "attention.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("paged_attention_rocm", &paged_attention,
          "paged_attention_rocm(Tensor! out, Tensor exp_sums,"
          "                Tensor max_logits, Tensor tmp_out,"
          "                Tensor query, Tensor key_cache,"
          "                Tensor value_cache, int num_kv_heads,"
          "                float scale, Tensor block_tables,"
          "                Tensor context_lens, int block_size,"
          "                int max_context_len,"
          "                Tensor? alibi_slopes,"
          "                str kv_cache_dtype,"
          "                float k_scale, float v_scale) -> ()");
}