#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void mla_decode_stage1_asm_fwd(torch::Tensor &Q,                 //   [num_seqs, num_heads, head_size]
                               torch::Tensor &KV,                //   [num_page, page_size, num_kv_heads, head_size]
                               torch::Tensor &qo_indptr,         //   [batch_size+1]
                               torch::Tensor &kv_indptr,         //   [batch_size+1]
                               torch::Tensor &kv_page_indices,   //   [num_page_used]
                               torch::Tensor &kv_last_page_lens, //   [batch_size]
                               int max_seqlen_q,
                               float softmax_scale,
                               // following are output
                               torch::Tensor &splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
                               torch::Tensor &splitLse   //[batch_size, num_kv_splits, num_heads,  1]
);

void mla_prefill_asm_fwd(torch::Tensor &Q,                 //   [num_seqs, num_heads, head_size]
                         torch::Tensor &KV,                //   [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
                         torch::Tensor &qo_indptr,         //   [batch_size+1]
                         torch::Tensor &kv_indptr,         //   [batch_size+1]
                         torch::Tensor &kv_page_indices,   //   [num_page_used]
                         torch::Tensor &kv_last_page_lens, //   [batch_size]
                         int max_seqlen_q,
                         float softmax_scale,
                         // following are output
                         torch::Tensor &splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
                         torch::Tensor &splitLse   //[batch_size, num_kv_splits, num_heads,  1]
);