#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void mla_decode_stage1_asm_fwd(
    torch::Tensor& Q,                    //   [num_seqs, num_heads, head_size]
    torch::Tensor& KV,                   //   [num_page, page_size, num_kv_heads, head_size]
    torch::Tensor& qo_indptr,            //   [batch_size+1]
    torch::Tensor& kv_indptr,            //   [batch_size+1]
    torch::Tensor& kv_page_indices,      //   [num_page_used]
    torch::Tensor& kv_last_page_lens,    //   [batch_size]
    std::optional<torch::Tensor>& num_kv_splits_indptr,   //   metadata
    std::optional<torch::Tensor>& work_meta_data,         //   metadata addr
    std::optional<torch::Tensor>& work_indptr,   //   metadata
    std::optional<torch::Tensor>& work_info_set, //   [batch_size+1]
    int max_seqlen_q,
    float softmax_scale,
    // following are output
    torch::Tensor& splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
    torch::Tensor& splitLse,  //[batch_size, num_kv_splits, num_heads,  1]
    torch::Tensor& output     //[batch_size, num_heads, v_head_dim]
);

void mla_prefill_asm_fwd(
    torch::Tensor& Q,  //   [num_seqs, num_heads, head_size]
    torch::Tensor& KV, //   [num_page, page_size, num_kv_heads, kv_lora_rank + qk_rope_head_dim]
    torch::Tensor& qo_indptr,         //   [batch_size+1]
    torch::Tensor& kv_indptr,         //   [batch_size+1]
    torch::Tensor& kv_page_indices,   //   [num_page_used]
    torch::Tensor& kv_last_page_lens, //   [batch_size]
    int max_seqlen_q,
    float softmax_scale,
    // following are output
    torch::Tensor& splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
    torch::Tensor& splitLse   //[batch_size, num_kv_splits, num_heads,  1]
);
