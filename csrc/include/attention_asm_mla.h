#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

void mla_decode_stage1_asm_fwd(
    torch::Tensor& Q,                 //   [num_seqs, num_heads, head_size]
    torch::Tensor& KV,                //   [num_page, page_size, num_kv_heads, head_size]
    torch::Tensor& qo_indptr,         //   [batch_size+1]
    torch::Tensor& kv_indptr,         //   [batch_size+1]
    torch::Tensor& kv_page_indices,   //   [num_page_used]
    torch::Tensor& kv_last_page_lens, //   [batch_size]
    std::optional<torch::Tensor>& num_kv_splits_indptr, //   metadata
    std::optional<torch::Tensor>& work_meta_data,       //   metadata addr
    std::optional<torch::Tensor>& work_indptr,          //   metadata
    std::optional<torch::Tensor>& work_info_set,        //   [batch_size+1]
    int max_seqlen_q,
    float softmax_scale,
    // following are output
    torch::Tensor& splitData, //[batch_size, num_kv_splits, num_heads, v_head_dim]
    torch::Tensor& splitLse,  //[batch_size, num_kv_splits, num_heads,  1]
    torch::Tensor& output,    //[batch_size, num_heads, v_head_dim]
    std::optional<torch::Tensor> q_scale  = std::nullopt, //   [1]
    std::optional<torch::Tensor> kv_scale = std::nullopt  //   [1]
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

void mla_prefill_ps_asm_fwd(
    torch::Tensor& Q,                            //  [num_seqs, num_q_heads, qk_hetad_size], fp8
    torch::Tensor& K,                            //   [num_page, num_kv_heads, qk_head_size], fp8
    torch::Tensor& V,                            //   [num_page, num_kv_heads, v_head_size], fp8
    torch::Tensor& qo_indptr,                    //   [batch_size+1], int
    torch::Tensor& kv_indptr,                    //   [batch_size+1], int
    torch::Tensor& kv_page_indices,              //   [num_page_used], int
    std::optional<torch::Tensor>& work_indptr,   //   [available_tgs+1], int
    std::optional<torch::Tensor>& work_info_set, //   [max_works], int
    int max_seqlen_q,
    float softmax_scale, // following are output
    bool is_causal,
    torch::Tensor& splitData, //[num_q_heads, num_seqs, max_kv_split, v_head_dim], fp32
    torch::Tensor& splitLse,  //[num_q_heads, num_seqs, max_kv_split,  1], fp32
    torch::Tensor& output,    //[num_seqs, num_q_heads, v_head_dim], bf16
    std::optional<torch::Tensor> q_scale = std::nullopt, // fp32, scalar
    std::optional<torch::Tensor> k_scale = std::nullopt, // fp32, scalar
    std::optional<torch::Tensor> v_scale = std::nullopt  // fp32, scalar
);
