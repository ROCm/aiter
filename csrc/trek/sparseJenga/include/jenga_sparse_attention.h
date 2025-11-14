#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <torch/extension.h>

torch::Tensor jenga_sparse_attention(
    torch::Tensor &TQ,
    torch::Tensor &TK,
    torch::Tensor &TV,
    torch::Tensor &Tblock_relation_onehot,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> lse,
    std::optional<torch::Tensor>  seqstart_q,
    std::optional<torch::Tensor>  seqstart_k,
    int bias_type,
    int batch,
    int nhead,
    int nhead_k,
    int seqlen_q,
    int seqlen_k,
    int hdim_q,
    int hdim_v,
    int mode,
    bool i_perm, 
    bool o_perm,
    int max_seqlen_q,
    int max_seqlen_k
);

torch::Tensor vsa_sparse_attention(
    torch::Tensor &TQ,
    torch::Tensor &TK,
    torch::Tensor &TV,
    torch::Tensor &TKV_block_idx,
    torch::Tensor &TKV_blocks,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    std::optional<torch::Tensor> lse,
    std::optional<torch::Tensor>  seqstart_q,
    std::optional<torch::Tensor>  seqstart_k,
    int bias_type,
    int batch,
    int nhead,
    int nhead_k,
    int seqlen_q,
    int seqlen_k,
    int hdim_q,
    int hdim_v,
    int mode,
    bool i_perm, 
    bool o_perm,
    int max_seqlen_q,
    int max_seqlen_k
);