#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

torch::Tensor pa_fwd(torch::Tensor& Q, //   [num_seqs, num_heads, head_size]
                     torch::Tensor& K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                     torch::Tensor& V, //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
                     torch::Tensor& block_tables, //   [num_seqs, max_num_blocks_per_seq]
                     torch::Tensor& context_lens, //   [num_seqs]
                     int max_num_blocks,
                     std::optional<torch::Tensor> K_QScale  = std::nullopt,
                     std::optional<torch::Tensor> V_QScale  = std::nullopt,
                     std::optional<torch::Tensor> out_      = std::nullopt,
                     std::optional<torch::Tensor> qo_indptr = std::nullopt,
                     std::string kernelName                 = "",
                     std::optional<int> high_precision      = 1,
                     std::optional<int> max_seqlen_q        = 1);