#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_enum.h"
#include <torch/extension.h>

torch::Tensor pa_fwd(torch::Tensor& Q, //   [num_seqs, num_heads, head_size]
                     torch::Tensor& K, //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
                     torch::Tensor& V, //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
                     torch::Tensor& block_tables, //   [num_seqs, max_num_blocks_per_seq]
                     torch::Tensor& context_lens, //   [num_seqs]
                     int block_tables_stride0,
                     int max_qlen                           = 1,
                     std::optional<torch::Tensor> K_QScale  = std::nullopt,
                     std::optional<torch::Tensor> V_QScale  = std::nullopt,
                     std::optional<torch::Tensor> out_      = std::nullopt,
                     std::optional<torch::Tensor> qo_indptr = std::nullopt,
                     std::optional<int> high_precision      = 1,
                     std::optional<std::string> kernelName_ = std::nullopt);

torch::Tensor
pa_ps_fwd(torch::Tensor& Q,            //   [num_seqs, num_heads, head_size]
          torch::Tensor& K,            //   [num_blocks, num_kv_heads, head_size/x, block_size, x]
          torch::Tensor& V,            //   [num_blocks, num_kv_heads, block_size/X, head_size, X]
          torch::Tensor& kv_indptr,    //   [batch_size+1], kvlen prefix sum
          torch::Tensor& kv_indices,   //   [sum_kvlen], packed kv ids
          torch::Tensor& context_lens, //   [batch_size]
          float softmax_scale,
          int max_qlen                           = 1,
          std::optional<torch::Tensor> K_QScale  = std::nullopt,
          std::optional<torch::Tensor> V_QScale  = std::nullopt,
          std::optional<torch::Tensor> out_      = std::nullopt,
          std::optional<torch::Tensor> qo_indptr = std::nullopt,
          //   std::optional<torch::Tensor> work_meta_data = std::nullopt,
          std::optional<torch::Tensor> work_indptr = std::nullopt,
          std::optional<torch::Tensor> work_info   = std::nullopt,
          std::optional<torch::Tensor> splitData   = std::nullopt,
          std::optional<torch::Tensor> splitLse    = std::nullopt,
          int mask                                 = 0,
          std::optional<int> high_precision        = 1,
          std::optional<std::string> kernelName_   = std::nullopt,
          QuantType quant_type                     = QuantType::per_Token);
