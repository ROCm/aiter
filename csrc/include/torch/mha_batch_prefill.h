#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
mha_batch_prefill(at::Tensor& q,                  // [total_q, hq, d]
                  const at::Tensor& k,            // [total_k, hk, d]
                  const at::Tensor& v,            // [total_k, hk, d]
                  const at::Tensor& cu_seqlens_q, // [b+1]
                  const at::Tensor& kv_indptr,    // [b+1]
                  const at::Tensor& kv_page_indices,
                  int max_seqlen_q,
                  int max_seqlen_k,
                  float p_dropout,
                  float softmax_scale,
                  float logits_soft_cap,
                  bool zero_tensors,
                  bool is_causal,
                  int window_size_left,
                  int window_size_right,
                  bool return_softmax_lse,
                  bool return_dropout_randval,
                  std::optional<at::Tensor> out_,                // [total_q, hq, d]
                  std::optional<const at::Tensor> bias_,         // [total_q, max_seqlen_k]
                  std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
                  std::optional<at::Generator> gen_);
} // namespace torch_itfs
} // namespace aiter
