#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
fmha_v3_varlen_fwd(at::Tensor& q,                                 // [total_q, hq, d]
                   const at::Tensor& k,                           // [total_k, hk, d]
                   const at::Tensor& v,                           // [total_k, hk, d]
                   const at::Tensor& cu_seqlens_q,                // [b+1]
                   const at::Tensor& cu_seqlens_k, // [b+1]
                   int max_seqlen_q,
                   int max_seqlen_k,
                   int min_seqlen_q,
                   float p_dropout,
                   float softmax_scale,
                   float logits_soft_cap,
                   bool zero_tensors,
                   bool is_causal,
                   int window_size_left,
                   int window_size_right,
                   bool return_softmax_lse,
                   bool return_dropout_randval,
                   int how_v3_bf16_cvt,
                   std::optional<at::Tensor> out,                // [total_q, hq, d]
                   std::optional<const at::Tensor> block_table,  // [hq] or [b, hq]
                   std::optional<const at::Tensor> bias,         // [total_q, max_seqlen_k]
                   std::optional<const at::Tensor> alibi_slopes, // [hq] or [b, hq]
                   std::optional<at::Generator> gen,
                   std::optional<const at::Tensor> cu_seqlens_q_padded, // [b+1]
                   std::optional<const at::Tensor> cu_seqlens_k_padded); // [b+1]);
} // namespace torch_itfs
} // namespace aiter
