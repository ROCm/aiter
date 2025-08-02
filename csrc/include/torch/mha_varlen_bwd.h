#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor& dout,         // [total_q, hq, d]
               const at::Tensor& q,            // [total_q, hq, d]
               const at::Tensor& k,            // [total_k, hk, d]
               const at::Tensor& v,            // [total_k, hk, d]
               const at::Tensor& out,          // [total_q, hq, d]
               const at::Tensor& softmax_lse,  // [b, hq, sq]
               const at::Tensor& cu_seqlens_q, // [b+1]
               const at::Tensor& cu_seqlens_k, // [b+1]
               const int max_seqlen_q,
               const int max_seqlen_k,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               const bool is_causal,
               int window_size_left,
               int window_size_right,
               const bool deterministic,
               std::optional<at::Tensor> dq,                 // [total_q, hq, d]
               std::optional<at::Tensor> dk,                 // [total_k, hk, d]
               std::optional<at::Tensor> dv,                 // [total_k, hk, d]
               std::optional<const at::Tensor> alibi_slopes, // [hq] or [b, hq]
               std::optional<const at::Tensor> rng_state,
               std::optional<at::Generator> gen);
} // namespace torch_itfs
} // namespace aiter
