#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
fmha_v3_varlen_bwd(const at::Tensor& dout,         // [total_q, hq, d_v]
                   const at::Tensor& q,            // [total_q, hq, d_q]
                   const at::Tensor& k,            // [total_k, hk, d_q]
                   const at::Tensor& v,            // [total_k, hk, d_v]
                   const at::Tensor& out,          // [total_q, hq, d_v]
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
                   bool is_v3_atomic_fp32,
                   int how_v3_bf16_cvt,
                   std::optional<at::Tensor> dq_,                 // [total_q, hq, d_q]
                   std::optional<at::Tensor> dk_,                 // [total_k, hk, d_q]
                   std::optional<at::Tensor> dv_,                 // [total_k, hk, d_v]
                   std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
                   std::optional<const at::Tensor> rng_state_,
                   std::optional<at::Generator> gen_);

} // namespace torch_itfs
} // namespace aiter
