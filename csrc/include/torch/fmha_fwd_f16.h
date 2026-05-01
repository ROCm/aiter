#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

// ASM FMHA forward (BF16, gfx1250).
//
// API contract: q/k/v have **bshd shape**:
//   q : [batch, seq_q, q_head_num,  qk_head_dim]
//   k : [batch, seq_k, kv_head_num, qk_head_dim]
//   v : [batch, seq_k, kv_head_num, v_head_dim]
//   out (returned): [batch, seq_q, q_head_num, v_head_dim]
//
// The kernel reads strides directly from `tensor.stride(...)`, so callers may
// pass a non-contiguous bshd-shaped view of an sbhd / bhsd allocation —
// strides will correctly reflect the underlying memory layout.  Only
// `tensor.stride(-1) == 1` (last-dim contiguous) is required.
//
// sink: optional per-Q-head fp32 tensor [q_head_num], AITER post-scale
//       convention (same domain as Q·K^T * softmax_scale).  Internally
//       converted to pre-scale: sink_raw = sink_user * sqrt(qk_head_dim).
std::vector<at::Tensor> fmha_fwd_f16(
    at::Tensor&                       q,
    const at::Tensor&                 k,
    const at::Tensor&                 v,
    float                             softmax_scale,
    bool                              is_causal,
    bool                              return_lse,
    std::optional<at::Tensor>         sink_  = std::nullopt,
    std::optional<at::Tensor>         out_   = std::nullopt);

} // namespace torch_itfs
} // namespace aiter
