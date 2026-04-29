#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

// ASM FMHA forward (BF16, gfx1250).
//
// Layout conventions (i_perm / o_perm):
//   0 = bshd  [batch, seq,  head, dim]
//   1 = bhsd  [batch, head, seq,  dim]
//   2 = sbhd  [seq,   batch,head, dim]   (defaults)
//
// q/k/v shapes are fully determined by i_perm:
//   i_perm=2: q [s,b,hq,d], k [s,b,hk,d], v [s,b,hk,d_v]
//   i_perm=1: q [b,hq,s,d], k [b,hk,s,d], v [b,hk,s,d_v]
//   i_perm=0: q [b,s,hq,d], k [b,s,hk,d], v [b,s,hk,d_v]
//
// out shape is determined by o_perm (default 0 → bshd [b,s,hq,d_v]).
//
// sink: optional per-head f32 tensor [q_head_num], post-scale AITER convention.
//       Internally converted to pre-scale: sink_raw = sink_user * sqrt(qk_head_dim).
std::vector<at::Tensor> fmha_fwd_f16(
    at::Tensor&                       q,
    const at::Tensor&                 k,
    const at::Tensor&                 v,
    float                             softmax_scale,
    bool                              is_causal,
    bool                              return_lse,
    int                               i_perm = 2,
    int                               o_perm = 0,
    std::optional<at::Tensor>         sink_  = std::nullopt,
    std::optional<at::Tensor>         out_   = std::nullopt);

} // namespace torch_itfs
} // namespace aiter
