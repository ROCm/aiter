// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

namespace aiter {

// ReplaySSM decode / speculative-verify step of KDA (Kimi Delta Attention), in place on the checkpoint and record
// buffers; same semantics as the Triton `replayssm_sigmoid_gating_delta_rule` (ATOM fla_ops/replayssm.py).
//   o          [T, HV, V] bf16             output
//   ckpt       [slot, HV, V, K] fp32       checkpoint (rewritten on a flush)
//   buf_k/u/g  [slot, HV, cache_len, K|V]  records (k, u bf16; g fp32 or bf16), this step's appended at base + t
//   q, k       [T, H, K] bf16;  v [T, HV, V] bf16;  a [T, HV, K] bf16 (gate input);  b [T, HV] bf16 (beta logits)
//   A_log [HV] fp32, dt_bias [HV * K] fp32;  write_pos [slot], slot_idx [N], cu_seqlens [N + 1] int32
// K = V = 128, max_query_len <= 8, cache_len <= 16.
void kda_replayssm_decode(aiter_tensor_t& o,
                          aiter_tensor_t& ckpt,
                          aiter_tensor_t& buf_k,
                          aiter_tensor_t& buf_u,
                          aiter_tensor_t& buf_g,
                          const aiter_tensor_t& q,
                          const aiter_tensor_t& k,
                          const aiter_tensor_t& v,
                          const aiter_tensor_t& a,
                          const aiter_tensor_t& b,
                          const aiter_tensor_t& A_log,
                          const aiter_tensor_t& dt_bias,
                          const aiter_tensor_t& write_pos,
                          const aiter_tensor_t& slot_idx,
                          const aiter_tensor_t& cu_seqlens,
                          int max_query_len,
                          float scale,
                          bool use_qk_l2norm,
                          bool use_lower_bound,
                          float lower_bound);

} // namespace aiter
