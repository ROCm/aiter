// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared public API for the OPUS gfx950 bf16 flash-attention forward kernels.
// A single entry point dispatches (by head dim, inferred from the tensors) to:
//   * the symmetric  D_QK=128 / D_V=128 kernel (gqa_d128_kernel), batch mode only, or
//   * the asymmetric D_QK=192 / D_V=128 kernel (gqa_d192_v128_kernel), batch + group.
//
// This replaces the former per-hd `fmha_fwd_hd128_bf16_opus_fwd`. The kernel/launch
// logic for D=128 is unchanged (only moved under this shared entry point).
#pragma once
#include "aiter_tensor.h"
#include <optional>

// Dense (batch) & varlen (group) GQA/MHA scaled-dot-product attention, bf16, gfx950.
//
// Tensor expectations (row-major, last dim contiguous):
//   Batch mode  : q [B, N, H, D_QK]  k [B, N, H_KV, D_QK]  v [B, N, H_KV, D_V]  out [B, N, H, D_V]
//   Group mode  : q [total_q, H, D_QK]  k [total_k, H_KV, D_QK]  v [total_k, H_KV, D_V]
//                 out [total_q, H, D_V]   (packed / varlen; group = num sequences)
//
// Supported head dims: (D_QK, D_V) in {(128,128), (192,128)}. Group mode requires
// (192,128) (the D=128 kernel is batch only).
//
// `causal` selects the causal mask (bottom-right aligned when seqlen_q != seqlen_kv).
// `softmax_scale` is applied to Q·K^T internally (pass <= 0 for the default 1/sqrt(D_QK)).
//
// Group / varlen (all four seqstart tensors are int32, length num_groups+1; pass
// std::nullopt for batch mode):
//   seqstart_q / seqstart_k          : cumulative REAL sequence lengths (mask / tile count)
//   seqstart_q_pad / seqstart_k_pad  : cumulative PHYSICAL row offsets (KV-padding variant;
//                                      equal to the non-pad arrays when there is no padding)
//   max_seqlen_q / max_seqlen_k      : upper bounds driving the grid (group mode only)
void fmha_fwd_bf16_opus_fwd(aiter_tensor_t& q,
                            aiter_tensor_t& k,
                            aiter_tensor_t& v,
                            aiter_tensor_t& out,
                            bool causal,
                            float softmax_scale,
                            std::optional<aiter_tensor_t> seqstart_q     = std::nullopt,
                            std::optional<aiter_tensor_t> seqstart_k     = std::nullopt,
                            std::optional<aiter_tensor_t> seqstart_q_pad = std::nullopt,
                            std::optional<aiter_tensor_t> seqstart_k_pad = std::nullopt,
                            int max_seqlen_q = 0,
                            int max_seqlen_k = 0);
