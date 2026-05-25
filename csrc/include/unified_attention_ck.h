#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include <optional>
#include <torch/extension.h>

void unified_attention_fwd(
    torch::Tensor& output,          // [num_tokens, num_heads_q, head_size]
    torch::Tensor& query,           // [num_tokens, num_heads_q, head_size]
    torch::Tensor& key_cache,       // [num_blks, blk_size, num_kv_heads, head_size]
    torch::Tensor& value_cache,     // [num_blks, blk_size, num_kv_heads, head_size]
    torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& seq_lens,        // [num_seqs]
    torch::Tensor& query_start_len, // [num_seqs + 1]
    int mask_type,
    float scale_s,
    float scale,
    float scale_k,
    float scale_v,
    float scale_out,
    bool cache_ptr_int32_overflow_possible = false,
    int num_splits                          = 1,
    std::optional<torch::Tensor> o_acc_workspace   = std::nullopt,
    std::optional<torch::Tensor> lse_acc_workspace = std::nullopt,
    // Per-tensor FP8 descales (a.k.a. "scales" in the Triton kernel). The
    // CK kernel folds Q*K descales into the softmax scale and applies V
    // descale once to o_acc after the 1/l normalisation — matches Triton
    // unified_attention's q_scale/k_scale/v_scale handling. For non-FP8
    // tensors these default to 1.0f (no-op).
    float q_descale = 1.0f,
    float k_descale = 1.0f,
    float v_descale = 1.0f,
    // Optional caller-side override for max_seqlen_q used by `select_config`.
    // 0 (default) keeps the conservative `num_tokens` heuristic. Pass the real
    // per-seq max here when known (e.g. uniform-sq benchmarks or when the
    // caller already has a host-side max) to let the dispatcher pick a tighter
    // tile-tier (e.g. decode_d128_m128 instead of prefill_d128).
    int64_t max_seqlen_q_override = 0);
