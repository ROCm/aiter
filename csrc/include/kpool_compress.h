// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <optional>

namespace aiter {

// kpool cache writers for the GLM-5.3-Flash sparse-attention indexer (see
// csrc/kernels/kpool_compress.cu for the semantics). All three are byte-exact
// with the vLLM Triton path.
//
// kv_cache: [num_blocks, page_size, 132] uint8 (128 fp8 e4m3fn keys per token
//           in the MFMA 16x16 preshuffled layout when page_size > 1, followed
//           by page_size fp32 scales); page_size must be 1 or a multiple of 16.
// ape:      [pool_size, 128] fp32 additive position embedding of the pool.
// pool_size in {2, 3, 4, 8, 16}; lanes_per_pool 0 (auto) or 16 / 32 / 64.

// Prefill insert: k, gate_score [n_tokens, 128] bf16 (rows may be strided),
// slot_mapping [n_tokens] int32 / int64 (>= 0 on the token that closes a pool).
void kpool_compress_insert(aiter_tensor_t& kv_cache,
                           const aiter_tensor_t& k,
                           const aiter_tensor_t& gate_score,
                           const aiter_tensor_t& ape,
                           const aiter_tensor_t& slot_mapping,
                           int pool_size,
                           bool round_scale   = true,
                           int lanes_per_pool = 0);

// Decode update: each token stashes its K / gate into the request's tail ring
// tail_kv_cache [num_blocks, 2, ring, 128] bf16 (ring a multiple of pool_size);
// a token that closes a pool compresses it into kv_cache. key, slot_score
// [num_requests, next_n, 128] bf16; tail_slot_mapping, slot_mapping, positions
// [num_requests, next_n] all int32 or all int64; next_n <= 16.
void kpool_decode_update(aiter_tensor_t& kv_cache,
                         aiter_tensor_t& tail_kv_cache,
                         const aiter_tensor_t& tail_slot_mapping,
                         const aiter_tensor_t& key,
                         const aiter_tensor_t& slot_score,
                         const aiter_tensor_t& ape,
                         const aiter_tensor_t& slot_mapping,
                         const aiter_tensor_t& positions,
                         int pool_size,
                         bool round_scale = true);

// Mixed step in one launch: the prefill insert on prefill_*, the tail seeding
// (kpool_seed_tail_cache) when prefill_tail_slot_mapping [n_tokens] is given,
// and the decode update on decode_*. Either side may be absent. One index
// dtype per launch: the seed slots and the decode metadata must have the
// dtype of prefill_slot_mapping (int32 or int64).
void kpool_mixed_write(aiter_tensor_t& kv_cache,
                       const aiter_tensor_t& ape,
                       int pool_size,
                       bool round_scale,
                       int lanes_per_pool,
                       std::optional<aiter_tensor_t> prefill_k,
                       std::optional<aiter_tensor_t> prefill_gate,
                       std::optional<aiter_tensor_t> prefill_slot_mapping,
                       std::optional<aiter_tensor_t> prefill_tail_slot_mapping,
                       std::optional<aiter_tensor_t> tail_kv_cache,
                       std::optional<aiter_tensor_t> decode_tail_slot_mapping,
                       std::optional<aiter_tensor_t> decode_key,
                       std::optional<aiter_tensor_t> decode_gate,
                       std::optional<aiter_tensor_t> decode_slot_mapping,
                       std::optional<aiter_tensor_t> decode_positions);

} // namespace aiter
