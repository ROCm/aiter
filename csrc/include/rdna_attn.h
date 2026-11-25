// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// RDNA decode and prefill attention over the paged KV cache
// (gfx1150..gfx1153, and RDNA4's gfx1200 and gfx1201), the kernels of
// csrc/kernels/rdna_attn.
//
// Each entry of a GPU family's tables (rdna_{decode,prefill}_configs.h) is one
// build of a kernel, its shape and launch knobs compile-time constants. The
// rdna_*_configs functions list the entries of the tables of a device's
// family, in every module; rdna_{decode,prefill,encoder}_attn launch one by
// its index, in the module built for its head dim and dtype, and refuse it
// unless its fields have the fingerprint of the entry the caller looked up
// (csrc/kernels/rdna_attn/rdna_attn_unit.cuh). aiter/ops/rdna_attn.py picks
// the entry and the module.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "aiter_tensor.h"

// The fields of every entry of the decode tables of `device_id`'s GPU family,
// one sequence's or (rdna_decode_batch_configs) a batch's, in DecodeConfig
// order; none on other GPUs.
std::vector<int64_t> rdna_decode_configs(int64_t device_id);
std::vector<int64_t> rdna_decode_batch_configs(int64_t device_id);

// Decode of up to 8 query tokens per sequence. q and out are
// (sequences * max_query_len, Hq, D), kv_cache the packed HND cache
// (pages, Hkv, page_size, 2 * D), block_table and seq_lens int32. acc, m, l
// and cnt are the caller's split-KV scratch (the kernel runs under CUDA-graph
// capture), cnt zeroed once. mm_ranges, per sequence, is int32
// [count, start0, end0, ...] in device memory.
void rdna_decode_attn(aiter_tensor_t& q,
                      aiter_tensor_t& kv_cache,
                      aiter_tensor_t& block_table,
                      aiter_tensor_t& out,
                      aiter_tensor_t& acc,
                      aiter_tensor_t& m,
                      aiter_tensor_t& l,
                      aiter_tensor_t& cnt,
                      aiter_tensor_t& seq_lens,
                      int64_t config,
                      int64_t fingerprint,
                      float scale,
                      std::optional<aiter_tensor_t> mm_ranges,
                      bool batched);

// Likewise for the prefill tables, in PrefillConfig order, and for the cells
// of the prefill batch tables, in PrefillBatchConfig order.
std::vector<int64_t> rdna_prefill_configs(int64_t device_id);
std::vector<int64_t> rdna_prefill_batch_configs(int64_t device_id);

// Causal prefill of one sequence, or of a batch with a plan
// (aiter.ops.rdna_attn.plan_prefill) and its query starts. Not for CUDA
// graphs: max_seq_len sizes the KV split.
void rdna_prefill_attn(aiter_tensor_t& q,
                       aiter_tensor_t& kv_cache,
                       aiter_tensor_t& block_table,
                       aiter_tensor_t& out,
                       aiter_tensor_t& partial_o,
                       aiter_tensor_t& partial_ml,
                       aiter_tensor_t& counters,
                       aiter_tensor_t& seq_lens,
                       int64_t config,
                       int64_t fingerprint,
                       int64_t max_seq_len,
                       float scale,
                       std::optional<aiter_tensor_t> mm_ranges,
                       std::optional<aiter_tensor_t> plan,
                       std::optional<aiter_tensor_t> query_start_loc,
                       int64_t head_group);

// Non-causal attention on a non-causal prefill build: every token sees every
// key of its sequence. q, k and v are (tokens, heads, D), read in place.
void rdna_encoder_attn(aiter_tensor_t& q,
                       aiter_tensor_t& k,
                       aiter_tensor_t& v,
                       aiter_tensor_t& out,
                       aiter_tensor_t& partial_o,
                       aiter_tensor_t& partial_ml,
                       aiter_tensor_t& counters,
                       aiter_tensor_t& seq_lens,
                       int64_t config,
                       int64_t fingerprint,
                       int64_t max_seq_len,
                       float scale,
                       std::optional<aiter_tensor_t> plan,
                       std::optional<aiter_tensor_t> query_start_loc,
                       int64_t head_group);
