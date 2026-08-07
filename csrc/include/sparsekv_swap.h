// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include <torch/extension.h>
#include <cstdint>

// Translate a pinned host allocation to a device-mapped pointer (int64 VA).
// On an xnack- agent a GPU kernel faults on a raw host VA, so the swap kernels
// must be fed the mapped pointer this returns instead of tensor.data_ptr().
int64_t sparsekv_host_get_device_pointer(at::Tensor pinned_host_tensor);

// Gather scattered tokens from one layer's pinned host cold pool into that
// layer's GPU hot buffer. One wavefront64 copies one token word-wise.
void sparsekv_swap_in(int64_t cold_pool_dev_ptr,
                      at::Tensor hot_buffer,
                      at::Tensor src_locs,
                      at::Tensor dst_locs,
                      int64_t item_size_bytes);

// Fused per-layer decode hot path for one layer: for every decode query token,
// miss-detect its logical top-k against the resident hot set, evict the
// least-recently-used slots for the misses, gather the missing tokens from the
// pinned host cold pool into the hot buffer, and write each top-k's resident
// hot-buffer absolute row into out_translated. All bookkeeping is on GPU and the
// launch shape is fixed (the data-dependent work is internal kernel control
// flow), with no host synchronization. Capturing this in a CUDAGraph additionally
// requires the caller to pass capture-stable input tensors (fixed addresses,
// updated in place) — that plumbing is the caller's responsibility.
void sparsekv_swap_and_translate(int64_t cold_pool_dev_ptr,
                                 at::Tensor hot_buffer,
                                 at::Tensor topk_logical,
                                 at::Tensor indptr,
                                 at::Tensor req_slots,
                                 at::Tensor slot_token,
                                 at::Tensor last_used,
                                 at::Tensor token_to_slot,
                                 at::Tensor recency,
                                 at::Tensor out_translated,
                                 at::Tensor host_cache_locs,
                                 int64_t host_stride,
                                 at::Tensor gpu_cache_locs,
                                 int64_t gpu_stride,
                                 int64_t skip_gather,
                                 int64_t item_size_bytes,
                                 int64_t hot_slots,
                                 int64_t cold_depth,
                                 int64_t topk);

// IndexShare variant of sparsekv_swap_and_translate that additionally records the
// miss plan it computed (per query token: the logical cold positions it gathered
// and the hot slots it assigned them, plus the count) so a group's shared-index
// layers can replay the identical IO. Same kernel, semantics, and side effects
// as sparsekv_swap_and_translate otherwise.
void sparsekv_swap_and_translate_record(int64_t cold_pool_dev_ptr,
                                        at::Tensor hot_buffer,
                                        at::Tensor topk_logical,
                                        at::Tensor indptr,
                                        at::Tensor req_slots,
                                        at::Tensor slot_token,
                                        at::Tensor last_used,
                                        at::Tensor token_to_slot,
                                        at::Tensor recency,
                                        at::Tensor out_translated,
                                        at::Tensor plan_miss_tok,
                                        at::Tensor plan_miss_slot,
                                        at::Tensor plan_miss_count,
                                        at::Tensor plan_miss_home,
                                        at::Tensor host_cache_locs,
                                        int64_t host_stride,
                                        at::Tensor gpu_cache_locs,
                                        int64_t gpu_stride,
                                        int64_t skip_gather,
                                        int64_t item_size_bytes,
                                        int64_t hot_slots,
                                        int64_t cold_depth,
                                        int64_t topk);

// Replay one home's share of a recorded miss plan into a layer's hot buffer
// (Design Y dual-source swap). Gathers only misses whose recorded home matches
// target_home (0=host, 1=gpu), indirecting through that home's translation table
// and cold pool base. The coordinator issues this twice per layer (host + gpu)
// after a record-only detect so a mixed-home top-k lands fully in the hot buffer.
// Pure IO, fixed launch shape (CUDAGraph-capturable).
void sparsekv_gather_planned(int64_t base_dev_ptr,
                             at::Tensor hot_buffer,
                             at::Tensor req_slots,
                             at::Tensor plan_miss_tok,
                             at::Tensor plan_miss_slot,
                             at::Tensor plan_miss_count,
                             at::Tensor plan_miss_home,
                             int64_t target_home,
                             at::Tensor cache_locs,
                             int64_t cache_stride,
                             int64_t item_size_bytes,
                             int64_t hot_slots,
                             int64_t cold_depth,
                             int64_t topk);

// Replay a recorded miss plan (from an anchor's swap_and_translate_record) into a
// shared-index layer's hot buffer. Pure host->device gather, no bookkeeping. One
// block per decode query token; fixed launch shape (CUDAGraph-capturable).
void sparsekv_copy_planned(int64_t cold_pool_dev_ptr,
                           at::Tensor hot_buffer,
                           at::Tensor req_slots,
                           at::Tensor plan_miss_tok,
                           at::Tensor plan_miss_slot,
                           at::Tensor plan_miss_count,
                           at::Tensor host_cache_locs,
                           int64_t host_stride,
                           int64_t item_size_bytes,
                           int64_t hot_slots,
                           int64_t cold_depth,
                           int64_t topk);

// Backup a shared-index layer's freshly generated token into the hot slot the
// anchor already assigned it (token_to_slot is the anchor's table). Data only:
// writes cold pool + the assigned hot slot, no LRU/recency mutation. One block
// per decode query token; no host sync.
void sparsekv_backup_into_assigned(int64_t cold_pool_dev_ptr,
                                   int64_t gpu_cold_pool_ptr,
                                   at::Tensor hot_buffer,
                                   at::Tensor layer_kv,
                                   at::Tensor src_slots,
                                   at::Tensor req_slots,
                                   at::Tensor logical_pos,
                                   at::Tensor token_to_slot,
                                   at::Tensor host_cache_locs,
                                   int64_t host_stride,
                                   at::Tensor gpu_cache_locs,
                                   int64_t gpu_stride,
                                   int64_t item_size_bytes,
                                   int64_t hot_slots,
                                   int64_t cold_depth);

// Persist a freshly generated token's KV (already resident in the layer cache at
// src_slot) into the cold pool and allocate it a hot-buffer slot with maximal
// recency. One block per decode query token; no host sync.
void sparsekv_backup_new_token(int64_t cold_pool_dev_ptr,
                               int64_t gpu_cold_pool_ptr,
                               at::Tensor hot_buffer,
                               at::Tensor layer_kv,
                               at::Tensor src_slots,
                               at::Tensor req_slots,
                               at::Tensor logical_pos,
                               at::Tensor slot_token,
                               at::Tensor last_used,
                               at::Tensor token_to_slot,
                               at::Tensor recency,
                               at::Tensor host_cache_locs,
                               int64_t host_stride,
                               at::Tensor gpu_cache_locs,
                               int64_t gpu_stride,
                               int64_t item_size_bytes,
                               int64_t hot_slots,
                               int64_t cold_depth);
