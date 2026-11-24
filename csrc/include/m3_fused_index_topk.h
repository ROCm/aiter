#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MiniMax-M3 lightning-indexer FP8 prefill (gfx950): each 128-token block chunk's top-16 packed
// selector keys per (index head, query token). Contract and key format: aiter/ops/m3_fused_index_topk.py.
#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

void m3_fused_index_topk_partial(const aiter_tensor_t& idx_q,        // [total_q, 4, 128] bf16
                                 const aiter_tensor_t& index_cache,  // [pages, 128, 128] e4m3
                                 aiter_tensor_t& partial,            // [4, total_q, nchunks * 16] i64
                                 const aiter_tensor_t& block_table,  // [batch, >= max_block] i32
                                 const aiter_tensor_t& cu_seqlens_q, // [batch + 1] i32
                                 const aiter_tensor_t& prefix_lens,  // [batch] i32
                                 int64_t max_query_len,
                                 int64_t nchunks,
                                 int64_t chunk_blocks,
                                 float scale);

} // namespace aiter
