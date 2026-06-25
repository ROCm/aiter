// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "aiter_tensor.h"

namespace aiter {

void minimax_m3_decode_index_score(
    const aiter_tensor_t& idx_q,
    const aiter_tensor_t& index_kv_cache,
    aiter_tensor_t& score,
    const aiter_tensor_t& block_table,
    const aiter_tensor_t& seq_lens,
    int64_t total_q,
    int64_t head_dim,
    int64_t init_blocks,
    int64_t local_blocks,
    double sm_scale,
    int64_t max_query_len,
    int64_t block_size,
    int64_t num_kv_chunks);

} // namespace aiter
