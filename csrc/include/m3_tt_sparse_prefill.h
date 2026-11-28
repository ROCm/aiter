#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include "aiter_tensor.h"
#include <cstdint>

namespace aiter {

// MiniMax-M3 TP1 FP8 sparse prefill attention, token-tiled (gfx950 only). Shapes and the tile-map
// contract: aiter/ops/m3_tt_sparse_prefill.py. Writes bf16 `out`.
void m3_tt_sparse_prefill(const aiter_tensor_t& q,
                          const aiter_tensor_t& k_cache,
                          const aiter_tensor_t& v_cache,
                          const aiter_tensor_t& k_scale,
                          const aiter_tensor_t& v_scale,
                          const aiter_tensor_t& topk_idx,
                          const aiter_tensor_t& block_table,
                          const aiter_tensor_t& row_req,
                          const aiter_tensor_t& row_kv_len,
                          const aiter_tensor_t& tile_t0,
                          const aiter_tensor_t& tile_n,
                          int64_t ntiles,
                          float sm_scale,
                          aiter_tensor_t& out);

} // namespace aiter
