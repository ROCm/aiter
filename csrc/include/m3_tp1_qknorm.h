// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"

namespace aiter {

// fused_qknorm_idxrqknorm(kv_cache_dtype="fp8", index_cache_dtype="fp8", asm_layout=True), byte
// for byte, for 64 q / 4 kv / 4 index heads of 128 and rotary 64 only; gfx950 only.
void m3_tp1_qknorm_v3(const aiter_tensor_t& qkv,           // [T, 9856] bf16
                      aiter_tensor_t& q_out,               // [T, 64 * 128] bf16
                      aiter_tensor_t& index_q_out,         // [T, 4 * 128] bf16
                      const aiter_tensor_t& q_norm_weight, // [128] bf16, each
                      const aiter_tensor_t& k_norm_weight,
                      const aiter_tensor_t& index_q_norm_weight,
                      const aiter_tensor_t& index_k_norm_weight,
                      const aiter_tensor_t& cos_sin_cache,      // [max_pos, 64] bf16
                      const aiter_tensor_t& positions,          // [T] int64
                      const aiter_tensor_t& slot_mapping,       // [T] int64, -1 skips the token
                      const aiter_tensor_t& index_slot_mapping, // [T] int64, -1 skips the token
                      aiter_tensor_t& k_cache,                  // [num_blocks, 4, 8, 16, 16] fp8
                      aiter_tensor_t& v_cache,                  // [num_blocks, 4, 1, 128, 16] fp8
                      aiter_tensor_t& index_cache,              // [rows, 128] fp8
                      aiter_tensor_t& k_scale,                  // [num_blocks, 4, 16] fp32
                      aiter_tensor_t& v_scale,                  // [num_blocks, 4, 16] fp32
                      double eps,
                      int64_t block_size); // 16

} // namespace aiter
