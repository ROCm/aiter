// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <optional>

namespace aiter {

// R1 NVFP4 MLA decode prep (gfx1250), one launch per decode step:
//   q_rope    [B, 128, 64]  bf16, rows 128 head strides apart
//   q_aux     [B, 128, 128] uint8: bytes 0..63 / 96..127 written as
//             mla_nvfp4_quantize_q writes them, 64..95 left as written
//   ring      [R, 16, 576]  bf16, the requests' tail tiles: the token's row
//             positions % 16 of ring[req_ids] is set, then rows 0..that row
//             are requantized into kv_nope [P, 16384] / kv_aux [P, 8192] at
//             page slots / 64, tile positions % 64 / 16, as
//             mla_nvfp4_update_cache writes them (N16 AUX)
//   req_ids   [B] int32;  k_nope [B, 512], k_rope [B, 64] bf16
//   positions, slots [B] int64; slot -1 is a padding row
//   cos/sin   [max_pos, 32] bf16 caches (GPT-J pairs): q_rope and k_rope are
//             rotated at positions here, rounding as aiter's Triton RoPE;
//             without them they are already rotated
void mla_nvfp4_decode_prep(aiter_tensor_t& q_rope,
                           aiter_tensor_t& q_aux,
                           aiter_tensor_t& ring,
                           aiter_tensor_t& req_ids,
                           aiter_tensor_t& k_nope,
                           aiter_tensor_t& k_rope,
                           aiter_tensor_t& positions,
                           aiter_tensor_t& slots,
                           aiter_tensor_t& kv_nope,
                           aiter_tensor_t& kv_aux,
                           std::optional<aiter_tensor_t> cos_cache,
                           std::optional<aiter_tensor_t> sin_cache);

} // namespace aiter
