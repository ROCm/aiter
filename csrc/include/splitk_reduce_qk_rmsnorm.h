#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"

namespace aiter {

// Adds six fp32 split-K partial planes of the MLA input projection in a fixed
// order, writes the bf16 sum, and applies RMSNorm to its q and kv columns.
//
//   partial  [6, M, 2624] fp32
//   out      [M, 2624]    bf16, the sum (q_lora 2048 | kv_lora 512 | rope 64)
//   q_out    [M, 2048]    bf16, RMSNorm(out[:, :2048]) * q_weight
//   k_out    [M, 512]     bf16, RMSNorm(out[:, 2048:2560]) * k_weight
//   q_weight [2048] bf16, k_weight [512] bf16
void splitk_reduce_qk_rmsnorm(const aiter_tensor_t& partial,
                              aiter_tensor_t& out,
                              aiter_tensor_t& q_out,
                              aiter_tensor_t& k_out,
                              const aiter_tensor_t& q_weight,
                              const aiter_tensor_t& k_weight,
                              double q_eps,
                              double k_eps);

} // namespace aiter
