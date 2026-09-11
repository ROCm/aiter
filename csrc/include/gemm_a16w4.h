#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include "aiter_tensor.h"

#include <cstdint>
#include <string>

namespace aiter {

// Dense weight-only int4 GEMM:  out = x @ dequant(weight)
//
//   x          [M, K]        bf16 or fp16
//   weight     [K/8, N]      i32, 8 int4 nibbles of ONE column packed along K
//   scales     [K/128, N]    same dtype as x
//   zeros      [K/128, N]    same dtype as x
//   out        [M, N]        same dtype as x, preallocated by the caller
//   workspace  [>= n]        fp32 scratch, n = gemm_a16w4_workspace_elems(...)
//
// dequant(weight)[k, n] = (nibble(k, n) - zeros[k/128, n]) * scales[k/128, n]
//
// The fp16 path uses a bit-magic dequant and needs a DIFFERENT weight
// bit order plus zeros pre-biased by +1024; aiter/ops/gemm_op_a16w4.py owns
// both conversions. The two weight layouts are not interchangeable and
// nothing downstream can detect a mix-up, so the dtype of `x` is the only
// switch.
//
// gfx1201 (Navi 48) only -- raises on any other architecture.
void gemm_a16w4(aiter_tensor_t x,
                aiter_tensor_t weight,
                aiter_tensor_t scales,
                aiter_tensor_t zeros,
                aiter_tensor_t out,
                aiter_tensor_t workspace);

// "" when (M, N, K) is supported, otherwise the constraint that failed.
// Pure host arithmetic; safe to cache on the Python side.
std::string gemm_a16w4_unsupported_reason(int64_t M, int64_t N, int64_t K, bool is_fp16);

// fp32 workspace elements gemm_a16w4() needs for this shape. 0 for shapes
// that take the prefill path, which has no split-K and no scratch.
int64_t gemm_a16w4_workspace_elems(int64_t M, int64_t N, int64_t K, bool is_fp16);

} // namespace aiter
