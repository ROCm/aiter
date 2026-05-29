#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

// Block-sparse Sage i8fp8 FMHA forward (hd=128, gfx950).
//
// Mirrors fmha_v3_fwd's contract but adds the 3 LUT tensors (see
// aiter/op_tests/op_benchmarks/triton/utils.py::block_attn_mask_to_ragged_lut).
// The LUT is consumed by the hand-written ASM kernel
// _ZN5aiter35fmha_fwd_hd128_i8fp8_sparse_gfx950E loaded from
// aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_i8fp8_sparse.co.
//
// Returns {out} (bf16, [b, sq, hq, d_v]).
std::vector<at::Tensor>
fmha_v3_fwd_sparse(at::Tensor& q,                  // [b, sq, hq, d], int8
                   const at::Tensor& k,            // [b, sk, hk, d], int8
                   const at::Tensor& v,            // [b, sk, hk, d_v], fp8
                   const at::Tensor& q_descale,    // [1] or [b, hk], fp32
                   const at::Tensor& k_descale,    // [1] or [b, hk], fp32
                   const at::Tensor& v_descale,    // [1] or [b, hk], fp32
                   const at::Tensor& kv_block_indices, // int32, ragged LUT data
                   const at::Tensor& lut_start,        // int32 [b*hq*num_q_blocks]
                   const at::Tensor& lut_count,        // int32 [b*hq*num_q_blocks]
                   float softmax_scale,
                   std::optional<at::Tensor> out_ = std::nullopt); // [b, sq, hq, d_v], bf16

} // namespace torch_itfs
} // namespace aiter
