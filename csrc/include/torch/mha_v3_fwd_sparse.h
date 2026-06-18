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

// Block-sparse fp8 sibling. Q/K/V are all fp8 (E4M3); descales are
// per-tensor (or [b, hk]) fp32. Same LUT triple and bf16 output as the
// i8fp8 variant -- the only contract difference is Q/K dtype (fp8 vs int8).
//
// Kernel: _ZN5aiter32fmha_fwd_hd128_fp8_sparse_gfx950E from
// aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_fp8_sparse.co.
std::vector<at::Tensor>
fmha_v3_fwd_fp8_sparse(at::Tensor& q,                  // [b, sq, hq, d], fp8
                       const at::Tensor& k,            // [b, sk, hk, d], fp8
                       const at::Tensor& v,            // [b, sk, hk, d_v], fp8
                       const at::Tensor& q_descale,    // [1] or [b, hk], fp32
                       const at::Tensor& k_descale,    // [1] or [b, hk], fp32
                       const at::Tensor& v_descale,    // [1] or [b, hk], fp32
                       const at::Tensor& kv_block_indices, // int32, ragged LUT data
                       const at::Tensor& lut_start,        // int32 [b*hq*num_q_blocks]
                       const at::Tensor& lut_count,        // int32 [b*hq*num_q_blocks]
                       float softmax_scale,
                       std::optional<at::Tensor> out_ = std::nullopt); // bf16

// Block-sparse mxfp4 sibling. Q/K are fp4-packed bytes (logical hd=128
// stored as byte[hd/2]); V is fp8; Q/K scales are E8M0 per-block
// uint8/int8 bytes; V descale is fp32 per output channel. Same LUT
// triple as the i8fp8 variant.
//
// Kernel: _ZN5aiter35fmha_fwd_hd128_mxfp4_sparse_gfx950E from
// aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_mxfp4_sparse.co.
// Block-sparse i8fp8 VFA ("frozen-max") sibling. Identical contract to
// fmha_v3_fwd_sparse (int8 Q/K, fp8 V, fp32 descales, int32 LUT, bf16 out);
// only the routed kernel differs -- it freezes the softmax running max on the
// no-mask inner blocks (mimics fav3_sage_attention.py FROZEN_MAX). Kernel:
// _ZN5aiter39fmha_fwd_hd128_i8fp8_sparse_vfa_gfx950E from
// aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_i8fp8_sparse_vfa.co.
std::vector<at::Tensor>
fmha_v3_fwd_i8fp8_sparse_vfa(at::Tensor& q,                  // [b, sq, hq, d], int8
                             const at::Tensor& k,            // [b, sk, hk, d], int8
                             const at::Tensor& v,            // [b, sk, hk, d_v], fp8
                             const at::Tensor& q_descale,    // [1] or [b, hk], fp32
                             const at::Tensor& k_descale,    // [1] or [b, hk], fp32
                             const at::Tensor& v_descale,    // [1] or [b, hk], fp32
                             const at::Tensor& kv_block_indices, // int32
                             const at::Tensor& lut_start,        // int32 [b*hq*num_q_blocks]
                             const at::Tensor& lut_count,        // int32 [b*hq*num_q_blocks]
                             float softmax_scale,
                             int64_t freeze_softmax_max_count,   // online blocks before freeze
                             std::optional<at::Tensor> out_ = std::nullopt); // bf16

// Block-sparse fp8 VFA ("frozen-max") sibling. Identical contract to
// fmha_v3_fwd_fp8_sparse (fp8 Q/K, fp8 V, fp32 descales, int32 LUT, bf16 out)
// plus freeze_softmax_max_count; routes to the frozen-max kernel. Kernel:
// _ZN5aiter36fmha_fwd_hd128_fp8_sparse_vfa_gfx950E from
// aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_fp8_sparse_vfa.co.
std::vector<at::Tensor>
fmha_v3_fwd_fp8_sparse_vfa(at::Tensor& q,                  // [b, sq, hq, d], fp8
                           const at::Tensor& k,            // [b, sk, hk, d], fp8
                           const at::Tensor& v,            // [b, sk, hk, d_v], fp8
                           const at::Tensor& q_descale,    // [1] or [b, hk], fp32
                           const at::Tensor& k_descale,    // [1] or [b, hk], fp32
                           const at::Tensor& v_descale,    // [1] or [b, hk], fp32
                           const at::Tensor& kv_block_indices, // int32
                           const at::Tensor& lut_start,        // int32 [b*hq*num_q_blocks]
                           const at::Tensor& lut_count,        // int32 [b*hq*num_q_blocks]
                           float softmax_scale,
                           int64_t freeze_softmax_max_count,   // online blocks before freeze
                           std::optional<at::Tensor> out_ = std::nullopt); // bf16

std::vector<at::Tensor>
fmha_v3_fwd_mxfp4_sparse(at::Tensor& q,                  // [b, sq, hq, d/2], int8/uint8
                        const at::Tensor& k,             // [b, sk, hk, d/2], int8/uint8
                        const at::Tensor& v,             // [b, sk, hk, d_v], fp8
                        const at::Tensor& q_descale,     // E8M0 bytes, [b, sq, hq, d/32]
                        const at::Tensor& k_descale,     // E8M0 bytes, [b, sk, hk, d/32]
                        const at::Tensor& v_descale,     // fp32 per output channel, [b*hk, d_v]
                        const at::Tensor& kv_block_indices, // int32
                        const at::Tensor& lut_start,        // int32 [b*hq*num_q_blocks]
                        const at::Tensor& lut_count,        // int32 [b*hq*num_q_blocks]
                        float softmax_scale,
                        std::optional<at::Tensor> out_ = std::nullopt); // [b, sq, hq, d_v], bf16

} // namespace torch_itfs
} // namespace aiter
