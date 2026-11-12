// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_tensor.h"
#include <cstdint>
#include <string>

namespace aiter {

// NVFP4 routed-expert MoE kernels for gfx950 (MiniMax-M3), see
// csrc/kernels/nvfp4_moe/moe_nvfp4_kernels.hip and aiter/ops/moe_nvfp4.py.

// Number of compiled kernels, the name of kernel i, and its field k (CfgField of
// moe_nvfp4_entry.h: s1, m_tiles, n_tiles, waves, gs, inbf16, outbf16, reduce). s1 is 1 exactly
// for the nvfp4_moe_s1_* names; gs: weight scale layout, K-tiles per scale group (2 pair, 4, 6 SG,
// 8); inbf16: a stage1 kernel that reads the bf16 activations; outbf16: a stage2 kernel with bf16
// output atomics (else f32 atomics into a workspace, cast to f16 by the caller); reduce: a stage2
// kernel that stores f32 [T * topk, H] rows for moe_nvfp4_reduce.
int64_t moe_nvfp4_num_cfgs();
std::string moe_nvfp4_cfg_name(int64_t i);
int64_t moe_nvfp4_cfg_field(int64_t i, int64_t k);
// Whether kernel i runs reduction length K (the kernel table rows must fit the weight shape).
bool moe_nvfp4_cfg_fits(int64_t i, int64_t K);

// Fused routing sort + bf16 -> f16 activation cast + zeroing of the stage2 atomics buffer.
//   x [T, H] bf16 or f16, x_f16 [T, H] f16 (bf16 x; empty: no cast), out [T, H] x dtype or f32
//   (zeroed; empty: none),
//   topk_ids [T, topk] i32, topk_w [T, topk] f32,
//   sorted_ids / sorted_w [nblk * BM] i32 / f32, sorted_eids [>= nblk] i32,
//   num_valid [1] i32.
void moe_nvfp4_sort(const aiter_tensor_t& x,
                    aiter_tensor_t& x_f16,
                    aiter_tensor_t& out,
                    const aiter_tensor_t& topk_ids,
                    const aiter_tensor_t& topk_w,
                    aiter_tensor_t& sorted_ids,
                    aiter_tensor_t& sorted_w,
                    aiter_tensor_t& sorted_eids,
                    aiter_tensor_t& num_valid,
                    int64_t num_experts,
                    int64_t block_m,
                    int64_t nblk);

// out [T, H] bf16 or f16 = sum over k of ws [T * topk, H] f32 rows t * topk + k (k in order,
// rounded once).
void moe_nvfp4_reduce(const aiter_tensor_t& ws, aiter_tensor_t& out, int64_t topk);

// One GEMM stage of kernel (index into the kernel table) on the sorted rows. Stage 2 out:
// [T, H], bf16 for _outbf16 kernels, else f32 (zeroed workspace of f16 output); _reduce kernels:
// f32 [T * topk, H] (moe_nvfp4_reduce).
void moe_nvfp4_gemm(int64_t kernel,
                    const aiter_tensor_t& x,
                    const aiter_tensor_t& w,
                    const aiter_tensor_t& w_sc,
                    const aiter_tensor_t& inv_gs,
                    const aiter_tensor_t& sorted_ids,
                    const aiter_tensor_t& sorted_w,
                    const aiter_tensor_t& sorted_eids,
                    const aiter_tensor_t& num_valid,
                    aiter_tensor_t& out,
                    int64_t T,
                    int64_t topk,
                    int64_t K,
                    int64_t ntiles,
                    int64_t n_out,
                    int64_t grid_x,
                    int64_t grid_y);

} // namespace aiter
