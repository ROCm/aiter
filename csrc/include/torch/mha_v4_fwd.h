#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

// Validate packed operands, select the exact format/scale manifest row, and launch its code object.
void fmha_v4_fwd(const at::Tensor& q,
                 const at::Tensor& k,
                 const at::Tensor& v,
                 const at::Tensor& q_descale,
                 const at::Tensor& k_descale,
                 const at::Tensor& v_descale,
                 at::Tensor out,
                 int64_t q_format,
                 int64_t k_format,
                 int64_t v_format,
                 int64_t v_pack,
                 int64_t q_scale_mode,
                 int64_t k_scale_mode,
                 int64_t v_scale_mode,
                 double softmax_scale,
                 std::optional<at::Tensor> seqlens_k = std::nullopt,
                 // Per-row log-sum-exp, [batch, nhead_q, seqlen_q] float32 and contiguous, or
                 // nullopt to write none. Only rows whose manifest lse column is 1 can fill it.
                 std::optional<at::Tensor> lse = std::nullopt);

// Sorted block-sparse sibling. Same packed operands as fmha_v4_fwd, plus a ragged LUT.
// Builds the work table internally: longest LUT first unless AITER_MHA_V4_WORK_ORDER says
// otherwise, and raster order when lut_count is uniform.
void fmha_v4_fwd_sparse(const at::Tensor& q,
                        const at::Tensor& k,
                        const at::Tensor& v,
                        const at::Tensor& q_descale,
                        const at::Tensor& k_descale,
                        const at::Tensor& v_descale,
                        at::Tensor out,
                        int64_t q_format,
                        int64_t k_format,
                        int64_t v_format,
                        int64_t v_pack,
                        int64_t q_scale_mode,
                        int64_t k_scale_mode,
                        int64_t v_scale_mode,
                        double softmax_scale,
                        const at::Tensor& kv_block_indices,
                        const at::Tensor& lut_start,
                        const at::Tensor& lut_count,
                        // Tile geometry of the manifest row to dispatch. The LUT is in units of
                        // kv_tile, so it must be the geometry the caller routed with.
                        int64_t q_tile,
                        int64_t kv_tile,
                        // As fmha_v4_fwd's, over only the KV blocks the LUT selects.
                        std::optional<at::Tensor> lse = std::nullopt);

// Sol-Attn sibling (arXiv 2607.24027): the LUT above still drives an EXACT pass, and a second pass
// then sweeps the pooled per-block K/V, masking off the blocks the LUT already covered via
// block_bitmap, so the below-threshold blocks contribute their zeroth-order term instead of nothing.
// Both passes share one online-softmax state, which is what normalizes the mix under a single L.
//
// mean_k / mean_v are K / V with seqlen -> num_kv_blocks in the SOURCE quantized dtype, so the
// approximate pass reuses k_descale / v_descale unchanged; block_bitmap is uint32
// [batch * query_heads * query_tiles, 4 * ceil(num_kv_blocks / 128)] with the bits at and above
// num_kv_blocks SET, which is what clips the last tile's overhang. aiter.ops.triton's
// sol_prepare() produces all six tensors consistently from one boolean mask.
//
// A row may select nothing. lut_count == 0 leaves the exact pass with no running max, and the
// approximate pass recovers one, so such a row lands on the pooled-only softmax over every block.
//
// Dispatches the raster 3-D grid, or on a row that declares sorted a 1-D grid ordered by lut_count
// (see sorted_dispatch). Threshold routing alone keeps per-row block counts near uniform, and ties
// keep raster order, so the sort mainly moves forced all-exact rows (sink queries) to the front.
void fmha_v4_fwd_sol(const at::Tensor& q,
                     const at::Tensor& k,
                     const at::Tensor& v,
                     const at::Tensor& q_descale,
                     const at::Tensor& k_descale,
                     const at::Tensor& v_descale,
                     at::Tensor out,
                     int64_t q_format,
                     int64_t k_format,
                     int64_t v_format,
                     // As in fmha_v4_fwd; mean_v takes V's packing.
                     int64_t v_pack,
                     int64_t q_scale_mode,
                     int64_t k_scale_mode,
                     int64_t v_scale_mode,
                     double softmax_scale,
                     const at::Tensor& kv_block_indices,
                     const at::Tensor& lut_start,
                     const at::Tensor& lut_count,
                     const at::Tensor& mean_k,
                     const at::Tensor& mean_v,
                     const at::Tensor& block_bitmap,
                     const std::optional<at::Tensor>& mean_k_scale,
                     const std::optional<at::Tensor>& mean_v_scale,
                     // As above; the pooled K/V and the selection bitmap are also in units of
                     // kv_tile, so all three of routing, pooling and dispatch must agree.
                     int64_t q_tile,
                     int64_t kv_tile,
                     // The joint softmax's log-sum-exp, over the exact columns AND the pooled
                     // proxy ones, which is what makes it mergeable: a merge adds numerators
                     // and denominators separately, so the proxy's contribution cancels in the
                     // merged ratio exactly as it does in one rank's own. It says nothing
                     // about WHICH blocks were proxied, and this launch routes from the blocks
                     // it is handed, so a caller sharding the KV owns that consistency.
                     std::optional<at::Tensor> lse = std::nullopt,
                     // Reset the softmax every this many keys inside the one launch, each
                     // range with its own pooled correction, merged by LSE. 0 is one range.
                     // Needs a row that declares kv_range and a multiple of 32 blocks.
                     int64_t kv_range_tokens = 0,
                     // Per-block population variance of K, mean_k's dtype, shape and block
                     // stride: adds 0.5 * scale^2 * sum_d q_d^2 * var[d] to each pooled logit.
                     // Needs a row that declares jensen.
                     const std::optional<at::Tensor>& mean_k_var = std::nullopt,
                     // Heavy-first work order from lut_count on a row that declares sorted:
                     // -1 wherever it applies, 0 raster, 1 required. Bitwise the same output.
                     int64_t sorted_dispatch = -1,
                     // mean_k_var's E8M0 scale in mean_k_scale's layout, exactly when K's
                     // scale mode is E8M0_PER_1X32 and mean_k_var is given.
                     const std::optional<at::Tensor>& mean_k_var_scale = std::nullopt,
                     // Low-rank tilt PoC: factor planes 1..rank behind mean_k / mean_v,
                     // tilt_plane_stride bytes apart, and a one- to eight-element int32 device
                     // tensor the kernel reads: the score and value GEMMs' E8M0 shifts, then the
                     // row's score multiplier and other plane strides (see fmha_v4_fwd_sol).
                     int64_t tilt_rank                              = 0,
                     int64_t tilt_plane_stride                      = 0,
                     const std::optional<at::Tensor>& tilt_shift = std::nullopt);

// The work table fmha_v4_fwd_sparse builds internally, exposed so its ordering can be tested.
// Reordering a permutation costs only load balance, but the table must stay a permutation: each
// entry names the tile one workgroup claims, so a duplicated entry leaves another tile unwritten.
// q_tile picks the default order the launchers would (see AITER_MHA_V4_WORK_ORDER), and
// kv_tiles the lut_count from which the head-promoted order starts a row first.
at::Tensor mha_v4_sparse_work_table(const at::Tensor& lut_count,
                                    int64_t batch,
                                    int64_t nhead,
                                    int64_t q_tiles,
                                    int64_t q_tile   = 256,
                                    int64_t kv_tiles = 0);

} // namespace torch_itfs
} // namespace aiter