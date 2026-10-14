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
// Builds the work table internally (identity raster if lut_count is uniform, else LPT).
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
                        // kv_tile, so this must match what the caller routed with; gfx950 ships
                        // both a 256x128 and a 64x64 FP8 block-sparse row.
                        int64_t q_tile,
                        int64_t kv_tile,
                        // As above. The LSE a sparse row writes covers the blocks its LUT
                        // selected, which is exactly the mass it computed, so ranks holding
                        // different KV shards can still be merged by it.
                        std::optional<at::Tensor> lse = std::nullopt);

// The work table fmha_v4_fwd_sparse builds internally, exposed so its ordering can be tested.
// Reordering a permutation costs only load balance, but the table must stay a permutation: each
// entry names the tile one workgroup claims, so a duplicated entry leaves another tile unwritten.
at::Tensor mha_v4_sparse_work_table(const at::Tensor& lut_count,
                                    int64_t batch,
                                    int64_t nhead,
                                    int64_t q_tiles);

} // namespace torch_itfs
} // namespace aiter