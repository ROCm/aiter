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
                 int64_t q_scale_mode,
                 int64_t k_scale_mode,
                 int64_t v_scale_mode,
                 double softmax_scale);

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
                        int64_t q_scale_mode,
                        int64_t k_scale_mode,
                        int64_t v_scale_mode,
                        double softmax_scale,
                        const at::Tensor& kv_block_indices,
                        const at::Tensor& lut_start,
                        const at::Tensor& lut_count);

// The work table fmha_v4_fwd_sparse builds internally, exposed so its ordering can be tested.
// Reordering a permutation costs only load balance, but the table must stay a permutation: each
// entry names the tile one workgroup claims, so a duplicated entry leaves another tile unwritten.
at::Tensor mha_v4_sparse_work_table(const at::Tensor& lut_count,
                                    int64_t batch,
                                    int64_t nhead,
                                    int64_t q_tiles);

} // namespace torch_itfs
} // namespace aiter