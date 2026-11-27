// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// gfx950 MoE token permute / unpermute entries (opus_moe_permute.cu). aiter_tensor_t keeps this header torch-free.
// Shapes and numeric rules: aiter/ops/opus/moe_permute.py.
#include "aiter_tensor.h"

#include <optional>

// Rows every expert's group is padded to (the grouped GEMMs' OPUS_A16W16_GROUP_ALIGN).
#define OPUS_MOE_PERMUTE_GROUP_ALIGN 128

// The plan, from expert_ids [N] (or [N, 1]): reverse_sorted [N] (row of token i, -1: its id is not an expert),
// sorted_token_ids [R] (token of the row, -1 padding), grouped_layout [R] (expert of the row, -1 padding),
// group_rows [E], num_valid_ids [1] (rows in use: up to the end of the last group); R = N rounded up to 128, + 128 (E - 1). Rows: experts
// ascending, tokens ascending inside an expert, every group padded to 128 rows, no rows for an expert without
// tokens. E up to 256. counts: workspace, int32 [E * blocks], blocks = ceil(N / 256) rounded up to 64.
void opus_moe_permute_plan(aiter_tensor_t& expert_ids,
                           aiter_tensor_t& counts,
                           aiter_tensor_t& reverse_sorted,
                           aiter_tensor_t& sorted_token_ids,
                           aiter_tensor_t& grouped_layout,
                           aiter_tensor_t& group_rows,
                           aiter_tensor_t& num_valid_ids);

// Scatter copy (K2): x_perm[r] = x[sorted_token_ids[r]], the rows with -1 (padding) 0. x [N, H], x_perm [R, H].
void opus_moe_permute_fwd(aiter_tensor_t& x, aiter_tensor_t& sorted_token_ids, aiter_tensor_t& x_perm);

// Gather copy (K3): dx[i] = d_perm[reverse_sorted[i]] (a token without a row: 0). d_perm [R, H], dx [N, H].
void opus_moe_permute_bwd(aiter_tensor_t& d_perm, aiter_tensor_t& reverse_sorted, aiter_tensor_t& dx);

// K4: out[i] = weights[i] * y_perm[reverse_sorted[i]], one rounding to bf16. weights fp32 [N] (or [N, 1]).
void opus_moe_unpermute_fwd(aiter_tensor_t& y_perm,
                            aiter_tensor_t& reverse_sorted,
                            aiter_tensor_t& weights,
                            aiter_tensor_t& out);

// K5: with i = sorted_token_ids[r]: d_perm[r] = weights[i] * d_out[i] (padding rows 0),
// d_weights[i] = dot(d_out[i], y_perm[r]). weights / d_weights fp32 [N] (or [N, 1]).
void opus_moe_unpermute_bwd(aiter_tensor_t& d_out,
                            aiter_tensor_t& y_perm,
                            aiter_tensor_t& weights,
                            aiter_tensor_t& sorted_token_ids,
                            aiter_tensor_t& d_perm,
                            aiter_tensor_t& d_weights);
