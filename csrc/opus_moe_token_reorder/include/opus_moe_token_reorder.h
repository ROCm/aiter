// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// gfx950 MoE token permute / unpermute: the entries (opus_moe_permute.cu) and the kernel arguments (host launcher
// and device code). aiter_tensor_t keeps this header torch-free. Shapes and numeric rules:
// aiter/ops/opus/moe_permute.py.
#include "aiter_tensor.h"

#include <optional>

// Rows every expert's group is padded to (the grouped GEMMs' OPUS_A16W16_GROUP_ALIGN).
#define OPUS_MOE_PERMUTE_GROUP_ALIGN 128

// Threads of a workgroup. One workgroup moves one row; thread i moves its 16-byte chunks i, i + 256, ...
#define OPUS_MOE_PERMUTE_THREADS 256
// bf16 elements of a chunk (one 16-byte load per lane).
#define OPUS_MOE_PERMUTE_VEC 8

// Last-level cache of gfx950 (bytes). A gather source that is read once per call and is larger than this is
// loaded non-temporal (measured: faster above, slower below, where repeated calls find the rows cached).
#define OPUS_MOE_PERMUTE_LLC_BYTES (256ll << 20)

// The plan kernels: one workgroup takes a block of OPUS_MOE_PERMUTE_PLAN_TOKENS tokens (one thread per token) and
// has one thread per expert: up to OPUS_MOE_PERMUTE_PLAN_EXPERTS experts.
#define OPUS_MOE_PERMUTE_PLAN_TOKENS 256
#define OPUS_MOE_PERMUTE_PLAN_EXPERTS 256
// The counts workspace of the plan holds the token blocks rounded up to this, for every expert (the blocks whose
// counts an expert's thread loads together).
#define OPUS_MOE_PERMUTE_PLAN_COUNT_BATCH 64
// Experts whose padding rows one fill workgroup of the plan writes (32 threads per expert).
#define OPUS_MOE_PERMUTE_PLAN_FILL_EXPERTS 8

// ---- kernel arguments -----------------------------------------------------------

// The plan: expert_ids -> the row of every token.
struct opus_moe_permute_plan_kargs {
    const int* expert_ids;   // [N]
    int* counts;             // [E, blocks rounded up]  workspace: tokens of each expert in each token block
    int* reverse_sorted;     // [N]  row of token i, -1: its id is not an expert
    int* sorted_token_ids;   // [R]  token of the row, -1: padding
    int* grouped_layout;     // [R]  expert of the row, -1: padding
    int* group_rows;         // [E]  rows of each expert, without padding
    int* num_valid_ids;      // [1]  rows in use: up to the end of the last group; the rows from there on are the tail
    int tokens, experts, rows;
    int blocks;              // token blocks: ceil(N / tokens of a block)
};

// The row movers (K2 / K3 / K4): one workgroup per row of dst. dst[b] = src[index[b]], times weights[b] if given;
// index[b] < 0: the row is zero.
struct opus_moe_permute_rows_kargs {
    const char* src;        // [rows of src, H]
    char* dst;              // [rows of dst, H]
    const int* index;       // [rows of dst]  K2: sorted_token_ids, K3 / K4: reverse_sorted
    const float* weights;   // [rows of dst]  K4, else nullptr
    int chunks;             // H / OPUS_MOE_PERMUTE_VEC
    int stream;             // non-temporal loads of src
};

// K5: one workgroup per row of d_perm.
struct opus_moe_unpermute_bwd_kargs {
    const char* d_out;             // [N, H]
    const char* y_perm;            // [R, H]
    const float* weights;          // [N]
    const int* sorted_token_ids;   // [R]  token of the row, -1: padding (the row of d_perm is zero)
    char* d_perm;                  // [R, H]
    float* d_weights;              // [N]
    int chunks;
    int stream;                    // non-temporal loads of d_out / y_perm
};

// ---- entries --------------------------------------------------------------------

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
