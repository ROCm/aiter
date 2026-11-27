// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

// Kernel arguments of the MoE token permute / unpermute kernels (host launcher and device code).
#include "opus/opus.hpp"

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
