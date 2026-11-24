// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// BF16 MQA logits fused with top-k (OPUS, gfx950): the DSA lightning indexer for training.
//
//   score[r, c] = sum_h relu(q[r, h] . k[c]) * weights[r, h],   c in [row_starts[r], row_ends[r])
//   indices[r]  = the top-k keys of row r
//
// The score matrix is never written. A call runs, in order on the current stream:
//   sample    score every 32nd key (with a fixed jitter) of each window;
//   threshold a per-row tau from the samples, and per (row, key chunk) candidate capacities;
//   emit      score every key, keep (index, score) above tau and the keys equal to it;
//   select    the exact top-k of each row's candidates by (score desc, index asc), ascending;
//   fallback  rescore the rows tau misjudged (too few or too many candidates) and select them.
//
// Outputs: indices [rows, min(topk, keys)] int32 absolute key rows, valid entries first in
// ascending order without duplicates, then -1; values (optional) fp32, -inf at -1.
//
// The caller owes what the device cannot check: windows are in key-row space (they are clamped
// to [0, keys)); q, k, weights and the outputs are contiguous; the workspace is at least the
// size aiter.ops.opus.mqa_topk_bf16 computes for the call's bounds.
#pragma once
#include "aiter_tensor.h"
#include <hip/hip_runtime.h>
#include <cstdint>

// -- the kernel arguments, shared by every stage; each kernel reads what it uses -----------------
struct mqa_topk_bf16_kargs {
    const void* ptr_q;         // [rows, 32, 128] bf16
    const void* ptr_k;         // [keys, 128] bf16
    const float* ptr_weights;  // [rows, 32]
    const int* ptr_row_starts; // [rows]
    const int* ptr_row_ends;   // [rows]
    int* ptr_indices;          // [rows, width]
    float* ptr_values;         // [rows, width], or null
    void* ptr_workspace;       // samples, thresholds, candidates, fallback rows and scratch
    int num_rows;
    int num_keys;
    int topk;
    int width;                 // min(topk, num_keys)
    int cap_row;               // candidate slots a row
    int sample_stride;         // keys a sample stands for
    int chunk_keys;            // keys a work item walks
    int flags;                 // bit 0: relu, bit 1: values
};
static_assert(sizeof(mqa_topk_bf16_kargs) <= 128, "kargs past 128 B are not preloaded");

// -- the ops (mqa_topk_bf16.cu) ---------------------------------------------------------------
// The device's instances as int records into a host int32 tensor; returns the ints it takes.
int mqa_topk_bf16_instances(int device_id, aiter_tensor_t& out);
