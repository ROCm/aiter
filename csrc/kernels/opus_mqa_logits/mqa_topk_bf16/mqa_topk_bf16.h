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
//   tau       a per-row threshold: the sample of rank e + 5 sqrt(e) + 8, e the samples the top-k
//             is expected to hold (-inf for rows of at most cap_row keys);
//   emit      score every key and append the (index, score) pairs at or above tau to the row's
//             candidate pool (cap_row slots), counting them all;
//   select    the exact top-k of each row's candidates by (score desc, index asc), stored in
//             ascending index order; rows whose count is under top-k or over cap_row go to the
//   fallback  list, rescored densely by a persistent kernel and selected the same way.
// The scores are bitwise the same in every stage and in mqa_logits_bf16, so the result is exact
// and deterministic whatever order the pools fill in.
//
// Outputs: indices [rows, min(topk, keys)] int32 absolute key rows, valid entries first in
// ascending order without duplicates, then -1; values (optional) fp32, -inf at -1.
//
// The caller owes what the device cannot check: windows are in key-row space (they are clamped
// to [0, keys)); q, k, weights and the outputs are contiguous; keys < 2^24; the workspace is at
// least mqa_topk_bf16_workspace_bytes for the call's bounds.
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
    float* ptr_values;         // [rows, width] or null; mqa_logits_bf16: [rows, keys]
    void* ptr_workspace;       // mqa_topk_ws
    int num_rows;
    int num_keys;
    int topk;
    int width;                 // min(topk, num_keys)
    int cap_row;               // candidate slots a row
    int chunk_keys;            // keys a work item walks
    int fb_slots;              // fallback workgroups, a scratch slice each
    int flags;                 // MQA_TOPK_RELU | MQA_TOPK_VALUES
    int sample_chunk;          // samples a sample-stage work item walks
};
static_assert(sizeof(mqa_topk_bf16_kargs) <= 128, "kargs past 128 B are not preloaded");

namespace opus_mqa_topk {

constexpr int HEADS         = 32;
constexpr int HEAD_DIM      = 128;
constexpr int SAMPLE_STRIDE = 32;        // keys a sample stands for
constexpr int MAX_KEYS      = 1 << 24;   // exclusive: keys x 256 B fits in 32 bits
constexpr int FB_ROWS       = 4;         // rows a fallback workgroup scores at once
constexpr int MQA_TOPK_RELU   = 1;
constexpr int MQA_TOPK_VALUES = 2;
// Samples a sample-stage item walks: the stage reads every row's Q, so its items are cut finer
// than emit's (a 32-row block's samples are one long item otherwise, and few such balance badly).
constexpr int SAMPLE_CHUNK = 1024;

// The workspace: byte offsets of its parts, the same on host and device.
struct mqa_topk_ws {
    int64_t samples;    // [rows, nsamp] fp32
    int64_t tau;        // [rows] fp32
    int64_t counts;     // [rows] int32: the emit's pool counters
    int64_t cands;      // [rows, cap_row] {int32 index, fp32 score}
    int64_t fb_list;    // [rows] int32: rows to rescore
    int64_t fb_ctrl;    // [4] int32: list length, cursor
    int64_t fb_scratch; // [fb_slots, FB_ROWS, keys] fp32
    int64_t total;
};

__host__ __device__ constexpr int num_samples(int keys) {
    return (keys + SAMPLE_STRIDE - 1) / SAMPLE_STRIDE;
}

__host__ __device__ constexpr mqa_topk_ws ws_layout(int rows, int keys, int cap_row, int fb_slots) {
    auto up = [](int64_t x) { return (x + 255) / 256 * 256; };
    mqa_topk_ws w{};
    int64_t at   = 0;
    w.samples    = at; at = up(at + (int64_t)rows * num_samples(keys) * 4);
    w.tau        = at; at = up(at + (int64_t)rows * 4);
    w.counts     = at; at = up(at + (int64_t)rows * 4);
    w.cands      = at; at = up(at + (int64_t)rows * cap_row * 8);
    w.fb_list    = at; at = up(at + (int64_t)rows * 4);
    w.fb_ctrl    = at; at = up(at + 16);
    w.fb_scratch = at; at = up(at + (int64_t)fb_slots * FB_ROWS * keys * 4);
    w.total      = at;
    return w;
}

// The stages, each a kernel; an instance registers the arch-specific ones.
enum mqa_topk_stage : int {
    STAGE_LOGITS   = 0,   // mqa_logits_bf16
    STAGE_SAMPLE   = 1,
    STAGE_EMIT     = 2,
    STAGE_FALLBACK = 3,
    SCAN_STAGES    = 4,
};

// -- the registry ---------------------------------------------------------------------------------
// One entry a compiled instance. Python's selection rules read these attributes, never names.
struct mqa_topk_instance {
    const char* arch;
    int rows_per_wave;   // query rows a wave scores at once
    int waves;           // logits / emit: a workgroup's waves, sharing its keys
    int sample_waves;    // sample: a workgroup's waves, each on its own
    int tile_keys;       // keys an MFMA tile covers
    hipError_t (*launch)(int stage, const mqa_topk_bf16_kargs& k, int grid, hipStream_t stream);
    hipError_t (*occupancy)(int stage, int* wgs_per_cu);
};

using mqa_topk_kernel = void (*)(mqa_topk_bf16_kargs);

// LOGITS, SAMPLE and EMIT run SCAN_BLOCK threads a workgroup, FALLBACK WAVE_BLOCK.
template<int SCAN_BLOCK, int WAVE_BLOCK, mqa_topk_kernel LOGITS, mqa_topk_kernel SAMPLE,
         mqa_topk_kernel EMIT, mqa_topk_kernel FALLBACK>
struct instance_kernels {
    static mqa_topk_kernel of(int stage) {
        switch (stage) {
        case STAGE_LOGITS:   return LOGITS;
        case STAGE_SAMPLE:   return SAMPLE;
        case STAGE_EMIT:     return EMIT;
        case STAGE_FALLBACK: return FALLBACK;
        default:             return nullptr;
        }
    }
    static int block(int stage) {
        return stage == STAGE_FALLBACK ? WAVE_BLOCK : SCAN_BLOCK;
    }
    static hipError_t launch(int stage, const mqa_topk_bf16_kargs& k, int grid,
                             hipStream_t stream) {
        const mqa_topk_kernel kernel = of(stage);
        if (!kernel) return hipErrorInvalidValue;
        void* args[] = {const_cast<mqa_topk_bf16_kargs*>(&k)};
        return hipLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(grid),
                               dim3(block(stage)), args, 0, stream);
    }
    static hipError_t occupancy(int stage, int* wgs_per_cu) {
        const mqa_topk_kernel kernel = of(stage);
        if (!kernel) return hipErrorInvalidValue;
        return hipOccupancyMaxActiveBlocksPerMultiprocessor(wgs_per_cu, kernel, block(stage), 0);
    }
};

const mqa_topk_instance* mqa_topk_gfx950_instances(int* count);

}  // namespace opus_mqa_topk

// -- the ops (mqa_topk_bf16.cu) ---------------------------------------------------------------
// The device's instances as int records, [ints a record, then the records], into a host int32
// tensor; returns the ints it takes (written only if `out` holds them all).
int mqa_topk_bf16_instances(int device_id, aiter_tensor_t& out);

// Bytes of workspace a call with these bounds needs.
int64_t mqa_topk_bf16_workspace_bytes(int rows, int keys, int cap_row, int fb_slots);

// The fused op from addresses and scalars: the Python wrapper checks the tensors. values == 0
// means no values. stream is the hipStream_t to launch on.
void mqa_topk_bf16_launch(int64_t q,
                          int64_t k,
                          int64_t weights,
                          int64_t row_starts,
                          int64_t row_ends,
                          int64_t indices,
                          int64_t values,
                          int64_t workspace,
                          int64_t workspace_bytes,
                          int rows,
                          int keys,
                          int topk,
                          int cap_row,
                          int chunk_keys,
                          int fb_slots,
                          int relu,
                          int instance,
                          int device,
                          int64_t stream);

// Dense scores [rows, keys] fp32, -inf outside each window.
void mqa_logits_bf16_launch(int64_t q,
                            int64_t k,
                            int64_t weights,
                            int64_t row_starts,
                            int64_t row_ends,
                            int64_t out,
                            int rows,
                            int keys,
                            int chunk_keys,
                            int relu,
                            int instance,
                            int device,
                            int64_t stream);
