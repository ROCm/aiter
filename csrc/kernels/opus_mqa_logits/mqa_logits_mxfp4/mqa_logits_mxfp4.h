// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (OPUS): one interface over every registered kernel instance.
//
//   out[r, c] = sum_h relu(q[r, h] . k[c]) * weights[r, h] * weight_scale,   c < row_ends[r]
//
// Rows are grouped into sequences by `query_start_loc`: sequence b holds rows
// [query_start_loc[b], query_start_loc[b + 1]) and reads keys through block_tables[b]. Every
// window is a prefix [0, row_ends[r]). A table entry names `pages_per_block` consecutive pages.
//
// The caller never names an instance: aiter.ops.opus.mqa_logits_mxfp4 picks one from the
// attributes the registry exports (`mqa_logits_mxfp4_instances`). An instance runs in one of
// three schedule modes:
//   Table    a 32-byte record a CTA slot, built on device once per forward (build_schedule);
//   Free     no records: every CTA works out its slot of the same kind of split (decode sizes);
//   Prefill  no records: each group of rows is cut into a static `prefill_parts` pieces.
//
// THE THINGS THE CALLER OWES (not checkable on device):
//   1. row_ends does not decrease along a sequence's rows (the CTA's union window is its last
//      row's; breaking this drops logits);
//   2. row_ends[r] <= the logits width and <= what block_tables[b] covers;
//   3. query_start_loc[-1] <= rows; rows past it are never read nor written;
//   4. page id x page rows fits 32 bits (2^32 rows of the plane).
#pragma once
#include "aiter_tensor.h"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

// -- the schedule record: one a CTA slot, loaded by one s_load_dwordx8 --------------------
// A record covers up to rows_per_cta consecutive rows of ONE sequence and a range of 64-key
// tiles of their union window. local_end (the union) alone sets the trip count, so it fixes
// the barrier count; each row masks its stores by its own row_ends.
struct opus_mqa_cta_record {
    int row_id;       // the group's first row
    int batch_id;     // its sequence: the block_tables row
    int chunk_start;  // first KV tile, absolute within the sequence
    int chunk_count;  // KV tiles it covers; 0 = surplus slot
    int local_start;  // 0: windows are prefixes
    int local_end;    // the group's union window end
    int group_rows;   // rows in the group, 1..rows_per_cta
    int _pad;         // a one-round schedule's longest chunk (the H64 kernel's wave priority), else 0
};
static_assert(sizeof(opus_mqa_cta_record) == 32, "the record must stay one s_load_dwordx8");
static_assert(alignof(opus_mqa_cta_record) == 4);

// -- the kernel arguments, one struct for every instance; each kernel reads what it uses ------
// Natural gfx1250 layouts: q [rows, H, D/2], q_scale [rows, H, 4], kv [pages, PAGE, D/2],
// kv_scale [pages, PAGE, 4] (dense pages, so a key is addressed by its plane row).
struct mqa_logits_mxfp4_kargs {
    const void* ptr_q;
    const void* ptr_q_scale;
    const void* ptr_kv;
    const void* ptr_kv_scale;
    const int* ptr_block_tables;           // [num_seqs, bt_stride]
    const void* ptr_weights;               // [rows, H] bf16
    float* ptr_out;                        // [rows, out_stride], or flat with ptr_out_offsets
    const long long* ptr_out_offsets;      // [rows] each row's first logit, or null
    const int* ptr_query_start_loc;        // [num_seqs + 1]
    const int* ptr_row_ends;               // [rows]
    const opus_mqa_cta_record* ptr_cta_info;  // Table only
    int bt_stride;                         // block_tables row stride
    int tpb_shift;                         // log2(64-key tiles a table entry); contiguous tiles only
    int out_stride;                        // elements between rows when ptr_out_offsets is null
    int num_rows;
    int num_seqs;
    float weight_scale;
    int cta_target;                        // Free: the split's target CTA count
    int prefill_parts;                     // Prefill: pieces a group is cut into
};
static_assert(sizeof(mqa_logits_mxfp4_kargs) == 120);

namespace opus_logits {

enum mqa_logits_mode : int {
    SCHED_TABLE   = 0,
    SCHED_FREE    = 1,
    SCHED_PREFILL = 2,
    SCHED_MODES   = 3,
};

// -- the registry ----------------------------------------------------------------------------
// One entry a compiled instance. Python's selection rules read these attributes, never names.
struct mqa_logits_instance {
    const char* arch;
    int heads;
    int rows_per_wave;
    int waves;            // a CTA's waves; rows_per_cta = rows_per_wave * waves
    int stages;           // the LDS ring
    int scale_tdm;        // 1: the key scales by a TDM op, 0: by per-lane loads
    int gather;           // 1: a 64-key tile gathers TILE_KEYS / entry_rows table entries
    int entry_rows_min;   // rows a table entry may name (page_size * pages_per_block)
    int entry_rows_max;
    int tile_keys;        // the KV tile, the Table builder's block_k
    int modes;            // bit m: schedule mode m is compiled
    int flat_out;         // reads out_offsets
    hipError_t (*launch)(int mode, const mqa_logits_mxfp4_kargs& k, int grid, hipStream_t stream);
    hipError_t (*occupancy)(int mode, int* ctas_per_cu);
};

// gfx1250's instances (mqa_logits_mxfp4_gfx1250.cu).
const mqa_logits_instance* mqa_logits_gfx1250_instances(int* count);

}  // namespace opus_logits

// -- the three ops ---------------------------------------------------------------------------
// The device's instances as fixed-size int records, [ints a record, then the records], into a
// host int32 tensor; returns how many ints that takes (written only if `out` holds them all). See
// MQA_LOGITS_INSTANCE_FIELDS in mqa_logits_mxfp4_launcher.cu for the order.
int mqa_logits_mxfp4_instances(int device_id, aiter_tensor_t& out);

// A Table schedule for `instance`'s rows_per_cta into caller-held buffers, on device, no sync.
void mqa_logits_mxfp4_build_schedule(aiter_tensor_t& query_start_loc,
                                     aiter_tensor_t& row_ends,
                                     aiter_tensor_t& cu_tiles,
                                     aiter_tensor_t& cta_info,
                                     int instance,
                                     int num_ctas,
                                     int cta_resident);

void mqa_logits_mxfp4_launch(int instance,
                             int mode,
                             aiter_tensor_t& q,
                             aiter_tensor_t& q_scale,
                             aiter_tensor_t& kv_cache,
                             aiter_tensor_t& kv_scale,
                             aiter_tensor_t& weights,
                             aiter_tensor_t& block_tables,
                             aiter_tensor_t& query_start_loc,
                             aiter_tensor_t& row_ends,
                             aiter_tensor_t& cta_info,
                             aiter_tensor_t& out,
                             aiter_tensor_t& out_offsets,
                             float weight_scale,
                             int tpb_shift,
                             int cta_target,
                             int prefill_parts,
                             int grid);
