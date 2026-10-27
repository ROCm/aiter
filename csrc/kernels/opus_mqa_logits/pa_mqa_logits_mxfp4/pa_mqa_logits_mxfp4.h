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
// The caller never names an instance: aiter.ops.opus.pa_mqa_logits_mxfp4 picks one from the
// attributes the registry exports (`mqa_logits_mxfp4_instances`). An instance runs in one of
// three schedule modes:
//   Table    a 32-byte record a CTA slot, built on device once per forward (build_schedule);
//   Free     no records: every CTA works out its slot of the same kind of split (decode sizes);
//   Prefill  no records: each group of rows is cut into a static `prefill_parts` pieces.
//
// The caller owes what the device cannot check:
//   1. row_ends does not decrease along a sequence's rows (a CTA runs its last row's window);
//   2. row_ends[r] <= the logits width and <= what block_tables[b] covers;
//   3. query_start_loc[-1] <= rows; rows past it are neither read nor written;
//   4. page id x page rows fits 32 bits.
#pragma once
#include "aiter_tensor.h"
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>

// -- the schedule record: one a CTA slot, loaded by one s_load_dwordx8 ----------------------------
// A record covers up to rows_per_cta consecutive rows of one sequence and a range of KV tiles of
// their union window. local_end (the union) alone sets the trip count, so it fixes the barrier
// count; each row masks its stores by its own row_ends.
struct opus_mqa_cta_record {
    int row_id;       // the group's first row
    int batch_id;     // its sequence: the block_tables row
    int chunk_start;  // first KV tile, absolute within the sequence
    int chunk_count;  // KV tiles it covers; 0 = surplus slot
    int reserved;     // 0
    int local_end;    // the group's union window end
    int group_rows;   // rows in the group, 1..rows_per_cta
    int chunk_max;    // one-round schedules: a bound on the longest chunk, else 0
};
static_assert(sizeof(opus_mqa_cta_record) == 32, "the record must stay one s_load_dwordx8");
static_assert(alignof(opus_mqa_cta_record) == 4);

// -- the kernel arguments, one struct for every instance; each kernel reads what it uses ----------
// The per-arch layouts of q_scale / kv / kv_scale: aiter/ops/opus/pa_mqa_logits_mxfp4.py.
struct mqa_logits_mxfp4_kargs {
    const void* ptr_q;
    const void* ptr_q_scale;
    const void* ptr_kv;
    const void* ptr_kv_scale;
    const int* ptr_block_tables;              // [num_seqs, bt_stride]
    const void* ptr_weights;                  // [rows, H] bf16
    float* ptr_out;                           // [rows, out_stride], or flat with ptr_out_offsets
    const long long* ptr_out_offsets;         // [rows] each row's first logit, or null
    const int* ptr_query_start_loc;           // [num_seqs + 1]
    const int* ptr_row_ends;                  // [rows]
    const opus_mqa_cta_record* ptr_cta_info;  // Table only
    int bt_stride;                            // block_tables row stride
    int tpb_shift;                            // log2(64-key tiles a table entry)
    int out_stride;                           // elements between rows when ptr_out_offsets is null
    int num_rows;
    int num_seqs;
    float weight_scale;
    int cta_target;                           // Free: the split's target CTA count
    int prefill_parts;                        // Prefill: pieces a group is cut into
};
static_assert(sizeof(mqa_logits_mxfp4_kargs) == 120);

namespace opus_logits {

enum mqa_logits_mode : int {
    SCHED_TABLE   = 0,
    SCHED_FREE    = 1,
    SCHED_PREFILL = 2,
    SCHED_MODES   = 3,
};

// -- the registry ---------------------------------------------------------------------------------
// One entry a compiled instance. Python's selection rules read these attributes, never names.
struct mqa_logits_instance {
    const char* arch;
    int heads;
    int rows_per_wave;
    int waves;            // a CTA's waves; rows_per_cta = rows_per_wave * waves
    int stages;           // the LDS ring
    int scale_tdm;        // 1: the key scales are copied with the tile, 0: loaded per lane
    int gather;           // 1: a 64-key tile gathers TILE_KEYS / entry_rows table entries
    int entry_rows_min;   // rows a table entry may name (page_size * pages_per_block)
    int entry_rows_max;
    int tile_keys;        // the KV tile, the Table builder's block_k
    int modes;            // bit m: schedule mode m is compiled
    int flat_out;         // reads out_offsets
    hipError_t (*launch)(int mode, const mqa_logits_mxfp4_kargs& k, int grid, hipStream_t stream);
    hipError_t (*occupancy)(int mode, int* ctas_per_cu);
};

using mqa_logits_kernel = void (*)(mqa_logits_mxfp4_kargs);

// An instance's registry hooks from its kernel in each schedule mode (null: not compiled).
template<int BLOCK_SIZE, mqa_logits_kernel TABLE, mqa_logits_kernel FREE, mqa_logits_kernel PREFILL>
struct instance_kernels {
    static constexpr int modes = (TABLE != nullptr ? 1 << SCHED_TABLE : 0) |
                                 (FREE != nullptr ? 1 << SCHED_FREE : 0) |
                                 (PREFILL != nullptr ? 1 << SCHED_PREFILL : 0);

    static mqa_logits_kernel of(int mode) {
        switch (mode) {
        case SCHED_TABLE:   return TABLE;
        case SCHED_FREE:    return FREE;
        case SCHED_PREFILL: return PREFILL;
        default:            return nullptr;
        }
    }

    static hipError_t launch(int mode, const mqa_logits_mxfp4_kargs& k, int grid,
                             hipStream_t stream) {
        const mqa_logits_kernel kernel = of(mode);
        if (!kernel) return hipErrorInvalidValue;
        void* args[] = {const_cast<mqa_logits_mxfp4_kargs*>(&k)};
        return hipLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(grid),
                               dim3(BLOCK_SIZE), args, 0, stream);
    }

    static hipError_t occupancy(int mode, int* ctas_per_cu) {
        const mqa_logits_kernel kernel = of(mode);
        if (!kernel) return hipErrorInvalidValue;
        return hipOccupancyMaxActiveBlocksPerMultiprocessor(ctas_per_cu, kernel, BLOCK_SIZE, 0);
    }
};

// Each arch's instances (pa_mqa_logits_mxfp4_<arch>.cu).
const mqa_logits_instance* mqa_logits_gfx950_instances(int* count);
const mqa_logits_instance* mqa_logits_gfx1250_instances(int* count);

}  // namespace opus_logits

// -- the three ops --------------------------------------------------------------------------------
// The device's instances as fixed-size int records, [ints a record, then the records], into a
// host int32 tensor; returns how many ints that takes (written only if `out` holds them all).
// MQA_LOGITS_INSTANCE_FIELDS in pa_mqa_logits_mxfp4.cu gives the order.
int mqa_logits_mxfp4_instances(int device_id, aiter_tensor_t& out);

// A Table schedule for `instance`'s rows_per_cta into caller-held buffers, on device, no sync.
void mqa_logits_mxfp4_build_schedule(aiter_tensor_t& query_start_loc,
                                     aiter_tensor_t& row_ends,
                                     aiter_tensor_t& cu_groups,
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
