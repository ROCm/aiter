// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits on gfx1250 (OPUS, wave32 / WMMA 32x16x128 / TDM).
//
//     q         [total_q, H, D/2]       u8    natural   (gfx950: same)
//     q_scale   [total_q, H, 4]         u8    natural   (gfx950: [T, 2, 32, 4])
//     kv_cache  [num_blocks, PAGE, 64]  u8    natural   (gfx950: [nb, 4, PAGE, 16])
//     kv_scale  [num_blocks, PAGE, 4]   u8    natural   (gfx950: [nb, 2, 32, 4])
//     weights   [total_q, H]            bf16  natural   (gfx950: same)
//     out       [total_q, max_seq_len]  fp32
//
// HANDING THIS KERNEL THE gfx950 ARRAYS IS SILENT: every fp4 scale layout has the same byte
// count, so the launcher catches a wrong array and never a wrong permutation. Only a
// random-data comparison against a dequantized reference does.
//
// `block_tables` is sized for the 128-token TILE, not the page -- a CTA rounds its window up
// to a whole tile, so it needs `ceil(max_window / 128) * 2` entries, one more than
// `ceil(max_window / PAGE)` whenever that is odd. NOTHING CHECKS IT, and the consequence is
// not a wrong answer: the extra entry feeds a KV fetch whose columns the store masks off, so
// what it risks is reading a page id the caller never populated.
#pragma once
#include "aiter_tensor.h"
#include <cstdint>
#include <cstddef>

// Prefill OR decode over one per-tile schedule table; one CTA per SLOT, a slot being up to
// Q_PER_BLOCK query rows sharing a KV window plus a contiguous run of that tile's KV tiles.
// `build_tiles` and `build_sched` are per FORWARD against a per-LAYER launch, and none of the
// three reads anything back, so the path stays cudagraph-safe.
//
// THE TWO THINGS THE CALLER OWES, neither checkable here. Broken, the kernel does not answer
// wrong: the waves of a CTA disagree about the trip count and DEADLOCK on the phase barrier.
//   1. THE WINDOW RULE IS NON-DECREASING IN THE ROW INDEX within a tile -- the loop bound is
//      the first row's start and the last row's end, which is the union only if it is monotone.
//   2. THE STORE IS BOUNDED BY THE WINDOW, NOT BY `max_seq_len`, which rides in kargs unread.
//      A `local_ends` entry past out.size(1) writes past the row.
//
// `cta_info` is caller-allocated int32, refreshed in place so a captured graph replays from one
// address, and holds `sched_buffer_records(num_ctas)` records -- slots PLUS the builder's own
// scratch -- so sizing it at `num_ctas * 8` is UNDER-allocation.
//
// `local_starts` and `row_to_batch` may each be EMPTY: starts then read as 0, and `batch_id` is
// the tile's FIRST ROW rather than a sequence. Handing a per-SEQUENCE map to a launch with a
// per-token block table, or the reverse, reads the wrong pages and looks plausible.
void pa_mqa_logits_mxfp4_gfx1250_build_tiles(aiter_tensor_t& cu_seq_q,
                                             aiter_tensor_t& cu_tiles,
                                             int total_q,
                                             int max_tiles);

void pa_mqa_logits_mxfp4_gfx1250_build_sched(aiter_tensor_t& cu_tiles,
                                             aiter_tensor_t& local_starts,
                                             aiter_tensor_t& local_ends,
                                             aiter_tensor_t& row_to_batch,
                                             aiter_tensor_t& cta_info,
                                             int num_tiles,
                                             int num_ctas,
                                             int cta_resident);

void pa_mqa_logits_mxfp4_gfx1250_fwd_sched(aiter_tensor_t& q,
                                           aiter_tensor_t& q_scale,
                                           aiter_tensor_t& kv_cache,
                                           aiter_tensor_t& kv_scale,
                                           aiter_tensor_t& block_tables,
                                           aiter_tensor_t& weights,
                                           aiter_tensor_t& local_starts,
                                           aiter_tensor_t& local_ends,
                                           aiter_tensor_t& cta_info,
                                           aiter_tensor_t& out,
                                           int num_rows,
                                           int num_ctas,
                                           float weight_scale,
                                           int kv_block_size,
                                           int max_seq_len);

#ifdef PA_MQA_LOGITS_MXFP4_GFX1250_IMPL

using bf16_t = __bf16;

// The relu must be the IEEE-754-2019 `maximum`, which PROPAGATES NaN, and not a
// compare-and-select, which returns 0 for it. An E8M0 scale of 0xFF is NaN, the indexer's KV
// scale pool carries such bytes at the first compressed row of essentially every sequence, and
// the engine's radix top-k ranks NaN above every finite value -- so propagating pins that
// column into the top-k and swallowing drops it. Worth ~10pp of DSpark speculative acceptance.
//
// REQUIRES `-fno-finite-math-only`: `-ffast-math` implies `-ffinite-math-only`, which lets the
// compiler assume no operand is NaN and fold the builtin back to the select. The `#error` below
// is the only thing that would say so.
#define OPUS_LOGITS_RELU(x) __builtin_elementwise_maximum((x), 0.0f)

// One 32-byte record per CTA slot, arriving in one `s_load_dwordx8` -- do not shrink it.
// `chunk_count == 0` marks a surplus slot; a slot is 1..Q_PER_BLOCK rows, `group_rows` many.
//
// `chunk_start` is ABSOLUTE, a tile index into the sequence and not an offset from
// `local_start`; the store's out-of-range proof depends on it.
//
// `local_start` / `local_end` are the tile's UNION window and set the LOOP BOUND, the only
// thing they may set: the trip count fixes the barrier count, so anything per-row reaching it
// deadlocks the CTA. The STORE MASK is separate and per row.
struct opus_mqa_cta_record {
    int row_id;       // FIRST packed query row of the tile
    int batch_id;     // block_tables row
    int chunk_start;  // first KV tile (block_k units) this CTA covers, absolute
    int chunk_count;  // KV tiles it covers; 0 = surplus slot
    int local_start;  // the tile's UNION window start
    int local_end;    // the tile's UNION window end
    int group_rows;   // query rows in the tile, 1..Q_PER_BLOCK
    int _pad;
};
static_assert(sizeof(opus_mqa_cta_record) == 32,
              "the record must stay one s_load_dwordx8");
static_assert(alignof(opus_mqa_cta_record) == 4);

struct opus_mqa_logits_kargs {
    const void* __restrict__ ptr_q;         // [total_tokens, H, D/2]                fp4 (E2M1)
    const void* __restrict__ ptr_q_scale;   // [total_tokens, H, 4]                  e8m0
    const void* __restrict__ ptr_kv;        // [num_blocks, PAGE, D/2]               fp4 (E2M1)
    const void* __restrict__ ptr_kv_scale;  // [num_blocks, PAGE, 4]                 e8m0
    const int*  __restrict__ ptr_block_tables; // [batch, max_blocks_per_seq] int32
    const void* __restrict__ ptr_weights;   // [total_tokens, H] bf16
    float* __restrict__ ptr_out;             // [total_tokens, max_seq_len] fp32

    // PER-ROW window arrays, each [total_q] int32, read for the STORE MASK only -- the loop bound
    // comes from the record's union. `ptr_local_starts` MAY BE NULL (every row starts at 0);
    // `ptr_local_ends` may not, being the store's upper bound.
    const int* __restrict__ ptr_row_to_batch;   // builder only; the record carries batch_id
    const int* __restrict__ ptr_local_starts;
    const int* __restrict__ ptr_local_ends;
    // Read by neither the kernel nor the schedule. `ptr_cu_seq_q` is the BATCH prefix sum and is
    // an input to the tile cut, not to the launch.
    const int* __restrict__ ptr_cu_seq_q;
    int   split_kv;            // context splits per row (>= 1); unused here
    int   num_rows;            // total query rows
    int   num_batches;         // real batch count

    int   max_seq_len;
    int   stride_out_row;      // out row stride in elements (== max_seq_len for dense out)
    float weight_scale;
    int   block_k;             // KV tile size along seq_kv (== Traits::KV_TILE_SIZE)
    int   kv_block_size;       // paged block (page) size (== Traits::PAGE_SIZE)
    int   max_blocks_per_seq;  // block_tables row stride

    // APPENDED, not grouped with the pointers above: every field before this keeps the kernarg
    // offset it has on gfx950, which the offsetof block below enforces.
    const opus_mqa_cta_record* __restrict__ ptr_cta_info;  // [num_ctas]
    int   num_ctas;            // grid.x of the launch; slots past the work idle
};

// A field reordered or retyped on either target fails here instead of producing a launcher that
// reads the wrong dword.
static_assert(sizeof(opus_mqa_logits_kargs)  == 144);   // 128 + the appended pair
static_assert(alignof(opus_mqa_logits_kargs) == 8);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_q             ) ==   0);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_q_scale       ) ==   8);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_kv            ) ==  16);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_kv_scale      ) ==  24);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_block_tables  ) ==  32);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_weights       ) ==  40);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_out           ) ==  48);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_row_to_batch  ) ==  56);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_local_starts  ) ==  64);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_local_ends    ) ==  72);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_cu_seq_q      ) ==  80);
static_assert(offsetof(opus_mqa_logits_kargs, split_kv          ) ==  88);
static_assert(offsetof(opus_mqa_logits_kargs, num_rows          ) ==  92);
static_assert(offsetof(opus_mqa_logits_kargs, num_batches       ) ==  96);
static_assert(offsetof(opus_mqa_logits_kargs, max_seq_len       ) == 100);
static_assert(offsetof(opus_mqa_logits_kargs, stride_out_row    ) == 104);
static_assert(offsetof(opus_mqa_logits_kargs, weight_scale      ) == 108);
static_assert(offsetof(opus_mqa_logits_kargs, block_k           ) == 112);
static_assert(offsetof(opus_mqa_logits_kargs, kv_block_size     ) == 116);
static_assert(offsetof(opus_mqa_logits_kargs, max_blocks_per_seq) == 120);
static_assert(offsetof(opus_mqa_logits_kargs, ptr_cta_info      ) == 128);
static_assert(offsetof(opus_mqa_logits_kargs, num_ctas          ) == 136);

__host__ __device__ inline int ceil_div_i(int a, int b) { return (a + b - 1) / b; }

// ── the schedule builder ──
// KEEP THIS BLOCK FIRST AMONG THE `namespace opus_logits` BLOCKS AND ITS CLOSING BRACE
// SPELLED WITH TWO SPACES: the out-of-tree equivalence check against its opus-ops twin locates
// it by those two strings, which is why the `mqa_logits_sched` enum sits after the builder.
namespace opus_logits {

constexpr int SCHED_BUILD_BLOCK      = 256;
constexpr int SCHED_BUILD_BLOCK_WIDE = 1024;

// CTAs resident on the part: 256 CUs x occupancy 1 x 4 waves per CTA -- a hardware number.
constexpr int SCHED_CTA_RESIDENT = 256;

// The CTA count the split aims at: a WHOLE NUMBER OF ROUNDS. One CTA per tile leaves the last
// round `resident - (nz mod resident)` CTAs empty, and that tail is not small -- 1025 tiles is
// four full rounds and one straggler. 0 means do not split.
__host__ __device__ inline int sched_target(int nz_tiles, int resident = SCHED_CTA_RESIDENT) {
    if (resident <= 0) return 0;
    if (nz_tiles <= resident) return resident;
    return ((nz_tiles + resident - 1) / resident) * resident;
}

// The GRID: the tile count rounded to a whole number of rounds, one round as the floor, and a
// host constant so a captured graph replays at one shape. Deliberately tight -- a surplus
// slot's CTA still reads a 32-byte record nobody else touches.
__host__ __device__ inline int sched_slots(int num_tiles, int resident = SCHED_CTA_RESIDENT) {
    const int n = num_tiles > resident ? num_tiles : resident;
    return (n + resident - 1) / resident * resident;
}

constexpr int SCHED_BUILD_MAX_BLOCKS = 256;
constexpr int SCHED_SCRATCH_INTS     = 3 * SCHED_BUILD_MAX_BLOCKS;
constexpr int SCHED_SCRATCH_RECORDS =
    (SCHED_SCRATCH_INTS * (int)sizeof(int) + (int)sizeof(opus_mqa_cta_record) - 1) /
    (int)sizeof(opus_mqa_cta_record);

__host__ __device__ inline int sched_buffer_records(int num_ctas) {
    return num_ctas + SCHED_SCRATCH_RECORDS;
}

struct sched_build_plan {
    int block;   // workgroup width
    int blocks;  // workgroups for the emit; 1 means the single-workgroup kernel
};

// How to launch, given the shape. Both hosts call this so neither can drift on the policy.
__host__ inline sched_build_plan sched_plan(int num_tiles, int num_ctas) {
    sched_build_plan p{SCHED_BUILD_BLOCK, 1};
    if (num_tiles < 512) return p;                 // narrow, one workgroup
    p.block = SCHED_BUILD_BLOCK_WIDE;
    if (num_tiles <= 4096) return p;               // wide, one workgroup
    p.blocks = (num_ctas + SCHED_BUILD_BLOCK - 1) / SCHED_BUILD_BLOCK;
    if (p.blocks < 1) p.blocks = 1;
    if (p.blocks > SCHED_BUILD_MAX_BLOCKS) p.blocks = SCHED_BUILD_MAX_BLOCKS;
    return p;
}

namespace sched_detail {

__device__ inline void sync_block() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
}

template<int BLOCK, bool IS_MAX>
__device__ inline int block_reduce(int v, int* s, int tid) {
    s[tid] = v;
    sync_block();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const int o = s[tid + off];
            s[tid] = IS_MAX ? (o > s[tid] ? o : s[tid]) : (s[tid] + o);
        }
        sync_block();
    }
    const int r = s[0];
    sync_block();
    return r;
}

template<int BLOCK>
__device__ inline void block_reduce_stats(int vmax, int vsum, int vnz, int* s, int tid,
                                          int& omax, int& osum, int& onz) {
    s[tid]             = vmax;
    s[BLOCK + tid]     = vsum;
    s[2 * BLOCK + tid] = vnz;
    sync_block();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const int om = s[tid + off];
            if (om > s[tid]) s[tid] = om;
            s[BLOCK + tid]     += s[BLOCK + tid + off];
            s[2 * BLOCK + tid] += s[2 * BLOCK + tid + off];
        }
        sync_block();
    }
    omax = s[0];
    osum = s[BLOCK];
    onz  = s[2 * BLOCK];
    sync_block();
}

template<int BLOCK>
__device__ inline int block_scan_excl(int v, int* s, int tid, int& total) {
    s[tid] = v;
    sync_block();
    for (int off = 1; off < BLOCK; off <<= 1) {
        const int add = (tid >= off) ? s[tid - off] : 0;
        sync_block();
        s[tid] += add;
        sync_block();
    }
    total = s[BLOCK - 1];
    const int incl = s[tid];
    sync_block();
    return incl - v;
}

// A tile's window is the UNION of its rows', and with a NON-DECREASING rule that is the first
// row's start and the LAST row's end. That contract is the caller's; nothing here checks it.
__device__ inline int tile_union_end(const int* __restrict__ local_ends, int r0, int r1) {
    return r1 > r0 ? local_ends[r1 - 1] : 0;
}
__device__ inline int tile_union_start(const int* __restrict__ local_starts, int r0, int r1) {
    return (local_starts && r1 > r0) ? local_starts[r0] : 0;
}

// BRANCHLESS ON PURPOSE. An `if (e <= 0) return 0;` here reads naturally and costs 38% at 16384
// tiles on the gfx950 sibling: the callers are latency-bound strided walks, and a branch on the
// value just loaded serializes what was a pipelined stream.
__device__ inline int tile_kv_tiles(const int* __restrict__ cu_tiles,
                                    const int* __restrict__ local_starts,
                                    const int* __restrict__ local_ends,
                                    int t, int block_k) {
    const int r0    = cu_tiles[t];
    const int r1    = cu_tiles[t + 1];
    const int e     = tile_union_end(local_ends, r0, r1);
    const int first = tile_union_start(local_starts, r0, r1) / block_k;
    const int end   = e > 0 ? ((e + block_k - 1) / block_k) : 0;
    return end > first ? end - first : 0;
}

// Read back out of the table `emit_fast` already wrote, rather than recomputed: with per-row
// windows the count needs a DEPENDENT global load. Safe only before `general_emit` writes its
// first slot -- **`general_emit` may NOT do this**, since it writes slots while reading tiles
// from the same array and a split makes those ranges overlap.
__device__ inline int tile_kv_tiles_cached(const opus_mqa_cta_record* __restrict__ cta_info,
                                           int t) {
    return cta_info[t].chunk_count;
}

// One strided walk that reduces AND emits: the one-CTA-per-tile record does not depend on the
// reduction, so it is written speculatively and overwritten only if a tile needs splitting.
__device__ inline void emit_fast(const int* __restrict__ cu_tiles,
                                 const int* __restrict__ local_starts,
                                 const int* __restrict__ local_ends,
                                 const int* __restrict__ row_to_batch,
                                 opus_mqa_cta_record* __restrict__ cta_info,
                                 int first_tile, int num_tiles, int stride, int block_k,
                                 int& part_max, int& part_sum, int& part_nz) {
    part_max = 0;
    part_sum = 0;
    part_nz  = 0;
    for (int t = first_tile; t < num_tiles; t += stride) {
        const int r0    = cu_tiles[t];
        const int r1    = cu_tiles[t + 1];
        const int e     = tile_union_end(local_ends, r0, r1);
        const int s     = tile_union_start(local_starts, r0, r1);
        const int b     = row_to_batch ? row_to_batch[r0] : r0;
        const int first = s / block_k;
        const int end   = e > 0 ? ((e + block_k - 1) / block_k) : 0;
        const int n     = end > first ? end - first : 0;
        if (n > part_max) part_max = n;
        part_sum += n;
        part_nz  += (n > 0);
        opus_mqa_cta_record rec{};
        rec.row_id      = r0;
        rec.batch_id    = b;
        rec.chunk_start = first;
        rec.chunk_count = n;
        rec.local_start = s;
        rec.local_end   = e;
        rec.group_rows  = r1 - r0;
        cta_info[t]     = rec;
    }
}

// Surplus slots. MARKED, not left stale: the table is reused in place across cudagraph replays,
// so a slot that carried work last forward must not carry it again this one.
__device__ inline void mark_surplus(opus_mqa_cta_record* __restrict__ cta_info,
                                    int first_slot, int num_ctas, int stride) {
    for (int slot = first_slot; slot < num_ctas; slot += stride) {
        opus_mqa_cta_record rec{};
        rec.chunk_count = 0;
        cta_info[slot]  = rec;
    }
}

template<int BLOCK>
__device__ inline int settle_safe(const opus_mqa_cta_record* __restrict__ cta_info,
                                  int num_tiles, int num_ctas,
                                  int safe, int max_tiles, int nz_tiles, int* smem, int tid) {
    if (nz_tiles >= num_ctas) return max_tiles;
    auto ctas_for = [&](int s) {
        int part = 0;
        for (int t = tid; t < num_tiles; t += BLOCK) {
            const int n = tile_kv_tiles_cached(cta_info, t);
            part += (n + s - 1) / s;
        }
        return block_reduce<BLOCK, false>(part, smem, tid);
    };
    if (ctas_for(safe) > num_ctas) {
        int lo = safe + 1, hi = max_tiles > safe ? max_tiles : safe + 1;
        while (lo < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (ctas_for(mid) <= num_ctas) hi = mid;
            else lo = mid + 1;
        }
        safe = lo;
    }
    return safe;
}

template<int BLOCK>
__device__ inline int general_emit(const int* __restrict__ cu_tiles,
                                   const int* __restrict__ local_starts,
                                   const int* __restrict__ local_ends,
                                   const int* __restrict__ row_to_batch,
                                   opus_mqa_cta_record* __restrict__ cta_info,
                                   int num_tiles, int num_ctas, int block_k, int safe,
                                   int* smem, int* s_excl, int* s_tiles, int* s_batch,
                                   int* s_end, int* s_ls, int* s_row, int* s_rows, int tid) {
    int carry = 0;
    for (int base = 0; base < num_tiles; base += BLOCK) {
        const int t = base + tid;
        const int tiles = (t < num_tiles)
                            ? tile_kv_tiles(cu_tiles, local_starts, local_ends, t, block_k) : 0;
        const int nc = (tiles + safe - 1) / safe;
        int block_total = 0;
        const int excl = block_scan_excl<BLOCK>(nc, smem, tid, block_total);
        const int r0 = (t < num_tiles) ? cu_tiles[t]     : 0;
        const int r1 = (t < num_tiles) ? cu_tiles[t + 1] : 0;
        s_excl[tid]  = excl;
        s_tiles[tid] = tiles;
        s_batch[tid] = (t < num_tiles) ? (row_to_batch ? row_to_batch[r0] : r0) : 0;
        s_end[tid]   = tile_union_end(local_ends, r0, r1);
        s_ls[tid]    = tile_union_start(local_starts, r0, r1);
        s_row[tid]   = r0;
        s_rows[tid]  = r1 - r0;
        sync_block();

        for (int j = tid; j < block_total; j += BLOCK) {
            int lo = 0, hi = BLOCK;
            while (lo < hi) {           // upper_bound(s_excl, j) - 1
                const int mid = (lo + hi) >> 1;
                if (s_excl[mid] <= j) lo = mid + 1; else hi = mid;
            }
            const int tl = lo - 1;      // >= 0: s_excl[0] is 0 and j >= 0
            const int i  = j - s_excl[tl];
            const int slot = carry + j;
            if (slot < num_ctas) {      // cannot fire while num_ctas >= the schedule's need
                // EVEN split: `safe` decides HOW MANY chunks a tile gets and must not also decide their size.
                // A phase costs the longest chunk, so 96 + 32 is the same CTA count as 64 + 64 and 44% slower.
                const int n  = s_tiles[tl];
                const int nc = (n + safe - 1) / safe;   // >= 1: a zero-chunk tile emits no slot
                const int q  = n / nc, r = n % nc;
                opus_mqa_cta_record rec{};
                rec.row_id      = s_row[tl];
                rec.batch_id    = s_batch[tl];
                rec.chunk_start = s_ls[tl] / block_k + i * q + (i < r ? i : r);
                rec.chunk_count = q + (i < r ? 1 : 0);
                rec.local_start = s_ls[tl];
                rec.local_end   = s_end[tl];
                rec.group_rows  = s_rows[tl];
                cta_info[slot]  = rec;
            }
        }
        carry += block_total;
        sync_block();
    }
    return carry;
}

__device__ inline int split_factor(int total_tiles, int cta_target) {
    const int safe = total_tiles > 0 ? ((total_tiles + cta_target - 1) / cta_target) : 1;
    return safe < 1 ? 1 : safe;
}

template<int BLOCK>
__device__ inline void settle_and_emit(const int* __restrict__ cu_tiles,
                                       const int* __restrict__ local_starts,
                                       const int* __restrict__ local_ends,
                                       const int* __restrict__ row_to_batch,
                                       opus_mqa_cta_record* __restrict__ cta_info,
                                       int num_tiles, int num_ctas, int block_k, int resident,
                                       int max_tiles, int total_tiles, int nz_tiles,
                                       bool surplus_done,
                                       int* smem, int* s_excl, int* s_tiles, int* s_batch,
                                       int* s_end, int* s_ls, int* s_row, int* s_rows, int tid) {
    const int aim   = (resident <= 0) ? 0 : sched_target(nz_tiles, resident);
    const int safe0 = (aim <= 0) ? max_tiles : split_factor(total_tiles, aim);
    if (max_tiles <= safe0) {
        if (!surplus_done) mark_surplus(cta_info, num_tiles + tid, num_ctas, BLOCK);
        return;
    }
    const int safe = settle_safe<BLOCK>(cta_info, num_tiles, num_ctas,
                                        safe0, max_tiles, nz_tiles, smem, tid);
    const int carry = general_emit<BLOCK>(cu_tiles, local_starts, local_ends, row_to_batch,
                                          cta_info, num_tiles, num_ctas, block_k, safe, smem,
                                          s_excl, s_tiles, s_batch, s_end, s_ls, s_row, s_rows,
                                          tid);
    mark_surplus(cta_info, carry + tid, num_ctas, BLOCK);
}

}  // namespace sched_detail

constexpr int GROUPS_BUILD_BLOCK     = 256;
constexpr int GROUPS_BUILD_MAX_BATCH = 2048;   // 8 KB of LDS for the per-batch offsets

// Tiles the cut can produce, from the STATIC shapes alone -- which is what keeps the launch
// cudagraph-safe. Never short, because a sum of per-batch roundings is at least the rounding of
// the sum; the slack is at most `batch - 1` tiles, each of which gets an empty record.
__host__ __device__ inline int max_tiles_for(int total_q, int batch, int qpb) {
    return (total_q + batch * (qpb - 1)) / qpb;
}

// Cut each batch's rows into runs of at most `qpb` and write the tile boundaries.
//
// **THIS RETIRES A CONTRACT THE CALLER CANNOT SAFELY BREAK.** "A tile is contiguous rows of
// ONE batch" gives no wrong answer when violated -- the CTA's waves disagree about the trip
// count and DEADLOCK on the phase barrier. One workgroup, so the scan in it caps the batch.
__global__ __launch_bounds__(GROUPS_BUILD_BLOCK)
void mqa_logits_build_tiles(const int* __restrict__ cu_seq_q,
                            int* __restrict__ cu_tiles,
                            int batch, int max_tiles, int qpb) {
    __shared__ int g_off[GROUPS_BUILD_MAX_BATCH + 1];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();

    for (int b = tid; b < batch; b += GROUPS_BUILD_BLOCK)
        g_off[b] = (cu_seq_q[b + 1] - cu_seq_q[b] + qpb - 1) / qpb;
    sched_detail::sync_block();

    if (tid == 0) {
        int acc = 0;
        for (int b = 0; b < batch; ++b) { const int c = g_off[b]; g_off[b] = acc; acc += c; }
        g_off[batch] = acc;
    }
    sched_detail::sync_block();

    const int num_tiles = g_off[batch];
    const int end_row   = cu_seq_q[batch];

    for (int t = tid; t <= max_tiles; t += GROUPS_BUILD_BLOCK) {
        if (t >= num_tiles) { cu_tiles[t] = end_row; continue; }
        int lo = 0, hi = batch - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (g_off[mid] <= t) lo = mid; else hi = mid - 1;
        }
        cu_tiles[t] = cu_seq_q[lo] + (t - g_off[lo]) * qpb;
    }
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK)
void mqa_logits_build_sched(const int* __restrict__ cu_tiles,
                            const int* __restrict__ local_starts,
                            const int* __restrict__ local_ends,
                            const int* __restrict__ row_to_batch,
                            opus_mqa_cta_record* __restrict__ cta_info,
                            int num_tiles, int num_ctas,
                            int block_k, int resident) {
    __shared__ int smem[3 * BLOCK];
    __shared__ int s_excl[BLOCK];
    __shared__ int s_tiles[BLOCK];
    __shared__ int s_batch[BLOCK];
    __shared__ int s_end[BLOCK];
    __shared__ int s_ls[BLOCK];
    __shared__ int s_row[BLOCK];
    __shared__ int s_rows[BLOCK];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();

    int part_max = 0, part_sum = 0, part_nz = 0;
    sched_detail::emit_fast(cu_tiles, local_starts, local_ends, row_to_batch, cta_info,
                            tid, num_tiles, BLOCK, block_k, part_max, part_sum, part_nz);
    int max_tiles = 0, total_tiles = 0, nz_tiles = 0;
    sched_detail::block_reduce_stats<BLOCK>(part_max, part_sum, part_nz, smem, tid,
                                            max_tiles, total_tiles, nz_tiles);
    sched_detail::settle_and_emit<BLOCK>(cu_tiles, local_starts, local_ends, row_to_batch,
                                         cta_info, num_tiles, num_ctas, block_k, resident,
                                         max_tiles, total_tiles, nz_tiles, /*surplus_done=*/false,
                                         smem, s_excl, s_tiles, s_batch, s_end, s_ls,
                                         s_row, s_rows, tid);
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK)
void mqa_logits_build_sched_emit(const int* __restrict__ cu_tiles,
                                 const int* __restrict__ local_starts,
                                 const int* __restrict__ local_ends,
                                 const int* __restrict__ row_to_batch,
                                 opus_mqa_cta_record* __restrict__ cta_info,
                                 int* __restrict__ scratch,
                                 int num_tiles, int num_ctas, int block_k, int blocks) {
    __shared__ int smem[3 * BLOCK];
    const int tid    = (int)__builtin_amdgcn_workitem_id_x();
    const int bid    = (int)__builtin_amdgcn_workgroup_id_x();
    const int stride = BLOCK * blocks;

    int part_max = 0, part_sum = 0, part_nz = 0;
    sched_detail::emit_fast(cu_tiles, local_starts, local_ends, row_to_batch, cta_info,
                            bid * BLOCK + tid, num_tiles, stride, block_k,
                            part_max, part_sum, part_nz);
    sched_detail::mark_surplus(cta_info, num_tiles + bid * BLOCK + tid, num_ctas, stride);

    int bmax = 0, bsum = 0, bnz = 0;
    sched_detail::block_reduce_stats<BLOCK>(part_max, part_sum, part_nz, smem, tid,
                                            bmax, bsum, bnz);
    if (tid == 0) {
        scratch[bid]                              = bmax;
        scratch[SCHED_BUILD_MAX_BLOCKS + bid]     = bsum;
        scratch[2 * SCHED_BUILD_MAX_BLOCKS + bid] = bnz;
    }
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK)
void mqa_logits_build_sched_finish(const int* __restrict__ cu_tiles,
                                   const int* __restrict__ local_starts,
                                   const int* __restrict__ local_ends,
                                   const int* __restrict__ row_to_batch,
                                   opus_mqa_cta_record* __restrict__ cta_info,
                                   const int* __restrict__ scratch,
                                   int num_tiles, int num_ctas, int block_k, int resident,
                                   int blocks) {
    __shared__ int smem[3 * BLOCK];
    __shared__ int s_excl[BLOCK];
    __shared__ int s_tiles[BLOCK];
    __shared__ int s_batch[BLOCK];
    __shared__ int s_end[BLOCK];
    __shared__ int s_ls[BLOCK];
    __shared__ int s_row[BLOCK];
    __shared__ int s_rows[BLOCK];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();

    int part_max = 0, part_sum = 0, part_nz = 0;
    for (int i = tid; i < blocks; i += BLOCK) {
        const int m = scratch[i];
        if (m > part_max) part_max = m;
        part_sum += scratch[SCHED_BUILD_MAX_BLOCKS + i];
        part_nz  += scratch[2 * SCHED_BUILD_MAX_BLOCKS + i];
    }
    int max_tiles = 0, total_tiles = 0, nz_tiles = 0;
    sched_detail::block_reduce_stats<BLOCK>(part_max, part_sum, part_nz, smem, tid,
                                            max_tiles, total_tiles, nz_tiles);
    sched_detail::settle_and_emit<BLOCK>(cu_tiles, local_starts, local_ends, row_to_batch,
                                         cta_info, num_tiles, num_ctas, block_k, resident,
                                         max_tiles, total_tiles, nz_tiles, /*surplus_done=*/true,
                                         smem, s_excl, s_tiles, s_batch, s_end, s_ls,
                                         s_row, s_rows, tid);
}

}  // namespace opus_logits

// Kept here, after the builder, for the reason the ordering note above gives -- it belongs with
// the ABI and cannot sit there.
namespace opus_logits {
// gfx950's two schedule-free modes are names only here; this target compiles `Table` and nothing
// else. Prefill gave every tile one CTA whatever its window length, which is the load imbalance
// the table exists to fix.
enum class mqa_logits_sched {
    Prefill,
    Decode,
    Table,
};
}

// ONE kv_cache layout, the natural one: with no FlyDSL kernel on this target there is nothing
// to be byte-compatible with, so the LDS address formulas below are that layout's, once.
template<int Q_PER_BLOCK_  = 4,
         int LDS_STAGES_   = 2,
         int KV_TILE_SIZE_ = 128,
         int PAGE_SIZE_    = 64,
         int HEAD_DIM_     = 128,
         int N_HEADS_      = 64>
struct opus_mqa_logits_fp4_qshare_traits {
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;  // block_k
    static constexpr int PAGE_SIZE    = PAGE_SIZE_;     // kv_block_size
    static constexpr int HEAD_DIM     = HEAD_DIM_;
    static constexpr int N_HEADS      = N_HEADS_;

    // A CONSTANT, never `opus::get_warp_size()`: that returns 64 in the host pass, which would
    // build the wave64 fragment layout with every byte count still matching.
    static constexpr int WAVE_SIZE   = 32;
    static constexpr int Q_PER_BLOCK = Q_PER_BLOCK_;               // query rows per CTA == waves
    static constexpr int NUM_WAVES   = Q_PER_BLOCK;
    static constexpr int BLOCK_SIZE  = NUM_WAVES * WAVE_SIZE;      // 128

    using D_WEIGHT = bf16_t;   // per-head weights
    using D_ACC    = float;    // WMMA accumulator (C)
    using D_OUT    = float;    // output logits
    using D_SCALE  = int;      // E8M0 blockscale, packed as one int32 dword per lane (BX32)

    static constexpr int MMA_M = 32;   // heads
    static constexpr int MMA_N = 16;   // tokens
    static constexpr int MMA_K = 128;  // head_dim

    static constexpr int ELEM_BITS   = 4;    // fp4: half a byte per element
    static constexpr int SCALE_BLOCK = 32;   // E8M0 blockscale granularity

    static constexpr int M_TILES  = N_HEADS / MMA_M;      // 2  (head tiles along M)
    static constexpr int K_TILES  = HEAD_DIM / MMA_K;     // 1  (no outer K loop)
    static constexpr int K_CHUNKS = MMA_K / SCALE_BLOCK;  // 4  (32-K scale blocks per instruction)
    static constexpr int SCALE_BLOCKS_ROW = HEAD_DIM / SCALE_BLOCK;  // 4 == K_TILES * K_CHUNKS

    static constexpr int N_TILES         = KV_TILE_SIZE / MMA_N;   // 8
    static constexpr int TILES_PER_PAGE  = PAGE_SIZE / MMA_N;      // 4
    static constexpr int PAGES_PER_TILE  = KV_TILE_SIZE / PAGE_SIZE;  // 2

    static constexpr int C_FRAG = MMA_M * MMA_N / WAVE_SIZE;   // 16 floats per lane per m-tile
    static constexpr int GRPN_C = MMA_N;                       // 16: n == lane % 16
    static constexpr int GRPM_C = WAVE_SIZE / GRPN_C;          // 2:  M spread over 2 lane groups
    static constexpr int PACK_C = C_FRAG / 2;                  // 8:  c[0..7] is m = base + i
    static constexpr int REPT_C = C_FRAG / PACK_C;             // 2:  and c[8..15] is base + 16
    static constexpr int HEADS_PER_LANE = M_TILES * C_FRAG;    // 32
    static constexpr int SWAP_DISTANCE  = GRPN_C;              // 16 == permlane16_swap's distance

    static constexpr int KV_GRP_ELEMS = SCALE_BLOCK;                     // 32 fp4 per block
    static constexpr int KV_GRP_BYTES = KV_GRP_ELEMS * ELEM_BITS / 8;    // 16 B == one ds_load_b128
    static constexpr int A_BYTES_PER_LANE = MMA_M * MMA_K / WAVE_SIZE * ELEM_BITS / 8;  // 64
    static constexpr int B_BYTES_PER_LANE = MMA_N * MMA_K / WAVE_SIZE * ELEM_BITS / 8;  // 32
    static constexpr int A_VGPR_GROUPS = A_BYTES_PER_LANE / KV_GRP_BYTES;   // 4
    static constexpr int B_VGPR_GROUPS = B_BYTES_PER_LANE / KV_GRP_BYTES;   // 2

    static constexpr int SCALE_BYTES_PER_DWORD = K_CHUNKS;   // 4
    static_assert(SCALE_BYTES_PER_DWORD == 4,
                  "a BX32 scale operand is one dword; byte b must be block b with nothing left over");

    static constexpr int TOKEN_BYTES  = HEAD_DIM * ELEM_BITS / 8;     // 64 == one token's fp4 row
    static constexpr int Q_ROW_BYTES  = N_HEADS * TOKEN_BYTES;        // 4096 == H * D/2
    static constexpr int QS_ROW_BYTES = N_HEADS * SCALE_BYTES_PER_DWORD;   // 256
    static constexpr int KV_PAGE_BYTES  = PAGE_SIZE * TOKEN_BYTES;    // 4096
    static constexpr int KVS_PAGE_BYTES = PAGE_SIZE * SCALE_BYTES_PER_DWORD;  // 256
    static constexpr int W_ROW_ELEMS    = N_HEADS;                    // natural [T, H] bf16

    static constexpr int KV_CHUNK_BYTES = PAGE_SIZE * KV_GRP_BYTES;   // 1024

    // ── the ABI, as byte offsets within one row / one page ──
    // `block` is the 32-element E8M0 block index, and it sits INSIDE the token: a token's 128 fp4
    // elements are 64 contiguous bytes and block `b` is the 16 B at `b * 16`. LDS_BLOCK_BYTES below
    // takes it from here rather than a literal at the read site -- reading the wrong 16 B is
    // invisible in the resource report.
    static constexpr int q_byte(int head, int block) {
        return head * TOKEN_BYTES + block * KV_GRP_BYTES;
    }
    static constexpr int q_scale_byte(int head) {
        return head * SCALE_BYTES_PER_DWORD;
    }
    static constexpr int kv_byte(int token, int block) {
        return token * TOKEN_BYTES + block * KV_GRP_BYTES;
    }
    static constexpr int kv_scale_byte(int token) {
        return token * SCALE_BYTES_PER_DWORD;
    }

    // ── the measured fragment maps, shared by the kernel and the op test's reference ──
    static constexpr int lane_m0(int lane) { return lane % GRPN_C; }   // L % 16
    static constexpr int lane_g (int lane) { return lane / GRPN_C; }   // L / 16

    static constexpr int frag_a_row  (int lane, int v) { return lane_m0(lane) + (v >= 2 ? 16 : 0); }
    static constexpr int frag_a_block(int lane, int v) { return lane_g(lane) + (v % 2 ? 2 : 0); }
    static constexpr int frag_b_col  (int lane)        { return lane_m0(lane); }
    static constexpr int frag_b_block(int lane, int v) { return lane_g(lane) + (v ? 2 : 0); }
    static constexpr int frag_c_n    (int lane)        { return lane_m0(lane); }
    static constexpr int frag_c_m    (int lane, int reg) {
        return (reg < PACK_C ? 0 : 16) + lane_g(lane) * PACK_C + reg % PACK_C;
    }
    static constexpr bool frag_a_is_bijection() {
        bool seen[MMA_M][K_CHUNKS] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int v = 0; v < A_VGPR_GROUPS; ++v) {
                const int r = frag_a_row(lane, v), b = frag_a_block(lane, v);
                if (r < 0 || r >= MMA_M || b < 0 || b >= K_CHUNKS || seen[r][b]) return false;
                seen[r][b] = true;
            }
        for (int r = 0; r < MMA_M; ++r)
            for (int b = 0; b < K_CHUNKS; ++b) if (!seen[r][b]) return false;
        return true;
    }
    static constexpr bool frag_b_is_bijection() {
        bool seen[MMA_N][K_CHUNKS] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int v = 0; v < B_VGPR_GROUPS; ++v) {
                const int n = frag_b_col(lane), b = frag_b_block(lane, v);
                if (n < 0 || n >= MMA_N || b < 0 || b >= K_CHUNKS || seen[n][b]) return false;
                seen[n][b] = true;
            }
        for (int n = 0; n < MMA_N; ++n)
            for (int b = 0; b < K_CHUNKS; ++b) if (!seen[n][b]) return false;
        return true;
    }
    static constexpr bool frag_c_is_bijection() {
        bool seen[MMA_M][MMA_N] = {};
        for (int lane = 0; lane < WAVE_SIZE; ++lane)
            for (int reg = 0; reg < C_FRAG; ++reg) {
                const int m = frag_c_m(lane, reg), n = frag_c_n(lane);
                if (m < 0 || m >= MMA_M || n < 0 || n >= MMA_N || seen[m][n]) return false;
                seen[m][n] = true;
            }
        for (int m = 0; m < MMA_M; ++m)
            for (int n = 0; n < MMA_N; ++n) if (!seen[m][n]) return false;
        return true;
    }
    static_assert(frag_a_is_bijection(), "A's (lane, VGPR group) -> (row, block) map is not a bijection");
    static_assert(frag_b_is_bijection(), "B's (lane, VGPR group) -> (col, block) map is not a bijection");
    static_assert(frag_c_is_bijection(), "C's (lane, reg) -> (m, n) map is not a bijection");

    static constexpr int scale_a_row(int lane) { return lane; }
    static constexpr int scale_b_sel (int n_tile) { return n_tile % 2; }
    static constexpr int scale_b_lane(int n_tile, int n) { return scale_b_sel(n_tile) * GRPN_C + n; }

    // ── LDS: the tile image, and the one padding policy that breaks the bank conflict ──
    // TDM copies a page as it is, so the pad is the ONLY LDS-side freedom: WMMA fixes which lane
    // reads which token, so the lane mapping cannot be rearranged instead.
    static constexpr int LDS_PAD_INTERVAL = 128;
    static constexpr int LDS_PAD_AMOUNT   =  16;
    static constexpr int LDS_PAD_STRIDE = LDS_PAD_INTERVAL ? LDS_PAD_INTERVAL : 1;

    static constexpr int lds_expand(int page_byte) {
        return page_byte + (page_byte / LDS_PAD_STRIDE) * LDS_PAD_AMOUNT;
    }
    static constexpr int lds_byte(int token_in_page, int block) {
        return lds_expand(kv_byte(token_in_page, block));
    }
    static constexpr int LDS_PAGE_BYTES = lds_expand(KV_PAGE_BYTES);  // 4608 == 4096 + one 16 B pad per 128 B
    static constexpr int LDS_TILE_BYTES  = PAGES_PER_TILE * LDS_PAGE_BYTES;   // 9216
    static constexpr int LDS_SCALE_BYTES = KV_TILE_SIZE * SCALE_BYTES_PER_DWORD;   // 512
    static constexpr int LDS_STAGE_BYTES = LDS_TILE_BYTES + LDS_SCALE_BYTES;
    static constexpr int LDS_STAGES      = LDS_STAGES_;
    static constexpr int LDS_BYTES       = LDS_STAGES * LDS_STAGE_BYTES;
    static constexpr size_t smem_size_bytes() { return (size_t)LDS_BYTES; }
    static_assert((LDS_STAGES & (LDS_STAGES - 1)) == 0,
                  "LDS_STAGES must be a power of two: the stage index is a mask, so a non-power "
                  "turns one `and` into a division in the phase's address arithmetic");

    static constexpr int lds_stage_byte(int tile) { return (tile & (LDS_STAGES - 1)) * LDS_STAGE_BYTES; }
    static constexpr int lds_page_byte (int page) { return page * LDS_PAGE_BYTES; }
    static constexpr int lds_scale_byte(int token_in_tile) {
        return LDS_TILE_BYTES + token_in_tile * SCALE_BYTES_PER_DWORD;
    }

    static constexpr int LDS_NTILE_BYTES = lds_byte(MMA_N, 0) - lds_byte(0, 0);   // 1152
    static constexpr bool lds_ntile_step_is_uniform() {
        for (int t = 0; t + MMA_N <= PAGE_SIZE - MMA_N; ++t)
            for (int b = 0; b < K_CHUNKS; ++b)
                if (lds_byte(t + MMA_N, b) - lds_byte(t, b) != LDS_NTILE_BYTES) return false;
        return true;
    }
    static_assert(lds_ntile_step_is_uniform(),
                  "the n-tile step in LDS is not lane-independent, so the B reads cannot share an "
                  "address register. The pad interval must divide MMA_N * TOKEN_BYTES.");

    static constexpr int LDS_BLOCK_BYTES = lds_byte(0, 1) - lds_byte(0, 0);   // 16
    static constexpr bool lds_block_step_is_uniform() {
        for (int t = 0; t < PAGE_SIZE; ++t)
            for (int b = 0; b + 1 < K_CHUNKS; ++b)
                if (lds_byte(t, b + 1) - lds_byte(t, b) != LDS_BLOCK_BYTES) return false;
        return true;
    }
    static_assert(lds_block_step_is_uniform(),
                  "the K-block step in LDS is not uniform -- under the natural layout that means "
                  "the pad is cutting a token in half -- so the B reads' block offset cannot be "
                  "an immediate");

    static constexpr int lds_bank_unit(int lane, int v) {
        return (lds_byte(frag_b_col(lane), frag_b_block(lane, v)) >> 4) % 8;
    }
    static constexpr bool lds_conflict_free() {
        for (int v = 0; v < B_VGPR_GROUPS; ++v)
            for (int base = 0; base < WAVE_SIZE; base += 8) {
                int seen = 0;
                for (int i = 0; i < 8; ++i) seen |= 1 << lds_bank_unit(base + i, v);
                if (seen != 0xFF) return false;
            }
        return true;
    }
    static constexpr bool lds_conflict_free_unpadded_control() {
        for (int base = 0; base < WAVE_SIZE; base += 8) {
            int seen = 0;
            for (int i = 0; i < 8; ++i) {
                const int lane = base + i;
                const int linear = frag_b_col(lane) * TOKEN_BYTES + frag_b_block(lane, 0) * KV_GRP_BYTES;
                seen |= 1 << ((linear >> 4) % 8);
            }
            if (seen != 0xFF) return false;
        }
        return true;
    }
    static_assert(!lds_conflict_free_unpadded_control(),
                  "the unpadded natural layout is supposed to CONFLICT; if it does not, this "
                  "checker cannot detect a conflict and the assert below proves nothing");
    static_assert(lds_conflict_free(),
                  "the LDS padding policy leaves a bank conflict. Unpadded, the natural layout's "
                  "64 B token stride puts every bank in {0-3} or {16-19} -- a 4-way conflict; the "
                  "pad of one read vector per 128 B is what makes the 8 lanes tile all 32 banks.");
    static_assert(LDS_PAD_INTERVAL != 0,
                  "the natural layout's lanes are 64 B apart and do NOT tile the banks unpadded, "
                  "so this layout may not run the pad off: see the control above.");

    // ── TDM. EVERY wave issues, and that is why the predicate is a compile-time constant and
    // the steady loop keeps no branch for it: two issuing waves out of four showed up as an
    // `s_cbranch_vccnz` per phase, splitting the loop into basic blocks. A page splits by
    // CONTIGUOUS BYTES -- not by token or by K block -- and wave w takes piece (w % SPLIT) of
    // page (w / SPLIT). ──
    static constexpr int TDM_ISSUE_WAVES = NUM_WAVES;
    static constexpr int TDM_PAGE_SPLIT  = TDM_ISSUE_WAVES / PAGES_PER_TILE;   // 2
    static constexpr int TDM_OPS_PER_TILE = 2 * TDM_ISSUE_WAVES;               // 8
    static constexpr int TDM_INFLIGHT_PER_WAVE = 3;                   // hardware, per opus.hpp
    static constexpr int TDM_OPS_PER_ISSUING_WAVE = TDM_OPS_PER_TILE / TDM_ISSUE_WAVES;   // 2
    static constexpr int TDM_INFLIGHT_CAP = TDM_INFLIGHT_PER_WAVE / TDM_OPS_PER_ISSUING_WAVE;

    static constexpr int TDM_ROWS        = PAGE_SIZE / TDM_PAGE_SPLIT;         // 32
    static constexpr int TDM_PIECE_BYTES = KV_PAGE_BYTES / TDM_PAGE_SPLIT;     // 2048, global side
    static constexpr int LDS_PIECE_BYTES = lds_expand(TDM_PIECE_BYTES);        // 2304
    static constexpr int TDM_SCALE_PIECE_BYTES = TDM_ROWS * SCALE_BYTES_PER_DWORD;   // 128

    static_assert(TDM_ISSUE_WAVES <= NUM_WAVES,
                  "there are not enough waves to spread a tile's TDM issues over");
    static_assert(TDM_ISSUE_WAVES % PAGES_PER_TILE == 0,
                  "the issuing waves must divide into whole pages, or a wave straddles two pages "
                  "and needs two page ids");
    static_assert(PAGE_SIZE % TDM_PAGE_SPLIT == 0 && KV_PAGE_BYTES % TDM_PAGE_SPLIT == 0,
                  "a page must split into equal pieces");
    static_assert(LDS_PIECE_BYTES * TDM_PAGE_SPLIT == LDS_PAGE_BYTES,
                  "the pieces must tile the padded LDS page exactly. They do not when a piece's "
                  "byte count is not a whole number of pad intervals, and the pieces then overlap "
                  "by the rounding -- silently, because each DMA on its own is still in range.");
    static_assert(TDM_OPS_PER_ISSUING_WAVE <= TDM_INFLIGHT_PER_WAVE,
                  "one tile's share already exceeds a wave's TDM queue depth");

    // ── pipeline depth: it follows from the stage count, not the other way round ──
    static constexpr int TILES_IN_FLIGHT = LDS_STAGES - 1;
    static constexpr int ISSUE_LEAD      = TILES_IN_FLIGHT;   // phase t issues tile t + this
    static_assert(LDS_STAGES >= 2, "need at least a double buffer");
    static_assert(TILES_IN_FLIGHT <= TDM_INFLIGHT_CAP,
                  "LDS_STAGES asks for more tiles in flight than the per-wave TDM queue admits. "
                  "Splitting a page more finely does NOT raise the cap: however finely it is "
                  "divided, every issuing wave still carries one data op and one scale op, so "
                  "the depth of 3 admits one tile. Reaching more needs the data and scale "
                  "descriptors MERGED, not a finer split.");
    static constexpr int TENSORCNT_KEEP = (TILES_IN_FLIGHT - 1) * TDM_OPS_PER_ISSUING_WAVE;

    // ── the accumulator: TWO copies, ping-ponged, so tile t+1's WMMAs issue while tile t is
    // reduced. Worth 20-52% wherever the grid fills the machine, which is why occupancy 1 is the
    // deliberate operating point. ──
    static constexpr int ACC_SETS = 2;
    static constexpr int ACC_VGPR_PER_SET = M_TILES * N_TILES * C_FRAG;   // 256
    static constexpr int ACC_VGPR = ACC_SETS * ACC_VGPR_PER_SET;          // 512
    static constexpr int WMMA_PER_TILE = M_TILES * N_TILES;               // 16
    static constexpr int WAVES_PER_EU = 1;
    static constexpr int VGPR_ADDRESSABLE = 1024;                    // wave32, per lane
    static_assert(ACC_VGPR < VGPR_ADDRESSABLE,
                  "the accumulator alone must not fill the register file");

    static_assert(ELEM_BITS == 4, "this traits set is fp4-only");
    static_assert(HEAD_DIM == MMA_K,
                  "K = 128 is the whole head_dim in one instruction; a HEAD_DIM past MMA_K needs "
                  "the outer kt loop back, and with it the kt term in both scale byte indices");
    static_assert(N_HEADS % MMA_M == 0, "N_HEADS must be a multiple of MMA_M (32)");
    static_assert(KV_TILE_SIZE % MMA_N == 0, "KV_TILE must be a multiple of MMA_N");
    static_assert(KV_TILE_SIZE % PAGE_SIZE == 0, "KV_TILE must be a whole number of pages");
    static_assert(PAGE_SIZE % MMA_N == 0, "PAGE must be a multiple of MMA_N");
    static_assert(M_TILES == 2, "the head reduction assumes 64 heads over 2 m-tiles");
    static_assert(GRPM_C == 2, "the head reduction emits exactly one permlane16_swap");
    static_assert(HEADS_PER_LANE * GRPM_C == N_HEADS,
                  "a lane and its swap partner must together hold every head, or the reduction "
                  "drops heads without any count changing");
    static_assert(A_BYTES_PER_LANE == 64 && B_BYTES_PER_LANE == 32,
                  "the dedicated f4 instruction takes i32x16 / i32x8 and uses all of both");
    static_assert(KV_GRP_BYTES == 16, "a 32-element fp4 block must be exactly one ds_load_b128");
    static_assert(Q_ROW_BYTES == N_HEADS * HEAD_DIM * ELEM_BITS / 8, "q must be size-preserving");
    static_assert(QS_ROW_BYTES == N_HEADS * SCALE_BLOCKS_ROW, "q_scale must be size-preserving");
    static_assert(KV_PAGE_BYTES == PAGE_SIZE * HEAD_DIM * ELEM_BITS / 8, "kv must be size-preserving");
    static_assert(Q_PER_BLOCK >= 1, "Q_PER_BLOCK must be positive");
};

using logits_fp4_qshare_traits_4q = opus_mqa_logits_fp4_qshare_traits<4>;

// A check rather than a comment, because a JSON build config cannot carry the reason:
// `-ffast-math` implies `-ffinite-math-only`, which folds OPUS_LOGITS_RELU's IEEE maximum back
// to a NaN-swallowing select. See that macro's definition for what that costs.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "build this TU with -fno-finite-math-only: -ffinite-math-only folds OPUS_LOGITS_RELU's \
IEEE maximum back to a compare-and-select, which SWALLOWS a NaN E8M0 scale instead of \
propagating it. See optCompilerConfig.json's flags_extra_hip for this module."
#endif

// The device pass on gfx1250 gets the real kernel; every other pass gets an empty stub so the
// launcher's `__device_stub__` reference still resolves. BOTH HALVES OF THE GUARD ARE
// LOAD-BEARING: `opus::get_warp_size()` answers 64 in the host pass, which would silently build
// the wave64 fragment layout with every byte count still matching.
#if !defined(__HIP_DEVICE_COMPILE__) || !defined(__gfx1250__)
namespace opus_logits {
namespace qshare {
template <class T, mqa_logits_sched SCHED = mqa_logits_sched::Table>
__global__ void mqa_logits_mxfp4_32x16x128_qshare_kernel(opus_mqa_logits_kargs)
{
}
} // namespace qshare
} // namespace opus_logits
#else
#include <opus/opus.hpp>
#include <bit>   // std::bit_cast -- see permlane_head_reduce for why not __builtin_bit_cast

namespace opus_logits {
namespace qshare {

// The write form ("+s") on purpose: the read-only one is a memory clobber and demotes the
// indexed s_loads. Do NOT pin a pointer -- that propagates inline-asm divergence into every
// address derived from it and costs occupancy.
template<class X>
__device__ inline void pin_sgpr(X& v) { asm("" : "+s"(v)); }

// Fenced on both sides so the scheduler cannot drag LDS traffic across it; callers owe the wait
// that gives the barrier something to publish. Lowers to TWO instructions on gfx1250.
__device__ inline void cta_sync() {
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
}

// Sum a per-lane value across the two C lane groups, whose partner sits at lane distance 16.
// TWO WAYS TO GET THIS WRONG, both silent: the builtin returns (self, partner), so taking only
// [0] computes v + v; and `std::bit_cast` is load-bearing -- `__builtin_bit_cast` miscompiled
// this swap to v + v, which uniform test data cannot distinguish from correct.
__device__ inline float permlane_head_reduce(float v) {
    opus::vector_t<opus::u32_t, 2> r = __builtin_amdgcn_permlane16_swap(
        std::bit_cast<opus::u32_t>(v), std::bit_cast<opus::u32_t>(v), false, false);
    return std::bit_cast<float>(r[0]) + std::bit_cast<float>(r[1]);
}

// Grid: one CTA per SCHEDULE SLOT (grid.x == kargs.num_ctas).
template<class T, mqa_logits_sched SCHED = mqa_logits_sched::Table>
__global__ __launch_bounds__(T::BLOCK_SIZE, T::WAVES_PER_EU)
void mqa_logits_mxfp4_32x16x128_qshare_kernel(opus_mqa_logits_kargs kargs) {
    static_assert(SCHED == mqa_logits_sched::Table,
                  "this target compiles the schedule-table mapping and nothing else. Decode needs "
                  "no separate mapping: a decode table is one whose tiles carry a zero start.");

    using D_BYTE  = opus::u8_t;
    using D_SCALE = typename T::D_SCALE;
    using D_OUT   = typename T::D_OUT;
    using D_WEIGHT = typename T::D_WEIGHT;
    using Mma = opus::wmma<opus::fp4_t, opus::fp4_t, float,
                          T::MMA_M, T::MMA_N, T::MMA_K, T::WAVE_SIZE>;
    Mma mma_op;
    static_assert(sizeof(typename Mma::vtype_a) == T::A_BYTES_PER_LANE);
    static_assert(sizeof(typename Mma::vtype_b) == T::B_BYTES_PER_LANE);
    static_assert(Mma::elem_c == T::C_FRAG);

    constexpr int WAVE   = T::WAVE_SIZE;
    constexpr int MT     = T::M_TILES;
    constexpr int NT     = T::N_TILES;
    constexpr int EC     = T::C_FRAG;
    constexpr int PAGES   = T::PAGES_PER_TILE;
    constexpr int PACK_W  = T::PACK_C;              // 8 bf16 = 16 B, one load per C run

    __shared__ __attribute__((aligned(16))) D_BYTE smem_buf[T::LDS_BYTES];

    const void* p_q        = kargs.ptr_q;
    const void* p_q_scale  = kargs.ptr_q_scale;
    const void* p_kv       = kargs.ptr_kv;
    const void* p_kv_scale  = kargs.ptr_kv_scale;
    const int*  p_bt        = kargs.ptr_block_tables;
    const void* p_weights   = kargs.ptr_weights;
    D_OUT*      p_out       = kargs.ptr_out;
    int   stride_out   = kargs.stride_out_row;
    float weight_scale = kargs.weight_scale;
    int   max_blk      = kargs.max_blocks_per_seq;
    pin_sgpr(stride_out); pin_sgpr(weight_scale); pin_sgpr(max_blk);
    const opus_mqa_cta_record* p_cta_info = kargs.ptr_cta_info;
    const int* p_local_starts = kargs.ptr_local_starts;
    const int* p_local_ends   = kargs.ptr_local_ends;

    const int tid = opus::thread_id_x();
    // readfirstlane, not a plain shift: the slot is wave-uniform in fact but divergent to LLVM,
    // and it feeds this wave's row base pointers and the TDM issue -- both must stay scalar or
    // every derived address goes through a waterfall.
    const int q_slot  = __builtin_amdgcn_readfirstlane(tid / WAVE);
    const int lane    = tid % WAVE;
    const int lane_m0 = lane % T::GRPN_C;    // L % 16: the n column / the m row within a tile
    const int lane_g  = lane / T::GRPN_C;    // L / 16: the K-block half, and the C m-group

    // TWO WINDOWS COME OUT OF THIS BLOCK AND THEY MUST NOT BE CONFUSED. `local_*` is the GROUP
    // UNION and is CTA-uniform: it, and only it, may reach chunk_start / tile_count, because
    // those set the barrier count. `own_*` is THIS WAVE'S row and reaches nothing but the store
    // mask and the out descriptor's size. A wave past the end of a short group does NOT return;
    // it runs with own_end = 0, so its out descriptor has zero records and every store it makes
    // is dropped. That is the whole of "short groups are masked, not padded".
    constexpr int kv_tile_size = T::KV_TILE_SIZE;
    // The whole assignment in one `s_load_dwordx8` off a blockIdx-uniform address. The
    // surplus-slot return is CTA-uniform, so it cannot deadlock the phase barrier.
    const opus_mqa_cta_record rec = p_cta_info[opus::block_id_x()];
    if (rec.chunk_count <= 0) return;                // surplus, or an empty window
    const bool slot_ok    = q_slot < rec.group_rows;
    // Clamped, not branched: q_slot is readfirstlane'd and group_rows comes off the record, so
    // the select is s_cselect and the row's base addresses below stay scalar.
    const int row_id      = rec.row_id + (slot_ok ? q_slot : 0);
    int       batch_id    = rec.batch_id;
    // The null-starts branch is on a kargs pointer, so it is uniform across the whole grid.
    const int own_start   = p_local_starts ? p_local_starts[row_id] : rec.local_start;
    const int own_end     = slot_ok ? p_local_ends[row_id] : 0;
    const int chunk_start = rec.chunk_start;         // ABSOLUTE tile index, not an offset
    const int tile_count  = rec.chunk_count;
    pin_sgpr(batch_id);

    const D_BYTE*   q_base  = reinterpret_cast<const D_BYTE*>(p_q) + (size_t)row_id * T::Q_ROW_BYTES;
    const D_SCALE*  qs_base = reinterpret_cast<const D_SCALE*>(p_q_scale) + (size_t)row_id * T::N_HEADS;
    const D_WEIGHT* w_base  = reinterpret_cast<const D_WEIGHT*>(p_weights) + (size_t)row_id * T::W_ROW_ELEMS;
    const int*      bt_base = p_bt + (size_t)batch_id * max_blk;

    // own_start folded into the base and num_records set to the window LENGTH, so BOTH bounds
    // are the hardware's: a token below the window gets a negative offset, which as an unsigned
    // byte offset is huge and fails the same test that catches one past the end. THIS WAVE'S
    // row throughout -- a wave whose window is shorter than the group's must not write the
    // difference.
    D_OUT*         out_base  = p_out + (size_t)row_id * stride_out + own_start;
    const unsigned out_bytes =
        (unsigned)(own_end > own_start ? own_end - own_start : 0) * (unsigned)sizeof(D_OUT);

    static_assert(T::LDS_STAGE_BYTES % 16 == 0 && T::LDS_PAGE_BYTES % 16 == 0
                      && T::TOKEN_BYTES % 16 == 0 && T::KV_GRP_BYTES % 16 == 0,
                  "a dword view needs every offset 4 B aligned; these are all 16 B aligned");
    constexpr int DW = (int)sizeof(D_SCALE);   // 4: the byte -> dword divisor, named so it reads
    auto g_q   = opus::make_gmem(reinterpret_cast<const D_SCALE*>(q_base));
    auto g_qs  = opus::make_gmem(qs_base);
    auto g_w   = opus::make_gmem(w_base);
    auto g_out = opus::make_gmem(out_base, out_bytes);
    auto s_dw  = opus::make_smem(reinterpret_cast<D_SCALE*>(smem_buf));

    typename Mma::vtype_a q_a[MT]{};
    int                   q_scale[MT]{};
    opus::vector_t<float, EC> w_pl[MT];
    // The two result matrices, as TWO NAMED VARIABLES and not `acc[2][MT][NT]`. That is not
    // style: a runtime subscript on the set puts the whole 512-VGPR accumulator in SCRATCH
    // before the register allocator ever sees it -- 2112 B/lane, and `.vgpr_spill_count` still
    // reads 0 because nothing was spilled, it was never in registers. The phase loop is
    // unrolled by two so the set stays a compile-time choice.
    typename Mma::vtype_c accA[MT][NT], accB[MT][NT];
    float outA[NT], outB[NT];

    // block_tables through the CONSTANT address space so it lowers to s_load_dword, which is
    // load-bearing: a TDM descriptor's fields are all uniform and live in SGPRs, so a page id in
    // a VGPR would put a readfirstlane waterfall around every issue. Neither opus::load nor a
    // plain dereference gets there -- the pass also has to prove the load is NOCLOBBER, and the
    // loop's store to `out` defeats that.
    using bt_scalar_ptr = const int __attribute__((address_space(4)))*;
    auto bt_c = reinterpret_cast<bt_scalar_ptr>(
        reinterpret_cast<__UINTPTR_TYPE__>(bt_base + chunk_start * PAGES));
    auto page_id = [&](int t, int p) { return bt_c[t * PAGES + p]; };

    // CU SCOPE, NOT opus's DEVICE default: device scope defeats the L2 and costs between 1.55x
    // and 7x. It is legal only because the KV cache and its scales are written by the indexer
    // BEFORE the launch and are read-only for its duration -- SO NOTHING MAY WRITE THE KV CACHE
    // CONCURRENTLY WITH IT, which is a condition on the caller and not on this kernel.
    using fp4x2 = opus::array<opus::fp4_t, 2>;
    constexpr int TDM_CACHE = opus::tdm_traits::make_cache_policy(
        opus::tdm_traits::load_temporal_hint::regular, opus::tdm_traits::scope::cu);
    using Cache = opus::tdm_traits::cache<TDM_CACHE>;
    using KvWin = opus::tdm<fp4x2, opus::seq<T::TOKEN_BYTES, T::TDM_ROWS>,
                            opus::tdm_traits::padding<fp4x2, T::LDS_PAD_INTERVAL, T::LDS_PAD_AMOUNT>,
                            Cache>;
    using KvsWin = opus::tdm<opus::u8_t, opus::seq<T::SCALE_BYTES_PER_DWORD, T::TDM_ROWS>, Cache>;

    // A CONSTANT because the window is built once and WALKED: `tensor_dim` only CLAMPS, and the
    // DMA moves `tile_dim1` rows regardless, so any extent past the largest legal origin is equal.
    constexpr opus::u32_t TDM_EXTENT = 1u << 30;   // rows; 2^30 tokens, and no u32 overflow

    // WHY THERE IS STILL A CLAMP. TDM does clamp an out-of-range ORIGIN to a zero-extent DMA,
    // but exploiting it needs the cache's true page count and THE ABI DOES NOT CARRY
    // `num_blocks`. So an over-issued tile re-fetches the last real page instead, which nobody
    // reads. Cheaper than a guarded address chain even so: the clamp lands on one origin.
    auto clamp_tile = [&](int t) { return t < tile_count ? t : tile_count - 1; };

    const int tdm_page  = q_slot / T::TDM_PAGE_SPLIT;
    const int tdm_piece = q_slot % T::TDM_PAGE_SPLIT;
    static_assert(T::TDM_ISSUE_WAVES == T::NUM_WAVES,
                  "the branch-free form needs one issuing wave per piece; with fewer issuers the "
                  "kernel needs the `q_slot < TDM_ISSUE_WAVES` guard this replaced");

    // BUILT ONCE AND WALKED: rebuilding per phase rewrites extent, stride and base when only
    // the origin changes. The LDS write point is deliberately NOT window state, so the base here
    // is this wave's fixed piece within a stage and the per-tile offset rides in async_load()'s
    // argument -- and it is the WRITE stream's map, not `lds_byte` of a token, because TDM inserts
    // the pad as it writes.
    const int kv_lds_base  = tdm_page * T::LDS_PAGE_BYTES + tdm_piece * T::LDS_PIECE_BYTES;
    const int kvs_lds_base = T::lds_scale_byte(tdm_page * T::PAGE_SIZE + tdm_piece * T::TDM_ROWS);
    auto kv_win = opus::make_tdm<KvWin>(
        (opus::u32_t)kv_lds_base, reinterpret_cast<const D_BYTE*>(p_kv),
        /*shape0*/ (opus::u32_t)T::TOKEN_BYTES, /*shape1*/ TDM_EXTENT,
        /*dim0_stride*/ (opus::u64_t)T::TOKEN_BYTES, /*origin0*/ 0u, /*origin1*/ 0u);
    auto kvs_win = opus::make_tdm<KvsWin>(
        (opus::u32_t)kvs_lds_base, reinterpret_cast<const D_BYTE*>(p_kv_scale),
        (opus::u32_t)T::SCALE_BYTES_PER_DWORD, TDM_EXTENT,
        (opus::u64_t)T::SCALE_BYTES_PER_DWORD, 0u, 0u);
    int tdm_origin = 0;
    auto issue_tile = [&](int t) {
        const int ct     = clamp_tile(t);
        const int pid    = page_id(ct, tdm_page);
        const int origin = pid * T::PAGE_SIZE + tdm_piece * T::TDM_ROWS;
        kv_win.move(opus::number<0>{}, origin - tdm_origin);
        kvs_win.move(opus::number<0>{}, origin - tdm_origin);
        tdm_origin = origin;
        const opus::u32_t stage = (opus::u32_t)T::lds_stage_byte(ct);
        kv_win.async_load(stage);
        kvs_win.async_load(stage);
    };

    constexpr int GRP_DW = T::KV_GRP_BYTES / 4;              // 4 dwords = one 32-element block
    const int lds_lane_dw = T::lds_byte(lane_m0, lane_g) / DW;
    auto read_kv_lds = [&](typename Mma::vtype_b (&kv)[NT], int stage_dw) {
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt   = ntc.value;
            constexpr int page = nt / T::TILES_PER_PAGE;
            constexpr int base = (nt % T::TILES_PER_PAGE) * T::LDS_NTILE_BYTES / 4;
            opus::static_for<T::B_VGPR_GROUPS>([&](auto vc) {
                constexpr int v = vc.value;
                // The fragment's second half is block g+2, and the step comes FROM THE TRAITS
                // because it is not the same in the two kv_cache layouts: 16 B when a token's
                // blocks are adjacent, 1024 when the block index sits outside the token.
                // Hardcoding either builds fine on one branch and reads the wrong 16 B on the
                // other.
                constexpr int imm = v * 2 * T::LDS_BLOCK_BYTES / 4;
                auto x = opus::load<GRP_DW>(
                    s_dw, stage_dw + T::lds_page_byte(page) / 4 + base + imm + lds_lane_dw);
                opus::set_slice(kv[nt], x, opus::number<v * GRP_DW>{},
                                opus::number<(v + 1) * GRP_DW>{});
            });
        });
    };
    auto read_kv_scale_lds = [&](int (&kvs)[NT / 2], int stage_dw) {
        opus::static_for<NT / 2>([&](auto uc) {
            constexpr int u = uc.value;
            kvs[u] = opus::load<1>(s_dw, stage_dw + T::lds_scale_byte(u * 2 * T::MMA_N) / DW + lane)[0];
        });
    };

    auto gemm_tile = [&](typename Mma::vtype_c (&a)[MT][NT],
                         typename Mma::vtype_b (&kv)[NT], int (&kvs)[NT / 2]) {
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            // a_scale_sel is 0 always: at M = 32 the 32 lanes x 4 B exactly fill the m rows
            // and the field has nothing left to select. b_scale_sel is the n-tile's parity,
            // which is what lets one dword of scale serve two n-tiles.
            constexpr int bsel = nt % 2;
            opus::static_for<MT>([&](auto mic) {
                constexpr int mi = mic.value;
                a[mi][nt] = mma_op(q_a[mi], kv[nt], q_scale[mi], kvs[nt / 2],
                                   opus::number<0>{}, opus::number<bsel>{});
            });
        });
    };

    auto reduce_tile = [&](typename Mma::vtype_c (&a)[MT][NT], float (&out)[NT]) {
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            opus::vector_t<float, 2> s2{0.0f, 0.0f};
            opus::static_for<MT>([&](auto mic) {
                constexpr int mi = mic.value;
                opus::static_for<EC / 2>([&](auto pc) {
                    constexpr int p = pc.value;
                    opus::vector_t<float, 2> w{ w_pl[mi][2 * p], w_pl[mi][2 * p + 1] };
                    // Packed two heads at a time so the sum vectorizes to v_pk_fma_f32.
                    opus::vector_t<float, 2> r{ OPUS_LOGITS_RELU(a[mi][nt][2 * p]),
                                                OPUS_LOGITS_RELU(a[mi][nt][2 * p + 1]) };
                    s2 = r * w + s2;
                });
            });
            out[nt] = permlane_head_reduce(s2[0] + s2[1]);
        });
    };

    // Only lane group 0 carries a useful logit after the symmetric butterfly, so group 1's stores
    // are pushed out of range by `lane_bias` instead of being branched around.
    const int lane_bias = (lane_g != 0) ? -(chunk_start + tile_count) * kv_tile_size : 0;
    auto do_store = [&](float (&out)[NT], int t) {
        const int tile_off = (chunk_start + t) * kv_tile_size - own_start + lane_bias;
        opus::static_for<NT>([&](auto ntc) {
            constexpr int nt = ntc.value;
            g_out.template store<1>(out[nt], tile_off + nt * T::MMA_N + lane_m0);
        });
    };
    auto fenced_store = [&](float (&out)[NT], int t) {
        __builtin_amdgcn_sched_barrier(0);
        do_store(out, t);
        __builtin_amdgcn_sched_barrier(0);
    };

    // The empty-assignment return sits at the record read, where `chunk_count <= 0` covers both the
    // surplus slot and the empty window BEFORE any TDM descriptor is built.

    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::A_VGPR_GROUPS>([&](auto vc) {
            constexpr int v = vc.value;
            const int head  = mi * T::MMA_M + lane_m0 + (v >= 2 ? 16 : 0);
            const int block = lane_g + (v % 2 ? 2 : 0);
            auto x = opus::load<GRP_DW>(g_q, T::q_byte(head, block) / DW);
            opus::set_slice(q_a[mi], x, opus::number<v * GRP_DW>{},
                            opus::number<(v + 1) * GRP_DW>{});
        });
    });
    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        q_scale[mi] = opus::load<1>(g_qs, mi * T::MMA_M + lane)[0];
    });
    opus::static_for<MT>([&](auto mic) {
        constexpr int mi = mic.value;
        opus::static_for<T::REPT_C>([&](auto rc) {
            constexpr int r = rc.value;
            const int h0 = mi * T::MMA_M + (r ? 16 : 0) + lane_g * PACK_W;
            auto wv = opus::load<PACK_W>(g_w, h0);   // 8 bf16 = 16 B
            opus::static_for<PACK_W>([&](auto jc) {
                constexpr int j = jc.value;
                w_pl[mi][r * PACK_W + j] = (float)wv[j] * weight_scale;
            });
        });
    });

    opus::static_for<T::TILES_IN_FLIGHT>([&](auto sc) { issue_tile(sc.value); });

    typename Mma::vtype_b kv_b[NT]{};
    int                   kvs_w[NT / 2]{};

    // One phase = one KV tile. The wait retires this wave's share of tile t's loads and the
    // barrier then does TWO duties: makes the tile visible CTA-wide, and separates the previous
    // phase's ds_loads from the issue that recycles the stage they freed.
    //
    // `acc_cur` takes this tile's WMMAs while `acc_prev` is reduced. They are separate locals with
    // no aliasing, so the halves have no dependency -- that is the ping-pong, and a sched_barrier
    // between them would serialise exactly what it buys.
    auto phase = [&](auto& acc_cur, auto& acc_prev, float (&out_prev)[NT], int t, bool has_prev) {
        opus::s_wait_tensorcnt<T::TENSORCNT_KEEP>();
        cta_sync();
        issue_tile(t + T::ISSUE_LEAD);
        const int stage_dw = T::lds_stage_byte(t) / DW;
        read_kv_scale_lds(kvs_w, stage_dw);
        read_kv_lds(kv_b, stage_dw);
        __builtin_amdgcn_sched_barrier(0);
        gemm_tile(acc_cur, kv_b, kvs_w);
        if (has_prev) reduce_tile(acc_prev, out_prev);
    };

    // Every branch below is on tile_count, which is CTA-uniform, so all four waves take the same
    // one -- a per-row condition here would deadlock them on the phase barrier.
    phase(accA, accB, outB, 0, false);
    int t = 1;
    for (; t + 1 < tile_count; t += 2) {
        phase(accB, accA, outA, t,     true);  fenced_store(outA, t - 1);
        phase(accA, accB, outB, t + 1, true);  fenced_store(outB, t);
    }
    if (t < tile_count) {   // t == tile_count - 1, and odd
        phase(accB, accA, outA, t, true);      fenced_store(outA, t - 1);
        reduce_tile(accB, outB);               fenced_store(outB, t);
    } else {                // tile_count - 1 was even, so accA holds it
        reduce_tile(accA, outA);               fenced_store(outA, tile_count - 1);
    }
}

} // namespace qshare
} // namespace opus_logits
#endif

#endif // PA_MQA_LOGITS_MXFP4_GFX1250_IMPL
