// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Device-side Table schedule builder (build_tiles + build_sched): arch-agnostic int-only code,
// included by the launcher TU after the record ABI (pa_mqa_logits_mxfp4.h); not standalone.
//
// A tile is up to rows_per_cta consecutive rows of ONE sequence (build_tiles cuts them from
// query_start_loc; at one row a CTA a tile is a row). Its block-table row is its sequence: the
// last b with query_start_loc[b] <= its first row, which skips empty sequences. build_tiles finds
// it while cutting and writes it beside the cut; at one row a tile, row r is sequence r whenever
// every sequence before it holds one row (checked, else searched). Rows at or past
// query_start_loc[num_seqs] belong to no sequence and get no work.
#pragma once

namespace opus_logits {

constexpr int SCHED_BUILD_BLOCK      = 256;
constexpr int SCHED_BUILD_BLOCK_WIDE = 1024;

// `resident` (CTAs the part holds at once) is per instance -- it follows occupancy -- so it is
// always passed explicitly, never defaulted.

// Split target: a whole number of rounds, so the last round is not left mostly empty.
// 0 means do not split.
__host__ __device__ inline int sched_target(int nz_tiles, int resident) {
    if (resident <= 0) return 0;
    if (nz_tiles <= resident) return resident;
    return ((nz_tiles + resident - 1) / resident) * resident;
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

// Every barrier here must be a full __syncthreads(), not a bare s_barrier: threads hand values
// through LDS (reductions, scan, general_emit staging) AND global memory (settle_safe and the
// priority pass read back cta_info records other threads stored). A bare s_barrier fences neither.

template<int BLOCK, bool IS_MAX>
__device__ inline int block_reduce(int v, int* s, int tid) {
    s[tid] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const int o = s[tid + off];
            s[tid] = IS_MAX ? (o > s[tid] ? o : s[tid]) : (s[tid] + o);
        }
        __syncthreads();
    }
    const int r = s[0];
    __syncthreads();
    return r;
}

template<int BLOCK>
__device__ inline void block_reduce_stats(int vmax, int vsum, int vnz, int* s, int tid,
                                          int& omax, int& osum, int& onz) {
    s[tid]             = vmax;
    s[BLOCK + tid]     = vsum;
    s[2 * BLOCK + tid] = vnz;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            const int om = s[tid + off];
            if (om > s[tid]) s[tid] = om;
            s[BLOCK + tid]     += s[BLOCK + tid + off];
            s[2 * BLOCK + tid] += s[2 * BLOCK + tid + off];
        }
        __syncthreads();
    }
    omax = s[0];
    osum = s[BLOCK];
    onz  = s[2 * BLOCK];
    __syncthreads();
}

template<int BLOCK>
__device__ inline int block_scan_excl(int v, int* s, int tid, int& total) {
    s[tid] = v;
    __syncthreads();
    for (int off = 1; off < BLOCK; off <<= 1) {
        const int add = (tid >= off) ? s[tid - off] : 0;
        __syncthreads();
        s[tid] += add;
        __syncthreads();
    }
    total = s[BLOCK - 1];
    const int incl = s[tid];
    __syncthreads();
    return incl - v;
}

// What a tile is cut from: the tile cut and each tile's sequence (null: a tile is a row), the
// sequences' bounds and the windows.
struct sched_rows {
    const int* cu_tiles;
    const int* tile_seq;
    const int* qsl;
    int        num_seqs;
    int        end_row;    // qsl[num_seqs]
    const int* row_ends;
};

__device__ inline sched_rows make_rows(const int* cu_tiles, const int* tile_seq, const int* qsl,
                                       int num_seqs, const int* row_ends) {
    return {cu_tiles, tile_seq, qsl, num_seqs, qsl[num_seqs], row_ends};
}

// The last b with qsl[b] <= r (qsl[num_seqs] > r): the sequence of a row below end_row.
__device__ inline int seq_search(const sched_rows& s, int r) {
    int lo = 0, hi = s.num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (s.qsl[mid] <= r) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// Tile t: its rows [r0, r1), its window end (the last row's: windows do not decrease along a
// sequence) and its sequence, 0 for an empty tile.
struct tile_facts {
    int r0, r1, end, seq;
};

// Every load a tile needs is issued before any is used: an index made safe, the value discarded
// after. A load behind the condition that picks it waits a round trip for that condition first.
__device__ inline tile_facts tile_facts_of(const sched_rows& s, int t) {
    if (s.cu_tiles) {
        // the cut wrote each tile's sequence; a tile past the cut (r0 == r1) never reads its own
        const int r0 = s.cu_tiles[t], r1 = s.cu_tiles[t + 1], seq = s.tile_seq[t];
        return {r0, r1, r1 > r0 ? s.row_ends[r1 - 1] : 0, r1 > r0 ? seq : 0};
    }
    // A tile is a row (t < the rows). Row t is sequence t whenever every sequence before it holds
    // one row (decode without speculation, candidate rows): its bounds load beside its window.
    const int ri = t < s.num_seqs ? t : s.num_seqs - 1;
    const int e = s.row_ends[t], q0 = s.qsl[ri], q1 = s.qsl[ri + 1];
    if (t >= s.end_row) return {s.end_row, s.end_row, 0, 0};
    const int seq = (ri == t && q0 == t && q1 == t + 1) ? t : seq_search(s, t);
    return {t, t + 1, e, seq};
}

__device__ inline int kv_tiles_of(int end, int block_k) {
    return end > 0 ? (end + block_k - 1) / block_k : 0;
}

// Reads the count emit_fast stored (recomputing needs a dependent global load). Valid only
// before general_emit writes: general_emit must not use it, as its slot writes overlap the tiles
// it reads. Not __restrict__ (nor in settle_safe): it aliases emit_fast's stores.
__device__ inline int tile_kv_tiles_cached(const opus_mqa_cta_record* cta_info, int t) {
    return cta_info[t].chunk_count;
}

// Reduces and emits in one walk: the one-CTA-per-tile record is written speculatively and
// overwritten only if splitting is needed.
__device__ inline void emit_fast(const sched_rows& s,
                                 opus_mqa_cta_record* __restrict__ cta_info,
                                 int first_tile, int num_tiles, int stride, int block_k,
                                 int& part_max, int& part_sum, int& part_nz) {
    part_max = 0;
    part_sum = 0;
    part_nz  = 0;
    for (int t = first_tile; t < num_tiles; t += stride) {
        const tile_facts f = tile_facts_of(s, t);
        const int n = kv_tiles_of(f.end, block_k);
        if (n > part_max) part_max = n;
        part_sum += n;
        part_nz  += (n > 0);
        opus_mqa_cta_record rec{};
        rec.row_id      = f.r0;
        rec.batch_id    = f.seq;
        rec.chunk_count = n;
        rec.local_end   = f.end;
        rec.group_rows  = f.r1 - f.r0;
        cta_info[t]     = rec;
    }
}

// Surplus slots are marked empty, not left stale: the table is reused across cudagraph replays.
__device__ inline void mark_surplus(opus_mqa_cta_record* __restrict__ cta_info,
                                    int first_slot, int num_ctas, int stride) {
    for (int slot = first_slot; slot < num_ctas; slot += stride) {
        opus_mqa_cta_record rec{};
        cta_info[slot] = rec;
    }
}

template<int BLOCK>
__device__ inline int settle_safe(const opus_mqa_cta_record* cta_info,
                                  int num_tiles, int num_ctas,
                                  int safe, int max_tiles, int nz_tiles, int* smem, int tid) {
    if (nz_tiles >= num_ctas) return max_tiles;
    auto ctas_for = [&](int sf) {
        int part = 0;
        for (int t = tid; t < num_tiles; t += BLOCK) {
            const int n = tile_kv_tiles_cached(cta_info, t);
            part += (n + sf - 1) / sf;
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

// Returns the slots used. Every record carries `pad` in its spare word.
template<int BLOCK>
__device__ inline int general_emit(const sched_rows& s,
                                   opus_mqa_cta_record* __restrict__ cta_info,
                                   int num_tiles, int num_ctas, int block_k, int safe, int pad,
                                   int* smem, int* s_excl, int* s_tiles, int* s_batch,
                                   int* s_end, int* s_row, int* s_rows, int tid) {
    int carry = 0;
    for (int base = 0; base < num_tiles; base += BLOCK) {
        const int t = base + tid;
        const tile_facts f = t < num_tiles ? tile_facts_of(s, t) : tile_facts{0, 0, 0, 0};
        const int tiles = kv_tiles_of(f.end, block_k);
        const int nc = (tiles + safe - 1) / safe;
        int block_total = 0;
        const int excl = block_scan_excl<BLOCK>(nc, smem, tid, block_total);
        s_excl[tid]  = excl;
        s_tiles[tid] = tiles;
        s_batch[tid] = f.seq;
        s_end[tid]   = f.end;
        s_row[tid]   = f.r0;
        s_rows[tid]  = f.r1 - f.r0;
        __syncthreads();

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
                // Even split: safe sets the chunk count, not the size (a phase costs the
                // longest chunk). n >= 1 because the search picks the last tile at each excl
                // value; guarded anyway against a divide by zero.
                const int n       = s_tiles[tl];
                const int nchunks = n > 0 ? (n + safe - 1) / safe : 1;
                const int q = n / nchunks, r = n % nchunks;
                opus_mqa_cta_record rec{};
                rec.row_id      = s_row[tl];
                rec.batch_id    = s_batch[tl];
                rec.chunk_start = i * q + (i < r ? i : r);
                rec.chunk_count = q + (i < r ? 1 : 0);
                rec.local_end   = s_end[tl];
                rec.group_rows  = s_rows[tl];
                rec._pad        = pad;
                cta_info[slot]  = rec;
            }
        }
        carry += block_total;
        __syncthreads();
    }
    return carry;
}

__device__ inline int split_factor(int total_tiles, int cta_target) {
    const int safe = total_tiles > 0 ? ((total_tiles + cta_target - 1) / cta_target) : 1;
    return safe < 1 ? 1 : safe;
}

// A one-round schedule (every slot resident at once) carries a bound on its longest chunk in
// every working record's spare word: the H64 kernel measures its wave priority against it. A
// split schedule's chunks are at most `safe` (Free's scale too), written as they are emitted; an
// unsplit one's at most its longest tile's count, patched into the tiles' records once known.
// Multi-round schedules keep 0, which leaves the priority alone (there the age order retires
// CTAs for the next round).
__device__ inline bool one_round(int num_ctas, int resident) {
    return resident > 0 && num_ctas <= resident;
}

template<int BLOCK>
__device__ inline void settle_and_emit(const sched_rows& s,
                                       opus_mqa_cta_record* __restrict__ cta_info,
                                       int num_tiles, int num_ctas, int block_k, int resident,
                                       int max_tiles, int total_tiles, int nz_tiles,
                                       bool surplus_done,
                                       int* smem, int* s_excl, int* s_tiles, int* s_batch,
                                       int* s_end, int* s_row, int* s_rows, int tid) {
    const int aim   = (resident <= 0) ? 0 : sched_target(nz_tiles, resident);
    const int safe0 = (aim <= 0) ? max_tiles : split_factor(total_tiles, aim);
    if (max_tiles <= safe0) {
        if (!surplus_done) mark_surplus(cta_info, num_tiles + tid, num_ctas, BLOCK);
        if (one_round(num_ctas, resident)) {
            __syncthreads();   // emit_fast's records of these tiles came from other threads
            for (int t = tid; t < num_tiles; t += BLOCK) cta_info[t]._pad = max_tiles;
        }
        return;
    }
    const int safe = settle_safe<BLOCK>(cta_info, num_tiles, num_ctas,
                                        safe0, max_tiles, nz_tiles, smem, tid);
    const int carry = general_emit<BLOCK>(s, cta_info, num_tiles, num_ctas, block_k, safe,
                                          one_round(num_ctas, resident) ? safe : 0, smem,
                                          s_excl, s_tiles, s_batch, s_end, s_row, s_rows, tid);
    mark_surplus(cta_info, carry + tid, num_ctas, BLOCK);
}

}  // namespace sched_detail

constexpr int GROUPS_BUILD_BLOCK     = 256;
constexpr int GROUPS_BUILD_MAX_BATCH = 2048;   // 8 KB of LDS for the per-batch offsets

// Cut each sequence's rows into tiles of at most qpb rows (unused at qpb == 1; see tile_facts_of), and
// write each tile's sequence beside the cut. Contract: a tile is contiguous rows of ONE sequence.
// Breaking it deadlocks: the CTA's waves disagree on the trip count at the phase barrier. One
// workgroup, so GROUPS_BUILD_MAX_BATCH caps the batch.
__global__ __launch_bounds__(GROUPS_BUILD_BLOCK)
void mqa_logits_build_tiles(const int* __restrict__ cu_seq_q,
                            int* __restrict__ cu_tiles,
                            int* __restrict__ tile_seq,
                            int batch, int max_tiles, int qpb) {
    __shared__ int g_off[GROUPS_BUILD_MAX_BATCH + 1];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();

    for (int b = tid; b < batch; b += GROUPS_BUILD_BLOCK)
        g_off[b] = (cu_seq_q[b + 1] - cu_seq_q[b] + qpb - 1) / qpb;
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int b = 0; b < batch; ++b) { const int c = g_off[b]; g_off[b] = acc; acc += c; }
        g_off[batch] = acc;
    }
    __syncthreads();

    const int num_tiles = g_off[batch];
    const int end_row   = cu_seq_q[batch];

    for (int t = tid; t <= max_tiles; t += GROUPS_BUILD_BLOCK) {
        if (t >= num_tiles) { cu_tiles[t] = end_row; continue; }   // empty: its sequence unread
        int lo = 0, hi = batch - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (g_off[mid] <= t) lo = mid; else hi = mid - 1;
        }
        cu_tiles[t] = cu_seq_q[lo] + (t - g_off[lo]) * qpb;
        tile_seq[t] = lo;
    }
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK)
void mqa_logits_build_sched(const int* __restrict__ cu_tiles,
                            const int* __restrict__ tile_seq,
                            const int* __restrict__ query_start_loc,
                            const int* __restrict__ row_ends,
                            opus_mqa_cta_record* __restrict__ cta_info,
                            int num_seqs, int num_tiles, int num_ctas,
                            int block_k, int resident) {
    __shared__ int smem[3 * BLOCK];
    __shared__ int s_excl[BLOCK];
    __shared__ int s_tiles[BLOCK];
    __shared__ int s_batch[BLOCK];
    __shared__ int s_end[BLOCK];
    __shared__ int s_row[BLOCK];
    __shared__ int s_rows[BLOCK];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const sched_detail::sched_rows s =
        sched_detail::make_rows(cu_tiles, tile_seq, query_start_loc, num_seqs, row_ends);

    int part_max = 0, part_sum = 0, part_nz = 0;
    sched_detail::emit_fast(s, cta_info, tid, num_tiles, BLOCK, block_k,
                            part_max, part_sum, part_nz);
    int max_tiles = 0, total_tiles = 0, nz_tiles = 0;
    sched_detail::block_reduce_stats<BLOCK>(part_max, part_sum, part_nz, smem, tid,
                                            max_tiles, total_tiles, nz_tiles);
    sched_detail::settle_and_emit<BLOCK>(s, cta_info, num_tiles, num_ctas, block_k, resident,
                                         max_tiles, total_tiles, nz_tiles, /*surplus_done=*/false,
                                         smem, s_excl, s_tiles, s_batch, s_end, s_row, s_rows, tid);
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK)
void mqa_logits_build_sched_emit(const int* __restrict__ cu_tiles,
                                 const int* __restrict__ tile_seq,
                                 const int* __restrict__ query_start_loc,
                                 const int* __restrict__ row_ends,
                                 opus_mqa_cta_record* __restrict__ cta_info,
                                 int* __restrict__ scratch,
                                 int num_seqs, int num_tiles, int num_ctas, int block_k,
                                 int blocks) {
    __shared__ int smem[3 * BLOCK];
    const int tid    = (int)__builtin_amdgcn_workitem_id_x();
    const int bid    = (int)__builtin_amdgcn_workgroup_id_x();
    const int stride = BLOCK * blocks;
    const sched_detail::sched_rows s =
        sched_detail::make_rows(cu_tiles, tile_seq, query_start_loc, num_seqs, row_ends);

    int part_max = 0, part_sum = 0, part_nz = 0;
    sched_detail::emit_fast(s, cta_info, bid * BLOCK + tid, num_tiles, stride, block_k,
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
                                   const int* __restrict__ tile_seq,
                                   const int* __restrict__ query_start_loc,
                                   const int* __restrict__ row_ends,
                                   opus_mqa_cta_record* __restrict__ cta_info,
                                   const int* __restrict__ scratch,
                                   int num_seqs, int num_tiles, int num_ctas, int block_k,
                                   int resident, int blocks) {
    __shared__ int smem[3 * BLOCK];
    __shared__ int s_excl[BLOCK];
    __shared__ int s_tiles[BLOCK];
    __shared__ int s_batch[BLOCK];
    __shared__ int s_end[BLOCK];
    __shared__ int s_row[BLOCK];
    __shared__ int s_rows[BLOCK];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const sched_detail::sched_rows s =
        sched_detail::make_rows(cu_tiles, tile_seq, query_start_loc, num_seqs, row_ends);

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
    sched_detail::settle_and_emit<BLOCK>(s, cta_info, num_tiles, num_ctas, block_k, resident,
                                         max_tiles, total_tiles, nz_tiles, /*surplus_done=*/true,
                                         smem, s_excl, s_tiles, s_batch, s_end, s_row, s_rows, tid);
}

}  // namespace opus_logits
