// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MXFP4 paged MQA logits (OPUS): the host side of pa_mqa_logits_mxfp4.h -- the registry query, the
// Table schedule builder and the launch. The kernels live in their arch's TU
// (pa_mqa_logits_mxfp4_<arch>.cu) and are reached only through their registry entries.
#include "pa_mqa_logits_mxfp4.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include <algorithm>
#include <string>
#include <vector>

// -- the Table schedule builder -------------------------------------------------------------------
// Device-side, arch-agnostic int-only code (build_groups + build_sched).
//
// A group is up to rows_per_cta consecutive rows of one sequence (build_groups cuts them from
// query_start_loc; at one row a CTA a group is a row). Its block-table row is its sequence: the
// last b with query_start_loc[b] <= its first row, which skips empty sequences. build_groups finds
// it while cutting and writes it beside the cut; at one row a group, row r is sequence r whenever
// every sequence before it holds one row (checked, else searched). Rows at or past
// query_start_loc[num_seqs] belong to no sequence and get no work. A group's KV tiles (block_k
// keys each) are split into chunks, one a record.

namespace opus_logits {

constexpr int SCHED_BUILD_BLOCK      = 256;
constexpr int SCHED_BUILD_BLOCK_WIDE = 1024;
constexpr int SCHED_BUILD_MAX_BLOCKS = 256;
constexpr int SCHED_SCRATCH_INTS     = 3 * SCHED_BUILD_MAX_BLOCKS;
constexpr int SCHED_SCRATCH_RECORDS =
    (SCHED_SCRATCH_INTS * (int)sizeof(int) + (int)sizeof(opus_mqa_cta_record) - 1) /
    (int)sizeof(opus_mqa_cta_record);

// Records a cta_info buffer holds: one a CTA slot, then the multi-block emit's scratch.
constexpr int sched_buffer_records(int num_ctas) { return num_ctas + SCHED_SCRATCH_RECORDS; }

constexpr int GROUPS_BUILD_BLOCK     = 256;
constexpr int GROUPS_BUILD_MAX_SEQS = 2048;   // 8 KB of LDS for the per-sequence offsets

struct sched_build_plan {
    int block;   // workgroup width
    int blocks;  // workgroups for the emit; 1 means the single-workgroup kernel
};

inline sched_build_plan sched_plan(int num_groups, int num_ctas) {
    sched_build_plan p{SCHED_BUILD_BLOCK, 1};
    if (num_groups < 512) return p;                // narrow, one workgroup
    p.block = SCHED_BUILD_BLOCK_WIDE;
    if (num_groups <= 4096) return p;              // wide, one workgroup
    p.blocks = (num_ctas + SCHED_BUILD_BLOCK - 1) / SCHED_BUILD_BLOCK;
    if (p.blocks < 1) p.blocks = 1;
    if (p.blocks > SCHED_BUILD_MAX_BLOCKS) p.blocks = SCHED_BUILD_MAX_BLOCKS;
    return p;
}

namespace sched_detail {

// Every barrier here must be a full __syncthreads(), not a bare s_barrier: threads hand values
// through LDS (reductions, scan, general_emit staging) and global memory (settle_safe and the
// one-round pass read back cta_info records other threads stored). A bare s_barrier fences neither.

template<int BLOCK>
__device__ inline int block_sum(int v, int* s, int tid) {
    s[tid] = v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) s[tid] += s[tid + off];
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

// What a group is cut from: the group cut and each group's sequence (null: a group is a row),
// the sequences' bounds and the windows.
struct sched_rows {
    const int* cu_groups;
    const int* group_seq;
    const int* qsl;
    int        num_seqs;
    int        end_row;    // qsl[num_seqs]
    const int* row_ends;
};

__device__ inline sched_rows make_rows(const int* cu_groups, const int* group_seq, const int* qsl,
                                       int num_seqs, const int* row_ends) {
    return {cu_groups, group_seq, qsl, num_seqs, qsl[num_seqs], row_ends};
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

// Group g: its rows [r0, r1), its window end (the last row's: windows do not decrease along a
// sequence) and its sequence, 0 for an empty group.
struct group_facts {
    int r0, r1, end, seq;
};

// Every load a group needs is issued before any is used: an index made safe, the value discarded
// after. A load behind the condition that picks it waits a round trip for that condition first.
__device__ inline group_facts group_facts_of(const sched_rows& s, int g) {
    if (s.cu_groups) {
        // the cut wrote each group's sequence; a group past the cut (r0 == r1) never reads its own
        const int r0 = s.cu_groups[g], r1 = s.cu_groups[g + 1], seq = s.group_seq[g];
        return {r0, r1, r1 > r0 ? s.row_ends[r1 - 1] : 0, r1 > r0 ? seq : 0};
    }
    // A group is a row (g < the rows). Row g is sequence g whenever every sequence before it holds
    // one row (decode without speculation, candidate rows): its bounds load beside its window.
    const int ri = g < s.num_seqs ? g : s.num_seqs - 1;
    const int e = s.row_ends[g], q0 = s.qsl[ri], q1 = s.qsl[ri + 1];
    if (g >= s.end_row) return {s.end_row, s.end_row, 0, 0};
    const int seq = (ri == g && q0 == g && q1 == g + 1) ? g : seq_search(s, g);
    return {g, g + 1, e, seq};
}

__device__ inline int kv_tiles_of(int end, int block_k) {
    return end > 0 ? (end + block_k - 1) / block_k : 0;
}

// Reduces and emits in one walk: the one-CTA-per-group record is written speculatively and
// overwritten only if splitting is needed.
__device__ inline void emit_fast(const sched_rows& s,
                                 opus_mqa_cta_record* __restrict__ cta_info,
                                 int first_group, int num_groups, int stride, int block_k,
                                 int& part_max, int& part_sum, int& part_nz) {
    part_max = 0;
    part_sum = 0;
    part_nz  = 0;
    for (int g = first_group; g < num_groups; g += stride) {
        const group_facts f = group_facts_of(s, g);
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
        cta_info[g]     = rec;
    }
}

// Surplus slots are marked empty, not left stale: the table is reused across graph replays.
__device__ inline void mark_surplus(opus_mqa_cta_record* __restrict__ cta_info,
                                    int first_slot, int num_ctas, int stride) {
    for (int slot = first_slot; slot < num_ctas; slot += stride) {
        opus_mqa_cta_record rec{};
        cta_info[slot] = rec;
    }
}

// The smallest chunk size (in KV tiles) >= safe whose split fits num_ctas slots. Reads the counts
// emit_fast stored (recomputing them needs a dependent global load), so it must run before
// general_emit overwrites them; not __restrict__, as it aliases emit_fast's stores.
template<int BLOCK>
__device__ inline int settle_safe(const opus_mqa_cta_record* cta_info,
                                  int num_groups, int num_ctas,
                                  int safe, int max_tiles, int nz_groups, int* smem, int tid) {
    if (nz_groups >= num_ctas) return max_tiles;
    auto ctas_for = [&](int sf) {
        int part = 0;
        for (int g = tid; g < num_groups; g += BLOCK) {
            const int n = cta_info[g].chunk_count;
            part += (n + sf - 1) / sf;
        }
        return block_sum<BLOCK>(part, smem, tid);
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

// Splits every group into ceil(tiles / safe) even chunks, one a slot; returns the slots used.
// Every record carries `chunk_max` in its last word.
template<int BLOCK>
__device__ inline int general_emit(const sched_rows& s,
                                   opus_mqa_cta_record* __restrict__ cta_info,
                                   int num_groups, int num_ctas, int block_k, int safe,
                                   int chunk_max, int* smem, int* s_excl, int* s_tiles,
                                   int* s_seq, int* s_end, int* s_row, int* s_rows, int tid) {
    int carry = 0;
    for (int base = 0; base < num_groups; base += BLOCK) {
        const int g = base + tid;
        const group_facts f = g < num_groups ? group_facts_of(s, g) : group_facts{0, 0, 0, 0};
        const int tiles = kv_tiles_of(f.end, block_k);
        const int nc = (tiles + safe - 1) / safe;
        int block_total = 0;
        const int excl = block_scan_excl<BLOCK>(nc, smem, tid, block_total);
        s_excl[tid]  = excl;
        s_tiles[tid] = tiles;
        s_seq[tid]   = f.seq;
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
            const int gl = lo - 1;      // >= 0: s_excl[0] is 0 and j >= 0
            const int i  = j - s_excl[gl];
            const int slot = carry + j;
            if (slot < num_ctas) {      // cannot fail while num_ctas >= the schedule's need
                // Even split: safe sets the chunk count, not the size (a phase costs the longest
                // chunk). n >= 1, as the search picks the last group at each excl value.
                const int n       = s_tiles[gl];
                const int nchunks = n > 0 ? (n + safe - 1) / safe : 1;
                const int q = n / nchunks, r = n % nchunks;
                opus_mqa_cta_record rec{};
                rec.row_id      = s_row[gl];
                rec.batch_id    = s_seq[gl];
                rec.chunk_start = i * q + (i < r ? i : r);
                rec.chunk_count = q + (i < r ? 1 : 0);
                rec.local_end   = s_end[gl];
                rec.group_rows  = s_rows[gl];
                rec.chunk_max   = chunk_max;
                cta_info[slot]  = rec;
            }
        }
        carry += block_total;
        __syncthreads();
    }
    return carry;
}

// The split's target CTA count: a whole number of rounds of `resident` CTAs, so the last round
// is not left mostly empty.
__device__ inline int sched_target(int nz_groups, int resident) {
    return nz_groups <= resident ? resident : (nz_groups + resident - 1) / resident * resident;
}

// A one-round schedule (every slot resident at once) carries a bound on its longest chunk in
// every working record's chunk_max: the gfx1250 H64 kernel scales its wave priority by it. A split
// schedule's chunks are at most `safe`, written as they are emitted; an unsplit one's at most its
// longest group's count, patched into the groups' records once known. Multi-round schedules keep
// 0, which leaves the priority alone (there the age order retires CTAs for the next round).
template<int BLOCK>
__device__ inline void settle_and_emit(const sched_rows& s,
                                       opus_mqa_cta_record* __restrict__ cta_info,
                                       int num_groups, int num_ctas, int block_k, int resident,
                                       int max_tiles, int total_tiles, int nz_groups,
                                       bool surplus_done,
                                       int* smem, int* s_excl, int* s_tiles, int* s_seq,
                                       int* s_end, int* s_row, int* s_rows, int tid) {
    const int  target    = sched_target(nz_groups, resident);
    const int  safe0     = total_tiles > 0 ? (total_tiles + target - 1) / target : 1;
    const bool one_round = num_ctas <= resident;
    if (max_tiles <= safe0) {
        if (!surplus_done) mark_surplus(cta_info, num_groups + tid, num_ctas, BLOCK);
        if (one_round) {
            __syncthreads();   // emit_fast's records of these groups came from other threads
            for (int g = tid; g < num_groups; g += BLOCK) cta_info[g].chunk_max = max_tiles;
        }
        return;
    }
    const int safe = settle_safe<BLOCK>(cta_info, num_groups, num_ctas,
                                        safe0, max_tiles, nz_groups, smem, tid);
    const int carry = general_emit<BLOCK>(s, cta_info, num_groups, num_ctas, block_k, safe,
                                          one_round ? safe : 0, smem,
                                          s_excl, s_tiles, s_seq, s_end, s_row, s_rows, tid);
    mark_surplus(cta_info, carry + tid, num_ctas, BLOCK);
}

}  // namespace sched_detail

// Cut each sequence's rows into groups of at most qpb rows (unused at qpb == 1; see
// group_facts_of), and write each group's sequence beside the cut. A group must be contiguous rows
// of one sequence: otherwise the CTA's waves disagree on the trip count at the phase barrier and
// deadlock. One workgroup, so GROUPS_BUILD_MAX_SEQS caps the sequences.
__global__ __launch_bounds__(GROUPS_BUILD_BLOCK)
void mqa_logits_build_groups(const int* __restrict__ query_start_loc,
                             int* __restrict__ cu_groups,
                             int* __restrict__ group_seq,
                             int num_seqs, int max_groups, int qpb) {
    __shared__ int g_off[GROUPS_BUILD_MAX_SEQS + 1];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();

    for (int b = tid; b < num_seqs; b += GROUPS_BUILD_BLOCK)
        g_off[b] = (query_start_loc[b + 1] - query_start_loc[b] + qpb - 1) / qpb;
    __syncthreads();

    if (tid == 0) {
        int acc = 0;
        for (int b = 0; b < num_seqs; ++b) { const int c = g_off[b]; g_off[b] = acc; acc += c; }
        g_off[num_seqs] = acc;
    }
    __syncthreads();

    const int num_groups = g_off[num_seqs];
    const int end_row    = query_start_loc[num_seqs];

    for (int g = tid; g <= max_groups; g += GROUPS_BUILD_BLOCK) {
        if (g >= num_groups) { cu_groups[g] = end_row; continue; }   // empty: its sequence unread
        int lo = 0, hi = num_seqs - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (g_off[mid] <= g) lo = mid; else hi = mid - 1;
        }
        cu_groups[g] = query_start_loc[lo] + (g - g_off[lo]) * qpb;
        group_seq[g] = lo;
    }
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK)
void mqa_logits_build_sched(const int* __restrict__ cu_groups,
                            const int* __restrict__ group_seq,
                            const int* __restrict__ query_start_loc,
                            const int* __restrict__ row_ends,
                            opus_mqa_cta_record* __restrict__ cta_info,
                            int num_seqs, int num_groups, int num_ctas,
                            int block_k, int resident) {
    __shared__ int smem[3 * BLOCK];
    __shared__ int s_excl[BLOCK];
    __shared__ int s_tiles[BLOCK];
    __shared__ int s_seq[BLOCK];
    __shared__ int s_end[BLOCK];
    __shared__ int s_row[BLOCK];
    __shared__ int s_rows[BLOCK];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const sched_detail::sched_rows s =
        sched_detail::make_rows(cu_groups, group_seq, query_start_loc, num_seqs, row_ends);

    int part_max = 0, part_sum = 0, part_nz = 0;
    sched_detail::emit_fast(s, cta_info, tid, num_groups, BLOCK, block_k,
                            part_max, part_sum, part_nz);
    int max_tiles = 0, total_tiles = 0, nz_groups = 0;
    sched_detail::block_reduce_stats<BLOCK>(part_max, part_sum, part_nz, smem, tid,
                                            max_tiles, total_tiles, nz_groups);
    sched_detail::settle_and_emit<BLOCK>(s, cta_info, num_groups, num_ctas, block_k, resident,
                                         max_tiles, total_tiles, nz_groups, /*surplus_done=*/false,
                                         smem, s_excl, s_tiles, s_seq, s_end, s_row, s_rows, tid);
}

template<int BLOCK>
__global__ __launch_bounds__(BLOCK)
void mqa_logits_build_sched_emit(const int* __restrict__ cu_groups,
                                 const int* __restrict__ group_seq,
                                 const int* __restrict__ query_start_loc,
                                 const int* __restrict__ row_ends,
                                 opus_mqa_cta_record* __restrict__ cta_info,
                                 int* __restrict__ scratch,
                                 int num_seqs, int num_groups, int num_ctas, int block_k,
                                 int blocks) {
    __shared__ int smem[3 * BLOCK];
    const int tid    = (int)__builtin_amdgcn_workitem_id_x();
    const int bid    = (int)__builtin_amdgcn_workgroup_id_x();
    const int stride = BLOCK * blocks;
    const sched_detail::sched_rows s =
        sched_detail::make_rows(cu_groups, group_seq, query_start_loc, num_seqs, row_ends);

    int part_max = 0, part_sum = 0, part_nz = 0;
    sched_detail::emit_fast(s, cta_info, bid * BLOCK + tid, num_groups, stride, block_k,
                            part_max, part_sum, part_nz);
    sched_detail::mark_surplus(cta_info, num_groups + bid * BLOCK + tid, num_ctas, stride);

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
void mqa_logits_build_sched_finish(const int* __restrict__ cu_groups,
                                   const int* __restrict__ group_seq,
                                   const int* __restrict__ query_start_loc,
                                   const int* __restrict__ row_ends,
                                   opus_mqa_cta_record* __restrict__ cta_info,
                                   const int* __restrict__ scratch,
                                   int num_seqs, int num_groups, int num_ctas, int block_k,
                                   int resident, int blocks) {
    __shared__ int smem[3 * BLOCK];
    __shared__ int s_excl[BLOCK];
    __shared__ int s_tiles[BLOCK];
    __shared__ int s_seq[BLOCK];
    __shared__ int s_end[BLOCK];
    __shared__ int s_row[BLOCK];
    __shared__ int s_rows[BLOCK];
    const int tid = (int)__builtin_amdgcn_workitem_id_x();
    const sched_detail::sched_rows s =
        sched_detail::make_rows(cu_groups, group_seq, query_start_loc, num_seqs, row_ends);

    int part_max = 0, part_sum = 0, part_nz = 0;
    for (int i = tid; i < blocks; i += BLOCK) {
        const int m = scratch[i];
        if (m > part_max) part_max = m;
        part_sum += scratch[SCHED_BUILD_MAX_BLOCKS + i];
        part_nz  += scratch[2 * SCHED_BUILD_MAX_BLOCKS + i];
    }
    int max_tiles = 0, total_tiles = 0, nz_groups = 0;
    sched_detail::block_reduce_stats<BLOCK>(part_max, part_sum, part_nz, smem, tid,
                                            max_tiles, total_tiles, nz_groups);
    sched_detail::settle_and_emit<BLOCK>(s, cta_info, num_groups, num_ctas, block_k, resident,
                                         max_tiles, total_tiles, nz_groups, /*surplus_done=*/true,
                                         smem, s_excl, s_tiles, s_seq, s_end, s_row, s_rows, tid);
}

}  // namespace opus_logits

// -- the registry query and the launch ------------------------------------------------------------

namespace {

namespace ol = opus_logits;

using registry_fn = const ol::mqa_logits_instance* (*)(int*);
// Every arch's instances; an instance's id is its index in this concatenation.
constexpr registry_fn REGISTRIES[] = {ol::mqa_logits_gfx950_instances,
                                       ol::mqa_logits_gfx1250_instances};

const ol::mqa_logits_instance& instance_at(int id) {
    AITER_CHECK(id >= 0, "no MXFP4 MQA-logits instance ", id);
    int base = 0;
    for (registry_fn fn : REGISTRIES) {
        int n = 0;
        const ol::mqa_logits_instance* table = fn(&n);
        if (id < base + n) return table[id - base];
        base += n;
    }
    AITER_CHECK(false, "no MXFP4 MQA-logits instance ", id, "; there are ", base);
    return *REGISTRIES[0](&base);   // unreachable
}

std::string device_arch(int device_id) {
    hipDeviceProp_t prop;
    HIP_CALL(hipGetDeviceProperties(&prop, device_id));
    std::string arch = prop.gcnArchName;
    return arch.substr(0, arch.find(':'));
}

// Groups the Table builder cuts `rows` rows of `seqs` sequences into, at most: one a row, or up
// to rows_per_cta rows of one sequence (each sequence's last group may be short).
int max_groups_for(int rows, int seqs, int qpb) {
    return qpb == 1 ? rows : (rows + seqs * (qpb - 1)) / qpb;
}

// int32 words a schedule record takes.
constexpr int64_t RECORD_INTS = sizeof(opus_mqa_cta_record) / sizeof(int);

void check_i32_vector(const aiter_tensor_t& t, const char* name) {
    AITER_CHECK(t.dtype() == AITER_DTYPE_i32 && t.dim() == 1 && t.is_contiguous(),
                name, " must be a contiguous 1-D int32 tensor");
}

}  // namespace

// The record layout of mqa_logits_mxfp4_instances (aiter/ops/opus/pa_mqa_logits_mxfp4.py reads it
// by the leading field count).
#define MQA_LOGITS_INSTANCE_FIELDS                                                              \
    X(id) X(heads) X(rows_per_wave) X(waves) X(stages) X(scale_tdm) X(gather) X(entry_rows_min) \
    X(entry_rows_max) X(tile_keys) X(modes) X(flat_out) X(resident_table) X(resident_free)      \
    X(resident_prefill)

int mqa_logits_mxfp4_instances(int device_id, aiter_tensor_t& out) {
    aiter_detail::AiterThrowGuard throw_guard;
    AITER_CHECK(out.is_cpu() && out.dtype() == AITER_DTYPE_i32 && out.is_contiguous(),
                "out must be a contiguous host int32 tensor");
    HipDeviceGuard guard(device_id);
    const std::string arch = device_arch(device_id);
    int cus = 0;
    HIP_CALL(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, device_id));

#define X(f) +1
    constexpr int fields = 0 MQA_LOGITS_INSTANCE_FIELDS;
#undef X
    std::vector<int> recs{fields};
    int id = 0;
    for (registry_fn fn : REGISTRIES) {
        int n = 0;
        const ol::mqa_logits_instance* table = fn(&n);
        for (int i = 0; i < n; ++i, ++id) {
            const ol::mqa_logits_instance& in = table[i];
            if (arch != in.arch) continue;
            // CTAs resident at once in each compiled mode: occupancy x CUs, queried, not
            // hard-coded (a stale figure silently under-fills the part).
            int resident[ol::SCHED_MODES] = {};
            for (int m = 0; m < ol::SCHED_MODES; ++m) {
                if (!(in.modes & (1 << m))) continue;
                int per_cu = 0;
                HIP_CALL(in.occupancy(m, &per_cu));
                resident[m] = per_cu * cus;
            }
            const int row[] = {id, in.heads, in.rows_per_wave, in.waves, in.stages, in.scale_tdm,
                               in.gather, in.entry_rows_min, in.entry_rows_max, in.tile_keys,
                               in.modes, in.flat_out, resident[ol::SCHED_TABLE],
                               resident[ol::SCHED_FREE], resident[ol::SCHED_PREFILL]};
            static_assert(sizeof(row) / sizeof(row[0]) == fields);
            recs.insert(recs.end(), row, row + fields);
        }
    }
    const int n = static_cast<int>(recs.size());
    if (static_cast<int64_t>(out.numel()) >= n)
        std::copy(recs.begin(), recs.end(), reinterpret_cast<int*>(out.data_ptr()));
    return n;
}

// Caller-held buffers and no device data read on the host (capture-safe); runs once per forward
// while the kernel runs per layer.
void mqa_logits_mxfp4_build_schedule(aiter_tensor_t& query_start_loc,
                                     aiter_tensor_t& row_ends,
                                     aiter_tensor_t& cu_groups,
                                     aiter_tensor_t& cta_info,
                                     int instance,
                                     int num_ctas,
                                     int cta_resident) {
    aiter_detail::AiterThrowGuard throw_guard;
    const ol::mqa_logits_instance& in = instance_at(instance);
    AITER_CHECK(in.modes & (1 << ol::SCHED_TABLE), "instance ", instance, " has no Table mode");
    check_i32_vector(query_start_loc, "query_start_loc");
    check_i32_vector(row_ends, "row_ends");
    AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous(),
                "cta_info must be contiguous int32");
    const int qpb      = in.rows_per_wave * in.waves;
    const int num_seqs = static_cast<int>(query_start_loc.numel()) - 1;
    const int num_rows = static_cast<int>(row_ends.numel());
    AITER_CHECK(num_seqs >= 1, "query_start_loc must hold num_seqs + 1 >= 2 entries");
    // cta_resident follows the kernel's occupancy; a wrong value idles the part silently.
    AITER_CHECK(cta_resident >= 1, "cta_resident must be >= 1, got ", cta_resident);
    const int num_groups = max_groups_for(num_rows, num_seqs, qpb);
    // Otherwise a group may get no CTA and its rows are silently never written.
    AITER_CHECK(num_ctas >= num_groups, "num_ctas (", num_ctas, ") must be >= the groups (",
                num_groups, ") of ", num_rows, " rows in ", num_seqs, " sequences");
    // The buffer also holds the multi-block emit's scratch past the num_ctas slots.
    const int64_t cta_info_ints = ol::sched_buffer_records(num_ctas) * RECORD_INTS;
    AITER_CHECK(static_cast<int64_t>(cta_info.numel()) >= cta_info_ints,
                "cta_info holds ", RECORD_INTS, " int32 a CTA slot plus a ",
                ol::SCHED_SCRATCH_RECORDS, "-record build scratch; need ", cta_info_ints, ", got ",
                cta_info.numel(), ". Take it from cfg.schedule_buffers()");
    if (qpb > 1) {
        // the cut [num_groups + 1], then each group's sequence [num_groups + 1]
        AITER_CHECK(cu_groups.dtype() == AITER_DTYPE_i32 && cu_groups.is_contiguous() &&
                        static_cast<int64_t>(cu_groups.numel()) >= 2 * (num_groups + 1),
                    "cu_groups must be contiguous int32 with ", 2 * (num_groups + 1), " entries");
        // The cut stages the sequences' group offsets in one workgroup's LDS.
        AITER_CHECK(num_seqs <= ol::GROUPS_BUILD_MAX_SEQS, "the Table builder cuts at most ",
                    ol::GROUPS_BUILD_MAX_SEQS, " sequences into groups, got ", num_seqs);
    }
    if (num_groups == 0 && num_ctas == 0) return;

    HipDeviceGuard guard(row_ends.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    const int* p_qsl = reinterpret_cast<const int*>(query_start_loc.data_ptr());
    const int* p_le  = reinterpret_cast<const int*>(row_ends.data_ptr());
    auto* p_cta      = reinterpret_cast<opus_mqa_cta_record*>(cta_info.data_ptr());
    const int* p_cut = nullptr;
    const int* p_seq = nullptr;
    if (qpb > 1 && num_groups > 0) {
        int* cut = reinterpret_cast<int*>(cu_groups.data_ptr());
        ol::mqa_logits_build_groups<<<1, ol::GROUPS_BUILD_BLOCK, 0, stream>>>(
            p_qsl, cut, cut + num_groups + 1, num_seqs, num_groups, qpb);
        HIP_CALL_LAUNCH(hipGetLastError());
        p_cut = cut;
        p_seq = cut + num_groups + 1;
    }
    constexpr int NARROW = ol::SCHED_BUILD_BLOCK, WIDE = ol::SCHED_BUILD_BLOCK_WIDE;
    const int block_k = in.tile_keys;
    const auto plan   = ol::sched_plan(num_groups, num_ctas);
    if (plan.blocks == 1 && plan.block == NARROW) {
        ol::mqa_logits_build_sched<NARROW><<<1, NARROW, 0, stream>>>(
            p_cut, p_seq, p_qsl, p_le, p_cta, num_seqs, num_groups, num_ctas, block_k,
            cta_resident);
    } else if (plan.blocks == 1) {
        ol::mqa_logits_build_sched<WIDE><<<1, WIDE, 0, stream>>>(
            p_cut, p_seq, p_qsl, p_le, p_cta, num_seqs, num_groups, num_ctas, block_k,
            cta_resident);
    } else {
        int* scratch = reinterpret_cast<int*>(p_cta + num_ctas);
        ol::mqa_logits_build_sched_emit<NARROW><<<plan.blocks, NARROW, 0, stream>>>(
            p_cut, p_seq, p_qsl, p_le, p_cta, scratch, num_seqs, num_groups, num_ctas, block_k,
            plan.blocks);
        ol::mqa_logits_build_sched_finish<WIDE><<<1, WIDE, 0, stream>>>(
            p_cut, p_seq, p_qsl, p_le, p_cta, scratch, num_seqs, num_groups, num_ctas, block_k,
            cta_resident, plan.blocks);
    }
    HIP_CALL_LAUNCH(hipGetLastError());
}

// The kernels stride every input by compile-time constants and read no runtime stride but the
// table's and out's, so a padded or permuted input would be silently wrong: layouts are required.
// Byte counts catch a wrong array, not a wrong permutation (the Python wrapper checks the ndim).
void mqa_logits_mxfp4_launch(aiter_tensor_t& q,
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
                             int instance,
                             int mode,
                             float weight_scale,
                             int tpb_shift,
                             int cta_target,
                             int prefill_parts,
                             int grid) {
    aiter_detail::AiterThrowGuard throw_guard;
    const ol::mqa_logits_instance& in = instance_at(instance);
    AITER_CHECK(mode >= 0 && mode < ol::SCHED_MODES && (in.modes & (1 << mode)),
                "instance ", instance, " has no schedule mode ", mode);

    constexpr int TOKEN_BYTES = 64;   // D = 128 fp4
    constexpr int SCALE_BYTES = 4;    // D / 32 E8M0
    const int64_t rows = q.dim() == 3 ? q.size(0) : -1;
    AITER_CHECK(rows >= 0 && q.size(1) == in.heads && q.size(2) == TOKEN_BYTES,
                "q must be [rows, ", in.heads, ", ", TOKEN_BYTES, "] (H x D/2 packed fp4)");
    AITER_CHECK(q.dtype() == AITER_DTYPE_fp4x2 || q.dtype() == AITER_DTYPE_u8,
                "q must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(kv_cache.dtype() == AITER_DTYPE_fp4x2 || kv_cache.dtype() == AITER_DTYPE_u8,
                "kv_cache must be fp4x2 (E2M1, 2/byte) or u8 bytes");
    AITER_CHECK(q_scale.dtype() == AITER_DTYPE_u8 && kv_scale.dtype() == AITER_DTYPE_u8,
                "q_scale / kv_scale are E8M0 bytes and must be u8");
    AITER_CHECK(weights.dtype() == AITER_DTYPE_bf16 && weights.dim() == 2 &&
                    weights.size(0) >= rows && weights.size(1) == in.heads,
                "weights must be bf16 [rows, ", in.heads, "]");
    AITER_CHECK(q.is_contiguous() && q_scale.is_contiguous() && kv_cache.is_contiguous() &&
                    kv_scale.is_contiguous() && weights.is_contiguous(),
                "q / q_scale / kv_cache / kv_scale / weights must be contiguous");
    AITER_CHECK(static_cast<int64_t>(q_scale.numel()) == rows * in.heads * SCALE_BYTES,
                "q_scale must hold ", in.heads * SCALE_BYTES, " bytes a row, got ", q_scale.numel(),
                " for ", rows, " rows");
    AITER_CHECK(kv_cache.numel() % TOKEN_BYTES == 0 &&
                    kv_scale.numel() == kv_cache.numel() / TOKEN_BYTES * SCALE_BYTES,
                "kv_cache must be whole ", TOKEN_BYTES, "-byte keys and kv_scale ", SCALE_BYTES,
                " bytes a key; got ", kv_cache.numel(), " and ", kv_scale.numel());
    AITER_CHECK(block_tables.dtype() == AITER_DTYPE_i32 && block_tables.dim() == 2 &&
                    block_tables.stride(1) == 1,
                "block_tables must be int32 [num_seqs, width] with unit column stride");
    check_i32_vector(query_start_loc, "query_start_loc");
    check_i32_vector(row_ends, "row_ends");
    const int num_seqs = static_cast<int>(query_start_loc.numel()) - 1;
    AITER_CHECK(num_seqs >= 1 && block_tables.size(0) >= num_seqs,
                "query_start_loc must hold num_seqs + 1 >= 2 entries and block_tables a row each");
    AITER_CHECK(static_cast<int64_t>(row_ends.numel()) >= rows,
                "row_ends needs ", rows, " entries");

    AITER_CHECK(out.dtype() == AITER_DTYPE_fp32, "out must be fp32");
    const bool flat = out_offsets.numel() > 0;
    if (flat) {
        AITER_CHECK(in.flat_out,
                    "instance ", instance, " writes [rows, width] only, not out_offsets");
        AITER_CHECK(out_offsets.dtype() == AITER_DTYPE_i64 && out_offsets.is_contiguous() &&
                        static_cast<int64_t>(out_offsets.numel()) >= rows,
                    "out_offsets must be contiguous int64 with an entry a row");
        AITER_CHECK(out.is_contiguous(), "a flat out must be contiguous");
    } else {
        AITER_CHECK(out.dim() == 2 && out.size(0) >= rows && out.stride(1) == 1,
                    "out must be [rows, width] with unit column stride");
    }
    if (mode == ol::SCHED_TABLE)
        AITER_CHECK(cta_info.dtype() == AITER_DTYPE_i32 && cta_info.is_contiguous() &&
                        static_cast<int64_t>(cta_info.numel()) >= grid * RECORD_INTS,
                    "a Table launch over ", grid, " slots needs that many records");
    else
        AITER_CHECK(cta_target >= 1 && prefill_parts >= 1,
                    "cta_target / prefill_parts must be >= 1");
    AITER_CHECK(grid >= 0 && tpb_shift >= 0 && tpb_shift <= ol::MAX_TPB_SHIFT,
                "bad grid / tpb_shift");
    if (rows == 0 || grid == 0) return;

    mqa_logits_mxfp4_kargs k{};
    k.ptr_q               = q.data_ptr();
    k.ptr_q_scale         = q_scale.data_ptr();
    k.ptr_kv              = kv_cache.data_ptr();
    k.ptr_kv_scale        = kv_scale.data_ptr();
    k.ptr_block_tables    = reinterpret_cast<const int*>(block_tables.data_ptr());
    k.ptr_weights         = weights.data_ptr();
    k.ptr_out             = reinterpret_cast<float*>(out.data_ptr());
    k.ptr_out_offsets     = flat ? reinterpret_cast<const long long*>(out_offsets.data_ptr())
                                 : nullptr;
    k.ptr_query_start_loc = reinterpret_cast<const int*>(query_start_loc.data_ptr());
    k.ptr_row_ends        = reinterpret_cast<const int*>(row_ends.data_ptr());
    k.ptr_cta_info        = mode == ol::SCHED_TABLE
                                ? reinterpret_cast<const opus_mqa_cta_record*>(cta_info.data_ptr())
                                : nullptr;
    k.bt_stride           = static_cast<int>(block_tables.stride(0));
    k.tpb_shift           = tpb_shift;
    k.out_stride          = flat ? 0 : static_cast<int>(out.stride(0));
    k.num_rows            = static_cast<int>(rows);
    k.num_seqs            = num_seqs;
    k.weight_scale        = weight_scale;
    k.cta_target          = cta_target;
    k.prefill_parts       = prefill_parts;

    HipDeviceGuard guard(q.device_id);
    HIP_CALL_LAUNCH(in.launch(mode, k, grid, aiter::getCurrentHIPStream()));
}
