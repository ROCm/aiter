#pragma once
// The per-tile schedule builder (build_tiles + build_sched and their sched_detail
// helpers), shared by BOTH arches and arch-agnostic (int-only device code, NOT under a
// `#if __gfxNNN__` guard). Split out of pa_mqa_logits_mxfp4_opus.h, which includes it under
// PA_MQA_LOGITS_MXFP4_IMPL after the kargs/record ABI it uses; NOT standalone.
//
// The KEEP-FIRST / two-space-brace markers below are load-bearing: an out-of-tree equivalence
// check locates this block by them (its path now points here rather than the opus header).

// ── the schedule builder ──
// KEEP THIS BLOCK FIRST AMONG THE `namespace opus_logits` BLOCKS AND ITS CLOSING BRACE
// SPELLED WITH TWO SPACES: the out-of-tree equivalence check against its opus-ops twin locates
// it by those two strings, which is why the `mqa_logits_sched` enum sits after the builder.
namespace opus_logits {

constexpr int SCHED_BUILD_BLOCK      = 256;
constexpr int SCHED_BUILD_BLOCK_WIDE = 1024;

// `resident` is the CTAs the part holds at once and is REQUIRED on both of these, with no
// default: it is PER INSTANCE -- it follows the accumulator width, hence the KV tile, hence the
// occupancy -- so a file-scope constant could only ever be right for one of them, and a default
// is how one gets launched with another's grid. It reaches both as an argument.

// The CTA count the split aims at: a WHOLE NUMBER OF ROUNDS. One CTA per tile leaves the last
// round `resident - (nz mod resident)` CTAs empty, and that tail is not small -- 1025 tiles is
// four full rounds and one straggler. 0 means do not split.
__host__ __device__ inline int sched_target(int nz_tiles, int resident) {
    if (resident <= 0) return 0;
    if (nz_tiles <= resident) return resident;
    return ((nz_tiles + resident - 1) / resident) * resident;
}

// The GRID: the tile count rounded to a whole number of rounds, one round as the floor, and a
// host constant so a captured graph replays at one shape. Deliberately tight -- a surplus
// slot's CTA still reads a 32-byte record nobody else touches.
__host__ __device__ inline int sched_slots(int num_tiles, int resident) {
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

// Tile `t`'s row bound, and the one place the cut is read. A NULL `cu_tiles` is the IDENTITY
// cut -- `q_per_block == 1`, where a tile IS a row. The predicate is a loop-invariant scalar,
// unlike the one the next comment is about.
__device__ inline int tile_row(const int* __restrict__ cu_tiles, int t) {
    return cu_tiles ? cu_tiles[t] : t;
}

// BRANCHLESS ON PURPOSE. An `if (e <= 0) return 0;` here reads naturally and costs 38% at 16384
// tiles on the gfx950 sibling: the callers are latency-bound strided walks, and a branch on the
// value just loaded serializes what was a pipelined stream.
__device__ inline int tile_kv_tiles(const int* __restrict__ cu_tiles,
                                    const int* __restrict__ local_starts,
                                    const int* __restrict__ local_ends,
                                    int t, int block_k) {
    const int r0    = tile_row(cu_tiles, t);
    const int r1    = tile_row(cu_tiles, t + 1);
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
        const int r0    = tile_row(cu_tiles, t);
        const int r1    = tile_row(cu_tiles, t + 1);
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
        const int r0 = (t < num_tiles) ? tile_row(cu_tiles, t)     : 0;
        const int r1 = (t < num_tiles) ? tile_row(cu_tiles, t + 1) : 0;
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

// Cut each batch's rows into runs of at most `qpb` and write the tile boundaries. NOT reached
// at `qpb == 1`, where a tile is a row and the schedule takes that mapping from `tile_row`.
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
