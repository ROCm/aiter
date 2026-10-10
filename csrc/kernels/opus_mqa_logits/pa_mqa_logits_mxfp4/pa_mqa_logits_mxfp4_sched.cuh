#pragma once
// Device-side schedule builder (build_tiles + build_sched), shared by gfx950 and gfx1250:
// arch-agnostic int-only code. Included by pa_mqa_logits_mxfp4_opus.h under
// PA_MQA_LOGITS_MXFP4_IMPL after the kargs/record ABI it uses; not standalone.

// Keep this opening line and the two-space closing brace byte-identical: an out-of-tree check
// locates the block by them.
namespace opus_logits {

constexpr int SCHED_BUILD_BLOCK      = 256;
constexpr int SCHED_BUILD_BLOCK_WIDE = 1024;

// `resident` (CTAs the part holds at once) is per instance -- it follows the KV tile, hence
// occupancy -- so it is always passed explicitly, never defaulted.

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

// Launch policy, shared by both hosts.
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
// through LDS (reductions, scan, general_emit staging) AND global memory (settle_safe reads back
// cta_info records other threads' emit_fast stored). A bare s_barrier fences neither.

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

// A tile's window is the union of its rows'; the caller guarantees a non-decreasing window rule,
// so that is the first row's start and the last row's end. Not checked here.
__device__ inline int tile_union_end(const int* __restrict__ local_ends, int r0, int r1) {
    return r1 > r0 ? local_ends[r1 - 1] : 0;
}
__device__ inline int tile_union_start(const int* __restrict__ local_starts, int r0, int r1) {
    return (local_starts && r1 > r0) ? local_starts[r0] : 0;
}

// Tile t's first row. NULL cu_tiles is the identity cut (q_per_block == 1: a tile is a row).
__device__ inline int tile_row(const int* __restrict__ cu_tiles, int t) {
    return cu_tiles ? cu_tiles[t] : t;
}

// Branchless on purpose: callers are latency-bound strided walks, and a branch on the
// just-loaded value serializes them.
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

// Reads the count emit_fast stored (recomputing needs a dependent global load). Valid only
// before general_emit writes: general_emit must not use it, as its slot writes overlap the tiles
// it reads. Not __restrict__ (nor in settle_safe): it aliases emit_fast's stores.
__device__ inline int tile_kv_tiles_cached(const opus_mqa_cta_record* cta_info, int t) {
    return cta_info[t].chunk_count;
}

// Reduces and emits in one walk: the one-CTA-per-tile record is written speculatively and
// overwritten only if splitting is needed.
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
        // r1 > r0 guard: an empty surplus tile has r0 == cu_seq_q[batch], past row_to_batch.
        const int b     = (row_to_batch && r1 > r0) ? row_to_batch[r0] : r0;
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

// Surplus slots are marked empty, not left stale: the table is reused across cudagraph replays.
__device__ inline void mark_surplus(opus_mqa_cta_record* __restrict__ cta_info,
                                    int first_slot, int num_ctas, int stride) {
    for (int slot = first_slot; slot < num_ctas; slot += stride) {
        opus_mqa_cta_record rec{};
        rec.chunk_count = 0;
        cta_info[slot]  = rec;
    }
}

template<int BLOCK>
__device__ inline int settle_safe(const opus_mqa_cta_record* cta_info,
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
        s_batch[tid] = (row_to_batch && r1 > r0) ? row_to_batch[r0] : r0;   // as in emit_fast
        s_end[tid]   = tile_union_end(local_ends, r0, r1);
        s_ls[tid]    = tile_union_start(local_starts, r0, r1);
        s_row[tid]   = r0;
        s_rows[tid]  = r1 - r0;
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
                rec.chunk_start = s_ls[tl] / block_k + i * q + (i < r ? i : r);
                rec.chunk_count = q + (i < r ? 1 : 0);
                rec.local_start = s_ls[tl];
                rec.local_end   = s_end[tl];
                rec.group_rows  = s_rows[tl];
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

// Cut each batch's rows into tiles of at most qpb rows (unused at qpb == 1; see tile_row).
// Contract: a tile is contiguous rows of ONE batch. Breaking it deadlocks: the CTA's waves
// disagree on the trip count at the phase barrier. One workgroup, so GROUPS_BUILD_MAX_BATCH
// caps the batch.
__global__ __launch_bounds__(GROUPS_BUILD_BLOCK)
void mqa_logits_build_tiles(const int* __restrict__ cu_seq_q,
                            int* __restrict__ cu_tiles,
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
