// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Prefill block-score kernel for nvfp4 index keys, H = 1, 2 or 4 index heads.
// 4 waves x NT query tiles of 16 rows per wave (NT = 2, 4 or 8, NT >= H);
// tile n holds 16 tokens of head n % H.
//
// Each page is decoded once to e4m3 in LDS (double buffered: the next page is
// fetched during the tile loop and decoded after it). Per wave and page, each
// of the 8 key tiles feeds NT MFMAs. Pages on a wave's causal diagonal use a
// masked copy of the tile loop.
#include "pa_sparse_block_select.hpp"

namespace aiter {
namespace sparse_attn {
namespace nvfp4_prefill {

constexpr int kWaves   = 4;
constexpr int kThreads = kWaves * kWave;
constexpr int kMT      = 8; // key tiles (MFMA M) per page
constexpr int kBlk     = SPARSE_BLOCK_SIZE;
constexpr int kD       = SPARSE_HEAD_DIM;
constexpr int kCodes   = kBlk * kD / 2;
constexpr int kRow     = kD + 16; // padded LDS row of one decoded key
constexpr int kPageLds = kBlk * kRow;

// NT query tiles (MFMA N) per wave; after the cross-lane max each lane holds
// kOut<NT> row maxima (NT = 2: half the lanes hold a duplicate).
template <int NT>
constexpr int kWaveRows = NT * 16;
template <int NT>
constexpr int kOut = NT >= 4 ? NT / 4 : 1;
// Query tokens per wave and per workgroup with H heads.
template <int NT, int H>
constexpr int kWaveTokens = kWaveRows<NT> / H;
template <int NT, int H>
constexpr int kGroupTokens = kWaves * kWaveTokens<NT, H>;

static_assert(kBlk == 128 && kD == 128, "tile counts assume 128 x 128 pages");
static_assert(kThreads * 32 == kCodes, "one lane decodes 64 dims of one token");

// Page id through a scalar (constant address space) load: lgkmcnt, so the
// wait for it does not also drain the score stores.
__device__ __forceinline__ int page_id(const int* __restrict__ bt_row, int blk)
{
    return ((const __attribute__((address_space(4))) int*)bt_row)[blk];
}

// One lane's share of a raw page: token tid / 2, dims [64 (tid & 1), +64):
// 32 B of e2m1 codes and the 4 scale bytes of those dims.
struct stage_t
{
    i32x4 v0, v1;
    uint32_t sc;
};

__device__ __forceinline__ void fetch(stage_t& st, const uint8_t* __restrict__ kp, int tid)
{
    st.sc = *(const uint32_t*)(kp + kCodes + tid * 4);
    st.v0 = *(const i32x4*)(kp + tid * 32);
    st.v1 = *(const i32x4*)(kp + tid * 32 + 16);
}

__device__ __forceinline__ void
commit(uint8_t* __restrict__ kb, const uint2* __restrict__ lut, int tid, const stage_t& st)
{
    const uint2 t0 = lut[st.sc & 0xff];
    const uint2 t1 = lut[(st.sc >> 8) & 0xff];
    const uint2 t2 = lut[(st.sc >> 16) & 0xff];
    const uint2 t3 = lut[st.sc >> 24];
    i32x4 lo0, hi0, lo1, hi1;
    nvfp4_decode(st.v0, t0, t1, lo0, hi0);
    nvfp4_decode(st.v1, t2, t3, lo1, hi1);
    uint8_t* row        = kb + (tid >> 1) * kRow + (tid & 1) * 64;
    *(i32x4*)(row + 0)  = lo0;
    *(i32x4*)(row + 16) = hi0;
    *(i32x4*)(row + 32) = lo1;
    *(i32x4*)(row + 48) = hi1;
}

// v_maximum3: no canonicalize of the MFMA outputs (fmaxf in IEEE mode adds
// one per operand).
__device__ __forceinline__ float max3f(float a, float b, float c)
{
    return __builtin_elementwise_maximum(__builtin_elementwise_maximum(a, b), c);
}

// Masked: element r of key tile m, token tile t (query tile n = t * H + h)
// is visible iff (m - t) * 16 + r < dlim (dlim per lane, see the caller).
template <bool Masked>
__device__ __forceinline__ void reduce_tile(float& tm, const f32x4& a, int m, int t, int dlim)
{
    if constexpr(!Masked)
        tm = max3f(max3f(tm, a[0], a[1]), a[2], a[3]);
    else
    {
        const int d = dlim - (m - t) * 16;
        tm          = max3f(tm, 0 < d ? a[0] : -INFINITY, 1 < d ? a[1] : -INFINITY);
        tm          = max3f(tm, 2 < d ? a[2] : -INFINITY, 3 < d ? a[3] : -INFINITY);
    }
}

// The 8 NT MFMAs of one page; A of key tile m + 1 is read while tile m runs.
template <bool Masked, int NT, int H>
__device__ __forceinline__ void page_tiles(const uint8_t* __restrict__ kb,
                                           const i32x8 (&qb)[NT],
                                           float (&tmax)[NT],
                                           int frag_rc,
                                           int kg,
                                           int dlim)
{
    i32x4 a[2][2];
    const uint8_t* ka = kb + frag_rc * kRow + kg * 32;
    a[0][0]           = *(const i32x4*)(ka);
    a[0][1]           = *(const i32x4*)(ka + 16);
    f32x4 acc[NT];
#pragma unroll
    for(int m = 0; m < kMT; ++m)
    {
        if(m + 1 < kMT)
        {
            const uint8_t* kn = ka + (m + 1) * 16 * kRow;
            a[(m + 1) & 1][0] = *(const i32x4*)(kn);
            a[(m + 1) & 1][1] = *(const i32x4*)(kn + 16);
        }
        const i32x8 av = join_frag(a[m & 1][0], a[m & 1][1]);
#pragma unroll
        for(int n = 0; n < NT; ++n)
        {
            if(m > 0)
                reduce_tile<Masked>(tmax[n], acc[n], m - 1, n / H, dlim);
            acc[n] = mfma_scale(av, qb[n], f32x4{0.f, 0.f, 0.f, 0.f});
        }
    }
#pragma unroll
    for(int n = 0; n < NT; ++n)
        reduce_tile<Masked>(tmax[n], acc[n], kMT - 1, n / H, dlim);
}

struct out_t
{
    float* score;
    int score_head_stride;
    int score_num_stride;
    int init_blocks;
    int local_blocks;
    int qlen;
    int q_base;
    int kvb; // wave token r sees kvb + r + 1 tokens
    int tok0;
};

__device__ __forceinline__ float sentinel(const out_t& o, int r, int blk, float m, bool& vis)
{
    const int kv_len   = max(o.kvb + r + 1, 0);
    const int nblk     = (kv_len + kBlk - 1) / kBlk;
    vis                = blk < nblk;
    const bool is_init = blk < o.init_blocks;
    const bool is_loc  = blk >= nblk - o.local_blocks;
    return is_loc ? 1e29f : (is_init ? 1e30f : m);
}

__device__ __forceinline__ float pmax32(float a, float b)
{
    auto r = __builtin_amdgcn_permlane32_swap(__float_as_uint(a), __float_as_uint(b), false, false);
    return fmaxf(__uint_as_float(r[0]), __uint_as_float(r[1]));
}

__device__ __forceinline__ float pmax16(float a, float b)
{
    auto r = __builtin_amdgcn_permlane16_swap(__float_as_uint(a), __float_as_uint(b), false, false);
    return fmaxf(__uint_as_float(r[0]), __uint_as_float(r[1]));
}

// tmax[NT] (per lane: max over its 4 tokens) -> w[kOut], lane L of w[k]
// holding the full max of query tile tile_of(k, L), column L & 15.
template <int NT>
__device__ __forceinline__ void xlane(const float (&tmax)[NT], float (&w)[kOut<NT>])
{
    float u[NT / 2];
#pragma unroll
    for(int j = 0; j < NT / 2; ++j)
        u[j] = pmax32(tmax[2 * j], tmax[2 * j + 1]);
    if constexpr(NT >= 4)
    {
#pragma unroll
        for(int k = 0; k < NT / 4; ++k)
            w[k] = pmax16(u[2 * k], u[2 * k + 1]);
    }
    else
        w[0] = pmax16(u[0], u[0]);
}

// NT >= 4: lane rows 0..3 -> tiles 0, 2, 1, 3 (+ 4 k). NT = 2: rows 0, 1 ->
// tile 0 and rows 2, 3 -> tile 1; rows 1 and 3 are copies and do not store.
template <int NT>
__device__ __forceinline__ int tile_of(int k, int lane)
{
    const int row = lane >> 4;
    if constexpr(NT >= 4)
        return 4 * k + ((row & 1) << 1) + (row >> 1);
    else
        return row >> 1;
}

template <int NT>
__device__ __forceinline__ bool lane_stores(int lane)
{
    return NT >= 4 || (lane & 16) == 0;
}

// Score row of query tile n (token tile n / H, head n % H) for lane column c.
template <int H>
__device__ __forceinline__ float* score_row(const out_t& o, int n, int c, int& r)
{
    r = o.tok0 + n / H * 16 + c;
    return o.score + (size_t)(n % H) * o.score_head_stride +
           (size_t)(o.q_base + r) * o.score_num_stride;
}

// Scores of the lane's rows for the last NS pages; bit j of m[k] marks slot
// j (block blk - NS + 1 + j at a flush) visible and in range.
template <int NS, int KO>
struct pend_t
{
    float v[KO][NS];
    uint32_t m[KO];
};

// One NS x 4 B store per row when all NS slots are live and aligned, else
// one store per live slot.
template <int NT, int NS, int H>
__device__ __forceinline__ void flush(const out_t& o, pend_t<NS, kOut<NT>>& pd, int blk, int lane)
{
    typedef float vec_t __attribute__((ext_vector_type(NS)));
#pragma unroll
    for(int k = 0; k < kOut<NT>; ++k)
    {
        if(pd.m[k] != 0)
        {
            int r;
            float* row   = score_row<H>(o, tile_of<NT>(k, lane), lane & 15, r);
            const int bb = blk - (NS - 1);
            if(pd.m[k] == (1u << NS) - 1 && (bb & (NS - 1)) == 0 &&
               ((uintptr_t)(row + bb) & (4 * NS - 1)) == 0)
            {
                vec_t x;
#pragma unroll
                for(int j = 0; j < NS; ++j)
                    x[j] = pd.v[k][j];
                *(vec_t*)(row + bb) = x;
            }
            else
            {
#pragma unroll
                for(int j = 0; j < NS; ++j)
                    if((pd.m[k] >> j) & 1)
                        row[bb + j] = pd.v[k][j];
            }
        }
        pd.m[k] = 0;
    }
}

// Writes the page's scores for the lane's rows: cross-lane max, init/local
// sentinels, then a store (NS == 1) or a queue slot (NS > 1, flushed every NS
// pages and after the wave's last active page).
template <int NT, int NS, int H>
__device__ __forceinline__ void store_scores(
    const out_t& o, const float (&tmax)[NT], int blk, int lane, pend_t<NS, kOut<NT>>& pd, bool last)
{
    float w[kOut<NT>];
    xlane<NT>(tmax, w);
#pragma unroll
    for(int k = 0; k < kOut<NT>; ++k)
    {
        int r;
        float* row = score_row<H>(o, tile_of<NT>(k, lane), lane & 15, r);
        bool vis;
        const float v = sentinel(o, r, blk, w[k], vis);
        const bool ok = r < o.qlen && vis && lane_stores<NT>(lane);
        if constexpr(NS == 1)
        {
            if(ok)
                row[blk] = v;
        }
        else
        {
#pragma unroll
            for(int j = 0; j + 1 < NS; ++j)
                pd.v[k][j] = pd.v[k][j + 1];
            pd.v[k][NS - 1] = v;
            pd.m[k]         = (pd.m[k] >> 1) | (ok ? (1u << (NS - 1)) : 0u);
        }
    }
    if constexpr(NS > 1)
        if((blk & (NS - 1)) == NS - 1 || last)
            flush<NT, NS, H>(o, pd, blk, lane);
}

// The workgroup order keys change once per 1 << kChunkKeyShift chunks.
constexpr int kChunkKeyShift = 2;

} // namespace nvfp4_prefill

// Workgroup -> (query group, request, chunk); see nvfp4_prefill_args.
// Workgroup l runs on XCD l % xcds, so with groups * requests % xcds == 0 the
// plain 3D grid keeps all chunks of a (group, request) on one XCD.
template <int NT, int NS, int H>
__global__
__launch_bounds__(nvfp4_prefill::kThreads) void pa_sparse_block_score_prefill_nvfp4_kernel(
    const uint8_t* __restrict__ q_idx,
    const uint8_t* __restrict__ key_cache_idx,
    float* __restrict__ score,
    const int* __restrict__ block_table,
    const int* __restrict__ cu_seqlens_q,
    const int* __restrict__ seq_lens,
    int block_table_stride,
    int wg_order,
    int score_head_stride,
    int score_num_stride,
    int chunk_blocks,
    int req_chunks,
    int init_blocks,
    int local_blocks,
    int page_stride,
    int num_q_groups,
    int num_reqs,
    int grid_chunks)
{
    using namespace nvfp4_prefill;
    static_assert(NT % H == 0, "a query tile holds tokens of one head");
    constexpr int kWT = kWaveTokens<NT, H>;
    constexpr int kGT = kGroupTokens<NT, H>;
    // wg_order packs the workgroup order into one argument early enough to be
    // preloaded, so the mapping waits on no kernel argument load.
    const int xcds      = wg_order & 0xff;
    const int group_xor = (wg_order >> 8) & 0xff;
    const int req_rot   = wg_order >> 16;
    int bx, by, bz;
    if(xcds > 0)
    {
        // 1D grid: each XCD takes a contiguous share of the work in
        // (group, request, chunk) order.
        const int n = num_q_groups * num_reqs * grid_chunks;
        const int l = blockIdx.x;
        const int p = (l % xcds) * ((n + xcds - 1) / xcds) + l / xcds;
        if(p >= n)
            return;
        bx           = p % num_q_groups;
        const int rr = p / num_q_groups;
        by           = rr % num_reqs;
        bz           = rr / num_reqs;
    }
    else
    {
        // group_xor + 1 divides num_q_groups, so the XOR is a permutation.
        bz            = blockIdx.z;
        const int key = bz >> kChunkKeyShift;
        bx            = (int)blockIdx.x ^ (key & group_xor);
        by            = req_rot ? ((int)blockIdx.y + key) % num_reqs : (int)blockIdx.y;
    }
    __shared__ __attribute__((aligned(16))) uint8_t kbuf[2 * kPageLds];
    __shared__ __attribute__((aligned(16))) uint2 lut[kNvfp4LutBytes / 8];

    const int tid   = threadIdx.x;
    i32x4 lut_chunk = {0, 0, 0, 0};
    if(tid < kNvfp4LutChunks)
        lut_chunk = nvfp4_lut_chunk(tid);

    const int lane    = tid & (kWave - 1);
    const int wave    = __builtin_amdgcn_readfirstlane(tid >> 6);
    const int frag_rc = lane & 15;
    const int kg      = lane >> 4;

    const int req    = by;
    const int q_base = cu_seqlens_q[req];
    const int qlen   = cu_seqlens_q[req + 1] - q_base;
    const int grp0   = bx * kGT;
    if(grp0 >= qlen)
        return;
    const int seq_len    = seq_lens[req];
    const int reach      = min(grp0 + kGT, qlen);
    const int kv_len_max = max(seq_len - qlen + reach, 0);
    const int nblk_max   = (kv_len_max + kBlk - 1) / kBlk;
    // Idle under any chunk size the rule below can pick.
    if(bz * chunk_blocks >= nblk_max && bz >= 2 * (req_chunks - 1))
        return;
    // req_chunks > 0: the host chunk rule (_score_split_prefill_nvfp4) on this
    // request's own block count, so a short request keeps req_chunks chunks;
    // grows past chunk_blocks only if grid_chunks would not cover it.
    int cb = chunk_blocks;
    if(req_chunks > 0)
    {
        const int nb = (seq_len + kBlk - 1) / kBlk;
        cb           = 1;
        while((cb < chunk_blocks && nb > 2 * cb * (req_chunks - 1)) || cb * grid_chunks < nb)
            cb *= 2;
    }
    const int b0 = bz * cb;
    if(b0 >= nblk_max)
        return;
    const int b1 = min(b0 + cb, nblk_max);

    const int tok0      = grp0 + wave * kWT;
    const int wave_rows = max(min(kWT, qlen - tok0), 0);
    const int kvb       = seq_len - qlen + tok0; // row tok0 + i sees kvb + i + 1 tokens
    const int full_blk  = max(kvb + 1, 0) / kBlk;
    const int wave_nblk = wave_rows > 0 ? (max(kvb + wave_rows, 0) + kBlk - 1) / kBlk : 0;
    const int* __restrict__ bt_row = block_table + (size_t)req * block_table_stride;

    out_t o;
    o.score             = score;
    o.score_head_stride = score_head_stride;
    o.score_num_stride  = score_num_stride;
    o.init_blocks       = init_blocks;
    o.local_blocks      = local_blocks;
    o.qlen              = qlen;
    o.q_base            = q_base;
    o.kvb               = kvb - tok0;
    o.tok0              = tok0;

    // First page in flight before Q.
    stage_t st;
    fetch(st, key_cache_idx + (size_t)page_id(bt_row, b0) * page_stride, tid);

    i32x8 qb[NT];
#pragma unroll
    for(int n = 0; n < NT; ++n)
    {
        const int r = tok0 + n / H * 16 + frag_rc;
        i32x4 lo = {0, 0, 0, 0}, hi = {0, 0, 0, 0};
        if(r < qlen)
        {
            const uint8_t* p = q_idx + ((size_t)(q_base + r) * H + n % H) * kD + kg * 32;
            lo               = *(const i32x4*)p;
            hi               = *(const i32x4*)(p + 16);
        }
        nvfp4_permute_q(lo, hi);
        qb[n] = join_frag(lo, hi);
    }

    if(tid < kNvfp4LutChunks)
        ((i32x4*)lut)[tid] = lut_chunk;
    __syncthreads();
    commit(kbuf, lut, tid, st);

    pend_t<NS, kOut<NT>> pd;
#pragma unroll
    for(int k = 0; k < kOut<NT>; ++k)
        pd.m[k] = 0;
    const int act_end = min(b1, wave_nblk);
    for(int blk = b0; blk < b1; ++blk)
    {
        const int buf = (blk - b0) & 1;
        __syncthreads();
        const bool more = blk + 1 < b1;
        if(more)
            fetch(st, key_cache_idx + (size_t)page_id(bt_row, blk + 1) * page_stride, tid);
        if(blk < wave_nblk)
        {
            const uint8_t* kb = kbuf + buf * kPageLds;
            float tmax[NT];
#pragma unroll
            for(int n = 0; n < NT; ++n)
                tmax[n] = -INFINITY;
            if(blk >= full_blk)
            {
                const int dlim = kvb + frag_rc + 1 - blk * kBlk - 4 * kg;
                page_tiles<true, NT, H>(kb, qb, tmax, frag_rc, kg, dlim);
            }
            else
                page_tiles<false, NT, H>(kb, qb, tmax, frag_rc, kg, 0);
            if(more)
                commit(kbuf + (buf ^ 1) * kPageLds, lut, tid, st);
            store_scores<NT, NS, H>(o, tmax, blk, lane, pd, blk + 1 == act_end);
        }
        else if(more)
            commit(kbuf + (buf ^ 1) * kPageLds, lut, tid, st);
    }
}

template <int NT, int NS, int H>
static void launch_nvfp4(SPARSE_SCORE_PREFILL_PARAMS, const nvfp4_prefill_args& a)
{
    dim3 grid(num_q_groups, num_reqs, a.grid_chunks);
    if(a.xcds > 0)
        grid = dim3((num_q_groups * num_reqs * a.grid_chunks + a.xcds - 1) / a.xcds * a.xcds);
    pa_sparse_block_score_prefill_nvfp4_kernel<NT, NS, H>
        <<<grid, dim3(nvfp4_prefill::kThreads), 0, stream>>>(q_idx,
                                                             key_cache_idx,
                                                             score,
                                                             block_table,
                                                             cu_seqlens_q,
                                                             seq_lens,
                                                             block_table_stride,
                                                             a.xcds | a.group_xor << 8 |
                                                                 (a.req_rot != 0) << 16,
                                                             score_head_stride,
                                                             score_num_stride,
                                                             chunk_blocks,
                                                             a.req_chunks,
                                                             init_blocks,
                                                             local_blocks,
                                                             a.page_stride,
                                                             num_q_groups,
                                                             num_reqs,
                                                             a.grid_chunks);
}

// Built: NT 2 (H <= 2), 4 and 8 with NS 1, NT 8 with NS 2.
template <int H>
static void launch_nvfp4_heads(SPARSE_SCORE_PREFILL_PARAMS, const nvfp4_prefill_args& a)
{
    if constexpr(H <= 2)
        if(a.wave_tiles == 2)
            return launch_nvfp4<2, 1, H>(SPARSE_SCORE_PREFILL_ARGS, a);
    if(a.wave_tiles == 4)
        return launch_nvfp4<4, 1, H>(SPARSE_SCORE_PREFILL_ARGS, a);
    if(a.store_pages == 2)
        return launch_nvfp4<8, 2, H>(SPARSE_SCORE_PREFILL_ARGS, a);
    launch_nvfp4<8, 1, H>(SPARSE_SCORE_PREFILL_ARGS, a);
}

void sparse_score_prefill_nvfp4(SPARSE_SCORE_PREFILL_PARAMS, const nvfp4_prefill_args& a)
{
    if(a.num_idx_heads == 4)
        return launch_nvfp4_heads<4>(SPARSE_SCORE_PREFILL_ARGS, a);
    if(a.num_idx_heads == 2)
        return launch_nvfp4_heads<2>(SPARSE_SCORE_PREFILL_ARGS, a);
    launch_nvfp4_heads<1>(SPARSE_SCORE_PREFILL_ARGS, a);
}

} // namespace sparse_attn
} // namespace aiter
