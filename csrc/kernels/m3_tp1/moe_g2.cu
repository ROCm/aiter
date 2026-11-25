// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

// MiniMax-M3.1 TP1 MoE GEMM2 (H 6144, I 3072, MXFP4 a4w4, sort block 128): out[tok * topk + slot] =
// bf16(sum_k A * W2 * route weight), fp32 accumulation in K order, as aiter's v2 "reduce" GEMM2.
// Tile = up to 256 sorted rows of one expert (two sort blocks) x 256 columns; 4 waves along N,
// 16-row MFMA groups past the tile's valid rows skipped. B comes from relayout_w2's preshuffle.
// moe_g2_gemm_w8 runs the same tiles on 8 waves per WG (2 per SIMD), from per-XCD work queues.
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "m3_tp1_prefill.h"

namespace aiter {
namespace {
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v2i __attribute__((ext_vector_type(2)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v2f __attribute__((ext_vector_type(2)));
typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

// fp32 pair -> packed bf16 pair, round-to-nearest-even (gfx950 v_cvt_pk_bf16_f32)
__device__ __forceinline__ uint32_t pk_bf16(float a, float b)
{
    return __builtin_bit_cast(uint32_t, __builtin_convertvector((v2f{a, b}), bf16x2));
}

constexpr int NH   = 6144;      // hidden size (GEMM2 N)
constexpr int KI   = 3072;      // MoE intermediate size (GEMM2 K)
constexpr int KT   = KI / 256;  // K-steps of 256
constexpr int KB   = KI / 2;    // bytes per A row
constexpr int NWB  = NH / 64;   // 64-column wave blocks per expert
constexpr int NBLK = NH / 256;  // 256-column WG blocks
constexpr int SLOT = 256 * 128; // LDS bytes of one A K-step slot

#define RFL(x) __builtin_amdgcn_readfirstlane(x)

__device__ __forceinline__ v8i ext8(v4i a) { return v8i{a.x, a.y, a.z, a.w, 0, 0, 0, 0}; }

#define MFMA(acc, a, b, sa, opa, sb, opb)                   \
    acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4( \
        ext8(a), ext8(b), acc, 4, 4, opa, sa, opb, sb)

__device__ __forceinline__ void lds_barrier()
{
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

typedef __amdgpu_buffer_rsrc_t rsrc_t;
__device__ __forceinline__ rsrc_t mk_rsrc(const void* p, uint32_t bytes)
{
    return __builtin_amdgcn_make_buffer_rsrc(
        const_cast<void*>(p), (short)0, (int)bytes, 0x00020000);
}
__device__ __forceinline__ v4i bload16(rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(v4i, __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0));
}
__device__ __forceinline__ v2i bload8(rsrc_t r, int voff, int soff)
{
    return __builtin_bit_cast(v2i, __builtin_amdgcn_raw_buffer_load_b64(r, voff, soff, 0));
}
__device__ __forceinline__ int bload4(rsrc_t r, int voff, int soff)
{
    return (int)__builtin_amdgcn_raw_buffer_load_b32(r, voff, soff, 0);
}

// Tile list: valid rows per sort block, then consecutive blocks of one expert paired.
// tiles[t] = {first row, valid rows (a prefix), expert, 0}; ntiles = {count, nq zeroed counters}.
__global__ void moe_g2_count(const int* __restrict__ sids,
                             const int* __restrict__ nvalid_ids,
                             int M,
                             int topk,
                             int* __restrict__ cnt)
{
    const int b     = blockIdx.x;
    const int total = nvalid_ids[0];
    int v           = 0;
    if(b * 128 < total)
    {
        int id = sids[b * 128 + threadIdx.x];
        v      = ((id & 0xFFFFFF) < M) && ((id >> 24) < topk);
    }
    int c = __syncthreads_count(v);
    if(threadIdx.x == 0)
        cnt[b] = c;
}

__global__ void __launch_bounds__(1024) moe_g2_tiles(const int* __restrict__ eids,
                                                     const int* __restrict__ nvalid_ids,
                                                     const int* __restrict__ cnt,
                                                     int4* __restrict__ tiles,
                                                     int* __restrict__ ntiles,
                                                     int nq)
{
    __shared__ int se[4096];
    __shared__ int wsum[16];
    const int nb = nvalid_ids[0] / 128;
    for(int b = threadIdx.x; b < nb; b += 1024)
        se[b] = eids[b];
    __syncthreads();
    int base = 0;
    for(int b0 = 0; b0 < nb; b0 += 1024)
    {
        const int b = b0 + threadIdx.x;
        int start   = 0;
        if(b < nb)
        {
            int r = 0;
            while(b - r - 1 >= 0 && se[b - r - 1] == se[b])
                ++r;
            start = (r & 1) == 0;
        }
        // block prefix sum of start flags
        const int lane = threadIdx.x & 63, w = threadIdx.x >> 6;
        unsigned long long m = __ballot(start);
        int pre              = __popcll(m & ((1ull << lane) - 1ull));
        if(lane == 0)
            wsum[w] = __popcll(m);
        __syncthreads();
        int off = 0, tot = 0;
        for(int i = 0; i < 16; ++i)
        {
            if(i < w)
                off += wsum[i];
            tot += wsum[i];
        }
        if(start)
        {
            const int t     = base + off + pre;
            const bool pair = (b + 1 < nb) && se[b + 1] == se[b];
            int nv          = cnt[b] + (pair ? cnt[b + 1] : 0);
            tiles[t]        = make_int4(b * 128, nv, se[b], 0);
        }
        base += tot;
        __syncthreads();
    }
    if(threadIdx.x == 0)
        ntiles[0] = base;
    else if(threadIdx.x <= nq)
        ntiles[threadIdx.x] = 0;
}

struct TileOff
{
    uint32_t b, bs, as, a; // byte offsets into Bw, Bs, As, A
    int row0, nb, nch, nrow16;
};

__device__ __forceinline__ TileOff tile_off(int t, const int4* tiles, int wave)
{
    TileOff o;
    const int mt   = t / NBLK;
    const int4 ti  = tiles[mt];
    o.row0         = RFL(ti.x);
    const int nval = RFL(ti.y), e = RFL(ti.z);
    o.nb              = t - mt * NBLK;
    o.nch             = (nval + 15) >> 4;
    o.nrow16          = o.nch * 16;
    const uint32_t wb = (uint32_t)(e * NWB + o.nb * 4 + wave);
    o.b               = wb * (uint32_t)(KT * 8192);
    o.bs              = wb * (uint32_t)(KT * 512);
    o.as              = (uint32_t)(o.row0 >> 5) * (uint32_t)(KT * 256);
    o.a               = (uint32_t)o.row0 * (uint32_t)KB;
    return o;
}

static_assert(KT % 4 == 0, "K-steps must be a multiple of 4");
__shared__ __attribute__((aligned(16))) uint8_t g2_a_slot0[SLOT];
__shared__ __attribute__((aligned(16))) uint8_t g2_a_slot1[SLOT];
__shared__ __attribute__((aligned(16))) uint8_t g2_a_slot2[SLOT];
__shared__ __attribute__((aligned(16))) uint8_t g2_a_slot3[SLOT];
__shared__ __attribute__((aligned(16))) int g2_row_id[2][256];
__shared__ __attribute__((aligned(16))) float g2_row_w[2][256];

// s is a constant after unrolling, so each use resolves to one LDS object
__device__ __forceinline__ uint8_t* a_slot(int s)
{
    switch(s & 3)
    {
    case 0: return g2_a_slot0;
    case 1: return g2_a_slot1;
    case 2: return g2_a_slot2;
    default: return g2_a_slot3;
    }
}

typedef __attribute__((address_space(3))) void* lds_ptr_t;

// Persistent: WG b walks tiles t = b, b + G, ... (t = m_tile * NBLK + n_block). A goes to
// LDS by DMA 3 K-steps ahead, B / scales to registers; the ring spans tile boundaries.
__global__ void __launch_bounds__(256, 1) moe_g2_gemm(const uint8_t* __restrict__ A,
                                                      const uint8_t* __restrict__ As,
                                                      const uint8_t* __restrict__ Bw,
                                                      const uint8_t* __restrict__ Bs,
                                                      const int4* __restrict__ tiles,
                                                      const int* __restrict__ ntiles,
                                                      const int* __restrict__ sids,
                                                      const float* __restrict__ sw,
                                                      uint16_t* __restrict__ out,
                                                      int M,
                                                      int topk)
{
#if defined(__gfx950__)
    const int nt = RFL(ntiles[0]);
    const int G  = gridDim.x;
    int t        = blockIdx.x;
    if(t / NBLK >= nt)
        return;
    const int tid = threadIdx.x, lane = tid & 63;
    const int wave   = RFL(tid >> 6);
    const rsrc_t rB  = mk_rsrc(Bw, 0xFFFFFFFFu);
    const rsrc_t rBs = mk_rsrc(Bs, 0xFFFFFFFFu);
    const rsrc_t rAs = mk_rsrc(As, 0xFFFFFFFFu);
    const rsrc_t rA  = mk_rsrc(A, 0xFFFFFFFFu);
    const rsrc_t rId = mk_rsrc(sids, 0xFFFFFFFFu);
    const rsrc_t rW  = mk_rsrc(sw, 0xFFFFFFFFu);
    TileOff cur      = tile_off(t, tiles, wave);

    // A DMA: lane l of wave w writes LDS row (j*32 + w*8 + l/8), physical chunk l%8, which holds
    // logical chunk (l%8) ^ ((row>>1)&7) of the (clamped) source row.
    const int drow = wave * 8 + (lane >> 3);
    const int dchk = ((lane & 7) ^ ((drow >> 1) & 7)) << 4;
    const int rl = lane & 15, kq = lane >> 4;
    const int ar0 = rl * 128 + (((0 + kq) ^ (rl >> 1)) << 4);
    const int ar1 = rl * 128 + (((4 + kq) ^ (rl >> 1)) << 4);

    v4f acc[16][4];
#pragma unroll
    for(int i = 0; i < 16; ++i)
#pragma unroll
        for(int J = 0; J < 4; ++J)
            acc[i][J] = v4f{0.f, 0.f, 0.f, 0.f};

    v4i bq[4][4][2];
    int sa[4][8];
    v2i sb[4];

#define LOAD_B(ob, obs, oas, nr16, kts, buf)                                                      \
    {                                                                                             \
        _Pragma("unroll") for(int J = 0; J < 4; ++J) _Pragma("unroll") for(int h = 0; h < 2; ++h) \
            bq[buf][J][h] = bload16(rB, lane * 16 + (J * 2 + h) * 1024, (ob) + (kts) * 8192);     \
        sb[buf]           = bload8(rBs, lane * 8, (obs) + (kts) * 512);                           \
        _Pragma("unroll") for(int cc = 0; cc < 8; ++cc) sa[buf][cc] =                             \
            bload4(rAs, lane * 4, (oas) + ((cc * 32 < (nr16) ? cc : 0) * KT + (kts)) * 256);      \
    }
#define LOAD_A(S, oa, nr16, kts)                                    \
    {                                                               \
        _Pragma("unroll") for(int j = 0; j < 8; ++j)                \
        {                                                           \
            const int row = j * 32 + drow;                          \
            __builtin_amdgcn_raw_ptr_buffer_load_lds(               \
                rA,                                                 \
                (lds_ptr_t)(a_slot(S) + (j * 32 + wave * 8) * 128), \
                16,                                                 \
                (row < (nr16) ? row : 0) * KB + dchk,               \
                (oa) + (kts) * 128,                                 \
                0,                                                  \
                0);                                                 \
        }                                                           \
    }

    int my_id = bload4(rId, (tid < cur.nrow16 ? tid : 0) * 4, cur.row0 * 4);
    int my_w  = bload4(rW, (tid < cur.nrow16 ? tid : 0) * 4, cur.row0 * 4);
    // prologue: A for K-steps 0..2 (DMA), B/scales for 0..2
    LOAD_A(0, cur.a, cur.nrow16, 0);
    LOAD_B(cur.b, cur.bs, cur.as, cur.nrow16, 0, 0);
    LOAD_A(1, cur.a, cur.nrow16, 1);
    LOAD_B(cur.b, cur.bs, cur.as, cur.nrow16, 1, 1);
    LOAD_A(2, cur.a, cur.nrow16, 2);
    LOAD_B(cur.b, cur.bs, cur.as, cur.nrow16, 2, 2);
    __builtin_amdgcn_s_waitcnt(0); // everything (gfx9 encoding: vmcnt 0, lgkmcnt 0)
    lds_barrier();
    v4i a_c[2], a_n[2];
    a_c[0] = *(const v4i*)(g2_a_slot0 + 0 * 2048 + ar0);
    a_c[1] = *(const v4i*)(g2_a_slot0 + 0 * 2048 + ar1);
    a_n[0] = *(const v4i*)(g2_a_slot0 + 1 * 2048 + ar0);
    a_n[1] = *(const v4i*)(g2_a_slot0 + 1 * 2048 + ar1);

    int par = 0;
    while(true)
    {
        const bool has_next = (t + G) / NBLK < nt;
        const TileOff nxt   = tile_off(has_next ? t + G : t, tiles, wave);
        g2_row_id[par][tid] = my_id;
        g2_row_w[par][tid]  = __int_as_float(my_w);
        __builtin_amdgcn_s_waitcnt(0xc07f); // lgkmcnt(0): ids visible before the next barrier
        my_id = bload4(rId, (tid < nxt.nrow16 ? tid : 0) * 4, nxt.row0 * 4);
        my_w  = bload4(rW, (tid < nxt.nrow16 ? tid : 0) * 4, nxt.row0 * 4);
#pragma unroll 1
        for(int kk = 0; kk < KT; kk += 4)
        {
#pragma unroll
            for(int u = 0; u < 4; ++u)
            {
                const int kt       = kk + u;
                const int tgt      = kt + 3;
                const bool here    = tgt < KT;
                const uint32_t ob  = here ? cur.b : nxt.b;
                const uint32_t obs = here ? cur.bs : nxt.bs;
                const uint32_t oas = here ? cur.as : nxt.as;
                const uint32_t oa  = here ? cur.a : nxt.a;
                const int nr16     = here ? cur.nrow16 : nxt.nrow16;
                const int kts      = here ? tgt : tgt - KT;
                const int nb3      = (u + 3) % 4;
                int nch_s          = cur.nch;
                asm volatile("" : "+s"(nch_s));
                const int bb = u;
#pragma unroll
                for(int i = 0; i < 16; ++i)
                {
                    // spread this step's prefetch (K-step kt+3) over the 16 blocks
                    if(i < 8)
                    {
                        const int row = i * 32 + drow;
                        __builtin_amdgcn_raw_ptr_buffer_load_lds(
                            rA,
                            (lds_ptr_t)(a_slot(u + 3) + (i * 32 + wave * 8) * 128),
                            16,
                            (row < nr16 ? row : 0) * KB + dchk,
                            oa + kts * 128,
                            0,
                            0);
                        // last A DMA: the 10 VMEM ops below and the next step's 25 stay after it
                        if(i == 7)
                            __builtin_amdgcn_sched_barrier(0);
                        bq[nb3][i >> 1][i & 1] = bload16(rB, lane * 16 + i * 1024, ob + kts * 8192);
                    }
                    else
                    {
                        if(i == 8)
                            sb[nb3] = bload8(rBs, lane * 8, obs + kts * 512);
                        const int cc = i - 8;
                        sa[nb3][cc]  = bload4(
                            rAs, lane * 4, oas + ((cc * 32 < nr16 ? cc : 0) * KT + kts) * 256);
                    }
                    v4i a_p[2];
                    if(i + 2 < 16)
                    {
                        a_p[0] = *(const v4i*)(a_slot(u) + (i + 2) * 2048 + ar0);
                        a_p[1] = *(const v4i*)(a_slot(u) + (i + 2) * 2048 + ar1);
                    }
                    else
                    {
                        a_p[0] = *(const v4i*)(a_slot(u + 1) + (i - 14) * 2048 + ar0);
                        a_p[1] = *(const v4i*)(a_slot(u + 1) + (i - 14) * 2048 + ar1);
                    }
                    if(__builtin_expect(i < nch_s, 1))
                    {
                        const int sA  = sa[bb][i >> 1];
                        const int sB0 = sb[bb].x, sB1 = sb[bb].y;
                        if((i & 1) == 0)
                        {
                            MFMA(acc[i][0], a_c[0], bq[bb][0][0], sA, 0, sB0, 0);
                            MFMA(acc[i][1], a_c[0], bq[bb][1][0], sA, 0, sB0, 1);
                            MFMA(acc[i][2], a_c[0], bq[bb][2][0], sA, 0, sB1, 0);
                            MFMA(acc[i][3], a_c[0], bq[bb][3][0], sA, 0, sB1, 1);
                            MFMA(acc[i][0], a_c[1], bq[bb][0][1], sA, 2, sB0, 2);
                            MFMA(acc[i][1], a_c[1], bq[bb][1][1], sA, 2, sB0, 3);
                            MFMA(acc[i][2], a_c[1], bq[bb][2][1], sA, 2, sB1, 2);
                            MFMA(acc[i][3], a_c[1], bq[bb][3][1], sA, 2, sB1, 3);
                        }
                        else
                        {
                            MFMA(acc[i][0], a_c[0], bq[bb][0][0], sA, 1, sB0, 0);
                            MFMA(acc[i][1], a_c[0], bq[bb][1][0], sA, 1, sB0, 1);
                            MFMA(acc[i][2], a_c[0], bq[bb][2][0], sA, 1, sB1, 0);
                            MFMA(acc[i][3], a_c[0], bq[bb][3][0], sA, 1, sB1, 1);
                            MFMA(acc[i][0], a_c[1], bq[bb][0][1], sA, 3, sB0, 2);
                            MFMA(acc[i][1], a_c[1], bq[bb][1][1], sA, 3, sB0, 3);
                            MFMA(acc[i][2], a_c[1], bq[bb][2][1], sA, 3, sB1, 2);
                            MFMA(acc[i][3], a_c[1], bq[bb][3][1], sA, 3, sB1, 3);
                        }
                    }
                    a_c[0] = a_n[0];
                    a_c[1] = a_n[1];
                    a_n[0] = a_p[0];
                    a_n[1] = a_p[1];
                }
                // publish step kt+2: its A DMA (block 7 of step kt-1) is followed by 10 VMEM ops
                // there and 25 here, so vmcnt(35) means it has landed before the barrier
                __builtin_amdgcn_s_waitcnt((35 & 15) | ((35 >> 4) << 14) | (7 << 4) | (15 << 8));
                // bare barrier: a release fence here would also wait for the A DMA issued this step
                asm volatile("s_barrier" ::: "memory");
            }
        }
        const int col = cur.nb * 256 + wave * 64 + rl * 4;
#pragma unroll
        for(int i = 0; i < 16; ++i)
        {
            if(i < cur.nch)
            {
                const v4i id4 = *(const v4i*)(&g2_row_id[par][i * 16 + kq * 4]);
                const v4f w4  = *(const v4f*)(&g2_row_w[par][i * 16 + kq * 4]);
#pragma unroll
                for(int v = 0; v < 4; ++v)
                {
                    const int id  = id4[v];
                    const int tok = id & 0xFFFFFF, sl = id >> 24;
                    if(tok < M && sl < topk)
                    {
                        const float wv    = w4[v];
                        const uint32_t lo = pk_bf16(acc[i][0][v] * wv, acc[i][1][v] * wv);
                        const uint32_t hi = pk_bf16(acc[i][2][v] * wv, acc[i][3][v] * wv);
                        *(v2i*)(out + ((size_t)(tok * topk + sl) * NH + col)) =
                            v2i{(int)lo, (int)hi};
                    }
                }
            }
#pragma unroll
            for(int J = 0; J < 4; ++J)
                acc[i][J] = v4f{0.f, 0.f, 0.f, 0.f};
        }
        if(!has_next)
            break;
        cur = nxt;
        t += G;
        par ^= 1;
    }
#else
    // gfx950-only (scaled f8f6f4 MFMA, 16-byte LDS DMA): compiled out for the other archs of a
    // multi-arch build; the host wrapper never launches it there.
    __builtin_trap();
#endif
}

// w8: moe_g2_gemm's tiles on 8 waves, wave w on rows (w >> 2) * 128 and columns (w & 3) * 64.
// Every accumulator gets moe_g2_gemm's MFMA sequence and epilogue: the same bytes.
constexpr int W8_NG       = 8;           // 16-row MFMA groups per wave (128 rows)
constexpr uint32_t W8_OOB = 0xFFFF0000u; // store offset of padding rows: the buffer store drops it
constexpr int W8_PUBCNT   = 36; // VMEM ops between a slot's last A DMA and its publish: 38 - 2

struct W8Off
{
    uint32_t b, bs, as, a;     // byte offsets into Bw, Bs, As, A
    int row0, nb, nch, nrow16; // nch: this wave's 16-row groups holding routes (0..8)
};

__device__ __forceinline__ W8Off w8_tile(int t, const int4* __restrict__ tiles, int wave)
{
    W8Off o;
    const int mt   = t / NBLK;
    const int4 ti  = tiles[mt];
    o.nb           = t - mt * NBLK;
    o.row0         = RFL(ti.x);
    const int nval = RFL(ti.y), e = RFL(ti.z);
    const int nch_t   = (nval + 15) >> 4;
    const int half    = wave >> 2;
    o.nrow16          = nch_t * 16;
    o.nch             = min(max(nch_t - half * 8, 0), 8);
    const uint32_t wb = (uint32_t)(e * NWB + o.nb * 4 + (wave & 3));
    o.b               = wb * (uint32_t)(KT * 8192);
    o.bs              = wb * (uint32_t)(KT * 512);
    o.as              = (uint32_t)((o.row0 >> 5) + half * 4) * (uint32_t)(KT * 256);
    o.a               = (uint32_t)o.row0 * (uint32_t)KB;
    return o;
}

// Work: 8 queues; queue x holds n-blocks x, x + 8, x + 16 of every tile. A WG drains its XCD's
// queue first, then steals, so any grid computes every item. -1 once every queue is drained.
__device__ __forceinline__ int w8_fetch(int* __restrict__ queues, int nitems, int& q)
{
    for(int n = 0; n < 8; ++n)
    {
        const int k = atomicAdd(&queues[q], 1);
        if(k < nitems)
            return (k / 3) * NBLK + q + 8 * (k % 3);
        q = (q + 1) & 7;
    }
    return -1;
}

// B goes to VGPRs in half-steps 1.5 K-steps ahead, the scales one step ahead, A to LDS by DMA 3
// steps ahead; slot kt + 1 is published by a barrier two group-halves before step kt ends.
__global__ void __launch_bounds__(512, 1) moe_g2_gemm_w8(const uint8_t* __restrict__ A,
                                                         const uint8_t* __restrict__ As,
                                                         const uint8_t* __restrict__ Bw,
                                                         const uint8_t* __restrict__ Bs,
                                                         const int4* __restrict__ tiles,
                                                         int* __restrict__ ctl,
                                                         const int* __restrict__ sids,
                                                         const float* __restrict__ sw,
                                                         uint16_t* __restrict__ out,
                                                         int M,
                                                         int topk)
{
#if defined(__gfx950__)
    __shared__ int w8_first[2];
    __shared__ int w8_next[2];
    const int nt      = RFL(ctl[0]);
    const int nitems  = nt * 3;
    int* const queues = ctl + 1;
    // only thread 0 fetches; q is its current queue, its XCD's first (L2 reuse of A and B)
    int q = (int)__builtin_amdgcn_s_getreg((3 << 11) | (0 << 6) | 20) & 7; // XCC_ID
    if(threadIdx.x == 0)
    {
        w8_first[0] = w8_fetch(queues, nitems, q);
        w8_first[1] = w8_fetch(queues, nitems, q);
    }
    __syncthreads();
    int t      = RFL(w8_first[0]);
    int t_next = RFL(w8_first[1]);
    if(t < 0)
        return;
    const int tid = threadIdx.x, lane = tid & 63;
    const int wave    = RFL(tid >> 6);
    const rsrc_t rB   = mk_rsrc(Bw, 0xFFFFFFFFu);
    const rsrc_t rBs  = mk_rsrc(Bs, 0xFFFFFFFFu);
    const rsrc_t rAs  = mk_rsrc(As, 0xFFFFFFFFu);
    const rsrc_t rA   = mk_rsrc(A, 0xFFFFFFFFu);
    const rsrc_t rId  = mk_rsrc(sids, 0xFFFFFFFFu);
    const rsrc_t rW   = mk_rsrc(sw, 0xFFFFFFFFu);
    const rsrc_t rOut = mk_rsrc(out, (uint32_t)M * (uint32_t)topk * (uint32_t)(NH * 2));
    W8Off cur         = w8_tile(t, tiles, wave);

    // A DMA: lane l of wave w writes LDS row (j * 64 + w * 8 + l / 8), physical chunk l % 8, which
    // holds logical chunk (l % 8) ^ ((row >> 1) & 7) of the (clamped) source row.
    const int drow = wave * 8 + (lane >> 3);
    const int half = wave >> 2;
    const int hro  = half * 8 * 2048; // this wave's 128 rows in an A slot
    const int dchk = ((lane & 7) ^ ((drow >> 1) & 7)) << 4;
    const int rl = lane & 15, kq = lane >> 4;
    const int ar0 = rl * 128 + (((0 + kq) ^ (rl >> 1)) << 4);
    const int ar1 = rl * 128 + (((4 + kq) ^ (rl >> 1)) << 4);

    v4f acc[W8_NG][4];
#pragma unroll
    for(int i = 0; i < W8_NG; ++i)
#pragma unroll
        for(int J = 0; J < 4; ++J)
            acc[i][J] = v4f{0.f, 0.f, 0.f, 0.f};

    v4i bq[4][4]; // B half-buffers: [(2 * kt + h) % 4][J]
    int sa[2][4]; // A scales per 32-row chunk, [kt % 2]
    v2i sb[2];    // B scales, [kt % 2]

#define W8_B(hb, ob, kts, h)                                 \
    _Pragma("unroll") for(int J = 0; J < 4; ++J) bq[hb][J] = \
        bload16(rB, lane * 16 + (J * 2 + (h)) * 1024, (ob) + (kts) * 8192);
#define W8_A(S, oa, nr16, kts, j)                                 \
    {                                                             \
        const int row = (j) * 64 + drow;                          \
        __builtin_amdgcn_raw_ptr_buffer_load_lds(                 \
            rA,                                                   \
            (lds_ptr_t)(a_slot(S) + ((j) * 64 + wave * 8) * 128), \
            16,                                                   \
            (row < (nr16) ? row : 0) * KB + dchk,                 \
            (oa) + (kts) * 128,                                   \
            0,                                                    \
            0);                                                   \
    }
#define W8_AS(buf, oas, nr16, kts, cc) \
    sa[buf][cc] = bload4(              \
        rAs, lane * 4, (oas) + ((half * 128 + (cc) * 32 < (nr16) ? (cc) : 0) * KT + (kts)) * 256)

    int my_id = 0, my_w = 0;
    if(tid < 256)
    {
        my_id = bload4(rId, (tid < cur.nrow16 ? tid : 0) * 4, cur.row0 * 4);
        my_w  = bload4(rW, (tid < cur.nrow16 ? tid : 0) * 4, cur.row0 * 4);
    }
    // prologue: A for steps 0..2, B halves (0, 0) (0, 1) (1, 0), scales for step 0
#pragma unroll
    for(int s = 0; s < 3; ++s)
#pragma unroll
        for(int j = 0; j < 4; ++j)
            W8_A(s, cur.a, cur.nrow16, s, j);
    W8_B(0, cur.b, 0, 0);
    W8_B(1, cur.b, 0, 1);
    W8_B(2, cur.b, 1, 0);
    sb[0] = bload8(rBs, lane * 8, cur.bs);
#pragma unroll
    for(int cc = 0; cc < 4; ++cc)
        W8_AS(0, cur.as, cur.nrow16, 0, cc);
    __builtin_amdgcn_s_waitcnt(0); // everything (gfx9 encoding: vmcnt 0, lgkmcnt 0)
    lds_barrier();
    v4i a_c = *(const v4i*)(g2_a_slot0 + hro + 0 * 2048 + ar0);
    v4i a_n = *(const v4i*)(g2_a_slot0 + hro + 1 * 2048 + ar0);

    int par = 0;
    while(true)
    {
        const bool has_next = t_next >= 0;
        // the item after next: its atomic's result is first used after the K loop
        int nn = 0;
        if(has_next && tid == 0)
            nn = w8_fetch(queues, nitems, q);
        const W8Off nxt = w8_tile(has_next ? t_next : t, tiles, wave);
        if(tid < 256)
        {
            const int tok = my_id & 0xFFFFFF, sl = my_id >> 24;
            g2_row_id[par][tid] = (tok < M && sl < topk)
                                      ? (int)((uint32_t)(tok * topk + sl) * (uint32_t)(NH * 2))
                                      : (int)W8_OOB;
            g2_row_w[par][tid]  = __int_as_float(my_w);
        }
        __builtin_amdgcn_s_waitcnt(0xc07f); // lgkmcnt(0): staged before the K-loop barriers
        if(tid < 256)
        {
            my_id = bload4(rId, (tid < nxt.nrow16 ? tid : 0) * 4, nxt.row0 * 4);
            my_w  = bload4(rW, (tid < nxt.nrow16 ? tid : 0) * 4, nxt.row0 * 4);
        }
#pragma unroll 1
        for(int kk = 0; kk < KT; kk += 4)
        {
#pragma unroll
            for(int u = 0; u < 4; ++u)
            {
                const int kt = kk + u;
                // prefetch: scales for kt + 1, B halves (kt + 1, 1) and (kt + 2, 0), A for kt + 3
                const bool h1 = kt + 1 < KT, h2 = kt + 2 < KT, h3 = kt + 3 < KT;
                const uint32_t ob1 = h1 ? cur.b : nxt.b, ob2 = h2 ? cur.b : nxt.b;
                const uint32_t obs1 = h1 ? cur.bs : nxt.bs, oas1 = h1 ? cur.as : nxt.as;
                const int nr1 = h1 ? cur.nrow16 : nxt.nrow16, nr3 = h3 ? cur.nrow16 : nxt.nrow16;
                const uint32_t oa3 = h3 ? cur.a : nxt.a;
                const int k1       = h1 ? kt + 1 : kt + 1 - KT;
                const int k2       = h2 ? kt + 2 : kt + 2 - KT;
                const int k3       = h3 ? kt + 3 : kt + 3 - KT;
                int nch_s          = cur.nch;
                asm volatile("" : "+s"(nch_s));
#pragma unroll
                for(int g = 0; g < 2 * W8_NG; ++g)
                {
                    const int h = g / W8_NG, i = g % W8_NG;
                    // issue: 17 VMEM ops per K-step
                    if(g == 0)
                    {
                        sb[(u + 1) & 1] = bload8(rBs, lane * 8, obs1 + k1 * 512);
                        W8_AS((u + 1) & 1, oas1, nr1, k1, 0);
                        W8_AS((u + 1) & 1, oas1, nr1, k1, 1);
                    }
                    else if(g == 1)
                    {
                        W8_AS((u + 1) & 1, oas1, nr1, k1, 2);
                        W8_AS((u + 1) & 1, oas1, nr1, k1, 3);
                    }
                    else if(g >= 2 && g < 6)
                    {
                        // B half (kt + 1, 1) -> half-buffer (2u + 3) % 4
                        const int J = g - 2;
                        bq[(2 * u + 3) & 3][J] =
                            bload16(rB, lane * 16 + (J * 2 + 1) * 1024, ob1 + k1 * 8192);
                    }
                    else if(g == 6 || g == 7)
                    {
                        // A for kt + 3 -> slot (u + 3) % 4, kept in program order (W8_PUBCNT)
                        __builtin_amdgcn_sched_barrier(0);
                        W8_A(u + 3, oa3, nr3, k3, 2 * (g - 6));
                        W8_A(u + 3, oa3, nr3, k3, 2 * (g - 6) + 1);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                    else if(g >= 8 && g < 12)
                    {
                        // B half (kt + 2, 0) -> half-buffer (2u) % 4 (half (kt, 0) is done)
                        const int J = g - 8;
                        bq[(2 * u) & 3][J] =
                            bload16(rB, lane * 16 + (J * 2 + 0) * 1024, ob2 + k2 * 8192);
                    }
                    // A operand two group-halves ahead
                    v4i a_p;
                    if(g + 2 < 2 * W8_NG)
                    {
                        const int hp = (g + 2) / W8_NG, ip = (g + 2) % W8_NG;
                        a_p = *(const v4i*)(a_slot(u) + hro + ip * 2048 + (hp ? ar1 : ar0));
                    }
                    else
                        a_p = *(const v4i*)(a_slot(u + 1) + hro + (g + 2 - 2 * W8_NG) * 2048 + ar0);
                    if(__builtin_expect(i < nch_s, 1))
                    {
                        const int hb  = (2 * u + h) & 3;
                        const int sA  = sa[u & 1][i >> 1];
                        const int sB0 = sb[u & 1].x, sB1 = sb[u & 1].y;
                        const int opa = (i & 1) + 2 * h;
                        if(opa == 0)
                        {
                            MFMA(acc[i][0], a_c, bq[hb][0], sA, 0, sB0, 0);
                            MFMA(acc[i][1], a_c, bq[hb][1], sA, 0, sB0, 1);
                            MFMA(acc[i][2], a_c, bq[hb][2], sA, 0, sB1, 0);
                            MFMA(acc[i][3], a_c, bq[hb][3], sA, 0, sB1, 1);
                        }
                        else if(opa == 1)
                        {
                            MFMA(acc[i][0], a_c, bq[hb][0], sA, 1, sB0, 0);
                            MFMA(acc[i][1], a_c, bq[hb][1], sA, 1, sB0, 1);
                            MFMA(acc[i][2], a_c, bq[hb][2], sA, 1, sB1, 0);
                            MFMA(acc[i][3], a_c, bq[hb][3], sA, 1, sB1, 1);
                        }
                        else if(opa == 2)
                        {
                            MFMA(acc[i][0], a_c, bq[hb][0], sA, 2, sB0, 2);
                            MFMA(acc[i][1], a_c, bq[hb][1], sA, 2, sB0, 3);
                            MFMA(acc[i][2], a_c, bq[hb][2], sA, 2, sB1, 2);
                            MFMA(acc[i][3], a_c, bq[hb][3], sA, 2, sB1, 3);
                        }
                        else
                        {
                            MFMA(acc[i][0], a_c, bq[hb][0], sA, 3, sB0, 2);
                            MFMA(acc[i][1], a_c, bq[hb][1], sA, 3, sB0, 3);
                            MFMA(acc[i][2], a_c, bq[hb][2], sA, 3, sB1, 2);
                            MFMA(acc[i][3], a_c, bq[hb][3], sA, 3, sB1, 3);
                        }
                    }
                    a_c = a_n;
                    a_n = a_p;
                    if(g == 2 * W8_NG - 3)
                    {
                        // publish slot kt + 1 (DMA'd at step kt - 2) after this wave's reads of
                        // slot kt; a bare barrier, as a fence would also wait for this step's DMA
                        __builtin_amdgcn_s_waitcnt((W8_PUBCNT & 15) | ((W8_PUBCNT >> 4) << 14) |
                                                   (7 << 4) | (0 << 8));
                        asm volatile("s_barrier" ::: "memory");
                    }
                }
            }
        }
        if(tid == 0)
            w8_next[par ^ 1] = has_next ? nn : -1;
        __syncthreads();
        // epilogue: bf16(acc * w) per route row; padding rows carry W8_OOB, which the store drops
        const uint32_t col = (uint32_t)(cur.nb * 512 + (wave & 3) * 128);
#pragma unroll
        for(int i = 0; i < W8_NG; ++i)
        {
            if(i < cur.nch)
            {
                const v4i off4 = *(const v4i*)(&g2_row_id[par][half * 128 + i * 16 + kq * 4]);
                const v4f w4   = *(const v4f*)(&g2_row_w[par][half * 128 + i * 16 + kq * 4]);
#pragma unroll
                for(int v = 0; v < 4; ++v)
                {
                    const float wv    = w4[v];
                    const uint32_t lo = pk_bf16(acc[i][0][v] * wv, acc[i][1][v] * wv);
                    const uint32_t hi = pk_bf16(acc[i][2][v] * wv, acc[i][3][v] * wv);
                    __builtin_amdgcn_raw_buffer_store_b64(
                        v2i{(int)lo, (int)hi}, rOut, off4[v] + rl * 8, col, 0);
                }
            }
#pragma unroll
            for(int J = 0; J < 4; ++J)
                acc[i][J] = v4f{0.f, 0.f, 0.f, 0.f};
        }
        if(!has_next)
            break;
        cur    = nxt;
        t      = t_next;
        t_next = RFL(w8_next[par ^ 1]);
        par ^= 1;
    }
#undef W8_B
#undef W8_A
#undef W8_AS
#else
    __builtin_trap(); // gfx950 only, as moe_g2_gemm
#endif
}

constexpr int kSortBlock     = 128;
constexpr int kMaxSortBlocks = 4096; // moe_g2_tiles stages the block experts in LDS

void check_i32(const aiter_tensor_t& t, size_t numel, const char* name)
{
    AITER_CHECK(t.dtype() == AITER_DTYPE_i32 && t.is_contiguous() && t.numel() >= numel,
                "m3_tp1_moe_g2: ",
                name,
                " must be a contiguous int32 tensor of at least ",
                numel,
                " elements");
}

void check_bytes(const aiter_tensor_t& t, size_t numel, const char* name)
{
    AITER_CHECK(t.element_size() == 1 && t.is_contiguous() && t.numel() >= numel,
                "m3_tp1_moe_g2: ",
                name,
                " must be a contiguous 1-byte tensor of at least ",
                numel,
                " elements");
}

void run_g2(const aiter_tensor_t& a2,
            const aiter_tensor_t& a2_scale,
            const aiter_tensor_t& w2_b,
            const aiter_tensor_t& w2_bs,
            const aiter_tensor_t& sorted_ids,
            const aiter_tensor_t& sorted_weights,
            const aiter_tensor_t& sorted_expert_ids,
            const aiter_tensor_t& num_valid_ids,
            aiter_tensor_t& cnt,
            aiter_tensor_t& tiles,
            aiter_tensor_t& ntiles,
            aiter_tensor_t& out,
            int64_t m,
            int64_t topk,
            bool w8)
{
    aiter_detail::AiterThrowGuard throw_guard;

    const size_t sorted      = sorted_ids.numel();
    const int64_t max_blocks = static_cast<int64_t>(sorted) / kSortBlock;
    AITER_CHECK(max_blocks <= kMaxSortBlocks,
                "m3_tp1_moe_g2: at most ",
                kMaxSortBlocks,
                " sort blocks of 128, got ",
                max_blocks);
    AITER_CHECK(m >= 0 && m < (1 << 24) && topk >= 1 && topk < 256,
                "m3_tp1_moe_g2: bad m ",
                m,
                " or topk ",
                topk);
    AITER_CHECK(out.dtype() == AITER_DTYPE_bf16 && out.is_contiguous() && out.dim() == 3 &&
                    out.size(0) == m && out.size(1) == topk && out.size(2) == NH,
                "m3_tp1_moe_g2: out must be a contiguous bf16 [m, topk, 6144] tensor");
    AITER_CHECK(a2.element_size() == 1 && a2.is_contiguous() && a2.dim() == 2 && a2.size(1) == KB &&
                    a2.size(0) >= max_blocks * kSortBlock,
                "m3_tp1_moe_g2: a2 must be the contiguous [sorted rows, 1536] MXFP4 GEMM1 output "
                "(I = 3072)");
    check_bytes(a2_scale, static_cast<size_t>(max_blocks) * kSortBlock * (KI / 32), "a2_scale");
    const size_t expert_bytes = static_cast<size_t>(NH) * KB;
    AITER_CHECK(w2_b.element_size() == 1 && w2_b.is_contiguous() && w2_b.numel() > 0 &&
                    w2_b.numel() % expert_bytes == 0,
                "m3_tp1_moe_g2: w2_b must be relayout_w2's weight copy of [E, 6144, 1536] bytes");
    check_bytes(w2_bs, w2_b.numel() / 16, "w2_bs");
    check_i32(sorted_ids, sorted, "sorted_ids");
    AITER_CHECK(
        sorted_weights.dtype() == AITER_DTYPE_fp32 && sorted_weights.is_contiguous() &&
            sorted_weights.numel() >= sorted,
        "m3_tp1_moe_g2: sorted_weights must be a contiguous fp32 tensor of sorted_ids' size");
    check_i32(sorted_expert_ids, max_blocks, "sorted_expert_ids");
    check_i32(num_valid_ids, 1, "num_valid_ids");
    check_i32(cnt, max_blocks, "cnt");
    check_i32(tiles, max_blocks * 4, "tiles");
    check_i32(ntiles, w8 ? 9 : 1, w8 ? "ctl" : "ntiles");
    AITER_CHECK(!w8 || m * topk * NH * 2 <= static_cast<int64_t>(W8_OOB),
                "m3_tp1_moe_g2_w8: ",
                m * topk,
                " route rows exceed its 32-bit store offsets");
    if(max_blocks == 0 || m == 0)
        return;

    const HipDeviceGuard guard(out.device_id);
    const hipStream_t stream = getCurrentHIPStream();
    moe_g2_count<<<dim3(max_blocks), dim3(kSortBlock), 0, stream>>>(
        static_cast<const int*>(sorted_ids.data_ptr()),
        static_cast<const int*>(num_valid_ids.data_ptr()),
        static_cast<int>(m),
        static_cast<int>(topk),
        static_cast<int*>(cnt.data_ptr()));
    moe_g2_tiles<<<dim3(1), dim3(1024), 0, stream>>>(
        static_cast<const int*>(sorted_expert_ids.data_ptr()),
        static_cast<const int*>(num_valid_ids.data_ptr()),
        static_cast<const int*>(cnt.data_ptr()),
        static_cast<int4*>(tiles.data_ptr()),
        static_cast<int*>(ntiles.data_ptr()),
        w8 ? 8 : 0);
    if(w8)
    {
        moe_g2_gemm_w8<<<dim3(get_num_cu_func()), dim3(512), 0, stream>>>(
            static_cast<const uint8_t*>(a2.data_ptr()),
            static_cast<const uint8_t*>(a2_scale.data_ptr()),
            static_cast<const uint8_t*>(w2_b.data_ptr()),
            static_cast<const uint8_t*>(w2_bs.data_ptr()),
            static_cast<const int4*>(tiles.data_ptr()),
            static_cast<int*>(ntiles.data_ptr()),
            static_cast<const int*>(sorted_ids.data_ptr()),
            static_cast<const float*>(sorted_weights.data_ptr()),
            static_cast<uint16_t*>(out.data_ptr()),
            static_cast<int>(m),
            static_cast<int>(topk));
        HIP_CALL(hipGetLastError());
        return;
    }
    // one persistent WG per CU of an MI355X
    moe_g2_gemm<<<dim3(256), dim3(256), 0, stream>>>(
        static_cast<const uint8_t*>(a2.data_ptr()),
        static_cast<const uint8_t*>(a2_scale.data_ptr()),
        static_cast<const uint8_t*>(w2_b.data_ptr()),
        static_cast<const uint8_t*>(w2_bs.data_ptr()),
        static_cast<const int4*>(tiles.data_ptr()),
        static_cast<const int*>(ntiles.data_ptr()),
        static_cast<const int*>(sorted_ids.data_ptr()),
        static_cast<const float*>(sorted_weights.data_ptr()),
        static_cast<uint16_t*>(out.data_ptr()),
        static_cast<int>(m),
        static_cast<int>(topk));
    HIP_CALL(hipGetLastError());
}

} // namespace

void m3_tp1_moe_g2(const aiter_tensor_t& a2,
                   const aiter_tensor_t& a2_scale,
                   const aiter_tensor_t& w2_b,
                   const aiter_tensor_t& w2_bs,
                   const aiter_tensor_t& sorted_ids,
                   const aiter_tensor_t& sorted_weights,
                   const aiter_tensor_t& sorted_expert_ids,
                   const aiter_tensor_t& num_valid_ids,
                   aiter_tensor_t& cnt,
                   aiter_tensor_t& tiles,
                   aiter_tensor_t& ntiles,
                   aiter_tensor_t& out,
                   int64_t m,
                   int64_t topk)
{
    run_g2(a2,
           a2_scale,
           w2_b,
           w2_bs,
           sorted_ids,
           sorted_weights,
           sorted_expert_ids,
           num_valid_ids,
           cnt,
           tiles,
           ntiles,
           out,
           m,
           topk,
           false);
}

void m3_tp1_moe_g2_w8(const aiter_tensor_t& a2,
                      const aiter_tensor_t& a2_scale,
                      const aiter_tensor_t& w2_b,
                      const aiter_tensor_t& w2_bs,
                      const aiter_tensor_t& sorted_ids,
                      const aiter_tensor_t& sorted_weights,
                      const aiter_tensor_t& sorted_expert_ids,
                      const aiter_tensor_t& num_valid_ids,
                      aiter_tensor_t& cnt,
                      aiter_tensor_t& tiles,
                      aiter_tensor_t& ctl,
                      aiter_tensor_t& out,
                      int64_t m,
                      int64_t topk)
{
    run_g2(a2,
           a2_scale,
           w2_b,
           w2_bs,
           sorted_ids,
           sorted_weights,
           sorted_expert_ids,
           num_valid_ids,
           cnt,
           tiles,
           ctl,
           out,
           m,
           topk,
           true);
}

} // namespace aiter
