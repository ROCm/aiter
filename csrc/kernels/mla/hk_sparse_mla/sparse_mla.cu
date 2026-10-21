// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx950 sparse MLA, H32/D512. Contract: sparse_mla.h; design: README.md.
// Q/KV are unscaled E4M3; scores accumulate in FP32, P is quantized as FP8(P*448).
// Each workgroup handles one (query, key split), double-buffering 128-key KV blocks.
// Small batches split keys and combine normalized outputs via log-sum-exp.
#include "sparse_mla.h"

#include <cmath>
#include <cstdint>

#include "kittens.cuh"

namespace {
constexpr int H = 32, D = 512, BI = 128;
// Wave w owns KPW keys and DPW output dimensions; NW=8 hides LDS/MFMA latency.
constexpr int NW = 8, NT = 64 * NW, KPW = BI / NW, KT = KPW / 16, DPW = D / NW, DT = DPW / 16;
static_assert(NW == 4 || NW == 8, "4 or 8 waves");
constexpr int MAX_TOPK  = 2304; // Padded top-k plus k-pool tail.
constexpr float P_SCALE = 448.f;

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));

// KV: four [128 keys,128 dims] tiles. P: [32 heads,128 keys].
using KvTile = kittens::st_fp8e4m3<BI, 128, kittens::ducks::st_shape::st_16x128>;
using PTile  = kittens::st_fp8e4m3<H, BI, kittens::ducks::st_shape::st_16x128>;
// Byte offset within a [R,128] st_16x128 tile.
template <typename ST>
__device__ __forceinline__ uint32_t st_off(int r, int c)
{
    return (r / 16) * ST::underlying_subtile_bytes + ST::swizzle({r % 16, c});
}

// FP8 lane l: row l%16, two 16-byte chunks at 16*(l/16) and +64.
// FP32 lane l: column l%16, four rows starting at 4*(l/16).
using OpTile = kittens::rt_fp8e4m3<32, 128>;
using KTile  = kittens::rt_fp8e4m3<16, 128>;
using VTile  = kittens::rt_fp8e4m3<16, 128, kittens::ducks::rt_layout::col>;
using SAcc   = kittens::rt_fl<16, 32, kittens::ducks::rt_layout::col>; // keys x heads
using OAcc   = kittens::rt_fl<32, 16, kittens::ducks::rt_layout::col>; // heads x dims
template <typename RT>
__device__ __forceinline__ v8i& frag(RT& t, int i)
{
    return *reinterpret_cast<v8i*>(t.tiles[i][0].data);
}
template <typename RT>
__device__ __forceinline__ v4f& acc4(RT& t, int i, int j)
{
    return *reinterpret_cast<v4f*>(t.tiles[i][j].data);
}
// Tie LDS waits to fragments so the compiler cannot move consumers before them.
template <int N, typename RT>
__device__ __forceinline__ void lds_wait(RT& t)
{
    asm volatile("s_waitcnt lgkmcnt(%1)" : "+v"(frag(t, 0)) : "n"(N));
}
template <int N>
__device__ __forceinline__ void lds_wait2(OpTile& t)
{
    asm volatile("s_waitcnt lgkmcnt(%2)" : "+v"(frag(t, 0)), "+v"(frag(t, 1)) : "n"(N));
}
// Reduce across lanes l, l^16, l^32, l^48.
__device__ __forceinline__ float rows_max(float m)
{
    auto s = __builtin_amdgcn_permlane16_swap(__float_as_uint(m), __float_as_uint(m), false, false);
    m      = fmaxf(__uint_as_float(s[0]), __uint_as_float(s[1]));
    auto t = __builtin_amdgcn_permlane32_swap(__float_as_uint(m), __float_as_uint(m), false, false);
    return fmaxf(__uint_as_float(t[0]), __uint_as_float(t[1]));
}
__device__ __forceinline__ float rows_sum(float m)
{
    auto s = __builtin_amdgcn_permlane16_swap(__float_as_uint(m), __float_as_uint(m), false, false);
    m      = __uint_as_float(s[0]) + __uint_as_float(s[1]);
    auto t = __builtin_amdgcn_permlane32_swap(__float_as_uint(m), __float_as_uint(m), false, false);
    return __uint_as_float(t[0]) + __uint_as_float(t[1]);
}
__device__ __forceinline__ uint32_t pack4_fp8(float a, float b, float c, float d)
{
    int w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    return (uint32_t)__builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
}

template <bool WIDE_POOL>
__global__ __launch_bounds__(NT, 1) void sparse_mla_fwd(const uint8_t* __restrict__ q,
                                                        const uint8_t* __restrict__ kv,
                                                        const int* __restrict__ idx,
                                                        uint16_t* __restrict__ out,
                                                        int topk,
                                                        long kv_stride,
                                                        int num_slots,
                                                        float sc /* sm_scale * log2(e) */,
                                                        int nsplit,
                                                        float* __restrict__ part_o,
                                                        float* __restrict__ part_lse)
{
    const auto one = kittens::base_types::constants<kittens::fp8e8m0_4>::one();

    // Splits write normalized FP32 [T,nsplit,H,D] outputs and [T,nsplit,H] log2-sums.
    __shared__ KvTile kvs[2][D / 128]; // 128 KB
    __shared__ PTile ps;               // 4 KB
    __shared__ int idx_all[MAX_TOPK];
    __shared__ float mx_s[NW][H];
    __shared__ float al_s[H];
    __shared__ float l_s[NW][H];
    __shared__ float m_s[H];

    const int tok = blockIdx.x / nsplit, split = blockIdx.x % nsplit, tid = threadIdx.x,
              w = tid / 64, l = tid % 64, lr = l % 16, lg = l / 16;
    const int nblk_all = topk / BI;
    const int b0       = (int)((long)nblk_all * split / nsplit),
              b1       = (int)((long)nblk_all * (split + 1) / nsplit);
    const int nblk     = b1 - b0;
    const int* ix      = idx + (size_t)tok * topk + b0 * BI; // Split-local keys.

    // Q is MFMA operand B: qt[ks] subtile ht covers heads 16*ht.., dims 128*ks...
    OpTile qt[4];
#pragma unroll
    for(int ht = 0; ht < 2; ht++)
#pragma unroll
        for(int ks = 0; ks < 4; ks++)
        {
            const uint4* p = reinterpret_cast<const uint4*>(
                q + ((size_t)tok * H + 16 * ht + lr) * D + ks * 128 + lg * 16);
            const uint4 x = p[0], y = p[4];
            frag(qt[ks], ht) =
                v8i{(int)x.x, (int)x.y, (int)x.z, (int)x.w, (int)y.x, (int)y.y, (int)y.z, (int)y.w};
        }

    // NW/4 waves fill each KV tile; transfer u covers rows g0+8*u .. +7.
    // The XOR swizzle is self-inverse.
    const int gc = w / (NW / 4), g0 = (w % (NW / 4)) * (BI * 4 / NW);
    // Masked keys load slot 0 to avoid stale LDS bytes; their probability is zero.
    const kittens::i32x4 kv_rsrc =
        kittens::make_srsrc(kv, WIDE_POOL ? 0u : (uint32_t)((long)num_slots * kv_stride));
    auto issue_gather = [&](int blk, int buf) {
#pragma unroll
        for(int u = 0; u < 64 / NW; u++)
        {
            const int key = g0 + 8 * u + (l >> 3);
            const int id  = idx_all[blk * BI + key];
            const int lc  = (KvTile::swizzle({key % 16, 16 * (l & 7)}) % 128) / 16;
            auto* lds     = &kvs[buf][gc].data[(g0 + 8 * u) * 128];
            if constexpr(WIDE_POOL)
            {
                const uint8_t* src =
                    kv + (size_t)(id < 0 ? 0 : id) * kv_stride + 128 * gc + lc * 16;
                __builtin_amdgcn_global_load_lds(
                    reinterpret_cast<const void*>(src),
                    (__attribute__((address_space(3))) void*)(uintptr_t)lds,
                    16,
                    0,
                    0);
            }
            else
            {
                const uint32_t off =
                    (uint32_t)(id < 0 ? 0 : id) * (uint32_t)kv_stride + 128 * gc + lc * 16;
                kittens::llvm_amdgcn_raw_buffer_load_lds(
                    kv_rsrc,
                    (kittens::as3_uint32_ptr)lds,
                    16,
                    off,
                    0,
                    0,
                    static_cast<int>(kittens::coherency::cache_all));
            }
        }
    };

    OAcc o[DT]; // Heads 16*ht.., dims DPW*w+16*dt...
#pragma unroll
    for(int dt = 0; dt < DT; dt++)
        kittens::zero(o[dt]);
    // Running max/sum for head 16*ht+lr.
    float m_run[2] = {-1073741824.f, -1073741824.f}, l_run[2] = {0.f, 0.f};

    // Stage indices once; mask out-of-pool slots.
    for(int i = tid; i < nblk * BI; i += NT)
    {
        const int v = ix[i];
        idx_all[i]  = (unsigned)v < (unsigned)num_slots ? v : -1;
    }
    __syncthreads();
    issue_gather(0, 0);
    for(int blk = 0; blk < nblk; blk++)
    {
        const int buf = blk & 1; // Split-local block.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads(); // KV ready; prior KV/P reads finished.
        if(blk + 1 < nblk)
            issue_gather(blk + 1, buf ^ 1);
        KvTile(&kb)[D / 128] = kvs[buf];

        // S^T = K Q^T: st[kt](0,ht) holds key KPW*w+16*kt+4*lg+r, head 16*ht+lr.
        SAcc st[KT];
#pragma unroll
        for(int kt = 0; kt < KT; kt++)
        {
            KTile ka[4]; // Two ds_read_b128 per tile.
            kittens::zero(st[kt]);
#pragma unroll
            for(int ks = 0; ks < 4; ks++)
                kittens::load(ka[ks], KvTile::subtile<16, 128>(kb[ks], {KPW / 16 * w + kt, 0}));
            lds_wait<6>(ka[0]);
            kittens::mma_ABt_scaled(st[kt], ka[0], qt[0], st[kt], &one, &one);
            lds_wait<4>(ka[1]);
            kittens::mma_ABt_scaled(st[kt], ka[1], qt[1], st[kt], &one, &one);
            lds_wait<2>(ka[2]);
            kittens::mma_ABt_scaled(st[kt], ka[2], qt[2], st[kt], &one, &one);
            lds_wait<0>(ka[3]);
            kittens::mma_ABt_scaled(st[kt], ka[3], qt[3], st[kt], &one, &one);
        }
// Mask invalid keys.
#pragma unroll
        for(int kt = 0; kt < KT; kt++)
#pragma unroll
            for(int r = 0; r < 4; r++)
                if(idx_all[blk * BI + KPW * w + 16 * kt + 4 * lg + r] < 0)
#pragma unroll
                    for(int ht = 0; ht < 2; ht++)
                        acc4(st[kt], 0, ht)[r] = -INFINITY;

// Per-head max: reduce within lanes, across lane rows, then across waves.
#pragma unroll
        for(int ht = 0; ht < 2; ht++)
        {
            float v = -INFINITY;
#pragma unroll
            for(int kt = 0; kt < KT; kt++)
                v = fmaxf(v,
                          fmaxf(fmaxf(acc4(st[kt], 0, ht)[0], acc4(st[kt], 0, ht)[1]),
                                fmaxf(acc4(st[kt], 0, ht)[2], acc4(st[kt], 0, ht)[3])));
            v = rows_max(v);
            if(lg == 0)
                mx_s[w][16 * ht + lr] = v;
        }
        __syncthreads();
        float alpha[2];
#pragma unroll
        for(int ht = 0; ht < 2; ht++)
        {
            const int h = 16 * ht + lr;
            float mb    = mx_s[0][h];
#pragma unroll
            for(int i = 1; i < NW; i++)
                mb = fmaxf(mb, mx_s[i][h]);
            const float mn = fmaxf(m_run[ht], mb);
            alpha[ht]      = exp2f((m_run[ht] - mn) * sc);
            m_run[ht]      = mn;
            if(lg == 0 && w == 0)
                al_s[h] = alpha[ht];
        }
// Accumulate row sums and pack FP8(P*448) into LDS: four keys per dword.
#pragma unroll
        for(int ht = 0; ht < 2; ht++)
        {
            const int h    = 16 * ht + lr;
            const float ms = m_run[ht] * sc;
            float sum      = 0.f;
#pragma unroll
            for(int kt = 0; kt < KT; kt++)
            {
                float p[4];
#pragma unroll
                for(int r = 0; r < 4; r++)
                {
                    p[r] = exp2f(acc4(st[kt], 0, ht)[r] * sc - ms);
                    sum += p[r];
                }
                const uint32_t pk = pack4_fp8(fminf(p[0] * P_SCALE, P_SCALE),
                                              fminf(p[1] * P_SCALE, P_SCALE),
                                              fminf(p[2] * P_SCALE, P_SCALE),
                                              fminf(p[3] * P_SCALE, P_SCALE));
                *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(ps.data) +
                                             st_off<PTile>(h, KPW * w + 16 * kt + 4 * lg)) = pk;
            }
            l_run[ht] = l_run[ht] * alpha[ht] + sum;
        }
        __syncthreads(); // P and alpha complete

// Rescale O and accumulate P*V for this wave's output dimensions.
#pragma unroll
        for(int ht = 0; ht < 2; ht++)
#pragma unroll
            for(int r = 0; r < 4; r++)
            {
                const float a = al_s[16 * ht + 4 * lg + r];
#pragma unroll
                for(int dt = 0; dt < DT; dt++)
                    acc4(o[dt], ht, 0)[r] *= a;
            }
        OpTile pt; // P: MFMA operand A, rows are heads.
        kittens::load(pt, ps);
        // Prefetch V^T one tile ahead with four ds_read_b64_tr_b8 instructions.
        const int vc = DPW * w / 128, v0 = DPW * w % 128;
        VTile vt[2];
        kittens::load(vt[0], kb[vc], v0);
        lds_wait2<4>(pt);
#pragma unroll
        for(int dt = 0; dt < DT; dt++)
        {
            const int cur = dt & 1;
            if(dt + 1 < DT)
            {
                kittens::load(vt[cur ^ 1], kb[vc], v0 + 16 * (dt + 1));
                lds_wait<4>(vt[cur]); // Only the four newer reads may remain.
            }
            else
            {
                lds_wait<0>(vt[cur]);
            }
            kittens::mma_ABt_scaled(o[dt], pt, vt[cur], o[dt], &one, &one);
        }
    }

// Normalize outputs; write BF16 or FP32 split partials.
#pragma unroll
    for(int ht = 0; ht < 2; ht++)
    {
        const float v = rows_sum(l_run[ht]);
        if(lg == 0)
            l_s[w][16 * ht + lr] = v;
        if(lg == 0 && w == 0)
            m_s[16 * ht + lr] = m_run[ht];
    }
    __syncthreads();
#pragma unroll
    for(int ht = 0; ht < 2; ht++)
#pragma unroll
        for(int r = 0; r < 4; r++)
        {
            const int h = 16 * ht + 4 * lg + r;
            float lt    = l_s[0][h];
#pragma unroll
            for(int i = 1; i < NW; i++)
                lt += l_s[i][h];
            const float inv = lt > 0.f ? 1.f / (lt * P_SCALE) : 0.f;
            if(nsplit == 1)
            {
                uint16_t* orow = out + ((size_t)tok * H + h) * D + DPW * w + lr;
#pragma unroll
                for(int dt = 0; dt < DT; dt++)
                {
                    const float v       = acc4(o[dt], ht, 0)[r] * inv;
                    const uint32_t bits = __float_as_uint(v);
                    orow[16 * dt]       = (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >>
                                               16); // BF16, round to nearest even.
                }
            }
            else
            {
                const size_t pr = ((size_t)tok * nsplit + split) * H + h;
                float* orow     = part_o + pr * D + DPW * w + lr;
#pragma unroll
                for(int dt = 0; dt < DT; dt++)
                    orow[16 * dt] = acc4(o[dt], ht, 0)[r] * inv;
                if(w == 0 && lr == 0)
                    part_lse[pr] = lt > 0.f ? log2f(lt) + m_s[h] * sc : -INFINITY;
            }
        }
}

// Merge partials weighted by exp2(lse_s - max_lse), then normalize.
// One workgroup per (token, four heads); preload all splits.
constexpr int MAX_SPLITS = MAX_TOPK / BI;
__global__ __launch_bounds__(256) void sparse_mla_combine(const float* __restrict__ part_o,
                                                          const float* __restrict__ part_lse,
                                                          uint16_t* __restrict__ out,
                                                          int nsplit)
{
    const int tok = blockIdx.x / (H / 4), h = (blockIdx.x % (H / 4)) * 4 + threadIdx.x / 64,
              c0     = (threadIdx.x % 64) * 8;
    const float* lse = part_lse + (size_t)tok * nsplit * H + h;
    float ls[MAX_SPLITS];
    float4 va[MAX_SPLITS], vb[MAX_SPLITS];
#pragma unroll
    for(int s = 0; s < MAX_SPLITS; s++)
        if(s < nsplit)
        {
            ls[s]             = lse[s * H];
            const float4* src = reinterpret_cast<const float4*>(
                part_o + (((size_t)tok * nsplit + s) * H + h) * D + c0);
            va[s] = src[0];
            vb[s] = src[1];
        }
    float mx = -INFINITY;
#pragma unroll
    for(int s = 0; s < MAX_SPLITS; s++)
        if(s < nsplit)
            mx = fmaxf(mx, ls[s]);
    float wsum = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for(int s = 0; s < MAX_SPLITS; s++)
        if(s < nsplit && ls[s] != -INFINITY)
        {
            const float wt = exp2f(ls[s] - mx);
            wsum += wt;
            acc[0] += wt * va[s].x;
            acc[1] += wt * va[s].y;
            acc[2] += wt * va[s].z;
            acc[3] += wt * va[s].w;
            acc[4] += wt * vb[s].x;
            acc[5] += wt * vb[s].y;
            acc[6] += wt * vb[s].z;
            acc[7] += wt * vb[s].w;
        }
    const float inv = wsum > 0.f ? 1.f / wsum : 0.f;
    uint32_t pk[4];
#pragma unroll
    for(int i = 0; i < 4; i++)
    {
        uint32_t b0 = __float_as_uint(acc[2 * i] * inv), b1 = __float_as_uint(acc[2 * i + 1] * inv);
        b0    = (b0 + 0x7fff + ((b0 >> 16) & 1)) >> 16;
        b1    = (b1 + 0x7fff + ((b1 >> 16) & 1)) >> 16;
        pk[i] = b0 | (b1 << 16);
    }
    *reinterpret_cast<uint4*>(out + ((size_t)tok * H + h) * D + c0) = {pk[0], pk[1], pk[2], pk[3]};
}
} // namespace

// Cover the CUs with (token,split) workgroups, capped by the number of key blocks.
extern "C" int hk_sparse_mla_splits(int T, int topk)
{
    if(T <= 0 || topk <= 0 || topk % BI || topk > MAX_TOPK)
        return 1;
    int ncu = 0, dev = 0;
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev);
    const int want = (ncu + T - 1) / T, maxs = topk / BI;
    return want < 1 ? 1 : (want > maxs ? maxs : want);
}
// FP32 partials; no workspace for one split.
extern "C" size_t hk_sparse_mla_workspace_bytes(int T, int topk)
{
    const int ns = hk_sparse_mla_splits(T, topk);
    return ns <= 1 ? 0 : (size_t)T * ns * H * (D + 1) * sizeof(float);
}
// Launch contract: sparse_mla.h.
extern "C" int hk_sparse_mla_fwd(const void* q,
                                 const void* kv,
                                 const int* idx,
                                 void* out,
                                 int T,
                                 int topk,
                                 long kv_stride_bytes,
                                 long num_slots,
                                 float sm_scale,
                                 void* workspace,
                                 size_t ws_bytes,
                                 hipStream_t stream)
{
    if(T <= 0)
        return HK_SPARSE_MLA_OK;
    // Gather transfers are 16-byte aligned.
    if(topk <= 0 || topk % BI || topk > MAX_TOPK || kv_stride_bytes % 16 || (uintptr_t)kv % 16)
        return HK_SPARSE_MLA_BAD_SHAPE;
    if(num_slots <= 0 || num_slots > INT32_MAX || kv_stride_bytes < D || !std::isfinite(sm_scale) ||
       sm_scale <= 0.f)
        return HK_SPARSE_MLA_BAD_SHAPE;
    // Pools beyond 32-bit buffer offsets need 64-bit global gathers.
    const bool wide_pool = (unsigned long)num_slots * (unsigned long)kv_stride_bytes > UINT32_MAX;
    const int ns         = hk_sparse_mla_splits(T, topk);
    float *part_o = nullptr, *part_lse = nullptr;
    if(ns > 1)
    {
        if(ws_bytes < hk_sparse_mla_workspace_bytes(T, topk))
            return HK_SPARSE_MLA_SMALL_WORKSPACE;
        part_o   = (float*)workspace;
        part_lse = part_o + (size_t)T * ns * H * D;
    }
    auto* fwd = wide_pool ? sparse_mla_fwd<true> : sparse_mla_fwd<false>;
    fwd<<<T * ns, NT, 0, stream>>>((const uint8_t*)q,
                                   (const uint8_t*)kv,
                                   idx,
                                   (uint16_t*)out,
                                   topk,
                                   kv_stride_bytes,
                                   (int)num_slots,
                                   sm_scale * 1.4426950408889634f,
                                   ns,
                                   part_o,
                                   part_lse);
    if(ns > 1)
        sparse_mla_combine<<<T*(H / 4), 256, 0, stream>>>(part_o, part_lse, (uint16_t*)out, ns);
    return hipGetLastError() == hipSuccess ? HK_SPARSE_MLA_OK : HK_SPARSE_MLA_LAUNCH_FAILED;
}
