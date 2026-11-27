// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// MiniMax-M3 lightning-indexer FP8 prefill block scorer that keeps each block chunk's top-16 in
// registers instead of writing the fp32 score plane (gfx950). Contract: aiter/ops/m3_fused_index_topk.py.
#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "m3_fused_index_topk.h"
#include <climits>
#include <cmath>
#include <cstdint>

namespace m3_fused_index_topk {

constexpr int HD         = 128; // index head dim
constexpr int PAGE       = 128; // tokens per index-cache page = sparse block
constexpr int NH         = 4;   // index heads: one wave each
constexpr int NF         = 4;   // 16-token feature tiles per wave
constexpr int OCC        = 2;
constexpr int TQ         = 16 * NF; // query tokens per CTA
constexpr int NTHR       = 64 * NH;
constexpr int PAGE_BYTES = PAGE * HD;
constexpr int TOPK       = 16;

struct Params
{
    const uint16_t* q;       // bf16 [total_q, NH, 128], token stride q_stride_n elements
    const uint8_t* k;        // e4m3 [pages, 128, 128]
    uint64_t* partial;       // [NH, total_q, nchunks * 16]
    const int* bt;           // [batch, bt_stride]
    const int* cu;           // [batch + 1]
    const int* prefix;       // [batch]
    long long k_page_stride; // bytes between pages
    int q_stride_n;
    int total_q, bt_stride, q_tiles, batch, nchunks, chunk_blocks, ncta_logical, ncta;
    float scale;
};

#if defined(__gfx950__)
typedef float v4f __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(3))) void* lds_ptr_t;
typedef __attribute__((address_space(1))) void* g_ptr_t;
#define WAIT_VM0() __builtin_amdgcn_s_waitcnt(0x0F70)

// The selector's packed key of a valid block: bit 48 | order-preserving score bits << 16 | id + 1.
__device__ __forceinline__ uint64_t pack_key(float s, int blk1)
{
    uint32_t bits    = __float_as_uint(s);
    bits             = bits == 0x80000000u ? 0u : bits;
    uint32_t ordered = bits ^ ((bits >> 31) ? 0xFFFFFFFFu : 0x80000000u);
    ordered          = (bits & 0x7FFFFFFFu) > 0x7F800000u ? 0u : ordered;
    return (1ull << 48) | ((uint64_t)ordered << 16) | (uint64_t)blk1;
}

// 16 bf16 -> 16 e4m3 bytes (RNE), element i at byte i
__device__ __forceinline__ v4i bf16x16_to_fp8(v4i lo, v4i hi)
{
    v4i out;
    const uint32_t w[8] = {(uint32_t)lo.x,
                           (uint32_t)lo.y,
                           (uint32_t)lo.z,
                           (uint32_t)lo.w,
                           (uint32_t)hi.x,
                           (uint32_t)hi.y,
                           (uint32_t)hi.z,
                           (uint32_t)hi.w};
#pragma unroll
    for(int d = 0; d < 4; ++d)
    {
        int r          = 0;
        const float a0 = __uint_as_float(w[2 * d] << 16), a1 = __uint_as_float(w[2 * d] & 0xFFFF0000u);
        const float b0 = __uint_as_float(w[2 * d + 1] << 16),
                    b1 = __uint_as_float(w[2 * d + 1] & 0xFFFF0000u);
        r      = __builtin_amdgcn_cvt_pk_fp8_f32(a0, a1, r, false);
        r      = __builtin_amdgcn_cvt_pk_fp8_f32(b0, b1, r, true);
        out[d] = r;
    }
    return out;
}

// compare-exchange: a <- max, b <- min
__device__ __forceinline__ void cx(uint64_t& a, uint64_t& b)
{
    const bool gt     = a > b;
    const uint64_t hi = gt ? a : b, lo = gt ? b : a;
    a                 = hi;
    b                 = lo;
}

// Merges 4 keys c[0..3] into the descending list e[0..15], keeping the top 16 (exact, branch-free):
// sort c, bitonic half-cleaner against e's tail, then a bitonic merge of 16.
__device__ __forceinline__ void merge4(uint64_t* e, uint64_t* c)
{
    cx(c[0], c[1]);
    cx(c[2], c[3]);
    cx(c[0], c[2]);
    cx(c[1], c[3]);
    cx(c[1], c[2]);
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        const uint64_t a = e[12 + i], b = c[3 - i];
        e[12 + i]        = a > b ? a : b;
    }
    cx(e[12], e[14]);
    cx(e[13], e[15]);
    cx(e[12], e[13]);
    cx(e[14], e[15]);
    uint64_t t;
    t     = e[12];
    e[12] = e[15];
    e[15] = t;
    t     = e[13];
    e[13] = e[14];
    e[14] = t;
#pragma unroll
    for(int d = 8; d >= 1; d >>= 1)
    {
#pragma unroll
        for(int i = 0; i < 16; ++i)
            if((i & d) == 0)
                cx(e[i], e[i + d]);
    }
}
#endif // __gfx950__

// CTA = the 4 index heads (one wave each; MQA, so they share every K page) x TQ tokens of one
// request x one chunk of chunk_blocks pages.
__global__ __launch_bounds__(NTHR, OCC) void kernel(Params p)
{
#if defined(__gfx950__)
    __shared__ __attribute__((aligned(16))) uint8_t s_k[2][PAGE_BYTES];
    const int tid = threadIdx.x, lane = tid & 63, w = tid >> 6; // w = head
    const int g = lane >> 4, n = lane & 15;

    // hardware CTA i runs on XCD i % 8: give each XCD a contiguous run of logical CTAs
    int bid = blockIdx.x;
    {
        const int per = p.ncta / 8;
        bid           = (bid & 7) * per + (bid >> 3);
    }
    if(bid >= p.ncta_logical)
        return;
    const int qt   = bid % p.q_tiles;
    const int rest = bid / p.q_tiles;
    const int b    = rest % p.batch;
    const int c    = rest / p.batch;

    const int seq_start = p.cu[b];
    const int q_len     = p.cu[b + 1] - seq_start;
    if(qt * TQ >= q_len)
        return;
    const int prefix  = p.prefix[b];
    const int t_hi    = min(q_len, (qt + 1) * TQ); // tokens [qt*TQ, t_hi)
    const int blk_end = (prefix + t_hi - 1) / PAGE + 1;
    const int blk_lo  = c * p.chunk_blocks;
    const int blk_hi  = min(blk_lo + p.chunk_blocks, blk_end);
    const int* bt_row = p.bt + (long long)b * p.bt_stride;

    // Q fragments: tile f, lane (g, n): token qt*TQ + 16f + n, k-bytes [16g,+16) and [64+16g,+16)
    v8i qf[NF];
    int qpos[NF];
#pragma unroll
    for(int f = 0; f < NF; ++f)
    {
        const int t = qt * TQ + 16 * f + n;
        qpos[f]     = t < q_len ? prefix + t : -1;
        if(t < q_len)
        {
            const uint16_t* qrow = p.q + (long long)(seq_start + t) * p.q_stride_n + w * HD;
            const v4i* src       = (const v4i*)(qrow + 16 * g);
            const v4i* src2      = (const v4i*)(qrow + 64 + 16 * g);
            const v4i a          = bf16x16_to_fp8(src[0], src[1]);
            const v4i a2         = bf16x16_to_fp8(src2[0], src2[1]);
            qf[f]                = (v8i){a.x, a.y, a.z, a.w, a2.x, a2.y, a2.z, a2.w};
        }
        else
        {
            qf[f] = (v8i){0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    // Lane (g, n) owns column (f = 4 kk + g, n): its top 16 sorted descending plus the last 4 pages'
    // keys. Seeds 15..0 lack bit 48, so every real key outranks them.
    constexpr int NK = NF / 4;
    uint64_t e[NK][16], cnd[NK][4];
#pragma unroll
    for(int kk = 0; kk < NK; ++kk)
    {
#pragma unroll
        for(int i = 0; i < 16; ++i)
            e[kk][i] = (uint64_t)(15 - i);
#pragma unroll
        for(int i = 0; i < 4; ++i)
            cnd[kk][i] = 0;
    }
    int my_qpos[NK];
#pragma unroll
    for(int kk = 0; kk < NK; ++kk)
    {
        const int t = qt * TQ + 16 * (4 * kk + g) + n;
        my_qpos[kk] = t < q_len ? prefix + t : -1;
    }

    // DMA a K page into LDS, 16 B a lane (wave w moves 1-KB pieces 4w..4w+3); LDS slot s holds
    // row s / 8, chunk (s % 8) ^ (row % 8).
    auto dma_page = [&](int blk, int buf) {
        const long long page = bt_row[blk];
        const uint8_t* src   = p.k + page * p.k_page_stride;
        asm volatile("" ::: "memory");
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            const int I = 4 * w + i, sl = I * 64 + lane, row = sl >> 3, pc = sl & 7;
            __builtin_amdgcn_global_load_lds((g_ptr_t)(src + row * HD + ((pc ^ (row & 7)) << 4)),
                                             (lds_ptr_t)(&s_k[buf][I * 1024]),
                                             16,
                                             0,
                                             0);
        }
        asm volatile("" ::: "memory");
    };

    if(blk_lo < blk_hi)
    {
        dma_page(blk_lo, 0);
        WAIT_VM0();
        __syncthreads();
    }
    const int diag_from = (prefix + qt * TQ) / PAGE; // first page that may hold a key past a query
    auto lds_a          = [&](int buf, int j) -> v8i {
        const int row     = 16 * j + n;
        const uint8_t* rb = &s_k[buf][row * HD];
        const v4i lo      = *(const v4i*)(rb + ((g ^ (row & 7)) << 4));
        const v4i hi      = *(const v4i*)(rb + (((g + 4) ^ (row & 7)) << 4));
        return (v8i){lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    };
    // one page: 8 key tiles x NF k=128 MFMAs (A double-buffered in registers), max folded per tile
    auto page_max = [&](int buf, int blk, bool diag, float* pm) {
#pragma unroll
        for(int f = 0; f < NF; ++f)
            pm[f] = -__builtin_inff();
        v8i a0 = lds_a(buf, 0), a1;
#pragma unroll
        for(int j = 0; j < 8; ++j)
        {
            const v8i a = (j & 1) ? a1 : a0;
            if(j + 1 < 8)
            {
                if(j & 1)
                    a0 = lds_a(buf, j + 1);
                else
                    a1 = lds_a(buf, j + 1);
            }
            v4f acc[NF];
#pragma unroll
            for(int f = 0; f < NF; ++f)
                acc[f] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    a, qf[f], (v4f){0.f, 0.f, 0.f, 0.f}, 0, 0, 0, 127, 0, 127);
            if(diag)
            {
                const int kp = blk * PAGE + 16 * j + 4 * g;
#pragma unroll
                for(int f = 0; f < NF; ++f)
#pragma unroll
                    for(int r = 0; r < 4; ++r)
                        acc[f][r] = (kp + r <= qpos[f]) ? acc[f][r] : -__builtin_inff();
            }
#pragma unroll
            for(int f = 0; f < NF; ++f)
            {
                pm[f] = __builtin_fmaxf(__builtin_fmaxf(pm[f], acc[f][0]), acc[f][1]);
                pm[f] = __builtin_fmaxf(__builtin_fmaxf(pm[f], acc[f][2]), acc[f][3]);
            }
        }
    };
    for(int blk = blk_lo; blk < blk_hi; ++blk)
    {
        const int buf   = (blk - blk_lo) & 1;
        const bool more = blk + 1 < blk_hi;
        if(more)
            dma_page(blk + 1, buf ^ 1);
        float pm[NF];
        if(blk >= diag_from)
            page_max(buf, blk, true, pm);
        else
            page_max(buf, blk, false, pm);
#pragma unroll
        for(int kk = 0; kk < NK; ++kk)
        {
            // 4x4 transpose-reduce over the lane groups: lane (g, n) ends with column (4kk+g, n)'s max
            float* q4 = pm + 4 * kk;
            auto s02  = __builtin_amdgcn_permlane32_swap(
                __float_as_uint(q4[0]), __float_as_uint(q4[2]), false, false);
            auto s13 = __builtin_amdgcn_permlane32_swap(
                __float_as_uint(q4[1]), __float_as_uint(q4[3]), false, false);
            const float r0 = __builtin_fmaxf(__uint_as_float(s02[0]), __uint_as_float(s02[1]));
            const float r1 = __builtin_fmaxf(__uint_as_float(s13[0]), __uint_as_float(s13[1]));
            auto s01       = __builtin_amdgcn_permlane16_swap(
                __float_as_uint(r0), __float_as_uint(r1), false, false);
            const float m    = __builtin_fmaxf(__uint_as_float(s01[0]), __uint_as_float(s01[1]));
            const int qp     = my_qpos[kk];
            const bool valid = qp >= 0 && blk * PAGE <= qp;
            // the query's own block is always selected
            const float sc   = (blk == qp / PAGE) ? 1e29f : m * p.scale;
            const uint64_t x = valid ? pack_key(sc, blk + 1) : 0ull;
            cnd[kk][3]       = cnd[kk][2];
            cnd[kk][2]       = cnd[kk][1];
            cnd[kk][1]       = cnd[kk][0];
            cnd[kk][0]       = x;
            if(((blk - blk_lo) & 3) == 3)
            {
                merge4(e[kk], cnd[kk]);
#pragma unroll
                for(int i = 0; i < 4; ++i)
                    cnd[kk][i] = 0;
            }
        }
        if(more)
        {
            WAIT_VM0();
            __syncthreads();
        }
    }

    if(((blk_hi - blk_lo) & 3) != 0 && blk_lo < blk_hi)
    {
#pragma unroll
        for(int kk = 0; kk < NK; ++kk)
            merge4(e[kk], cnd[kk]);
    }
#pragma unroll
    for(int kk = 0; kk < NK; ++kk)
    {
        const int t = qt * TQ + 16 * (4 * kk + g) + n;
        if(t < q_len)
        {
            uint64_t* dst = p.partial + ((long long)w * p.total_q + seq_start + t) * (p.nchunks * 16) +
                            c * 16;
#pragma unroll
            for(int i = 0; i < 16; i += 2)
                *(ulonglong2*)(dst + i) = make_ulonglong2(e[kk][i], e[kk][i + 1]);
        }
    }
#else
    // gfx950-only (scaled f8f6f4 MFMA, 16-byte global_load_lds, permlane swaps): compiled out for
    // the other archs of a multi-arch build; the Python wrapper never launches it there.
    __builtin_trap();
#endif
}

} // namespace m3_fused_index_topk

namespace aiter {

void m3_fused_index_topk_partial(const aiter_tensor_t& idx_q,
                                 const aiter_tensor_t& index_cache,
                                 aiter_tensor_t& partial,
                                 const aiter_tensor_t& block_table,
                                 const aiter_tensor_t& cu_seqlens_q,
                                 const aiter_tensor_t& prefix_lens,
                                 int64_t max_query_len,
                                 int64_t nchunks,
                                 int64_t chunk_blocks,
                                 float scale)
{
    using namespace m3_fused_index_topk;
    aiter_detail::AiterThrowGuard throw_guard;

    AITER_CHECK(idx_q.dtype() == AITER_DTYPE_bf16 && idx_q.dim() == 3 && idx_q.size(1) == NH &&
                    idx_q.size(2) == HD && idx_q.stride(1) == HD && idx_q.stride(2) == 1 &&
                    idx_q.stride(0) % 8 == 0 &&
                    reinterpret_cast<uintptr_t>(idx_q.data_ptr()) % 16 == 0,
                "idx_q must be bf16 [total_q, 4, 128], head stride 128, 16-byte aligned token rows");
    AITER_CHECK(index_cache.dtype() == AITER_DTYPE_fp8 && index_cache.dim() == 3 &&
                    index_cache.size(1) == PAGE && index_cache.size(2) == HD &&
                    index_cache.stride(1) == HD && index_cache.stride(2) == 1 &&
                    index_cache.stride(0) >= PAGE_BYTES && index_cache.stride(0) % 16 == 0 &&
                    reinterpret_cast<uintptr_t>(index_cache.data_ptr()) % 16 == 0,
                "index_cache must be e4m3 [pages, 128, 128] with contiguous, 16-byte aligned pages");
    AITER_CHECK(nchunks >= 1 && chunk_blocks >= 1, "nchunks and chunk_blocks must be positive");
    const int64_t total_q = idx_q.size(0);
    AITER_CHECK(partial.dtype() == AITER_DTYPE_i64 && partial.is_contiguous() && partial.dim() == 3 &&
                    partial.size(0) == NH && partial.size(1) == total_q &&
                    partial.size(2) == nchunks * TOPK &&
                    reinterpret_cast<uintptr_t>(partial.data_ptr()) % 16 == 0,
                "partial must be a contiguous int64 [4, total_q, nchunks * 16] tensor");
    AITER_CHECK(cu_seqlens_q.dtype() == AITER_DTYPE_i32 && cu_seqlens_q.dim() == 1 &&
                    cu_seqlens_q.is_contiguous() && cu_seqlens_q.size(0) >= 2,
                "cu_seqlens_q must be a contiguous int32 [batch + 1] tensor");
    const int64_t batch = cu_seqlens_q.size(0) - 1;
    AITER_CHECK(prefix_lens.dtype() == AITER_DTYPE_i32 && prefix_lens.dim() == 1 &&
                    prefix_lens.is_contiguous() && prefix_lens.size(0) >= batch,
                "prefix_lens must be a contiguous int32 [batch] tensor");
    AITER_CHECK(block_table.dtype() == AITER_DTYPE_i32 && block_table.dim() == 2 &&
                    block_table.size(0) >= batch && block_table.stride(1) == 1,
                "block_table must be int32 [batch, max_blocks] with unit column stride");
    AITER_CHECK(scale > 0.f && std::isfinite(scale), "scale must be positive and finite");
    AITER_CHECK(idx_q.device_id == partial.device_id &&
                    index_cache.device_id == partial.device_id &&
                    block_table.device_id == partial.device_id &&
                    cu_seqlens_q.device_id == partial.device_id &&
                    prefix_lens.device_id == partial.device_id && partial.is_gpu(),
                "all tensors must be on the same GPU");
    const int64_t q_tiles      = (max_query_len + TQ - 1) / TQ;
    const int64_t ncta_logical = q_tiles * batch * nchunks;
    if(total_q == 0 || ncta_logical <= 0)
        return;
    AITER_CHECK(ncta_logical <= INT_MAX - 8, "grid too large");

    Params p;
    p.q             = static_cast<const uint16_t*>(idx_q.data_ptr());
    p.k             = static_cast<const uint8_t*>(index_cache.data_ptr());
    p.partial       = static_cast<uint64_t*>(partial.data_ptr());
    p.bt            = static_cast<const int*>(block_table.data_ptr());
    p.cu            = static_cast<const int*>(cu_seqlens_q.data_ptr());
    p.prefix        = static_cast<const int*>(prefix_lens.data_ptr());
    p.k_page_stride = index_cache.stride(0);
    p.q_stride_n    = static_cast<int>(idx_q.stride(0));
    p.total_q       = static_cast<int>(total_q);
    p.bt_stride     = static_cast<int>(block_table.stride(0));
    p.q_tiles       = static_cast<int>(q_tiles);
    p.batch         = static_cast<int>(batch);
    p.nchunks       = static_cast<int>(nchunks);
    p.chunk_blocks  = static_cast<int>(chunk_blocks);
    p.ncta_logical  = static_cast<int>(ncta_logical);
    p.ncta          = (p.ncta_logical + 7) / 8 * 8;
    p.scale         = scale;

    const HipDeviceGuard guard(partial.device_id);
    const hipStream_t stream = aiter::getCurrentHIPStream();
    m3_fused_index_topk::kernel<<<p.ncta, NTHR, 0, stream>>>(p);
    HIP_CALL_LAUNCH(hipGetLastError());
}

} // namespace aiter
