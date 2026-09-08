// gfx942 experimental V-independent BT64 Aqk producer.
//
// The direct-RTP output kernels currently rebuild the same causal
//
//   Aqk = tril(qd_segment @ ki_segment^T)
//
// in every V16 CTA.  This producer evaluates it once per (head, BT64 segment)
// and writes the ten lower 16x16 BF16 blocks in triangular order:
//
//   0:A00, 1:A10, 2:A11, 3:A20, 4:A21,
//   5:A22, 6:A30, 7:A31, 8:A32, 9:A33.
//
// The output is 5 KiB per segment and is deliberately pointer-agnostic.  The
// fused P3 experiment can place it at the beginning of that segment's cs_sin
// arena, whose existing 32 KiB stride is ample.  The legacy scan must not run
// concurrently with that alias because it writes the complete cs_sin arena.
//
// Numerical contract: this is the QK half of
// k2_kda_csplit_bt64_out_bk32_kernel, lifted out without changing its stable
// scales, explicit BF16 operand rounding, or K16 accumulation order.
#pragma once

#include <hip/hip_runtime.h>

#include "../mfma.hpp"

namespace flashkda_hip::gfx942 {

namespace k1_bt64_aqk_producer_detail {

constexpr int C = 16;
constexpr int BT = 64;
constexpr int D = 128;
constexpr int BK = 32;
constexpr int NW = 4;
constexpr int STRIDE_K = BK + 4;
constexpr int FACTOR_GROUPS = BT / C;
constexpr int TRI_TILES = FACTOR_GROUPS * (FACTOR_GROUPS + 1) / 2;
constexpr int TILE_ELEMS = C * C;

__device__ __forceinline__ constexpr int tri_tile(int r, int c) {
    return r * (r + 1) / 2 + c;
}

// Keep scale ownership identical to direct-RTP P4.  With four waves here, each
// thread evaluates two of the original 128-thread cohorts.
__device__ __forceinline__ constexpr int factor_owner(int pair) {
    return pair < 3 ? 0 :
           pair < 5 ? 1 :
           pair < 7 ? 2 :
           pair == 8 ? 2 : 3;
}

__device__ __forceinline__ int tile_row(int tile) {
    return tile < 1 ? 0 : tile < 3 ? 1 : tile < 6 ? 2 : 3;
}

__device__ __forceinline__ int tile_col(int tile) {
    const int r = tile_row(tile);
    return tile - r * (r + 1) / 2;
}

__device__ __forceinline__ bf16x4 load_fragment(
        const __bf16* __restrict__ x,
        int row_base, int col_base, int stride, int lane) {
    const int row = row_base + (lane & 15);
    const int col4 = col_base + ((lane >> 4) << 2);
    bf16x4 v;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] = x[row * stride + col4 + i];
    return v;
}

struct alignas(16) SharedStorage {
    __bf16 q_local_panel[BT * STRIDE_K];
    __bf16 kr_panel[BT * STRIDE_K];
    float a_scale[TRI_TILES * D];
};

static_assert(sizeof(SharedStorage) == 14 * 1024,
              "Aqk producer must retain four-CTA LDS residency on gfx942");

}  // namespace k1_bt64_aqk_producer_detail

constexpr int kK1Bt64AqkProducerSmemBytes =
    sizeof(k1_bt64_aqk_producer_detail::SharedStorage);

template <bool VL = false>
__global__ void __launch_bounds__(256)
k1_kda_bt64_aqk_producer_kernel(
        const __bf16* __restrict__ ws_qd,   // [tile,16,128], chunk-local
        const __bf16* __restrict__ ws_kr,   // [tile,16,128], bounded suffix K
        const float* __restrict__ ws_gt,    // [tile,128], base-2 log decay
        __bf16* __restrict__ aqk_out,       // [H*segments,10,16,16]
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT) {
    using namespace k1_bt64_aqk_producer_detail;

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;

    int h;
    int ht_base;
    int ss;
    int alen;
    int chunks;
    if constexpr (VL) {
        const int gsi = static_cast<int>(blockIdx.x);
        h = static_cast<int>(blockIdx.y);
        int lo = 0;
        int hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (segment_prefix[mid] <= gsi)
                lo = mid;
            else
                hi = mid;
        }
        const int local_seg = gsi - segment_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int seq_len = static_cast<int>(cu_seqlens[lo + 1] - bos);
        const int nseg = (seq_len + BT - 1) / BT;
        if (local_seg < 0 || local_seg >= nseg)
            return;
        ht_base = h * total_tiles + tile_prefix[lo] + local_seg * FACTOR_GROUPS;
        ss = h * total_segments + gsi;
        alen = min(BT, seq_len - local_seg * BT);
        chunks = (alen + C - 1) / C;
    } else {
        const int seg = static_cast<int>(blockIdx.x);
        const int bh = static_cast<int>(blockIdx.y);
        const int nseg = (NT + FACTOR_GROUPS - 1) / FACTOR_GROUPS;
        if (seg >= nseg)
            return;
        const int b = bh / H;
        (void)b;
        h = bh % H;
        ht_base = bh * NT + seg * FACTOR_GROUPS;
        ss = bh * nseg + seg;
        alen = min(BT, T_seq - seg * BT);
        chunks = (alen + C - 1) / C;
    }

    __shared__ SharedStorage smem;

    // Reproduce P4's A-factor construction verbatim.  Aqk consumes P4's
    // q_local_panel, not the q_scale-multiplied q_panel used by qd@H.  The
    // pair scale alone carries the stable cross-chunk relationship.  Each
    // thread owns one feature d and two original cohorts, so gt is fetched once.
    const int cohort0 = tid / D;
    const int d = tid - cohort0 * D;
    float gt[FACTOR_GROUPS];
#pragma unroll
    for (int r = 0; r < FACTOR_GROUPS; ++r) {
        gt[r] = r < chunks
            ? ws_gt[int64_t(ht_base + r) * D + d]
            : 0.0f;
    }

#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
        const int group = cohort0 + pass * 2;
#pragma unroll
        for (int r = 0; r < FACTOR_GROUPS; ++r) {
#pragma unroll
            for (int c = 0; c <= r; ++c) {
                const int pair = tri_tile(r, c);
                if (factor_owner(pair) == group) {
                    float exponent = 0.0f;
                    if (r == c) {
                        exponent = -gt[c];
                    } else {
#pragma unroll
                        for (int u = 0; u < FACTOR_GROUPS; ++u) {
                            if (u > c && u < r)
                                exponent += gt[u];
                        }
                    }
                    smem.a_scale[pair * D + d] = ex2(exponent);
                }
            }
        }
    }
    __syncthreads();

    // tile = wave + 4*slot balances the ten lower blocks as 3/3/2/2.
    constexpr int TILES_PER_WAVE = 3;
    f32x4 acc[TILES_PER_WAVE];
#pragma unroll
    for (int slot = 0; slot < TILES_PER_WAVE; ++slot)
        acc[slot] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};

    constexpr int PANEL_VECS = BT * (BK / 4);
    static_assert(PANEL_VECS == 2 * 256);

#pragma unroll
    for (int panel = 0; panel < D / BK; ++panel) {
        // Coalesced BF16x4 staging: every q/kr element is fetched once by this
        // V-independent CTA.  q remains the exact chunk-local BF16 qsrc used by
        // P4's q_local_panel; kr stays bounded until its pair scale below.
        for (int vi = tid; vi < PANEL_VECS; vi += 256) {
            const int row = vi / (BK / 4);
            const int col4 = (vi % (BK / 4)) * 4;
            const int chunk = row / C;
            const int local_row = row & (C - 1);
            const int d0 = panel * BK + col4;
            bf16x4 qv{};
            bf16x4 kv{};
            if (row < alen) {
                const int64_t off =
                    (int64_t(ht_base + chunk) * C + local_row) * D + d0;
                qv = *reinterpret_cast<const bf16x4*>(ws_qd + off);
                kv = *reinterpret_cast<const bf16x4*>(ws_kr + off);
            }
            *reinterpret_cast<bf16x4*>(
                smem.q_local_panel + row * STRIDE_K + col4) = qv;
            *reinterpret_cast<bf16x4*>(
                smem.kr_panel + row * STRIDE_K + col4) = kv;
        }
        __syncthreads();

#pragma unroll
        for (int slot = 0; slot < TILES_PER_WAVE; ++slot) {
            const int tile = wave + slot * NW;
            if (tile < TRI_TILES) {
                const int r = tile_row(tile);
                const int c = tile_col(tile);
                if (r < chunks) {
#pragma unroll
                    for (int ek = 0; ek < BK / C; ++ek) {
                        const bf16x4 af = load_fragment(
                            smem.q_local_panel, r * C, ek * C,
                            STRIDE_K, lane);
                        bf16x4 bf = load_fragment(
                            smem.kr_panel, c * C, ek * C, STRIDE_K, lane);
                        const int d0 = panel * BK + ek * C
                            + ((lane >> 4) << 2);
#pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            bf[p] = f32_to_bf16(
                                bf16_to_f32(bf[p]) *
                                smem.a_scale[tile * D + d0 + p]);
                        }
                        acc[slot] = mfma_bf16(af, bf, acc[slot]);
                    }
                }
            }
        }

        // Panels 0..2 need a reuse fence.  The last panel remains read-only
        // until all stores below and needs no trailing CTA rendezvous.
        if (panel + 1 < D / BK)
            __syncthreads();
    }

    // Store all ten tiles, explicitly zeroing absent tail blocks/rows so a
    // fixed-shape fused P3 consumer never observes stale cs_sin contents.
#pragma unroll
    for (int slot = 0; slot < TILES_PER_WAVE; ++slot) {
        const int tile = wave + slot * NW;
        if (tile < TRI_TILES) {
            const int r = tile_row(tile);
            const int c = tile_col(tile);
            const int m0 = (lane >> 4) << 2;
            const int n = lane & 15;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int m = m0 + p;
                const int global_m = r * C + m;
                const int global_n = c * C + n;
                const bool valid = r < chunks && global_m < alen &&
                    global_n < alen && (r != c || m >= n);
                aqk_out[(int64_t(ss) * TRI_TILES + tile) * TILE_ELEMS
                         + m * C + n] = valid
                    ? f32_to_bf16(acc[slot][p])
                    : (__bf16)0.0f;
            }
        }
    }
}

}  // namespace flashkda_hip::gfx942
