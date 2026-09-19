// gfx950-private BT64 factor merge for the plain C-split pipeline.
//
// CDNA4's native K32 BF16 MFMA halves the K128 cross contractions.  The
// pair-local K operands and triangular factors use aligned 128-bit transfers,
// and the remaining 16x16 products use transposed LDS reads.  Keep independent
// K16 products separate so the optimized factor is bit-exact with the shared
// implementation.
#pragma once

#include "mfma_gfx950.hpp"

namespace flashkda_hip::gfx950 {

template <bool USE_TR>
__device__ __forceinline__ f32x4 merge_mm16(
        const __bf16* a, const __bf16* b, int lane) {
    if constexpr (USE_TR)
        return mm_std_16_tr(a, b, lane);
    return mm_std_tile_bf16(a, b, 0, 16, lane);
}

template <bool VL, bool USE_X32 = true, bool USE_VEC = true,
          bool USE_TR = true>
__global__ void __launch_bounds__(256)
k1_kda_bt64_merge_kernel(
        const float* __restrict__ beta_g,
        __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ cross32,
        __bf16* __restrict__ cross64,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ pair_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_pairs, int total_segments,
        int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    int h, ht0, xp0, xs, t2, alen;

    if constexpr (VL) {
        const int gsi = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (segment_prefix[mid] <= gsi) lo = mid; else hi = mid;
        }
        const int local_seg = gsi - segment_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int len = int(cu_seqlens[lo + 1] - bos);
        if (local_seg >= (len + 63) / 64) return;
        alen = min(64, len - local_seg * 64);
        if (alen <= 32) return;
        ht0 = h * total_tiles + tile_prefix[lo] + local_seg * 4;
        xp0 = h * total_pairs + pair_prefix[lo] + local_seg * 2;
        xs = h * total_segments + gsi;
        t2 = int(bos) + local_seg * 64 + 32;
    } else {
        const int seg = blockIdx.x, bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        alen = min(64, T_seq - seg * 64);
        if (alen <= 32) return;
        ht0 = bh * NT + seg * 4;
        xp0 = bh * ((NT + 1) / 2) + seg * 2;
        xs = bh * ((NT + 3) / 4) + seg;
        t2 = b * T_seq + seg * 64 + 32;
    }
    const bool has_chunk3 = alen > 48;

    __shared__ __bf16 kdB[32 * D], krA[32 * D];
    __shared__ __bf16 ca[3 * C * C], cb[3 * C * C];
    __shared__ __bf16 lm[4 * C * C], tmp[4 * C * C];
    __shared__ float betaB[32], decay1[D], decayA[D];

    if constexpr (USE_VEC) {
        if (tid < D / 4) {
            const int d0 = tid * 4;
            const f32x4 g0 = *reinterpret_cast<const f32x4*>(
                ws_gt + int64_t(ht0) * D + d0);
            const f32x4 g1 = *reinterpret_cast<const f32x4*>(
                ws_gt + int64_t(ht0 + 1) * D + d0);
            f32x4 e1, ea;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                e1[i] = ex2(g1[i]);
                ea[i] = ex2(g0[i] + g1[i]);
            }
            *reinterpret_cast<f32x4*>(decay1 + d0) = e1;
            *reinterpret_cast<f32x4*>(decayA + d0) = ea;
        }
    } else {
        for (int d = tid; d < D; d += 256) {
            const float g0 = ws_gt[int64_t(ht0) * D + d];
            const float g1 = ws_gt[int64_t(ht0 + 1) * D + d];
            decay1[d] = ex2(g1);
            decayA[d] = ex2(g0 + g1);
        }
    }
    for (int m = tid; m < 32; m += 256)
        betaB[m] = m + 32 < alen
            ? sigmoid_tanh(beta_g[int64_t(t2 + m) * H + h]) : 0.0f;

    if constexpr (USE_VEC) {
        for (int vi = tid; vi < (3 * C * C) / 8; vi += 256) {
            const int tile = vi / ((C * C) / 8);
            const int e = (vi % ((C * C) / 8)) * 8;
            bf16x8 av, bv{};
            if (tile == 0) {
                av = *reinterpret_cast<const bf16x8*>(
                    ws_inv + int64_t(ht0) * C * C + e);
                bv = *reinterpret_cast<const bf16x8*>(
                    ws_inv + int64_t(ht0 + 2) * C * C + e);
            } else if (tile == 1) {
                av = *reinterpret_cast<const bf16x8*>(
                    cross32 + int64_t(xp0) * C * C + e);
                if (has_chunk3)
                    bv = *reinterpret_cast<const bf16x8*>(
                        cross32 + int64_t(xp0 + 1) * C * C + e);
            } else {
                av = *reinterpret_cast<const bf16x8*>(
                    ws_inv + int64_t(ht0 + 1) * C * C + e);
                if (has_chunk3)
                    bv = *reinterpret_cast<const bf16x8*>(
                        ws_inv + int64_t(ht0 + 3) * C * C + e);
            }
            *reinterpret_cast<bf16x8*>(ca + tile * C * C + e) = av;
            *reinterpret_cast<bf16x8*>(cb + tile * C * C + e) = bv;
        }
    } else {
        for (int i = tid; i < 3 * C * C; i += 256) {
            const int tile = i / (C * C), e = i % (C * C);
            if (tile == 0) {
                ca[i] = ws_inv[int64_t(ht0) * C * C + e];
                cb[i] = ws_inv[int64_t(ht0 + 2) * C * C + e];
            } else if (tile == 1) {
                ca[i] = cross32[int64_t(xp0) * C * C + e];
                cb[i] = has_chunk3
                    ? cross32[int64_t(xp0 + 1) * C * C + e] : (__bf16)0.0f;
            } else {
                ca[i] = ws_inv[int64_t(ht0 + 1) * C * C + e];
                cb[i] = has_chunk3
                    ? ws_inv[int64_t(ht0 + 3) * C * C + e] : (__bf16)0.0f;
            }
        }
    }
    __syncthreads();

    if constexpr (USE_VEC) {
        for (int vi = tid; vi < (32 * D) / 8; vi += 256) {
            const int m = vi / (D / 8), d0 = (vi % (D / 8)) * 8;
            const int cm = m & 15;
            bf16x8 kv{};
            if (m + 32 < alen)
                kv = *reinterpret_cast<const bf16x8*>(
                    ws_kd + (int64_t(ht0 + 2 + (m >> 4)) * C + cm) * D + d0);
            *reinterpret_cast<bf16x8*>(kdB + m * D + d0) = kv;

            bf16x8 rv = *reinterpret_cast<const bf16x8*>(
                ws_kr + (int64_t(ht0 + (m >> 4)) * C + cm) * D + d0);
            if (m < C) {
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    rv[i] = f32_to_bf16(bf16_to_f32(rv[i]) * decay1[d0 + i]);
            }
            *reinterpret_cast<bf16x8*>(krA + m * D + d0) = rv;
        }
    } else {
        for (int idx = tid; idx < 32 * D; idx += 256) {
            const int m = idx / D, d = idx % D, cm = m & 15;
            if (m + 32 < alen)
                kdB[idx] = ws_kd[(int64_t(ht0 + 2 + (m >> 4)) * C + cm) * D + d];
            else
                kdB[idx] = (__bf16)0.0f;
            __bf16 x = ws_kr[(int64_t(ht0 + (m >> 4)) * C + cm) * D + d];
            krA[idx] = m < C
                ? f32_to_bf16(bf16_to_f32(x) * decay1[d]) : x;
        }
    }
    __syncthreads();

    const int rb = wave >> 1, cbk = wave & 1;
    f32x4 l;
    if constexpr (USE_X32)
        l = contract_last_x32<D, D, D>(
            kdB + rb * C * D, krA + cbk * C * D, lane);
    else
        l = gemm_contract_last<__bf16, D>(
            kdB + rb * C * D, krA + cbk * C * D, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        lm[(wave * C + m) * C + n] =
            f32_to_bf16(l[i] * betaB[rb * C + m]);
    }
    __syncthreads();

    f32x4 t;
    if (cbk == 0) {
        f32x4 x0 = merge_mm16<USE_TR>(lm + (rb * 2) * C * C, ca, lane);
        f32x4 x1 = merge_mm16<USE_TR>(
            lm + (rb * 2 + 1) * C * C, ca + C * C, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) t[i] = x0[i] + x1[i];
    } else {
        t = merge_mm16<USE_TR>(
            lm + (rb * 2 + 1) * C * C, ca + 2 * C * C, lane);
    }
    store_acc_16x16(tmp + wave * C * C, t, lane);
    __syncthreads();

    f32x4 z;
    if (rb == 0) {
        z = merge_mm16<USE_TR>(cb, tmp + cbk * C * C, lane);
    } else {
        f32x4 x0 = merge_mm16<USE_TR>(cb + C * C, tmp + cbk * C * C, lane);
        f32x4 x1 = merge_mm16<USE_TR>(
            cb + 2 * C * C, tmp + (2 + cbk) * C * C, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) z[i] = x0[i] + x1[i];
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        cross64[(int64_t(xs) * 4 + wave) * C * C + m * C + n] =
            f32_to_bf16(-z[i]);
    }
    __syncthreads();

    if constexpr (USE_VEC) {
        for (int vi = tid; vi < (32 * D) / 8; vi += 256) {
            const int m = vi / (D / 8), d0 = (vi % (D / 8)) * 8;
            if (m + 32 < alen) {
                bf16x8 x = *reinterpret_cast<const bf16x8*>(kdB + m * D + d0);
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    x[i] = f32_to_bf16(bf16_to_f32(x[i]) * decayA[d0 + i]);
                const int cm = m & 15;
                *reinterpret_cast<bf16x8*>(
                    ws_kd + (int64_t(ht0 + 2 + (m >> 4)) * C + cm) * D + d0) = x;
            }
        }
    } else {
        for (int idx = tid; idx < 32 * D; idx += 256) {
            const int m = idx / D, d = idx % D, cm = m & 15;
            if (m + 32 < alen)
                ws_kd[(int64_t(ht0 + 2 + (m >> 4)) * C + cm) * D + d] =
                    f32_to_bf16(bf16_to_f32(kdB[idx]) * decayA[d]);
        }
    }
}

}  // namespace flashkda_hip::gfx950
