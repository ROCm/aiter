// Merge two BT32 KDA factors into one BT64 lower-triangular inverse.
//
// The two diagonal BT32 factors are already represented by their three
// 16x16 lower tiles.  This pass computes the 32x32 cross block
//
//   C_BA = -C_BB * L_BA * C_AA,   C = (I + L)^-1,
//
// using four waves (one per 16x16 output tile).  It also materializes chunks
// 2/3's kd relative to the BT64 segment start; K2 output does not consume kd.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

template <bool VL>
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

    for (int d = tid; d < D; d += 256) {
        const float g0 = ws_gt[int64_t(ht0) * D + d];
        const float g1 = ws_gt[int64_t(ht0 + 1) * D + d];
        decay1[d] = ex2(g1);
        decayA[d] = ex2(g0 + g1);
    }
    for (int m = tid; m < 32; m += 256)
        betaB[m] = m + 32 < alen
            ? sigmoid_tanh(beta_g[int64_t(t2 + m) * H + h]) : 0.0f;
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
    __syncthreads();

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
    __syncthreads();

    const int rb = wave >> 1, cbk = wave & 1;
    f32x4 l = gemm_contract_last<__bf16, D>(
        kdB + rb * C * D, krA + cbk * C * D, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        lm[(wave * C + m) * C + n] =
            f32_to_bf16(l[i] * betaB[rb * C + m]);
    }
    __syncthreads();

    // tmp = L_BA * C_AA. ca tiles are [A00, A10, A11].
    f32x4 t;
    if (cbk == 0) {
        f32x4 x0 = mm_std_tile_bf16(
            lm + (rb * 2) * C * C, ca, 0, C, lane);
        f32x4 x1 = mm_std_tile_bf16(
            lm + (rb * 2 + 1) * C * C, ca + C * C, 0, C, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) t[i] = x0[i] + x1[i];
    } else {
        t = mm_std_tile_bf16(
            lm + (rb * 2 + 1) * C * C, ca + 2 * C * C, 0, C, lane);
    }
    store_acc_16x16(tmp + wave * C * C, t, lane);
    __syncthreads();

    // C_BA = -C_BB * tmp. cb tiles are [B00, B10, B11].
    f32x4 z;
    if (rb == 0) {
        z = mm_std_tile_bf16(cb, tmp + cbk * C * C, 0, C, lane);
    } else {
        f32x4 x0 = mm_std_tile_bf16(
            cb + C * C, tmp + cbk * C * C, 0, C, lane);
        f32x4 x1 = mm_std_tile_bf16(
            cb + 2 * C * C, tmp + (2 + cbk) * C * C, 0, C, lane);
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

    // kdB was pair-local; make it relative to this BT64 segment's start.
    for (int idx = tid; idx < 32 * D; idx += 256) {
        const int m = idx / D, d = idx % D, cm = m & 15;
        if (m + 32 < alen)
            ws_kd[(int64_t(ht0 + 2 + (m >> 4)) * C + cm) * D + d] =
                f32_to_bf16(bf16_to_f32(kdB[idx]) * decayA[d]);
    }
}

}  // namespace flashkda_hip
