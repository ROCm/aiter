// gfx950-private BT32 factor merge for the plain C-split pipeline.
//
// The K128 cross contraction uses CDNA4's native K32 BF16 MFMA.  All four
// aligned input tiles stream directly from HBM to LDS, while transposed LDS
// reads make the two dependent 16x16 products conflict-free.
#pragma once

#include "mfma_gfx950.hpp"

namespace flashkda_hip::gfx950 {

template <bool VL, bool USE_X32 = true, bool USE_GLL = true,
          bool USE_TR = true>
__global__ void __launch_bounds__(64)
k1_kda_bt32_merge_kernel(
        const float* __restrict__ beta_g,
        __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ cross_inv10,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ pair_prefix,
        int N, int total_tiles, int total_pairs, int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;
    int h, ht0, xp, t1, alen1;

    if constexpr (VL) {
        const int gpi = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (pair_prefix[mid] <= gpi) lo = mid; else hi = mid;
        }
        const int local_pair = gpi - pair_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int len = int(cu_seqlens[lo + 1] - bos);
        const int np = (len + 31) / 32;
        if (local_pair >= np || local_pair * 32 + C >= len) return;
        ht0 = h * total_tiles + tile_prefix[lo] + local_pair * 2;
        xp = h * total_pairs + gpi;
        t1 = int(bos) + local_pair * 32 + C;
        alen1 = min(C, len - local_pair * 32 - C);
    } else {
        const int p = blockIdx.x, bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        if (p * 2 + 1 >= NT) return;
        ht0 = bh * NT + p * 2;
        xp = bh * ((NT + 1) / 2) + p;
        t1 = b * T_seq + p * 32 + C;
        alen1 = min(C, T_seq - p * 32 - C);
    }

    __shared__ __bf16 kd1[C * D], kr0[C * D];
    __shared__ __bf16 c00[C * C], c11[C * C];
    __shared__ __bf16 l10[C * C], tmp[C * C];
    __shared__ float beta1[C], decay0[D];

    if constexpr (USE_GLL) {
        gll_bf16_vec(kd1, ws_kd + int64_t(ht0 + 1) * C * D, C * D, lane);
        gll_bf16_vec(kr0, ws_kr + int64_t(ht0) * C * D, C * D, lane);
        gll_bf16_vec(c00, ws_inv + int64_t(ht0) * C * C, C * C, lane);
        gll_bf16_vec(c11, ws_inv + int64_t(ht0 + 1) * C * C, C * C, lane);
    } else {
        copy_bf16_vec(kd1, ws_kd + int64_t(ht0 + 1) * C * D, C * D, lane);
        copy_bf16_vec(kr0, ws_kr + int64_t(ht0) * C * D, C * D, lane);
        copy_bf16_vec(c00, ws_inv + int64_t(ht0) * C * C, C * C, lane);
        copy_bf16_vec(c11, ws_inv + int64_t(ht0 + 1) * C * C, C * C, lane);
    }
    if (lane < C)
        beta1[lane] = lane < alen1
            ? sigmoid_tanh(beta_g[int64_t(t1 + lane) * H + h]) : 0.0f;
    for (int d = lane; d < D; d += 64)
        decay0[d] = ex2(ws_gt[int64_t(ht0) * D + d]);
    __syncthreads();

    f32x4 ll;
    if constexpr (USE_X32)
        ll = contract_last_x32<D, D, D>(kd1, kr0, lane);
    else
        ll = gemm_contract_last<__bf16, D>(kd1, kr0, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        l10[m * C + n] = f32_to_bf16(ll[i] * beta1[m]);
    }
    __syncthreads();

    f32x4 x = USE_TR ? mm_std_16_tr(l10, c00, lane)
                     : mm_std_tile_bf16(l10, c00, 0, C, lane);
    store_acc_16x16(tmp, x, lane);
    __syncthreads();
    x = USE_TR ? mm_std_16_tr(c11, tmp, lane)
               : mm_std_tile_bf16(c11, tmp, 0, C, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        cross_inv10[int64_t(xp) * C * C + m * C + n] = f32_to_bf16(-x[i]);
    }
    __syncthreads();

    for (int idx = lane; idx < C * D; idx += 64) {
        const int d = idx % D;
        ws_kd[int64_t(ht0 + 1) * C * D + idx] =
            f32_to_bf16(bf16_to_f32(kd1[idx]) * decay0[d]);
    }
}

}  // namespace flashkda_hip::gfx950
