// Pair two already-prepared BT16 KDA chunks into one BT32 triangular system.
//
// The fast split K1 front-end remains BT16, but the recurrent path consumes a
// genuine 32-token factorization.  For adjacent chunks 0/1:
//
//   L10 = beta1 * kd1 @ kr0^T
//   C10 = -C11 * L10 * C00,      C = (I + L)^-1
//
// kr0 already contains the first chunk's total decay, so it is exactly the
// cross-block inverse-decay operand.  The two cross tiles are kept separately
// from the BT16 workspace and consumed by the BT32 scan/output kernels.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

template <bool VL>
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
        if (local_pair >= np) return;
        // A tail containing only the first BT16 tile has no cross block.
        if (local_pair * 32 + C >= len) return;
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

    copy_bf16_vec(kd1, ws_kd + int64_t(ht0 + 1) * C * D, C * D, lane);
    copy_bf16_vec(kr0, ws_kr + int64_t(ht0) * C * D, C * D, lane);
    copy_bf16_vec(c00, ws_inv + int64_t(ht0) * C * C, C * C, lane);
    copy_bf16_vec(c11, ws_inv + int64_t(ht0 + 1) * C * C, C * C, lane);
    if (lane < C)
        beta1[lane] = lane < alen1
            ? sigmoid_tanh(beta_g[int64_t(t1 + lane) * H + h]) : 0.0f;
    for (int d = lane; d < D; d += 64)
        decay0[d] = ex2(ws_gt[int64_t(ht0) * D + d]);
    __syncthreads();

    f32x4 ll = gemm_contract_last<__bf16, D>(kd1, kr0, lane);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        l10[m * C + n] = f32_to_bf16(ll[i] * beta1[m]);
    }
    __syncthreads();

    // BT16 builds I-L+L^2-... = (I+L)^-1, hence the negative Schur tile.
    { f32x4 x = mm_std_tile_bf16(l10, c00, 0, C, lane);
      store_acc_16x16(tmp, x, lane); }
    __syncthreads();
    { f32x4 x = mm_std_tile_bf16(c11, tmp, 0, C, lane);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
          const int m = (lane >> 4) * 4 + i, n = lane & 15;
          cross_inv10[int64_t(xp) * C * C + m * C + n] = f32_to_bf16(-x[i]);
      } }
    __syncthreads();

    // Materialize the second half's decay relative to the BT32 pair start once.
    // Eight V-split scan CTAs reuse this in-place operand, avoiding duplicated
    // elementwise reconstruction on the serial path.  K2 output never reads kd.
    for (int idx = lane; idx < C * D; idx += 64) {
        const int d = idx % D;
        ws_kd[int64_t(ht0 + 1) * C * D + idx] =
            f32_to_bf16(bf16_to_f32(kd1[idx]) * decay0[d]);
    }
}

}  // namespace flashkda_hip
