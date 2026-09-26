// gfx950-private BT16 triangular solve for the plain C-split pipeline.
//
// CDNA4's native K32 BF16 MFMA halves both K128 contractions.  The three
// aligned workspace tiles stream directly from HBM to LDS, and transposed
// f16 LDS reads remove the strided B-fragment loads from the polynomial solve.
#pragma once

#include "k1_kda_common.hpp"

namespace flashkda_hip::gfx950 {

template <bool VL, bool USE_X32 = true, bool USE_GLL = true,
          bool USE_TR_F16 = true>
__global__ void __launch_bounds__(64)
k1_kda_split_solve_kernel(
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ tmp_kinv,
        __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ ws_mqk,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        int N, int total_tiles, int T_seq, int H) {
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;
    int h, ht, t0, alen;
    if constexpr (VL) {
        const int gti = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (tile_prefix[mid] <= gti) lo = mid; else hi = mid;
        }
        const int local = gti - tile_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int len = int(cu_seqlens[lo + 1] - bos);
        if (local >= (len + C - 1) / C) return;
        ht = h * total_tiles + gti;
        t0 = int(bos) + local * C;
        alen = min(C, len - local * C);
    } else {
        const int nt = blockIdx.x, bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        ht = bh * gridDim.x + nt;
        t0 = b * T_seq + nt * C;
        alen = min(C, T_seq - nt * C);
    }

    __shared__ __bf16 kd[C * D], qd[C * D], ki[C * D], mqk[C * C];
    __shared__ float beta[C];
    __shared__ _Float16 lm[C * C], inv[C * C], lk[C * C];
    if constexpr (USE_GLL) {
        gll_bf16_vec(kd, ws_kd + int64_t(ht) * C * D, C * D, lane);
        gll_bf16_vec(qd, ws_qd + int64_t(ht) * C * D, C * D, lane);
        gll_bf16_vec(ki, tmp_kinv + int64_t(ht) * C * D, C * D, lane);
    } else {
        copy_bf16_vec(kd, ws_kd + int64_t(ht) * C * D, C * D, lane);
        copy_bf16_vec(qd, ws_qd + int64_t(ht) * C * D, C * D, lane);
        copy_bf16_vec(ki, tmp_kinv + int64_t(ht) * C * D, C * D, lane);
    }
    if (lane < C)
        beta[lane] = lane < alen
            ? sigmoid_tanh(beta_g[int64_t(t0 + lane) * H + h]) : 0.0f;
    __syncthreads();

    f32x4 cl;
    f32x4 cm;
    if constexpr (USE_X32) {
        cl = contract_last_x32<D, D, D>(kd, ki, lane);
        cm = contract_last_x32<D, D, D>(qd, ki, lane);
    } else {
        cl = gemm_contract_last<__bf16, D>(kd, ki, lane);
        cm = gemm_contract_last<__bf16, D>(qd, ki, lane);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int m = (lane >> 4) * 4 + i, n = lane & 15;
        lm[m * C + n] = m > n
            ? f32_to_f16(cl[i]) * f32_to_f16(beta[m]) : (_Float16)0.0f;
        mqk[m * C + n] = m >= n ? f32_to_bf16(cm[i]) : (__bf16)0.0f;
        inv[m * C + n] = (_Float16)(m == n ? 1.0f : 0.0f) - lm[m * C + n];
    }
    __syncthreads();

    auto mm = [&](const _Float16* a, const _Float16* b) {
        if constexpr (USE_TR_F16)
            return gemm_std_f16_tr(a, b, lane);
        else
            return gemm_std_f16(a, b, lane);
    };
    { f32x4 c = mm(lm, lm); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = mm(inv, lk); __syncthreads();
      for (int i=0;i<4;++i){int m=(lane>>4)*4+i,n=lane&15;inv[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();
    { f32x4 c = mm(lm, lm); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = mm(lk, lk); __syncthreads(); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = mm(inv, lk); __syncthreads();
      for (int i=0;i<4;++i){int m=(lane>>4)*4+i,n=lane&15;inv[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();
    { f32x4 c = mm(lm, lm); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = mm(lk, lk); __syncthreads(); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = mm(lk, lk); __syncthreads(); store_acc_16x16(lk, c, lane); }
    __syncthreads();
    { f32x4 c = mm(inv, lk); __syncthreads();
      for (int i=0;i<4;++i){int m=(lane>>4)*4+i,n=lane&15;inv[m*C+n]+=f32_to_f16(c[i]);} }
    __syncthreads();

    for (int idx = lane; idx < C * C; idx += 64) {
        ws_inv[int64_t(ht) * C * C + idx] = f32_to_bf16(f16_to_f32(inv[idx]));
        ws_mqk[int64_t(ht) * C * C + idx] = mqk[idx];
    }
}

}  // namespace flashkda_hip::gfx950
