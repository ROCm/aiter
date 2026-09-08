// FlashKDA K2 (recurrence) — V-split HIP/MFMA, CHUNK=16, D=128.
// The recurrence is independent across the V (value) dimension: U=INV@v, the two
// output GEMMs, and the K-dim state decay all act column-wise in V. So we split
// V into groups of BV and give each group its own block, multiplying grid
// parallelism by D/BV over the baseline (which used one block per (seq,head) and
// starved the GPU). Each block owns state rows S[v0:v0+BV, :K] (bf16 LDS here;
// the fp32 register-resident variant is M2b). Grid (N*H, D/BV), block 64.
//
// State convention matches the baseline: logical S_vk[V,K], row-major S[v*K + k],
// so S_kv[k,v]=S[v,k]. This block holds only its BV V-rows: Sv[vloc*K + k].
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

template <int BV>
__global__ void __launch_bounds__(64)
k2_kda_vsplit_kernel(
        const __bf16* __restrict__ v_g,     // [T_total, H, D]
        const float*  __restrict__ beta_g,  // [T_total, H]
        __bf16* __restrict__ out_g,         // [T_total, H, D]
        const __bf16* __restrict__ ws_kd,   // [n_ht, 16, 128]
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float*  __restrict__ ws_gt,   // [n_ht, 128]
        const __bf16* __restrict__ ws_inv,  // [n_ht, 16, 16]
        const __bf16* __restrict__ ws_mqk,
        int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    constexpr int SD = D + 4;                 // padded LDS row pitch: 16 rows -> 16
                                              // distinct banks (132/2 %32 = 2), 8B aligned.
    constexpr int NVT = BV / C;              // V-subtiles owned by this block
    const int lane = threadIdx.x;            // 0..63
    const int bh = blockIdx.x;               // sequence*H + head
    const int vgrp = blockIdx.y;             // which BV-group of V
    const int b = bh / H, h = bh % H;
    const int v0 = vgrp * BV;                // first V column this block owns

    __shared__ __bf16 Sv[BV * SD];           // this block's state rows [BV, K], padded
    __shared__ __bf16 kd[C * SD];            // padded (read row-wise in contract_last)
    __shared__ __bf16 qd[C * SD];            // padded
    __shared__ __bf16 kr[C * D];             // unpadded (only read in contract_first)
    __shared__ __bf16 vmat[C * BV];          // v slice, then reused for U-input
    __shared__ __bf16 Umat[C * BV];
    __shared__ __bf16 INV[C * C];
    __shared__ __bf16 Mqk[C * C];
    __shared__ float  gtot[D];
    __shared__ float  beta[C];

    for (int idx = lane; idx < BV * SD; idx += 64) Sv[idx] = (__bf16)0.0f;
    __syncthreads();

    for (int nt = 0; nt < NT; nt++) {
        const int ht = bh * NT + nt;
        const int t0 = b * T_seq + nt * C;
        const int alen = min(C, T_seq - nt * C);

        // ---- load shared workspace tiles (re-read per V-group) + v slice + beta ----
        copy_bf16_rows(kd, SD, ws_kd + (int64_t)ht*C*D, D, C, D, lane);
        copy_bf16_rows(qd, SD, ws_qd + (int64_t)ht*C*D, D, C, D, lane);
        copy_bf16_vec(kr, ws_kr + (int64_t)ht*C*D, C * D, lane);
        for (int idx = lane; idx < C * BV; idx += 64) {   // v slice: strided gather + tail mask
            int m = idx / BV, vloc = idx % BV;
            vmat[idx] = (m < alen) ? v_g[(t0 + m)*H*D + h*D + v0 + vloc] : (__bf16)0.0f;
        }
        copy_bf16_vec(INV, ws_inv + (int64_t)ht*C*C, C * C, lane);
        copy_bf16_vec(Mqk, ws_mqk + (int64_t)ht*C*C, C * C, lane);
        copy_f32_vec(gtot, ws_gt + (int64_t)ht*D, D, lane);
        if (lane < C) beta[lane] = (lane < alen)
            ? sigmoid_tanh(beta_g[(t0 + lane)*H + h]) : 0.0f;
        __syncthreads();

        // ---- v = (v - kd @ S_kv) * beta ;  tmp[c,vloc]=sum_k kd[c,k]*Sv[vloc,k] ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 c = gemm_contract_last<__bf16, D, SD>(kd, &Sv[vt*C*SD], lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                float nv = (bf16_to_f32(vmat[m*BV + vloc]) - c[i]) * beta[m];
                vmat[m*BV + vloc] = f32_to_bf16(nv);
            }
        }
        __syncthreads();

        // ---- U = INV @ v ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 c;
            if constexpr (BV == C) c = mm_std_16_tr(INV, vmat, lane);          // tr-read B
            else                   c = mm_std_tile_bf16(INV, vmat, vt*C, BV, lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                Umat[m*BV + vloc] = f32_to_bf16(c[i]);
            }
        }
        __syncthreads();

        // ---- out = qd @ S_kv + Mqk @ U  (write straight to global) ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 o1 = gemm_contract_last<__bf16, D, SD>(qd, &Sv[vt*C*SD], lane);
            f32x4 o2;
            if constexpr (BV == C) o2 = mm_std_16_tr(Mqk, Umat, lane);         // tr-read B
            else                   o2 = mm_std_tile_bf16(Mqk, Umat, vt*C, BV, lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                if (m < alen) {
                    __bf16 a = f32_to_bf16(o1[i]);
                    __bf16 bb = f32_to_bf16(o2[i]);
                    out_g[(t0 + m)*H*D + h*D + v0 + vloc] =
                        f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(bb));
                }
            }
        }

        // ---- state update: delta_s[k,vloc]=sum_c kr[c,k]*U[c,vloc];
        //      Sv[vloc,k] = delta_s[k,vloc] + Sv[vloc,k]*ex2(g_total[k]) ----
        #pragma unroll
        for (int kt = 0; kt < D / C; kt++) {
            #pragma unroll
            for (int vt = 0; vt < NVT; vt++) {
                f32x4 c;
                if constexpr (BV == C) c = mm_cf_trB(kr, D, kt*C, Umat, lane);  // tr-read B(U)
                else                   c = mm_contract_first_bf16(kr, Umat, kt*C, vt*C, D, BV, lane);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k = kt*C + (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                    float sv = bf16_to_f32(Sv[vloc*SD + k]) * ex2(gtot[k]) + c[i];
                    Sv[vloc*SD + k] = f32_to_bf16(sv);
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace flashkda_hip
