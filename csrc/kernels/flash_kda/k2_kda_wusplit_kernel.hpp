// FlashKDA K2 (recurrence) — WU + output-split HIP/MFMA, CHUNK=16, D=128.
//
// Ports aiter-GDN's "transport C across K1/K2" idea to KDA. FlashKDA already
// transports the intra-chunk inverse INV=(I-L)^-1 (GDN's "C") from K1 to K2 and
// applies it there as U=INV@v. This variant goes one step further, splitting K2
// into three passes so the serial state carry is as short as possible:
//
//   1. wu_prep  (chunk-parallel, grid (NT, N*H)): form the WU factors
//         u_bar = INV @ (beta * v)     [C,V]
//         w_bar = INV @ (beta * kd)    [C,K]
//      These fold the INV application off the serial path (GDN's w_bar/u_bar).
//   2. wu_carry (serial per (seq,head), V-split): the ONLY serial pass. Per
//      chunk it does just the state carry —
//         v_new = u_bar - w_bar @ S_kv          (1 GEMM, contract K)
//         snapshot incoming state  S_in[chunk]  (for the parallel output pass)
//         S_kv  = diag(decay) @ S_kv + kr^T @ v_new
//      Carry critical chain is 2 GEMMs/chunk (w_bar@S -> kr^T@v_new) vs the
//      baseline's 3 (kd@S -> INV@v -> kr^T@U); the two output GEMMs are gone.
//   3. wu_out   (fully chunk-parallel, grid (N*H, NT, D/BV)): recompute v_new
//      from the snapshot and emit
//         out = q_decayed @ S_in + Mqk @ v_new
//      This fills the GPU with the output work that used to sit on the serial
//      path, which is the whole point for grid-starved small-N*H shapes.
//
// State convention matches the baseline/vsplit kernels: logical S_vk[V,K],
// row-major S[v*D + k], compute view S_kv[k,v] = S[v*D + k]. Numerics track the
// baseline (all fp32 MFMA accumulate, bf16 round between steps) so the output
// rel_err stays ~3e-3 vs the portable reference. Behind FLASH_KDA_K2=wusplit.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

// ---- pass 1: WU factors (chunk-parallel prepass) --------------------------
// grid (NT, N*H), block 64. ht = bh*NT + nt, matching K1.
__global__ void __launch_bounds__(64)
k2_wu_prep_kernel(
        const __bf16* __restrict__ v_g,     // [T_total, H, D]
        const float*  __restrict__ beta_g,  // [T_total, H]
        const __bf16* __restrict__ ws_kd,   // [n_ht, C, D]
        const __bf16* __restrict__ ws_inv,  // [n_ht, C, C]
        __bf16* __restrict__ wu_ubar,       // [n_ht, C, D]  out
        __bf16* __restrict__ wu_wbar,       // [n_ht, C, D]  out
        int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;
    const int nt = blockIdx.x;
    const int bh = blockIdx.y;                 // sequence*H + head
    const int b = bh / H, h = bh % H;
    const int ht = bh * NT + nt;
    const int t0 = b * T_seq + nt * C;
    const int alen = min(C, T_seq - nt * C);

    __shared__ __bf16 INV[C * C];
    __shared__ __bf16 bkd[C * D];              // beta * k_decayed
    __shared__ __bf16 bv[C * D];               // beta * v
    __shared__ float  beta[C];

    if (lane < C) beta[lane] = (lane < alen)
        ? sigmoid_tanh(beta_g[(t0 + lane)*H + h]) : 0.0f;
    copy_bf16_vec(INV, ws_inv + (int64_t)ht*C*C, C * C, lane);
    __syncthreads();

    for (int idx = lane; idx < C * D; idx += 64) {
        int m = idx / D, d = idx % D;
        bkd[idx] = f32_to_bf16(bf16_to_f32(ws_kd[ht*C*D + idx]) * beta[m]);
        bv[idx]  = (m < alen)
            ? f32_to_bf16(bf16_to_f32(v_g[(t0 + m)*H*D + h*D + d]) * beta[m])
            : (__bf16)0.0f;
    }
    __syncthreads();

    // u_bar = INV @ bv, w_bar = INV @ bkd  (each [C,D], 8 K=16 output tiles)
    #pragma unroll
    for (int vt = 0; vt < D / C; vt++) {
        f32x4 cu = mm_std_tile_bf16(INV, bv,  vt*C, D, lane);
        f32x4 cw = mm_std_tile_bf16(INV, bkd, vt*C, D, lane);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int m = (lane >> 4) * 4 + i, col = vt*C + (lane & 15);
            wu_ubar[ht*C*D + m*D + col] = f32_to_bf16(cu[i]);
            wu_wbar[ht*C*D + m*D + col] = f32_to_bf16(cw[i]);
        }
    }
}

// ---- pass 2: serial state carry (V-split) ---------------------------------
// grid (N*H, D/BV), block 64. Each block owns state rows S[v0:v0+BV, :K].
template <int BV>
__global__ void __launch_bounds__(64)
k2_wu_carry_kernel(
        const __bf16* __restrict__ wu_ubar, // [n_ht, C, D]
        const __bf16* __restrict__ wu_wbar, // [n_ht, C, D]
        const __bf16* __restrict__ ws_kr,   // [n_ht, C, D]
        const float*  __restrict__ ws_gt,   // [n_ht, D]
        __bf16* __restrict__ wu_sin,        // [n_ht, V, K]  out (S_vk snapshot)
        int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    constexpr int SD = D + 4;                   // padded LDS row pitch (bank-conflict-free)
    constexpr int NVT = BV / C;
    const int lane = threadIdx.x;
    const int bh = blockIdx.x;
    const int vgrp = blockIdx.y;
    const int v0 = vgrp * BV;

    __shared__ __bf16 Sv[BV * SD];             // state rows [BV, K], padded
    __shared__ __bf16 wbar[C * SD];            // padded (read row-wise in contract_last)
    __shared__ __bf16 ubar[C * BV];
    __shared__ __bf16 vnew[C * BV];
    __shared__ __bf16 kr[C * D];               // unpadded (contract_first only)
    __shared__ float  gtot[D];

    for (int idx = lane; idx < BV * SD; idx += 64) Sv[idx] = (__bf16)0.0f;
    __syncthreads();

    for (int nt = 0; nt < NT; nt++) {
        const int ht = bh * NT + nt;

        copy_bf16_rows(wbar, SD, wu_wbar + (int64_t)ht*C*D, D, C, D, lane);
        copy_bf16_vec(kr,   ws_kr   + (int64_t)ht*C*D, C * D, lane);
        for (int idx = lane; idx < C * BV; idx += 64) {   // ubar: strided gather
            int m = idx / BV, vloc = idx % BV;
            ubar[idx] = wu_ubar[ht*C*D + m*D + v0 + vloc];
        }
        copy_f32_vec(gtot, ws_gt + (int64_t)ht*D, D, lane);
        __syncthreads();

        // v_new = u_bar - w_bar @ S_kv ; (w_bar@S)[c,vloc]=sum_k wbar[c,k]*Sv[vloc,k]
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 c = gemm_contract_last<__bf16, D, SD>(wbar, &Sv[vt*C*SD], lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                vnew[m*BV + vloc] = f32_to_bf16(bf16_to_f32(ubar[m*BV + vloc]) - c[i]);
            }
        }
        __syncthreads();

        // snapshot incoming state (pre-update) for the parallel output pass (Sv padded)
        for (int idx = lane; idx < BV * D; idx += 64) {
            int vloc = idx / D, k = idx % D;
            wu_sin[(ht*D + v0 + vloc)*D + k] = Sv[vloc*SD + k];
        }
        __syncthreads();

        // state: Sv[vloc,k] = decay[k]*Sv[vloc,k] + sum_c kr[c,k]*vnew[c,vloc]
        #pragma unroll
        for (int kt = 0; kt < D / C; kt++) {
            #pragma unroll
            for (int vt = 0; vt < NVT; vt++) {
                f32x4 c = mm_contract_first_bf16(kr, vnew, kt*C, vt*C, D, BV, lane);
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

// ---- pass 3: parallel output ----------------------------------------------
// grid (N*H, NT, D/BV), block 64. Fully independent across all chunks.
template <int BV>
__global__ void __launch_bounds__(64)
k2_wu_out_kernel(
        const __bf16* __restrict__ wu_ubar, // [n_ht, C, D]
        const __bf16* __restrict__ wu_wbar, // [n_ht, C, D]
        const __bf16* __restrict__ wu_sin,  // [n_ht, V, K]
        const __bf16* __restrict__ ws_qd,   // [n_ht, C, D]
        const __bf16* __restrict__ ws_mqk,  // [n_ht, C, C]
        __bf16* __restrict__ out_g,         // [T_total, H, D]
        int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    constexpr int SD = D + 4;                   // padded LDS row pitch (bank-conflict-free)
    constexpr int NVT = BV / C;
    const int lane = threadIdx.x;
    const int bh = blockIdx.x;                 // sequence*H + head
    const int nt = blockIdx.y;
    const int vgrp = blockIdx.z;
    const int b = bh / H, h = bh % H;
    const int v0 = vgrp * BV;
    const int ht = bh * NT + nt;
    const int t0 = b * T_seq + nt * C;
    const int alen = min(C, T_seq - nt * C);

    __shared__ __bf16 Sin[BV * SD];            // padded
    __shared__ __bf16 wbar[C * SD];            // padded
    __shared__ __bf16 qd[C * SD];              // padded
    __shared__ __bf16 ubar[C * BV];
    __shared__ __bf16 vnew[C * BV];
    __shared__ __bf16 Mqk[C * C];

    // Sin rows are contiguous in HBM (stride D); load into padded LDS (stride SD).
    copy_bf16_rows(Sin,  SD, wu_sin  + (int64_t)(ht*D + v0)*D, D, BV, D, lane);
    copy_bf16_rows(wbar, SD, wu_wbar + (int64_t)ht*C*D,        D, C,  D, lane);
    copy_bf16_rows(qd,   SD, ws_qd   + (int64_t)ht*C*D,        D, C,  D, lane);
    copy_bf16_vec(Mqk,  ws_mqk  + (int64_t)ht*C*C,        C * C,  lane);
    for (int idx = lane; idx < C * BV; idx += 64) {   // ubar: strided gather, scalar
        int m = idx / BV, vloc = idx % BV;
        ubar[idx] = wu_ubar[ht*C*D + m*D + v0 + vloc];
    }
    __syncthreads();

    // v_new = u_bar - w_bar @ Sin  (recompute; parallel, off the serial path)
    #pragma unroll
    for (int vt = 0; vt < NVT; vt++) {
        f32x4 c = gemm_contract_last<__bf16, D, SD>(wbar, &Sin[vt*C*SD], lane);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
            vnew[m*BV + vloc] = f32_to_bf16(bf16_to_f32(ubar[m*BV + vloc]) - c[i]);
        }
    }
    __syncthreads();

    // out = q_decayed @ Sin + Mqk @ v_new
    #pragma unroll
    for (int vt = 0; vt < NVT; vt++) {
        f32x4 o1 = gemm_contract_last<__bf16, D, SD>(qd, &Sin[vt*C*SD], lane);
        f32x4 o2 = mm_std_tile_bf16(Mqk, vnew, vt*C, BV, lane);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
            if (m < alen) {
                __bf16 a  = f32_to_bf16(o1[i]);
                __bf16 bb = f32_to_bf16(o2[i]);
                out_g[(t0 + m)*H*D + h*D + v0 + vloc] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(bb));
            }
        }
    }
}

}  // namespace flashkda_hip
