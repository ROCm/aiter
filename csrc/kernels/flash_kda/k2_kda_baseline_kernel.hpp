// FlashKDA K2 (recurrence) — baseline HIP/MFMA, CHUNK=16, D=128.
// One wavefront (64 lanes) per (sequence, head); each block walks its NT chunks
// serially, carrying the [V,K] state in LDS as bf16 (zero initial state, no
// final-state store — the M1B milestone). Consumes the six K1 workspace
// intermediates and produces `out`. Math mirrors tests/torch_ref.py per chunk.
//
// State convention (matches torch_ref work_state[seq,h]): S is logical S_vk
// with shape [V,K], row-major S[v*D + k]. The compute view S_kv = S^T, so
// S_kv[k,v] = S[v*D + k]. Both v-projection GEMMs contract over K reading S with
// row = v, i.e. "contract last dim" of an operand[C,K] against S[V,K].
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

// ht index of this block's chunk = (b*H + h)*NT + nt  (matches K1).
__global__ void __launch_bounds__(64)
k2_kda_baseline_kernel(
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
    const int lane = threadIdx.x;      // 0..63
    const int bh = blockIdx.x;         // sequence*H + head
    const int b = bh / H, h = bh % H;

    __shared__ __bf16 S[D * D];         // logical S_vk [V,K], row-major [v][k]
    __shared__ __bf16 kd[C * D];
    __shared__ __bf16 qd[C * D];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 vmat[C * D];      // v-chunk, later reused for the output
    __shared__ __bf16 Umat[C * D];
    __shared__ __bf16 INV[C * C];
    __shared__ __bf16 Mqk[C * C];
    __shared__ float  gtot[D];
    __shared__ float  beta[C];

    // ---- zero initial state ----
    for (int idx = lane; idx < D * D; idx += 64) S[idx] = (__bf16)0.0f;
    __syncthreads();

    for (int nt = 0; nt < NT; nt++) {
        const int ht = bh * NT + nt;
        const int t0 = b * T_seq + nt * C;
        const int alen = min(C, T_seq - nt * C);

        // ---- load workspace tiles + v + beta ----
        for (int idx = lane; idx < C * D; idx += 64) {
            kd[idx] = ws_kd[ht*C*D + idx];
            qd[idx] = ws_qd[ht*C*D + idx];
            kr[idx] = ws_kr[ht*C*D + idx];
            int m = idx / D, d = idx % D;
            vmat[idx] = (m < alen) ? v_g[(t0 + m)*H*D + h*D + d] : (__bf16)0.0f;
        }
        for (int idx = lane; idx < C * C; idx += 64) {
            INV[idx] = ws_inv[ht*C*C + idx];
            Mqk[idx] = ws_mqk[ht*C*C + idx];
        }
        for (int d = lane; d < D; d += 64) gtot[d] = ws_gt[ht*D + d];
        if (lane < C) beta[lane] = (lane < alen)
            ? sigmoid_tanh(beta_g[(t0 + lane)*H + h]) : 0.0f;
        __syncthreads();

        // ---- v = (v - k_decayed @ S_kv) * beta ; tmp[c,v]=sum_k kd[c,k]*S[v,k] ----
        #pragma unroll
        for (int vt = 0; vt < D / C; vt++) {
            f32x4 c = gemm_contract_last<__bf16, D>(kd, &S[vt*C*D], lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vcol = vt*C + (lane & 15);
                float nv = (bf16_to_f32(vmat[m*D + vcol]) - c[i]) * beta[m];
                vmat[m*D + vcol] = f32_to_bf16(nv);
            }
        }
        __syncthreads();

        // ---- U = INV @ v ----
        #pragma unroll
        for (int vt = 0; vt < D / C; vt++) {
            f32x4 c = mm_std_tile_bf16(INV, vmat, vt*C, D, lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vcol = vt*C + (lane & 15);
                Umat[m*D + vcol] = f32_to_bf16(c[i]);
            }
        }
        __syncthreads();

        // ---- out = q_decayed @ S_kv + Mqk @ U  (reuse vmat for the output) ----
        #pragma unroll
        for (int vt = 0; vt < D / C; vt++) {
            f32x4 o1 = gemm_contract_last<__bf16, D>(qd, &S[vt*C*D], lane);
            f32x4 o2 = mm_std_tile_bf16(Mqk, Umat, vt*C, D, lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vcol = vt*C + (lane & 15);
                __bf16 a = f32_to_bf16(o1[i]);
                __bf16 bb = f32_to_bf16(o2[i]);
                vmat[m*D + vcol] = f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(bb));
            }
        }
        __syncthreads();

        // ---- write output rows ----
        for (int idx = lane; idx < C * D; idx += 64) {
            int m = idx / D, d = idx % D;
            if (m < alen) out_g[(t0 + m)*H*D + h*D + d] = vmat[idx];
        }

        // ---- state update: delta_s[k,v]=sum_c kr[c,k]*U[c,v];
        //      S_vk[v,k] = delta_s[k,v] + S_vk[v,k]*ex2(g_total[k]) ----
        #pragma unroll
        for (int kt = 0; kt < D / C; kt++) {
            #pragma unroll
            for (int vt = 0; vt < D / C; vt++) {
                f32x4 c = mm_contract_first_bf16(kr, Umat, kt*C, vt*C, D, D, lane);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k = kt*C + (lane >> 4) * 4 + i, v = vt*C + (lane & 15);
                    float sv = bf16_to_f32(S[v*D + k]) * ex2(gtot[k]) + c[i];
                    S[v*D + k] = f32_to_bf16(sv);
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace flashkda_hip
