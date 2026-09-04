// FlashKDA K2 (recurrence) — SPLIT-SCAN over the time/chunk dimension.
//
// Motivation: the baseline / V-split kernels use one block per (seq,head) whose
// NT-chunk loop is a serial dependency chain. For small N*H the grid is far under
// 128 CUs and latency-bound (measured ~14x headroom at N*H=4). Splitting the time
// dimension into G segments manufactures N*H*G blocks to fill the GPU.
//
// Enabling algebra (validated in tests/proto_splitscan.py, bit-exact): within a
// chunk the state S_kv[K,V] enters every op LINEARLY on the left, so the chunk
// transition is a [K,K] matrix:
//     S_new_kv = M @ S_kv + dS0,   M = diag(decay) - kr^T @ INV @ diag(beta) @ kd
// Transitions compose, so a segmented scan is exact:
//   1. mseg : per segment, M_seg = M_{L-1}@...@M_0            (phase 1a)
//   2. sloc : per segment, Sloc  = state from ZERO over its chunks (phase 1b)
//   3. scan : serial over G segments, Sin[g+1] = M_seg[g]@Sin[g] + Sloc[g]
//   4. apply: rerun the baseline recurrence seeded with Sin[g] -> outputs
// Output numerics equal the baseline (phase 4 IS the baseline recurrence); only
// the cross-segment state carry uses the matrix form.
//
// State layouts (all row-major, D=128): baseline S is S_vk[v*D+k]. Scratch here
// uses S_kv[k*D+v] for Sin/Sloc/M_seg to keep the scan a plain [K,K]@[K,V].
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

// ---- Phase 1a: per-segment transition matrix M_seg [K,K] (row k1, col k2).
// grid (N*H, nseg_max), block 64. Segments past NT early-return.
__global__ void __launch_bounds__(64)
k2_ss_mseg_kernel(
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float*  __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        const float*  __restrict__ beta_g,   // [T_total, H]
        __bf16* __restrict__ ss_mseg,        // [N*H, nseg, D, D]
        int T_seq, int H, int NT, int L, int nseg) {
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;
    const int bh = blockIdx.x, seg = blockIdx.y;
    const int b = bh / H, h = bh % H;
    const int c0 = seg * L;
    if (c0 >= NT) return;
    const int c1 = min(NT, c0 + L);

    __shared__ __bf16 P[D * D];      // running product, row k1, col k2
    __shared__ __bf16 kd[C * D];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 INV[C * C];
    __shared__ __bf16 Yb[C * D];     // Y=kd@P then scaled by beta (Z)
    __shared__ __bf16 Qb[C * D];     // Q=INV@Z
    __shared__ float  gtot[D];
    __shared__ float  beta[C];

    for (int idx = lane; idx < D * D; idx += 64) {
        int r = idx / D, c = idx % D;
        P[idx] = (__bf16)(r == c ? 1.0f : 0.0f);
    }
    __syncthreads();

    for (int nt = c0; nt < c1; nt++) {
        const int ht = bh * NT + nt;
        const int t0 = b * T_seq + nt * C;
        const int alen = min(C, T_seq - nt * C);
        for (int idx = lane; idx < C * D; idx += 64) {
            kd[idx] = ws_kd[ht*C*D + idx];
            kr[idx] = ws_kr[ht*C*D + idx];
        }
        for (int idx = lane; idx < C * C; idx += 64) INV[idx] = ws_inv[ht*C*C + idx];
        for (int d = lane; d < D; d += 64) gtot[d] = ws_gt[ht*D + d];
        if (lane < C) beta[lane] = (lane < alen)
            ? sigmoid_tanh(beta_g[(t0 + lane)*H + h]) : 0.0f;
        __syncthreads();

        // Y = kd @ P ([C,K]@[K,K]); Z = beta ⊙ Y  (store in Yb).
        #pragma unroll
        for (int vt = 0; vt < D / C; vt++) {
            f32x4 c = mm_std_bigK_bf16(kd, P, D, vt*C, D, lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, col = vt*C + (lane & 15);
                Yb[m*D + col] = f32_to_bf16(c[i] * beta[m]);
            }
        }
        __syncthreads();
        // Q = INV @ Z ([C,C]@[C,K]).
        #pragma unroll
        for (int vt = 0; vt < D / C; vt++) {
            f32x4 c = mm_std_tile_bf16(INV, Yb, vt*C, D, lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, col = vt*C + (lane & 15);
                Qb[m*D + col] = f32_to_bf16(c[i]);
            }
        }
        __syncthreads();
        // P <- diag(decay) @ P - kr^T @ Q   (in place; Y,Q already captured P).
        #pragma unroll
        for (int kt = 0; kt < D / C; kt++) {
            #pragma unroll
            for (int vt = 0; vt < D / C; vt++) {
                f32x4 c = mm_contract_first_bf16(kr, Qb, kt*C, vt*C, D, D, lane);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k1 = kt*C + (lane >> 4) * 4 + i, k2 = vt*C + (lane & 15);
                    float pv = bf16_to_f32(P[k1*D + k2]) * ex2(gtot[k1]) - c[i];
                    P[k1*D + k2] = f32_to_bf16(pv);
                }
            }
        }
        __syncthreads();
    }

    __bf16* dst = ss_mseg + (int64_t)(bh * nseg + seg) * D * D;
    for (int idx = lane; idx < D * D; idx += 64) dst[idx] = P[idx];
}

// ---- Phase 1b: per-segment local state Sloc [K,V] from ZERO (baseline state).
// grid (N*H, nseg_max), block 64. Stored transposed as S_kv[k*D+v].
__global__ void __launch_bounds__(64)
k2_ss_sloc_kernel(
        const __bf16* __restrict__ v_g,      // [T_total, H, D]
        const float*  __restrict__ beta_g,   // [T_total, H]
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float*  __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ ss_sloc,        // [N*H, nseg, D, D]  (S_kv[k*D+v])
        int T_seq, int H, int NT, int L, int nseg) {
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;
    const int bh = blockIdx.x, seg = blockIdx.y;
    const int b = bh / H, h = bh % H;
    const int c0 = seg * L;
    if (c0 >= NT) return;
    const int c1 = min(NT, c0 + L);

    __shared__ __bf16 S[D * D];      // S_vk[v*D+k]
    __shared__ __bf16 kd[C * D];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 vmat[C * D];
    __shared__ __bf16 Umat[C * D];
    __shared__ __bf16 INV[C * C];
    __shared__ float  gtot[D];
    __shared__ float  beta[C];

    for (int idx = lane; idx < D * D; idx += 64) S[idx] = (__bf16)0.0f;
    __syncthreads();

    for (int nt = c0; nt < c1; nt++) {
        const int ht = bh * NT + nt;
        const int t0 = b * T_seq + nt * C;
        const int alen = min(C, T_seq - nt * C);
        for (int idx = lane; idx < C * D; idx += 64) {
            kd[idx] = ws_kd[ht*C*D + idx];
            kr[idx] = ws_kr[ht*C*D + idx];
            int m = idx / D, d = idx % D;
            vmat[idx] = (m < alen) ? v_g[(t0 + m)*H*D + h*D + d] : (__bf16)0.0f;
        }
        for (int idx = lane; idx < C * C; idx += 64) INV[idx] = ws_inv[ht*C*C + idx];
        for (int d = lane; d < D; d += 64) gtot[d] = ws_gt[ht*D + d];
        if (lane < C) beta[lane] = (lane < alen)
            ? sigmoid_tanh(beta_g[(t0 + lane)*H + h]) : 0.0f;
        __syncthreads();

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

    __bf16* dst = ss_sloc + (int64_t)(bh * nseg + seg) * D * D;
    for (int idx = lane; idx < D * D; idx += 64) {
        int v = idx / D, k = idx % D;      // reading S as [v][k]
        dst[k*D + v] = S[v*D + k];
    }
}

// ---- Phase 2: serial scan over segments. grid (N*H), block 64.
// ss_sin[g] = incoming state to segment g (S_kv[k*D+v]); ss_sin[0]=0.
__global__ void __launch_bounds__(64)
k2_ss_scan_kernel(
        const __bf16* __restrict__ ss_mseg,  // [N*H, nseg, D, D]
        const __bf16* __restrict__ ss_sloc,
        __bf16* __restrict__ ss_sin,         // [N*H, nseg, D, D]
        int NT, int L, int nseg) {
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;
    const int bh = blockIdx.x;
    const int nseg_act = min(nseg, (NT + L - 1) / L);

    __shared__ __bf16 Scur[D * D];   // Sin[g], S_kv[k*D+v]
    __shared__ __bf16 Mband[C * D];

    for (int idx = lane; idx < D * D; idx += 64) Scur[idx] = (__bf16)0.0f;
    __syncthreads();

    // ss_sin[0] = 0
    __bf16* sin0 = ss_sin + (int64_t)(bh * nseg) * D * D;
    for (int idx = lane; idx < D * D; idx += 64) sin0[idx] = (__bf16)0.0f;
    __syncthreads();

    for (int g = 0; g + 1 < nseg_act; g++) {
        const __bf16* M_g  = ss_mseg + (int64_t)(bh * nseg + g) * D * D;
        const __bf16* Sl_g = ss_sloc + (int64_t)(bh * nseg + g) * D * D;
        __bf16* next = ss_sin + (int64_t)(bh * nseg + g + 1) * D * D;
        // next[k1,v] = sum_k2 M_g[k1,k2]*Scur[k2,v] + Sloc[k1,v], written to HBM.
        #pragma unroll
        for (int kt = 0; kt < D / C; kt++) {
            for (int idx = lane; idx < C * D; idx += 64) Mband[idx] = M_g[kt*C*D + idx];
            __syncthreads();
            #pragma unroll
            for (int vt = 0; vt < D / C; vt++) {
                f32x4 c = mm_std_bigK_bf16(Mband, Scur, D, vt*C, D, lane);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k1 = kt*C + (lane >> 4) * 4 + i, v = vt*C + (lane & 15);
                    float sv = c[i] + bf16_to_f32(Sl_g[k1*D + v]);
                    next[k1*D + v] = f32_to_bf16(sv);
                }
            }
            __syncthreads();
        }
        // reload Scur <- next (own block's global writes; fence for visibility)
        __threadfence();
        __syncthreads();
        for (int idx = lane; idx < D * D; idx += 64) Scur[idx] = next[idx];
        __syncthreads();
    }
}

// ---- Phase 4: rerun baseline recurrence seeded with Sin[g] -> outputs.
// grid (N*H, nseg_max), block 64.
__global__ void __launch_bounds__(64)
k2_ss_apply_kernel(
        const __bf16* __restrict__ v_g,      // [T_total, H, D]
        const float*  __restrict__ beta_g,   // [T_total, H]
        __bf16* __restrict__ out_g,          // [T_total, H, D]
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float*  __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ ws_mqk,
        const __bf16* __restrict__ ss_sin,   // [N*H, nseg, D, D] (S_kv[k*D+v])
        int T_seq, int H, int NT, int L, int nseg) {
    constexpr int C = 16, D = 128;
    const int lane = threadIdx.x;
    const int bh = blockIdx.x, seg = blockIdx.y;
    const int b = bh / H, h = bh % H;
    const int c0 = seg * L;
    if (c0 >= NT) return;
    const int c1 = min(NT, c0 + L);

    __shared__ __bf16 S[D * D];         // S_vk[v*D+k]
    __shared__ __bf16 kd[C * D];
    __shared__ __bf16 qd[C * D];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 vmat[C * D];
    __shared__ __bf16 Umat[C * D];
    __shared__ __bf16 INV[C * C];
    __shared__ __bf16 Mqk[C * C];
    __shared__ float  gtot[D];
    __shared__ float  beta[C];

    // initial state S_vk[v,k] = Sin[k,v]
    const __bf16* sin = ss_sin + (int64_t)(bh * nseg + seg) * D * D;
    for (int idx = lane; idx < D * D; idx += 64) {
        int v = idx / D, k = idx % D;
        S[v*D + k] = sin[k*D + v];
    }
    __syncthreads();

    for (int nt = c0; nt < c1; nt++) {
        const int ht = bh * NT + nt;
        const int t0 = b * T_seq + nt * C;
        const int alen = min(C, T_seq - nt * C);
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
        for (int idx = lane; idx < C * D; idx += 64) {
            int m = idx / D, d = idx % D;
            if (m < alen) out_g[(t0 + m)*H*D + h*D + d] = vmat[idx];
        }
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
