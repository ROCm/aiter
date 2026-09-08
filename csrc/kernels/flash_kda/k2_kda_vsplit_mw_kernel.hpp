// FlashKDA K2 (recurrence) — MULTI-WAVE V-split (occupancy + shared workspace).
//
// The single-wave V-split launches one block per (seq,head,V-group); at small N*H
// that is grid-starved (few CUs, 1 wave each -> only intra-wave MLP hides the HBM
// latency) AND every V-group block independently re-reads the SAME per-chunk
// workspace (kd/qd/kr/INV/Mqk/gtot/beta are all V-INDEPENDENT), an 8x redundant
// HBM load at BV=16.
//
// This kernel folds NW V-groups into ONE block of NW waves:
//   * the V-independent workspace is loaded ONCE per block (split across all
//     NW*64 threads) into shared LDS -> NW x less workspace HBM traffic, and
//   * NW waves co-reside on the CU -> they hide each other's load latency
//     (occupancy), the lever the profiling pointed to for the latency-bound case.
// Only Sv / vmat / Umat / the output are per-wave (per-V-group).
//
// Bit-identical math/layout/MFMA order to vsplit_db (same recurrence, same
// fragment reads). Grid (N*H, (D/BV)/NW), block NW*64. Selected via
// FLASH_KDA_K2=vsplit_mw, waves/block via FLASH_KDA_MW (default 4).
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

template <int BV, int NW>
__global__ void __launch_bounds__(NW * 64)
k2_kda_vsplit_mw_kernel(
        const __bf16* __restrict__ v_g,
        const float*  __restrict__ beta_g,
        __bf16* __restrict__ out_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float*  __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ ws_mqk,
        int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    constexpr int SD = D + 4;
    constexpr int NVT = BV / C;
    const int tid = threadIdx.x;             // 0 .. NW*64-1
    const int w = tid >> 6;                  // wave id (V-group within block)
    const int lane = tid & 63;               // lane within wave
    const int bh = blockIdx.x;
    const int b = bh / H, h = bh % H;
    const int vg = blockIdx.y * NW + w;      // this wave's V-group
    const int v0 = vg * BV;
    constexpr int NTHREADS = NW * 64;

    // ---- shared per-block (V-independent) workspace, loaded once per chunk ----
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 qd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 INV[C * C];
    __shared__ __bf16 Mqk[C * C];
    __shared__ float  gtot[D];
    __shared__ float  beta[C];
    // ---- per-wave (per-V-group) state / scratch ----
    __shared__ __bf16 Sv[NW][BV * SD];
    __shared__ __bf16 vmat[NW][C * BV];
    __shared__ __bf16 Umat[NW][C * BV];

    // load the V-independent tiles once, split across the whole block (tid).
    auto load_ws = [&](int ht) {
        // kd/qd: row-major [C,D] -> padded LDS [C,SD], dwordx4 (8 bf16) chunks.
        constexpr int WV = D >> 3;                 // 16 chunks per row
        for (int g = tid; g < C * WV; g += NTHREADS) {
            int r = g / WV, c = g % WV;
            reinterpret_cast<bf16x8*>(kd + r*SD)[c] =
                reinterpret_cast<const bf16x8*>(ws_kd + (int64_t)ht*C*D + r*D)[c];
            reinterpret_cast<bf16x8*>(qd + r*SD)[c] =
                reinterpret_cast<const bf16x8*>(ws_qd + (int64_t)ht*C*D + r*D)[c];
        }
        // kr: flat contiguous [C,D], dwordx4.
        for (int g = tid; g < (C*D) >> 3; g += NTHREADS)
            reinterpret_cast<bf16x8*>(kr)[g] =
                reinterpret_cast<const bf16x8*>(ws_kr + (int64_t)ht*C*D)[g];
        // INV/Mqk: flat contiguous [C,C], dwordx4.
        for (int g = tid; g < (C*C) >> 3; g += NTHREADS) {
            reinterpret_cast<bf16x8*>(INV)[g] =
                reinterpret_cast<const bf16x8*>(ws_inv + (int64_t)ht*C*C)[g];
            reinterpret_cast<bf16x8*>(Mqk)[g] =
                reinterpret_cast<const bf16x8*>(ws_mqk + (int64_t)ht*C*C)[g];
        }
        // gtot: [D] f32, dwordx4.
        for (int g = tid; g < D >> 2; g += NTHREADS)
            reinterpret_cast<f32x4*>(gtot)[g] =
                reinterpret_cast<const f32x4*>(ws_gt + (int64_t)ht*D)[g];
    };

    // load beta (V-independent) + this wave's vmat (per-V, tail-masked).
    auto load_vbeta = [&](int t0, int alen) {
        if (tid < C) beta[tid] = (tid < alen)
            ? sigmoid_tanh(beta_g[(t0 + tid)*H + h]) : 0.0f;
        #pragma unroll
        for (int j = 0; j < (C*BV)/64; j++) {
            int idx = lane + j*64, m = idx / BV, vloc = idx % BV;
            vmat[w][idx] = (m < alen) ? v_g[(t0 + m)*H*D + h*D + v0 + vloc] : (__bf16)0.0f;
        }
    };

    for (int idx = lane; idx < BV * SD; idx += 64) Sv[w][idx] = (__bf16)0.0f;

    for (int nt = 0; nt < NT; nt++) {
        const int ht = bh * NT + nt;
        const int t0 = b * T_seq + nt * C;
        const int alen = min(C, T_seq - nt * C);
        load_ws(ht);
        load_vbeta(t0, alen);
        __syncthreads();

        // ---- v = (v - kd @ S_kv) * beta ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 c = gemm_contract_last<__bf16, D, SD>(kd, &Sv[w][vt*C*SD], lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                float nv = (bf16_to_f32(vmat[w][m*BV + vloc]) - c[i]) * beta[m];
                vmat[w][m*BV + vloc] = f32_to_bf16(nv);
            }
        }
        __syncthreads();

        // ---- U = INV @ v ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 c;
            if constexpr (BV == C) c = mm_std_16_tr(INV, vmat[w], lane);
            else                   c = mm_std_tile_bf16(INV, vmat[w], vt*C, BV, lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                Umat[w][m*BV + vloc] = f32_to_bf16(c[i]);
            }
        }
        __syncthreads();

        // ---- out = qd @ S_kv + Mqk @ U ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 o1 = gemm_contract_last<__bf16, D, SD>(qd, &Sv[w][vt*C*SD], lane);
            f32x4 o2;
            if constexpr (BV == C) o2 = mm_std_16_tr(Mqk, Umat[w], lane);
            else                   o2 = mm_std_tile_bf16(Mqk, Umat[w], vt*C, BV, lane);
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

        // ---- state update ----
        #pragma unroll
        for (int kt = 0; kt < D / C; kt++) {
            #pragma unroll
            for (int vt = 0; vt < NVT; vt++) {
                f32x4 c;
                if constexpr (BV == C) c = mm_cf_trB(kr, D, kt*C, Umat[w], lane);
                else                   c = mm_contract_first_bf16(kr, Umat[w], kt*C, vt*C, D, BV, lane);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k = kt*C + (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                    float sv = bf16_to_f32(Sv[w][vloc*SD + k]) * ex2(gtot[k]) + c[i];
                    Sv[w][vloc*SD + k] = f32_to_bf16(sv);
                }
            }
        }
        __syncthreads();     // all waves done reading shared ws -> safe to reload
    }
}

}  // namespace flashkda_hip
