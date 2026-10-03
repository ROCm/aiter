// FlashKDA K2 (recurrence) — V-split with direct global->LDS (`global_load_lds`)
// double-buffered prefetch. Step E: same math/layout/MFMA order as
// k2_kda_vsplit_db_kernel<BV>, but the per-chunk workspace tiles stream straight
// from HBM into a SECOND LDS buffer via `global_load_lds` (AMD's cp.async) while
// the current chunk computes out of the first buffer. Versus vsplit_db this
//   (1) removes the global->reg->LDS round trip for the six big tiles (no staging
//       VGPRs, no ds_write commit) -> lower VGPR pressure / higher occupancy, and
//   (2) removes the commit's `s_waitcnt` from the critical path (the confirmed
//       largest single stall in vsplit_db's ATT: the commit waiting on vmcnt).
// Only vmat/beta keep the register path (they need the v tail-zero mask and the
// beta sigmoid, which a raw DMA cannot do); their commit is tiny.
//
// Bit-identical to vsplit_db (same values, same fragment reads, same MFMA order);
// only the load MECHANISM and buffering change. Selected via FLASH_KDA_K2=vsplit_lds.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

template <int BV>
__global__ void __launch_bounds__(64)
k2_kda_vsplit_lds_kernel(
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
    constexpr int SD = D + 4;                 // padded LDS row pitch (see vsplit)
    constexpr int NVT = BV / C;
    const int lane = threadIdx.x;
    const int bh = blockIdx.x;
    const int vgrp = blockIdx.y;
    const int b = bh / H, h = bh % H;
    const int v0 = vgrp * BV;

    // ---- double-buffered per-chunk tiles (DMA'd straight from HBM) ----
    __shared__ __bf16 kd[2][C * SD];
    __shared__ __bf16 qd[2][C * SD];
    __shared__ __bf16 kr[2][C * D];
    __shared__ __bf16 INV[2][C * C];
    __shared__ __bf16 Mqk[2][C * C];
    __shared__ float  gtot[2][D];
    __shared__ __bf16 vmat[2][C * BV];        // filled via reg-stage (tail mask)
    __shared__ float  beta[2][C];             // filled via reg-stage (sigmoid)
    // ---- single-buffered persistent state / scratch ----
    __shared__ __bf16 Sv[BV * SD];
    __shared__ __bf16 Umat[C * BV];

    // vmat/beta register staging (mirror vsplit_db; vmat needs the v tail mask).
    constexpr int VR = (C * BV) / 64;
    __bf16 vR[VR];
    float  betaR;

    // dma(buf, chunk): issue the six big tiles straight into LDS buffer `buf`.
    auto dma = [&](int buf, int htc) {
        gll_rows_pad(kd[buf], SD, ws_kd + (int64_t)htc*C*D, D, C, lane);
        gll_rows_pad(qd[buf], SD, ws_qd + (int64_t)htc*C*D, D, C, lane);
        gll_bf16_vec(kr[buf], ws_kr + (int64_t)htc*C*D, C*D, lane);
        gll_bf16_vec(INV[buf], ws_inv + (int64_t)htc*C*C, C*C, lane);
        gll_bf16_vec(Mqk[buf], ws_mqk + (int64_t)htc*C*C, C*C, lane);
        gll_f32_vec (gtot[buf], ws_gt + (int64_t)htc*D, D, lane);
    };
    // stage(chunk): load v (tail-masked) + beta (sigmoid) into registers.
    auto stage = [&](int t0c, int alenc) {
        #pragma unroll
        for (int j = 0; j < VR; j++) {
            int idx = lane + j*64, m = idx / BV, vloc = idx % BV;
            vR[j] = (m < alenc) ? v_g[(t0c + m)*H*D + h*D + v0 + vloc] : (__bf16)0.0f;
        }
        betaR = (lane < C && lane < alenc)
            ? sigmoid_tanh(beta_g[(t0c + lane)*H + h]) : 0.0f;
    };
    // commit(buf): write staged v/beta registers into LDS buffer `buf`.
    auto commit = [&](int buf) {
        #pragma unroll
        for (int j = 0; j < VR; j++) vmat[buf][lane + j*64] = vR[j];
        if (lane < C) beta[buf][lane] = betaR;
    };

    for (int idx = lane; idx < BV * SD; idx += 64) Sv[idx] = (__bf16)0.0f;

    // prologue: fill buffer 0 for chunk 0.
    int t0_cur   = b * T_seq;
    int alen_cur = min(C, T_seq);
    dma(0, bh * NT);
    stage(t0_cur, alen_cur);
    commit(0);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");   // chunk-0 DMA landed in LDS
    __syncthreads();

    for (int nt = 0; nt < NT; nt++) {
        const int cur = nt & 1, nxt = cur ^ 1;
        const int t0 = t0_cur, alen = alen_cur;
        const bool has_nx = (nt + 1 < NT);

        // issue next chunk's DMA + v/beta loads NOW — in flight during compute.
        if (has_nx) {
            const int ht_nx   = bh * NT + (nt + 1);
            const int t0_nx   = b * T_seq + (nt + 1) * C;
            const int alen_nx = min(C, T_seq - (nt + 1) * C);
            dma(nxt, ht_nx);
            stage(t0_nx, alen_nx);
            t0_cur = t0_nx; alen_cur = alen_nx;
        }

        // ---- v = (v - kd @ S_kv) * beta ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 c = gemm_contract_last<__bf16, D, SD>(kd[cur], &Sv[vt*C*SD], lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                float nv = (bf16_to_f32(vmat[cur][m*BV + vloc]) - c[i]) * beta[cur][m];
                vmat[cur][m*BV + vloc] = f32_to_bf16(nv);
            }
        }
        __syncthreads();

        // ---- U = INV @ v ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 c;
            if constexpr (BV == C) c = mm_std_16_tr(INV[cur], vmat[cur], lane);
            else                   c = mm_std_tile_bf16(INV[cur], vmat[cur], vt*C, BV, lane);
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int m = (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                Umat[m*BV + vloc] = f32_to_bf16(c[i]);
            }
        }
        __syncthreads();

        // ---- out = qd @ S_kv + Mqk @ U ----
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 o1 = gemm_contract_last<__bf16, D, SD>(qd[cur], &Sv[vt*C*SD], lane);
            f32x4 o2;
            if constexpr (BV == C) o2 = mm_std_16_tr(Mqk[cur], Umat, lane);
            else                   o2 = mm_std_tile_bf16(Mqk[cur], Umat, vt*C, BV, lane);
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
                if constexpr (BV == C) c = mm_cf_trB(kr[cur], D, kt*C, Umat, lane);
                else                   c = mm_contract_first_bf16(kr[cur], Umat, kt*C, vt*C, D, BV, lane);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k = kt*C + (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                    float sv = bf16_to_f32(Sv[vloc*SD + k]) * ex2(gtot[cur][k]) + c[i];
                    Sv[vloc*SD + k] = f32_to_bf16(sv);
                }
            }
        }
        __syncthreads();     // chunk-nt LDS reads complete -> buffers free to reuse

        // land the (in-flight) next chunk: commit v/beta regs, wait DMA, publish.
        if (has_nx) {
            commit(nxt);
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");   // next DMA landed
            __syncthreads();
        }
    }
}

}  // namespace flashkda_hip
