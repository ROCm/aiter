// FlashKDA K2 (recurrence) — V-split, DEPTH-2 software-pipelined prefetch.
// Same mechanism as k2_kda_vsplit_db_kernel<BV> (register-staged dwordx4 loads,
// single LDS buffer committed per chunk) but keeps TWO chunks' worth of HBM loads
// in flight at once: at the top of chunk nt it issues chunk (nt+2)'s loads while
// chunk (nt+1)'s loads — issued last iter — are still in flight. This doubles the
// outstanding memory requests (MLP), directly attacking the residual vmcnt stall
// that dominates vsplit_db's ATT at small N*H (latency-bound, grid-starved).
//
// The two staging slots MUST be indexed by a COMPILE-TIME constant — indexing a
// register array with a runtime value forces it out of the register file into
// movrel/select sequences (catastrophic). So the main loop is unrolled by 2 and
// the slot is always a 0/1 literal, passed to stage/commit as an integral_constant.
//
// Bit-identical math/layout/MFMA order to vsplit_db. Selected via FLASH_KDA_K2=vsplit_db2.
#pragma once
#include <hip/hip_runtime.h>
#include <type_traits>
#include "mfma.hpp"

namespace flashkda_hip {

template <int BV>
__global__ void __launch_bounds__(64)
k2_kda_vsplit_db2_kernel(
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
    const int lane = threadIdx.x;
    const int bh = blockIdx.x;
    const int vgrp = blockIdx.y;
    const int b = bh / H, h = bh % H;
    const int v0 = vgrp * BV;

    __shared__ __bf16 Sv[BV * SD];
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 qd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 vmat[C * BV];
    __shared__ __bf16 Umat[C * BV];
    __shared__ __bf16 INV[C * C];
    __shared__ __bf16 Mqk[C * C];
    __shared__ float  gtot[D];
    __shared__ float  beta[C];

    constexpr int RW = (C * D) / 8 / 64;   // 4
    constexpr int VR = (C * BV) / 64;
    bf16x8 kdR[2][RW], qdR[2][RW], krR[2][RW];
    bf16x4 invR[2], mqkR[2];
    f32x2  gtotR[2];
    __bf16 vR[2][VR];
    float  betaR[2];

    // stage<S>(chunk c): issue chunk c's global loads into staging slot S (literal).
    auto stage = [&](auto Sc, int htc, int t0c, int alenc) {
        constexpr int s = decltype(Sc)::value;
        auto* skd = reinterpret_cast<const bf16x8*>(ws_kd + (int64_t)htc*C*D);
        auto* sqd = reinterpret_cast<const bf16x8*>(ws_qd + (int64_t)htc*C*D);
        auto* skr = reinterpret_cast<const bf16x8*>(ws_kr + (int64_t)htc*C*D);
        #pragma unroll
        for (int j = 0; j < RW; j++) {
            int g = lane + j*64;
            kdR[s][j] = skd[g]; qdR[s][j] = sqd[g]; krR[s][j] = skr[g];
        }
        invR[s]  = reinterpret_cast<const bf16x4*>(ws_inv + (int64_t)htc*C*C)[lane];
        mqkR[s]  = reinterpret_cast<const bf16x4*>(ws_mqk + (int64_t)htc*C*C)[lane];
        gtotR[s] = reinterpret_cast<const f32x2*>(ws_gt + (int64_t)htc*D)[lane];
        #pragma unroll
        for (int j = 0; j < VR; j++) {
            int idx = lane + j*64, m = idx / BV, vloc = idx % BV;
            vR[s][j] = (m < alenc) ? v_g[(t0c + m)*H*D + h*D + v0 + vloc] : (__bf16)0.0f;
        }
        betaR[s] = (lane < C && lane < alenc)
            ? sigmoid_tanh(beta_g[(t0c + lane)*H + h]) : 0.0f;
    };

    // commit<S>(): write staging slot S (literal) into the single LDS buffer.
    auto commit = [&](auto Sc) {
        constexpr int s = decltype(Sc)::value;
        #pragma unroll
        for (int j = 0; j < RW; j++) {
            int g = lane + j*64, r = g >> 4, cc = g & 15;
            reinterpret_cast<bf16x8*>(kd + r*SD)[cc] = kdR[s][j];
            reinterpret_cast<bf16x8*>(qd + r*SD)[cc] = qdR[s][j];
            reinterpret_cast<bf16x8*>(kr)[g]         = krR[s][j];
        }
        reinterpret_cast<bf16x4*>(INV)[lane] = invR[s];
        reinterpret_cast<bf16x4*>(Mqk)[lane] = mqkR[s];
        reinterpret_cast<f32x2*>(gtot)[lane] = gtotR[s];
        #pragma unroll
        for (int j = 0; j < VR; j++) vmat[lane + j*64] = vR[s][j];
        if (lane < C) beta[lane] = betaR[s];
    };

    // compute chunk with token base t0 and valid length alen (reads LDS only).
    auto compute = [&](int t0, int alen) {
        // ---- v = (v - kd @ S_kv) * beta ----
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
            if constexpr (BV == C) c = mm_std_16_tr(INV, vmat, lane);
            else                   c = mm_std_tile_bf16(INV, vmat, vt*C, BV, lane);
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
            f32x4 o1 = gemm_contract_last<__bf16, D, SD>(qd, &Sv[vt*C*SD], lane);
            f32x4 o2;
            if constexpr (BV == C) o2 = mm_std_16_tr(Mqk, Umat, lane);
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
        // ---- state update ----
        #pragma unroll
        for (int kt = 0; kt < D / C; kt++) {
            #pragma unroll
            for (int vt = 0; vt < NVT; vt++) {
                f32x4 c;
                if constexpr (BV == C) c = mm_cf_trB(kr, D, kt*C, Umat, lane);
                else                   c = mm_contract_first_bf16(kr, Umat, kt*C, vt*C, D, BV, lane);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int k = kt*C + (lane >> 4) * 4 + i, vloc = vt*C + (lane & 15);
                    float sv = bf16_to_f32(Sv[vloc*SD + k]) * ex2(gtot[k]) + c[i];
                    Sv[vloc*SD + k] = f32_to_bf16(sv);
                }
            }
        }
        __syncthreads();     // chunk LDS reads done -> safe to commit next into LDS
    };

    for (int idx = lane; idx < BV * SD; idx += 64) Sv[idx] = (__bf16)0.0f;

    auto t0_of   = [&](int c){ return b * T_seq + c * C; };
    auto alen_of = [&](int c){ return min(C, T_seq - c * C); };
    constexpr auto S0 = std::integral_constant<int,0>{};
    constexpr auto S1 = std::integral_constant<int,1>{};

    // prologue: chunk0 -> slot0, chunk1 -> slot1 (both in flight); commit chunk0.
    stage(S0, bh * NT + 0, t0_of(0), alen_of(0));
    if (1 < NT) stage(S1, bh * NT + 1, t0_of(1), alen_of(1));
    commit(S0);
    __syncthreads();

    // Unrolled by 2: even chunk uses slot0, odd chunk uses slot1 (compile-time).
    for (int nt = 0; nt < NT; nt += 2) {
        // ----- chunk nt (in LDS; slot0 free after its prologue/prev commit) -----
        if (nt + 2 < NT) stage(S0, bh * NT + nt+2, t0_of(nt+2), alen_of(nt+2));
        compute(t0_of(nt), alen_of(nt));
        if (nt + 1 < NT) { commit(S1); __syncthreads(); }   // publish chunk nt+1

        // ----- chunk nt+1 (in LDS; slot1 free after the commit above) -----
        if (nt + 1 < NT) {
            if (nt + 3 < NT) stage(S1, bh * NT + nt+3, t0_of(nt+3), alen_of(nt+3));
            compute(t0_of(nt+1), alen_of(nt+1));
            if (nt + 2 < NT) { commit(S0); __syncthreads(); } // publish chunk nt+2
        }
    }
}

}  // namespace flashkda_hip
