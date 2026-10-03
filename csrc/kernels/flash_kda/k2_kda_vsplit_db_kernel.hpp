// FlashKDA K2 (recurrence) — V-split with software-pipelined HBM prefetch.
// Identical math/layout to k2_kda_vsplit_kernel<BV>, but restructured to hide the
// per-chunk workspace HBM load latency (the confirmed bottleneck: the kernel is
// latency-bound, using ~0.2% of HBM bandwidth — grid-starved at small N*H, ATT
// 53% VMEM-wait). One LDS buffer as before; each iteration issues chunk nt+1's
// global loads into REGISTERS at the loop top (non-blocking, in flight during the
// chunk-nt compute), then commits them to LDS after the chunk-nt LDS reads are
// done. This overlaps the long global-load latency with MFMA compute, raising the
// number of in-flight memory requests (MLP) instead of stalling on lgkmcnt/vmcnt.
//
// Bit-identical to vsplit (same values, same fragment reads, same MFMA order);
// only the *timing* of the loads changes. Selected via FLASH_KDA_K2=vsplit_db.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

template <int BV>
__global__ void __launch_bounds__(64)
k2_kda_vsplit_db_kernel(
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

    // ---- per-lane register staging tiles for the next chunk (issued in flight) ----
    // Layouts mirror the copy_* helpers so the commit lowers to the same ds_write.
    constexpr int RW = (C * D) / 8 / 64;         // bf16x8 chunks/lane for kd/qd/kr = 4
    constexpr int VR = (C * BV) / 64;            // vmat scalars/lane (BV=16 -> 4)
    bf16x8 kdR[RW], qdR[RW], krR[RW];
    bf16x4 invR, mqkR;
    f32x2  gtotR;
    __bf16 vR[VR];
    float  betaR;

    // stage(chunk c): issue all of chunk c's global loads into the registers above.
    // t0c/alenc are chunk c's token base and valid length (for the v tail mask).
    auto stage = [&](int htc, int t0c, int alenc) {
        auto* skd = reinterpret_cast<const bf16x8*>(ws_kd + (int64_t)htc*C*D);
        auto* sqd = reinterpret_cast<const bf16x8*>(ws_qd + (int64_t)htc*C*D);
        auto* skr = reinterpret_cast<const bf16x8*>(ws_kr + (int64_t)htc*C*D);
        #pragma unroll
        for (int j = 0; j < RW; j++) {
            int g = lane + j*64;               // 0..255 = (row*16 + col8) over [16,16] of 8
            kdR[j] = skd[g]; qdR[j] = sqd[g]; krR[j] = skr[g];
        }
        invR  = reinterpret_cast<const bf16x4*>(ws_inv + (int64_t)htc*C*C)[lane];
        mqkR  = reinterpret_cast<const bf16x4*>(ws_mqk + (int64_t)htc*C*C)[lane];
        gtotR = reinterpret_cast<const f32x2*>(ws_gt + (int64_t)htc*D)[lane];
        #pragma unroll
        for (int j = 0; j < VR; j++) {
            int idx = lane + j*64, m = idx / BV, vloc = idx % BV;
            vR[j] = (m < alenc) ? v_g[(t0c + m)*H*D + h*D + v0 + vloc] : (__bf16)0.0f;
        }
        betaR = (lane < C && lane < alenc)
            ? sigmoid_tanh(beta_g[(t0c + lane)*H + h]) : 0.0f;
    };

    // commit(): write the staged registers into the (single) LDS buffers. kd/qd are
    // row-padded to SD; kr is flat; INV/Mqk/gtot/vmat flat; beta on lanes<C.
    auto commit = [&]() {
        #pragma unroll
        for (int j = 0; j < RW; j++) {
            int g = lane + j*64, r = g >> 4, cc = g & 15;   // 16 cols of 8 per row
            reinterpret_cast<bf16x8*>(kd + r*SD)[cc] = kdR[j];
            reinterpret_cast<bf16x8*>(qd + r*SD)[cc] = qdR[j];
            reinterpret_cast<bf16x8*>(kr)[g]         = krR[j];
        }
        reinterpret_cast<bf16x4*>(INV)[lane] = invR;
        reinterpret_cast<bf16x4*>(Mqk)[lane] = mqkR;
        reinterpret_cast<f32x2*>(gtot)[lane] = gtotR;
        #pragma unroll
        for (int j = 0; j < VR; j++) vmat[lane + j*64] = vR[j];
        if (lane < C) beta[lane] = betaR;
    };

    for (int idx = lane; idx < BV * SD; idx += 64) Sv[idx] = (__bf16)0.0f;

    // prologue: stage + commit chunk 0
    int t0_cur   = b * T_seq;
    int alen_cur = min(C, T_seq);
    stage(bh * NT, t0_cur, alen_cur);
    commit();
    __syncthreads();

    for (int nt = 0; nt < NT; nt++) {
        const int t0 = t0_cur, alen = alen_cur;
        const bool has_nx = (nt + 1 < NT);

        // issue next chunk's loads NOW — in flight during the compute below
        if (has_nx) {
            const int ht_nx   = bh * NT + (nt + 1);
            const int t0_nx   = b * T_seq + (nt + 1) * C;
            const int alen_nx = min(C, T_seq - (nt + 1) * C);
            stage(ht_nx, t0_nx, alen_nx);
            t0_cur = t0_nx; alen_cur = alen_nx;
        }

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
                int kbase = kt*C + (lane >> 4) * 4, vloc = vt*C + (lane & 15);
                state_decay_acc(Sv, vloc*SD, gtot, kbase, c);
            }
        }
        __syncthreads();     // chunk-nt LDS reads complete -> safe to overwrite

        // commit the (already in-flight) next chunk into LDS
        if (has_nx) {
            commit();
            __syncthreads();
        }
    }
}

}  // namespace flashkda_hip
