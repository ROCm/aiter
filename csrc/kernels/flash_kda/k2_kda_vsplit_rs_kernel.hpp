// FlashKDA K2 (recurrence) — V-split with fp32 REGISTER-RESIDENT state (M2b).
// Same data flow and software-pipelined HBM prefetch as vsplit_db, but the
// recurrence state S no longer lives in LDS as bf16. Instead each lane keeps its
// own slice in fp32 VGPRs:
//     Sreg[vt][kt][i] = state[ V = vt*16 + (lane&15) ][ K = kt*16 + (lane>>4)*4 + i ]
// which is EXACTLY the element that lane (a) reads as the B-fragment in the
// kd@S / qd@S gemms and (b) writes in the K-dim decay carry — zero reshuffle
// (mapping verified against gemm_contract_last / mm_cf_trB fragment layouts).
//
// Why: vsplit_db rounds the whole state to bf16 in LDS every chunk, so the
// carry error compounds over NT chunks. Here the MFMA operands are still cast to
// bf16 (hardware MFMA is bf16), but the CARRY stays fp32 across chunks — the
// per-chunk bf16 rounding of the state is eliminated. Accuracy win (grows with
// T); not expected to be a speed win (kernel is HBM-latency-bound). Selected via
// FLASH_KDA_K2=vsplit_rs. Validated against the fp32-state oracle (torch_ref
// fp32_state=True).
//
// VGPR state cost: 32 fp32/lane per BV=16 (NVT*8*4). BV=32 -> 64, BV=64 -> 128
// (spill risk); guarded to the small-BV configs that fit.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

namespace vsplit_rs_detail {

struct RegBPair {
    f32x4 first;
    f32x4 second;
};

// Architecture-neutral register-state contraction.  Architecture policies may
// replace only this operator while reusing the complete recurrence body below.
// Keeping the default here also preserves every existing kernel instantiation.
struct RegBX16 {
    template <int LD, int NKB>
    static __device__ __forceinline__ f32x4 run(
            const __bf16* __restrict__ a,
            const float (&state)[NKB][4],
            int lane) {
        return gemm_regB<LD, NKB>(a, state, lane);
    }

    template <int LD, int NKB>
    static __device__ __forceinline__ RegBPair run_pair(
            const __bf16* __restrict__ a0,
            const __bf16* __restrict__ a1,
            const float (&state)[NKB][4],
            int lane) {
        const int row = lane & 15;
        const int kb = (lane >> 4) * 4;
        RegBPair out{{0, 0, 0, 0}, {0, 0, 0, 0}};
        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt) {
            bf16x4 af0, af1, sf;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = kt * 16 + kb + i;
                af0[i] = a0[row * LD + col];
                af1[i] = a1[row * LD + col];
                // Share the fp32->bf16 state conversion between the Kd and
                // Qd contractions; their independent accumulators also let
                // the scheduler interleave two otherwise 8-deep MFMA chains.
                sf[i] = f32_to_bf16(state[kt][i]);
            }
            out.first = mfma_bf16(af0, sf, out.first);
            out.second = mfma_bf16(af1, sf, out.second);
        }
        return out;
    }
};

// Default row-major kr publication and K16 state carry.  A private policy may
// replace this independently of RegBGemm, which keeps MFMA-width and LDS-layout
// experiments orthogonal instead of coupling both decisions in one kernel.
struct LinearKrCarry {
    template <int C, int D, int RW>
    static __device__ __forceinline__ void store(
            __bf16* __restrict__ kr,
            const bf16x8 (&staged)[RW],
            int lane) {
        (void)C;
        (void)D;
        #pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int g = lane + j * 64;
            reinterpret_cast<bf16x8*>(kr)[g] = staged[j];
        }
    }

    template <int C, int D, int BV>
    static __device__ __forceinline__ f32x4 run(
            const __bf16* __restrict__ kr,
            const __bf16* __restrict__ umat,
            int kt,
            int vt,
            int lane) {
        if constexpr (BV == C)
            return mm_cf_trB(kr, D, kt * C, umat, lane);
        else
            return mm_contract_first_bf16(
                kr, umat, kt * C, vt * C, D, BV, lane);
    }
};

}  // namespace vsplit_rs_detail

// HI/HO = has initial/final state; SFP32 = external state tensor dtype is fp32
// (bf16 otherwise). State layout [N,H,V,K] (row V, col K), matching torch_ref's
// work_state and the CUDA backend. The internal carry is always fp32 (register-
// resident); MFMA reads cast to bf16. init/final state are void* (dtype chosen by
// SFP32 at compile time).
template <
    int BV,
    bool HI = false,
    bool HO = false,
    bool SFP32 = false,
    bool VL = false,
    typename RegBGemm = vsplit_rs_detail::RegBX16,
    typename KrCarry = vsplit_rs_detail::LinearKrCarry>
__global__ void __launch_bounds__(64)
k2_kda_vsplit_rs_kernel(
        const __bf16* __restrict__ v_g,     // [T_total, H, D]
        const float*  __restrict__ beta_g,  // [T_total, H]
        __bf16* __restrict__ out_g,         // [T_total, H, D]
        const __bf16* __restrict__ ws_kd,   // [n_ht, 16, 128]
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float*  __restrict__ ws_gt,   // [n_ht, 128]
        const __bf16* __restrict__ ws_inv,  // [n_ht, 16, 16]
        const __bf16* __restrict__ ws_mqk,
        const void* __restrict__ init_state,  // [N,H,D,D] ([V,K]) or nullptr
        void* __restrict__ final_state,       // [N,H,D,D] ([V,K]) or nullptr
        const int32_t* __restrict__ cu_seqlens,  // varlen only
        const int* __restrict__ tile_prefix,     // varlen only [N+1]
        int total_tiles,                          // varlen only (ht column pitch)
        int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    constexpr int SD = D + 4;                 // padded LDS row pitch (kd/qd)
    constexpr int NVT = BV / C;
    constexpr int NKB = D / C;                 // K-tiles per V-row (=8)
    const int lane = threadIdx.x;
    const int bh = blockIdx.x;
    const int vgrp = blockIdx.y;
    const int v0 = vgrp * BV;
    // Per-block sequence geometry: VL derives (h, chunks, ht/t0 bases) from
    // cu_seqlens/tile_prefix; the batched path keeps the uniform-length layout.
    int h, seq_len_eff, NT_eff, ht_base, t0_base;
    if constexpr (VL) {
        const int seq_idx = bh / H; h = bh % H;
        const int64_t bos = cu_seqlens[seq_idx];
        seq_len_eff = int(cu_seqlens[seq_idx + 1] - bos);
        NT_eff  = (seq_len_eff + C - 1) / C;
        ht_base = h * total_tiles + tile_prefix[seq_idx];
        t0_base = int(bos);
    } else {
        const int b = bh / H; h = bh % H;
        seq_len_eff = T_seq;
        NT_eff  = NT;
        ht_base = bh * NT;
        t0_base = b * T_seq;
    }

    // NOTE: no __shared__ Sv — the state is in registers below.
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 qd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 vmat[C * BV];
    __shared__ __bf16 Umat[C * BV];
    __shared__ __bf16 INV[C * C];
    __shared__ __bf16 Mqk[C * C];
    __shared__ float  gtot[D];
    __shared__ float  beta[C];

    // ---- fp32 register-resident recurrence state ----
    // Sreg[vt][kt][i] holds state[V = v0+vt*16+(lane&15)][K = kt*16+(lane>>4)*4+i].
    // Load initial_state (per-lane scalar gather) when HI, else zero.
    float Sreg[NVT][NKB][4];
    const int64_t st_base = (int64_t)bh * D * D;   // [V,K] slab for this (b,h)
    #pragma unroll
    for (int vt = 0; vt < NVT; vt++)
        #pragma unroll
        for (int kt = 0; kt < NKB; kt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                if constexpr (HI) {
                    int V = v0 + vt*C + (lane & 15);
                    int K = kt*C + (lane >> 4) * 4 + i;
                    int64_t idx = st_base + (int64_t)V * D + K;
                    Sreg[vt][kt][i] = SFP32
                        ? reinterpret_cast<const float*>(init_state)[idx]
                        : bf16_to_f32(reinterpret_cast<const __bf16*>(init_state)[idx]);
                } else {
                    Sreg[vt][kt][i] = 0.0f;
                }
            }

    // ---- per-lane register staging tiles for the next chunk (issued in flight) ----
    constexpr int RW = (C * D) / 8 / 64;         // bf16x8 chunks/lane for kd/qd/kr = 4
    constexpr int VR = (C * BV) / 64;            // vmat scalars/lane
    bf16x8 kdR[RW], qdR[RW], krR[RW];
    bf16x4 invR, mqkR;
    f32x2  gtotR;
    __bf16 vR[VR];
    float  betaR;

    auto stage = [&](int htc, int t0c, int alenc) {
        auto* skd = reinterpret_cast<const bf16x8*>(ws_kd + (int64_t)htc*C*D);
        auto* sqd = reinterpret_cast<const bf16x8*>(ws_qd + (int64_t)htc*C*D);
        auto* skr = reinterpret_cast<const bf16x8*>(ws_kr + (int64_t)htc*C*D);
        #pragma unroll
        for (int j = 0; j < RW; j++) {
            int g = lane + j*64;
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

    auto commit = [&]() {
        #pragma unroll
        for (int j = 0; j < RW; j++) {
            int g = lane + j*64, r = g >> 4, cc = g & 15;
            reinterpret_cast<bf16x8*>(kd + r*SD)[cc] = kdR[j];
            reinterpret_cast<bf16x8*>(qd + r*SD)[cc] = qdR[j];
        }
        KrCarry::template store<C, D>(kr, krR, lane);
        reinterpret_cast<bf16x4*>(INV)[lane] = invR;
        reinterpret_cast<bf16x4*>(Mqk)[lane] = mqkR;
        reinterpret_cast<f32x2*>(gtot)[lane] = gtotR;
        #pragma unroll
        for (int j = 0; j < VR; j++) vmat[lane + j*64] = vR[j];
        if (lane < C) beta[lane] = betaR;
    };

    // prologue: stage + commit chunk 0
    int t0_cur   = t0_base;
    int alen_cur = min(C, seq_len_eff);
    stage(ht_base, t0_cur, alen_cur);
    commit();
    __syncthreads();

    for (int nt = 0; nt < NT_eff; nt++) {
        const int t0 = t0_cur, alen = alen_cur;
        const bool has_nx = (nt + 1 < NT_eff);

        if (has_nx) {
            const int ht_nx   = ht_base + (nt + 1);
            const int t0_nx   = t0_base + (nt + 1) * C;
            const int alen_nx = min(C, seq_len_eff - (nt + 1) * C);
            stage(ht_nx, t0_nx, alen_nx);
            t0_cur = t0_nx; alen_cur = alen_nx;
        }

        // ---- v = (v - kd @ S_kv) * beta ---- (S from registers)
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 c = RegBGemm::template run<SD, NKB>(
                kd, Sreg[vt], lane);
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

        // ---- out = qd @ S_kv + Mqk @ U ---- (qd@S from registers)
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++) {
            f32x4 o1 = RegBGemm::template run<SD, NKB>(
                qd, Sreg[vt], lane);
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
        // no barrier here: state carry re-reads Umat/kr/gtot, none written since.

        // ---- state carry (fp32, in registers): S[v,k] = delta_s[k,v] + S[v,k]*ex2(g_total[k]) ----
        #pragma unroll
        for (int kt = 0; kt < NKB; kt++) {
            #pragma unroll
            for (int vt = 0; vt < NVT; vt++) {
                f32x4 c = KrCarry::template run<C, D, BV>(
                    kr, Umat, kt, vt, lane);
                int kbase = kt*C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; i++)
                    Sreg[vt][kt][i] = Sreg[vt][kt][i] * ex2(gtot[kbase + i]) + c[i];
            }
        }
        __syncthreads();     // kr/gtot/Umat reads done -> safe for commit to overwrite

        if (has_nx) {
            commit();
            __syncthreads();
        }
    }

    // ---- final state store: Sreg (post last-chunk carry) -> final_state[V,K] ----
    if constexpr (HO) {
        #pragma unroll
        for (int vt = 0; vt < NVT; vt++)
            #pragma unroll
            for (int kt = 0; kt < NKB; kt++)
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    int V = v0 + vt*C + (lane & 15);
                    int K = kt*C + (lane >> 4) * 4 + i;
                    int64_t idx = st_base + (int64_t)V * D + K;
                    if constexpr (SFP32)
                        reinterpret_cast<float*>(final_state)[idx] = Sreg[vt][kt][i];
                    else
                        reinterpret_cast<__bf16*>(final_state)[idx] = f32_to_bf16(Sreg[vt][kt][i]);
                }
    }
}

// Multi-wave packing of the BV16 register-state kernel.  Each
// wave still owns an independent V16 slice and keeps the same fp32 recurrence
// state as k2_kda_vsplit_rs_kernel<16>; the CTA only shares V-independent KDA
// operands between the waves.  A separate symbol keeps the established
// single-wave specialization and its resource allocation independent.
template <int NW, bool HI = false, bool HO = false,
          bool SFP32 = false, bool VL = false,
          bool ACTIVATED_BETA = false>
__global__ void __launch_bounds__(NW * 64)
k2_kda_vsplit_rs_mw_kernel(
        const __bf16* __restrict__ v_g,
        const float*  __restrict__ beta_g,
        __bf16* __restrict__ out_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float*  __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ ws_mqk,
        const void* __restrict__ init_state,
        void* __restrict__ final_state,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        int total_tiles,
        int T_seq, int H, int NT) {
    static_assert(NW == 1 || NW == 2 || NW == 4,
                  "the rs_mw probe supports one, two, or four waves");
    constexpr int C = 16, D = 128, BV = 16, SD = D + 4;
    constexpr int NKB = D / C;
    constexpr int NTHREADS = NW * 64;

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int bh = blockIdx.x;
    const int vgrp = blockIdx.y * NW + wave;
    const int v0 = vgrp * BV;

    int h, seq_len_eff, NT_eff, ht_base, t0_base;
    if constexpr (VL) {
        const int seq_idx = bh / H;
        h = bh % H;
        const int64_t bos = cu_seqlens[seq_idx];
        seq_len_eff = int(cu_seqlens[seq_idx + 1] - bos);
        NT_eff = (seq_len_eff + C - 1) / C;
        ht_base = h * total_tiles + tile_prefix[seq_idx];
        t0_base = int(bos);
    } else {
        const int b = bh / H;
        h = bh % H;
        seq_len_eff = T_seq;
        NT_eff = NT;
        ht_base = bh * NT;
        t0_base = b * T_seq;
    }

    // The waves share only V-independent operands.  Per-wave residual/U
    // panels stay disjoint, and the recurrent state remains entirely in VGPRs.
    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 qd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 vmat[NW][C * BV];
    __shared__ __bf16 umat[NW][C * BV];
    __shared__ __bf16 inv[C * C];
    __shared__ __bf16 mqk[C * C];
    __shared__ float gtot[D];
    __shared__ float beta[C];

    float sreg[NKB][4];
    const int64_t state_base = int64_t(bh) * D * D;
    #pragma unroll
    for (int kt = 0; kt < NKB; ++kt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = kt * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_base + int64_t(vv) * D + kk;
            if constexpr (HI)
                sreg[kt][i] = SFP32
                    ? reinterpret_cast<const float*>(init_state)[idx]
                    : bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[idx]);
            else
                sreg[kt][i] = 0.0f;
        }

    // One-chunk register prefetch.  Splitting the three CxD inputs across NW
    // waves reduces their staging footprint per lane relative to NW1.
    constexpr int ROW_VECS = (C * D) / 8;
    constexpr int RW = (ROW_VECS + NTHREADS - 1) / NTHREADS;
    constexpr int VR = (C * BV) / 64;
    static_assert(((NW == 1 && RW == 4) || (NW == 2 && RW == 2) ||
                   (NW == 4 && RW == 1)) &&
                  VR == 4);
    bf16x8 kd_r[RW], qd_r[RW], kr_r[RW];
    bf16x8 inv_r, mqk_r;
    f32x4 gt_r;
    __bf16 v_r[VR];
    float beta_r;

    auto stage = [&](int ht, int t0, int alen) {
        const auto* skd = reinterpret_cast<const bf16x8*>(
            ws_kd + int64_t(ht) * C * D);
        const auto* sqd = reinterpret_cast<const bf16x8*>(
            ws_qd + int64_t(ht) * C * D);
        const auto* skr = reinterpret_cast<const bf16x8*>(
            ws_kr + int64_t(ht) * C * D);
        #pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int idx = tid + j * NTHREADS;
            if (idx < ROW_VECS) {
                kd_r[j] = skd[idx];
                qd_r[j] = sqd[idx];
                kr_r[j] = skr[idx];
            }
        }
        if (tid < (C * C) / 8) {
            inv_r = reinterpret_cast<const bf16x8*>(
                ws_inv + int64_t(ht) * C * C)[tid];
            mqk_r = reinterpret_cast<const bf16x8*>(
                ws_mqk + int64_t(ht) * C * C)[tid];
        }
        if (tid < D / 4)
            gt_r = reinterpret_cast<const f32x4*>(
                ws_gt + int64_t(ht) * D)[tid];
        #pragma unroll
        for (int j = 0; j < VR; ++j) {
            const int idx = lane + j * 64;
            const int m = idx / BV;
            const int vv = idx % BV;
            v_r[j] = m < alen
                ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                : (__bf16)0.0f;
        }
        if (tid < C) {
            if (tid < alen) {
                if constexpr (ACTIVATED_BETA)
                    beta_r = beta_g[int64_t(ht) * C + tid];
                else
                    beta_r = sigmoid_tanh(
                        beta_g[int64_t(t0 + tid) * H + h]);
            } else {
                beta_r = 0.0f;
            }
        }
    };

    auto commit = [&]() {
        #pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int idx = tid + j * NTHREADS;
            if (idx < ROW_VECS) {
                const int row = idx >> 4;
                const int col8 = idx & 15;
                reinterpret_cast<bf16x8*>(kd + row * SD)[col8] = kd_r[j];
                reinterpret_cast<bf16x8*>(qd + row * SD)[col8] = qd_r[j];
                reinterpret_cast<bf16x8*>(kr)[idx] = kr_r[j];
            }
        }
        if (tid < (C * C) / 8) {
            reinterpret_cast<bf16x8*>(inv)[tid] = inv_r;
            reinterpret_cast<bf16x8*>(mqk)[tid] = mqk_r;
        }
        if (tid < D / 4) {
            // The decay is V-independent.  Materialize its 128 unique values
            // once per chunk instead of re-evaluating ex2 for every recurrent
            // state element owned by the CTA's one, two, or four V16 waves.
            f32x4 decay = gt_r;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                decay[i] = ex2(decay[i]);
            reinterpret_cast<f32x4*>(gtot)[tid] = decay;
        }
        #pragma unroll
        for (int j = 0; j < VR; ++j)
            vmat[wave][lane + j * 64] = v_r[j];
        if (tid < C)
            beta[tid] = beta_r;
    };

    int t0_cur = t0_base;
    int alen_cur = min(C, seq_len_eff);
    stage(ht_base, t0_cur, alen_cur);
    commit();
    if constexpr (NW == 1)
        __syncwarp();
    else
        __syncthreads();

    for (int nt = 0; nt < NT_eff; ++nt) {
        const int t0 = t0_cur;
        const int alen = alen_cur;
        const bool has_next = nt + 1 < NT_eff;
        if (has_next) {
            const int ht_next = ht_base + nt + 1;
            const int t0_next = t0_base + (nt + 1) * C;
            const int alen_next = min(C, seq_len_eff - (nt + 1) * C);
            stage(ht_next, t0_next, alen_next);
            t0_cur = t0_next;
            alen_cur = alen_next;
        }

        const auto state_products =
            vsplit_rs_detail::RegBX16::template run_pair<SD, NKB>(
                kd, qd, sreg, lane);
        const f32x4 residual = state_products.first;
        const f32x4 out_state = state_products.second;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i;
            const int vv = lane & 15;
            const float x =
                (bf16_to_f32(vmat[wave][m * BV + vv]) - residual[i]) *
                beta[m];
            vmat[wave][m * BV + vv] = f32_to_bf16(x);
        }
        // The waves own disjoint vmat panels.  Only lanes in this wave
        // consume the residual update below, so a CTA rendezvous needlessly
        // couples two independent V16 recurrences on every chunk.
        __syncwarp();

        f32x4 u = mm_std_16_tr(inv, vmat[wave], lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i;
            const int vv = lane & 15;
            umat[wave][m * BV + vv] = f32_to_bf16(u[i]);
        }
        // umat is likewise wave-private; make its LDS publication visible to
        // this wave's output/carry MFMAs without stalling on the sibling wave.
        __syncwarp();

        f32x4 out_local = mm_std_16_tr(mqk, umat[wave], lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i;
            const int vv = lane & 15;
            if (m < alen) {
                const __bf16 a = f32_to_bf16(out_state[i]);
                const __bf16 b = f32_to_bf16(out_local[i]);
                out_g[(int64_t(t0 + m) * H + h) * D + v0 + vv] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
        }

        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt) {
            f32x4 carry = vsplit_rs_detail::LinearKrCarry::template run<
                C, D, BV>(kr, umat[wave], kt, 0, lane);
            const int kbase = kt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = sreg[kt][i] * gtot[kbase + i] + carry[i];
        }
        if constexpr (NW == 1)
            __syncwarp();
        else
            __syncthreads();
        if (has_next) {
            commit();
            if constexpr (NW == 1)
                __syncwarp();
            else
                __syncthreads();
        }
    }

    if constexpr (HO) {
        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = kt * C + (lane >> 4) * 4 + i;
                const int64_t idx = state_base + int64_t(vv) * D + kk;
                if constexpr (SFP32)
                    reinterpret_cast<float*>(final_state)[idx] = sreg[kt][i];
                else
                    reinterpret_cast<__bf16*>(final_state)[idx] =
                        f32_to_bf16(sreg[kt][i]);
            }
    }
}

}  // namespace flashkda_hip
