// FlashKDA K2 C-split: serial state scan + chunk-parallel output.
//
// K1 has already materialized all chunk-local KDA factors.  The scan keeps the
// recurrent state in fp32 registers, stores the state at each chunk boundary,
// and materializes the corrected value U.  The second kernel consumes those
// two tensors and moves both output GEMMs off the serial recurrence path.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

// Four-wave K5 scan for gfx942.  A CTA owns V16; each wave owns K32 of the
// fp32 recurrent state.  The kd@S reduction is combined through fp32 LDS and
// the dominant kr^T@U carry runs four K tiles in parallel.
template <int NW, bool HI = false, bool HO = false, bool SFP32 = false, bool VL = false>
__global__ void __launch_bounds__(NW * 64)
k2_kda_csplit_scan_mw_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ cs_u,
        __bf16* __restrict__ cs_sin,
        const void* __restrict__ init_state,
        void* __restrict__ final_state,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int total_tiles, int total_segments, int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128, BV = 16, SD = D + 4;
    static_assert(NW == 2 || NW == 4 || NW == 8);
    constexpr int KTW = 8 / NW;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const int bh = blockIdx.x, v0 = blockIdx.y * BV;
    int h, seq_len, nt_eff, ht_base, t0_base;
    if constexpr (VL) {
        const int seq = bh / H;
        h = bh % H;
        const int64_t bos = cu_seqlens[seq];
        seq_len = int(cu_seqlens[seq + 1] - bos);
        nt_eff = (seq_len + C - 1) / C;
        ht_base = h * total_tiles + tile_prefix[seq];
        t0_base = int(bos);
    } else {
        const int b = bh / H;
        h = bh % H;
        seq_len = T_seq;
        nt_eff = NT;
        ht_base = bh * NT;
        t0_base = b * T_seq;
    }

    // The standard varlen prefix launch has already published the recurrent
    // identity for an empty owner.  Return before loading/staging state: a
    // BF16 -> FP32 -> BF16 round trip is not bitwise neutral for subnormals
    // under the backend's FTZ mode, and the first speculative stage otherwise
    // reads an unused gap tile.
    if constexpr (VL) {
        if (nt_eff == 0) return;
    }

    __shared__ __bf16 kd[C * SD], kr[C * D];
    __shared__ __bf16 vmat[C * BV], umat[C * BV], inv[C * C];
    __shared__ float gtot[D], beta[C];
    __shared__ float partial[NW * C * BV];

    float sreg[KTW][4];
    const int64_t state_base = int64_t(bh) * D * D;
    #pragma unroll
    for (int kt = 0; kt < KTW; ++kt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = (wave * KTW + kt) * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_base + int64_t(vv) * D + kk;
            if constexpr (HI)
                sreg[kt][i] = SFP32
                    ? reinterpret_cast<const float*>(init_state)[idx]
                    : bf16_to_f32(reinterpret_cast<const __bf16*>(init_state)[idx]);
            else
                sreg[kt][i] = 0.0f;
        }

    constexpr int NTH = NW * 64;
    constexpr int WSR = ((C * D) / 8 + NTH - 1) / NTH;
    constexpr int VR = (C * BV + NTH - 1) / NTH;
    bf16x8 kd_r[WSR], kr_r[WSR], inv_r;
    f32x4 gt_r;
    __bf16 v_r[VR];
    float beta_r;
    auto stage = [&](int ht, int t0, int alen) {
        #pragma unroll
        for (int j = 0; j < WSR; ++j) {
            const int idx = tid + j * NTH;
            if (idx < (C * D) / 8) {
                kd_r[j] = reinterpret_cast<const bf16x8*>(
                    ws_kd + int64_t(ht) * C * D)[idx];
                kr_r[j] = reinterpret_cast<const bf16x8*>(
                    ws_kr + int64_t(ht) * C * D)[idx];
            }
        }
        if (tid < (C * C) / 8)
            inv_r = reinterpret_cast<const bf16x8*>(
                ws_inv + int64_t(ht) * C * C)[tid];
        if (tid < D / 4)
            gt_r = reinterpret_cast<const f32x4*>(
                ws_gt + int64_t(ht) * D)[tid];
        #pragma unroll
        for (int j = 0; j < VR; ++j) {
            const int idx = tid + j * NTH;
            if (idx < C * BV) {
                const int m = idx / BV, vv = idx % BV;
                v_r[j] = m < alen
                    ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                    : (__bf16)0.0f;
            }
        }
        if (tid < C)
            beta_r = tid < alen
                ? sigmoid_tanh(beta_g[int64_t(t0 + tid) * H + h]) : 0.0f;
    };
    auto commit = [&]() {
        #pragma unroll
        for (int j = 0; j < WSR; ++j) {
            const int idx = tid + j * NTH;
            if (idx < (C * D) / 8) {
                const int row = idx >> 4, col8 = idx & 15;
                reinterpret_cast<bf16x8*>(kd + row * SD)[col8] = kd_r[j];
                reinterpret_cast<bf16x8*>(kr)[idx] = kr_r[j];
            }
        }
        if (tid < (C * C) / 8)
            reinterpret_cast<bf16x8*>(inv)[tid] = inv_r;
        if (tid < D / 4)
            reinterpret_cast<f32x4*>(gtot)[tid] = gt_r;
        #pragma unroll
        for (int j = 0; j < VR; ++j) {
            const int idx = tid + j * NTH;
            if (idx < C * BV) vmat[idx] = v_r[j];
        }
        if (tid < C) beta[tid] = beta_r;
    };
    stage(ht_base, t0_base, min(C, seq_len));
    commit();
    __syncthreads();

    for (int nt = 0; nt < nt_eff; ++nt) {
        const int ht = ht_base + nt;
        const int t0 = t0_base + nt * C;
        const int alen = min(C, seq_len - nt * C);
        const bool has_next = nt + 1 < nt_eff;
        if ((nt & 3) == 0) {
            int ss;
            if constexpr (VL)
                ss = h * total_segments + segment_prefix[bh / H] + nt / 4;
            else
                ss = bh * ((NT + 3) / 4) + nt / 4;
            #pragma unroll
            for (int kt = 0; kt < KTW; ++kt) {
                const int vv = v0 + (lane & 15);
                const int kk = (wave * KTW + kt) * C + (lane >> 4) * 4;
                bf16x4 packed;
                #pragma unroll
                for (int i = 0; i < 4; ++i) packed[i] = f32_to_bf16(sreg[kt][i]);
                *reinterpret_cast<bf16x4*>(
                    cs_sin + (int64_t(ss) * D + vv) * D + kk) = packed;
            }
        }

        if (has_next)
            stage(ht + 1, t0 + C, min(C, seq_len - (nt + 1) * C));

        f32x4 p = gemm_regB<SD, KTW>(kd + wave * KTW * C, sreg, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i, vv = lane & 15;
            partial[(wave * C + m) * BV + vv] = p[i];
        }
        __syncthreads();

        if (wave == 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i, vv = lane & 15;
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < NW; ++w)
                    sum += partial[(w * C + m) * BV + vv];
                vmat[m * BV + vv] = f32_to_bf16(
                    (bf16_to_f32(vmat[m * BV + vv]) - sum) * beta[m]);
            }
        }
        __syncthreads();

        if (wave == 0) {
            f32x4 u = mm_std_16_tr(inv, vmat, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i, vv = lane & 15;
                const __bf16 x = f32_to_bf16(u[i]);
                umat[m * BV + vv] = x;
                cs_u[(int64_t(ht) * C + m) * D + v0 + vv] = x;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kt = 0; kt < KTW; ++kt) {
            const int gkt = wave * KTW + kt;
            f32x4 c = mm_cf_trB(kr, D, gkt * C, umat, lane);
            const int kbase = gkt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = sreg[kt][i] * ex2(gtot[kbase + i]) + c[i];
        }
        __syncthreads();
        if (has_next) {
            commit();
            __syncthreads();
        }
    }

    if constexpr (HO) {
        #pragma unroll
        for (int kt = 0; kt < KTW; ++kt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk = (wave * KTW + kt) * C + (lane >> 4) * 4 + i;
                const int64_t idx = state_base + int64_t(vv) * D + kk;
                if constexpr (SFP32)
                    reinterpret_cast<float*>(final_state)[idx] = sreg[kt][i];
                else
                    reinterpret_cast<__bf16*>(final_state)[idx] = f32_to_bf16(sreg[kt][i]);
            }
    }
}

template <int BV, bool HI = false, bool HO = false, bool SFP32 = false,
          bool VL = false>
__global__ void __launch_bounds__(64)
k2_kda_csplit_scan_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        __bf16* __restrict__ cs_u,       // [n_ht, C, V]
        __bf16* __restrict__ cs_sin,     // [n_ht, V, K]
        const void* __restrict__ init_state,
        void* __restrict__ final_state,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int total_tiles, int total_segments, int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128;
    constexpr int SD = D + 4;
    constexpr int NVT = BV / C;
    constexpr int NKB = D / C;
    const int lane = threadIdx.x;
    const int bh = blockIdx.x;
    const int vgrp = blockIdx.y;
    const int v0 = vgrp * BV;

    int h, seq_len, nt_eff, ht_base, t0_base;
    if constexpr (VL) {
        const int seq = bh / H;
        h = bh % H;
        const int64_t bos = cu_seqlens[seq];
        seq_len = int(cu_seqlens[seq + 1] - bos);
        nt_eff = (seq_len + C - 1) / C;
        ht_base = h * total_tiles + tile_prefix[seq];
        t0_base = int(bos);
    } else {
        const int b = bh / H;
        h = bh % H;
        seq_len = T_seq;
        nt_eff = NT;
        ht_base = bh * NT;
        t0_base = b * T_seq;
    }

    // Empty packed sequences are initialized by k1_build_tile_prefix.  Avoid
    // both the unused speculative stage and a lossy FTZ state round trip.
    if constexpr (VL) {
        if (nt_eff == 0) return;
    }

    __shared__ __bf16 kd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 vmat[C * BV];
    __shared__ __bf16 umat[C * BV];
    __shared__ __bf16 inv[C * C];
    __shared__ float gtot[D];
    __shared__ float beta[C];

    float sreg[NVT][NKB][4];
    const int64_t state_base = int64_t(bh) * D * D;
    #pragma unroll
    for (int vt = 0; vt < NVT; ++vt)
        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + vt * C + (lane & 15);
                const int kk = kt * C + (lane >> 4) * 4 + i;
                const int64_t idx = state_base + int64_t(vv) * D + kk;
                if constexpr (HI) {
                    sreg[vt][kt][i] = SFP32
                        ? reinterpret_cast<const float*>(init_state)[idx]
                        : bf16_to_f32(reinterpret_cast<const __bf16*>(init_state)[idx]);
                } else {
                    sreg[vt][kt][i] = 0.0f;
                }
            }

    // Match the production register-state kernel's one-chunk software
    // pipeline: global reads for chunk n+1 overlap the MFMA work of chunk n.
    constexpr int RW = (C * D) / 8 / 64;
    constexpr int VR = (C * BV) / 64;
    bf16x8 kd_r[RW], kr_r[RW];
    bf16x4 inv_r;
    f32x2 gt_r;
    __bf16 v_r[VR];
    float beta_r;

    auto stage = [&](int ht, int t0, int alen) {
        const auto* kd_src = reinterpret_cast<const bf16x8*>(
            ws_kd + int64_t(ht) * C * D);
        const auto* kr_src = reinterpret_cast<const bf16x8*>(
            ws_kr + int64_t(ht) * C * D);
        #pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int idx = lane + j * 64;
            kd_r[j] = kd_src[idx];
            kr_r[j] = kr_src[idx];
        }
        inv_r = reinterpret_cast<const bf16x4*>(
            ws_inv + int64_t(ht) * C * C)[lane];
        gt_r = reinterpret_cast<const f32x2*>(
            ws_gt + int64_t(ht) * D)[lane];
        #pragma unroll
        for (int j = 0; j < VR; ++j) {
            const int idx = lane + j * 64;
            const int m = idx / BV, vv = idx % BV;
            v_r[j] = m < alen
                ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                : (__bf16)0.0f;
        }
        beta_r = lane < C && lane < alen
            ? sigmoid_tanh(beta_g[int64_t(t0 + lane) * H + h]) : 0.0f;
    };
    auto commit = [&]() {
        #pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int idx = lane + j * 64;
            const int row = idx >> 4, col8 = idx & 15;
            reinterpret_cast<bf16x8*>(kd + row * SD)[col8] = kd_r[j];
            reinterpret_cast<bf16x8*>(kr)[idx] = kr_r[j];
        }
        reinterpret_cast<bf16x4*>(inv)[lane] = inv_r;
        reinterpret_cast<f32x2*>(gtot)[lane] = gt_r;
        #pragma unroll
        for (int j = 0; j < VR; ++j)
            vmat[lane + j * 64] = v_r[j];
        if (lane < C) beta[lane] = beta_r;
    };

    stage(ht_base, t0_base, min(C, seq_len));
    commit();
    __syncthreads();

    for (int nt = 0; nt < nt_eff; ++nt) {
        const int ht = ht_base + nt;
        const int t0 = t0_base + nt * C;
        const int alen = min(C, seq_len - nt * C);
        const bool has_next = nt + 1 < nt_eff;

        // One snapshot per four BT16 chunks (a BT64 output segment).  K6
        // reconstructs the three internal chunk states from U/kr/g_total.
        if ((nt & 3) == 0) {
            int ss;
            if constexpr (VL)
                ss = h * total_segments + segment_prefix[bh / H] + nt / 4;
            else
                ss = bh * ((NT + 3) / 4) + nt / 4;
            #pragma unroll
            for (int vt = 0; vt < NVT; ++vt)
                #pragma unroll
                for (int kt = 0; kt < NKB; ++kt)
                    {
                        const int vv = v0 + vt * C + (lane & 15);
                        const int kk = kt * C + (lane >> 4) * 4;
                        bf16x4 packed;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                            packed[i] = f32_to_bf16(sreg[vt][kt][i]);
                        *reinterpret_cast<bf16x4*>(
                            cs_sin + (int64_t(ss) * D + vv) * D + kk) = packed;
                    }
        }

        if (has_next)
            stage(ht + 1, t0 + C, min(C, seq_len - (nt + 1) * C));

        // beta * (v - kd @ S_in)
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt) {
            f32x4 c = gemm_regB<SD, NKB>(kd, sreg[vt], lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const int vv = vt * C + (lane & 15);
                vmat[m * BV + vv] = f32_to_bf16(
                    (bf16_to_f32(vmat[m * BV + vv]) - c[i]) * beta[m]);
            }
        }
        __syncthreads();

        // U = INV @ beta*(v-kd@S); materialize it for the parallel output.
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt) {
            f32x4 c = BV == C ? mm_std_16_tr(inv, vmat, lane)
                              : mm_std_tile_bf16(inv, vmat, vt * C, BV, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = (lane >> 4) * 4 + i;
                const int vv = vt * C + (lane & 15);
                const __bf16 u = f32_to_bf16(c[i]);
                umat[m * BV + vv] = u;
                cs_u[(int64_t(ht) * C + m) * D + v0 + vv] = u;
            }
        }
        __syncthreads();

        // S_out = exp(g_total) * S_in + kr^T @ U.
        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt)
            #pragma unroll
            for (int vt = 0; vt < NVT; ++vt) {
                f32x4 c = BV == C
                    ? mm_cf_trB(kr, D, kt * C, umat, lane)
                    : mm_contract_first_bf16(kr, umat, kt * C, vt * C, D, BV, lane);
                const int kbase = kt * C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    sreg[vt][kt][i] = sreg[vt][kt][i] * ex2(gtot[kbase + i]) + c[i];
            }
        __syncthreads();
        if (has_next) {
            commit();
            __syncthreads();
        }
    }

    if constexpr (HO) {
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt)
            #pragma unroll
            for (int kt = 0; kt < NKB; ++kt)
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + vt * C + (lane & 15);
                    const int kk = kt * C + (lane >> 4) * 4 + i;
                    const int64_t idx = state_base + int64_t(vv) * D + kk;
                    if constexpr (SFP32)
                        reinterpret_cast<float*>(final_state)[idx] = sreg[vt][kt][i];
                    else
                        reinterpret_cast<__bf16*>(final_state)[idx] = f32_to_bf16(sreg[vt][kt][i]);
                }
    }
}

// BT64 K6: one 8-wave CTA owns all V=128 rows of one (sequence, head,
// segment).  Each wave holds V=16 state rows in fp32 registers and replays up
// to four cheap kr^T@U carries to recover the BT16 entry states.
template <bool VL = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_segment_out_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_mqk,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128, SD = D + 4, NKB = D / C;
    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int v0 = wave * C;

    int h, seq_idx = 0, ht_base, ss, t0_base, chunks;
    if constexpr (VL) {
        const int gsi = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (segment_prefix[mid] <= gsi) lo = mid; else hi = mid;
        }
        seq_idx = lo;
        const int local_seg = gsi - segment_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int seq_len = int(cu_seqlens[lo + 1] - bos);
        const int nseg = (seq_len + 63) / 64;
        if (local_seg >= nseg) return;
        const int local_chunk = local_seg * 4;
        ht_base = h * total_tiles + tile_prefix[lo] + local_chunk;
        ss = h * total_segments + gsi;
        t0_base = int(bos) + local_chunk * C;
        chunks = min(4, (seq_len + C - 1) / C - local_chunk);
    } else {
        const int seg = blockIdx.x;
        const int bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        ht_base = bh * NT + seg * 4;
        ss = bh * ((NT + 3) / 4) + seg;
        t0_base = b * T_seq + seg * 64;
        chunks = min(4, NT - seg * 4);
    }

    __shared__ __bf16 qd[C * SD];
    __shared__ __bf16 kr[C * D];
    __shared__ __bf16 umat[C * D];
    __shared__ __bf16 mqk[C * C];
    __shared__ float gtot[D];

    float sreg[NKB][4];
    #pragma unroll
    for (int kt = 0; kt < NKB; ++kt)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = kt * C + (lane >> 4) * 4 + i;
            sreg[kt][i] = bf16_to_f32(
                cs_sin[(int64_t(ss) * D + vv) * D + kk]);
        }

    for (int j = 0; j < chunks; ++j) {
        const int ht = ht_base + j;
        for (int idx = tid; idx < C * D; idx += 512) {
            const int m = idx / D, d = idx % D;
            qd[m * SD + d] = ws_qd[int64_t(ht) * C * D + idx];
            kr[idx] = ws_kr[int64_t(ht) * C * D + idx];
            umat[idx] = cs_u[int64_t(ht) * C * D + idx];
        }
        for (int idx = tid; idx < C * C; idx += 512)
            mqk[idx] = ws_mqk[int64_t(ht) * C * C + idx];
        for (int idx = tid; idx < D; idx += 512)
            gtot[idx] = ws_gt[int64_t(ht) * D + idx];
        __syncthreads();

        f32x4 o1 = gemm_regB<SD, NKB>(qd, sreg, lane);
        f32x4 o2 = mm_std_tile_bf16(mqk, umat, v0, D, lane);
        const int seq_remaining = VL
            ? int(cu_seqlens[seq_idx + 1] - (t0_base + j * C))
            : T_seq - (blockIdx.x * 64 + j * C);
        const int alen = min(C, seq_remaining);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i;
            const int vv = v0 + (lane & 15);
            if (m < alen) {
                const __bf16 a = f32_to_bf16(o1[i]);
                const __bf16 b = f32_to_bf16(o2[i]);
                out_g[(int64_t(t0_base + j * C + m) * H + h) * D + vv] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
        }

        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt) {
            f32x4 c = mm_contract_first_bf16(
                kr, umat, kt * C, v0, D, D, lane);
            const int kbase = kt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = sreg[kt][i] * ex2(gtot[kbase + i]) + c[i];
        }
        __syncthreads();
    }
}

template <int BV, bool VL = false>
__global__ void __launch_bounds__(64)
k2_kda_csplit_out_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_mqk,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        int N, int total_tiles, int T_seq, int H, int NT) {
    constexpr int C = 16, D = 128, SD = D + 4;
    constexpr int NVT = BV / C;
    const int lane = threadIdx.x;
    const int v0 = blockIdx.z * BV;
    int h, ht, t0, alen;

    if constexpr (VL) {
        const int gti = blockIdx.x;
        h = blockIdx.y;
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (tile_prefix[mid] <= gti) lo = mid; else hi = mid;
        }
        const int local_nt = gti - tile_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int seq_len = int(cu_seqlens[lo + 1] - bos);
        if (local_nt >= (seq_len + C - 1) / C) return;
        ht = h * total_tiles + gti;
        t0 = int(bos) + local_nt * C;
        alen = min(C, seq_len - local_nt * C);
    } else {
        const int nt = blockIdx.x;
        const int bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        ht = bh * NT + nt;
        t0 = b * T_seq + nt * C;
        alen = min(C, T_seq - nt * C);
    }

    __shared__ __bf16 sin[BV * SD];
    __shared__ __bf16 qd[C * SD];
    __shared__ __bf16 umat[C * BV];
    __shared__ __bf16 mqk[C * C];
    copy_bf16_rows(sin, SD, cs_sin + (int64_t(ht) * D + v0) * D,
                   D, BV, D, lane);
    copy_bf16_rows(qd, SD, ws_qd + int64_t(ht) * C * D, D, C, D, lane);
    copy_bf16_vec(mqk, ws_mqk + int64_t(ht) * C * C, C * C, lane);
    for (int idx = lane; idx < C * BV; idx += 64) {
        const int m = idx / BV, vv = idx % BV;
        umat[idx] = cs_u[(int64_t(ht) * C + m) * D + v0 + vv];
    }
    __syncthreads();

    #pragma unroll
    for (int vt = 0; vt < NVT; ++vt) {
        f32x4 o1 = gemm_contract_last<__bf16, D, SD>(qd, sin + vt * C * SD, lane);
        f32x4 o2 = mm_std_tile_bf16(mqk, umat, vt * C, BV, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = (lane >> 4) * 4 + i;
            const int vv = vt * C + (lane & 15);
            if (m < alen) {
                const __bf16 a = f32_to_bf16(o1[i]);
                const __bf16 b = f32_to_bf16(o2[i]);
                out_g[(int64_t(t0 + m) * H + h) * D + v0 + vv] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
        }
    }
}

}  // namespace flashkda_hip
