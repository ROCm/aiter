// FlashKDA BT32 C-split recurrence and output kernels for gfx942.
//
// Adjacent BT16 K1 tiles are coalesced into one 32-token recurrence step.  Four
// waves split K=128 into K32 slices, cutting the serial scan length and the
// number of CTA-wide barriers in half while retaining an fp32 register state.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

template <bool HI = false, bool HO = false, bool SFP32 = false, bool VL = false>
__global__ void __launch_bounds__(256)
k2_kda_csplit_bt32_scan_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ cross_inv10,
        __bf16* __restrict__ cs_u,
        __bf16* __restrict__ cs_sin,
        const void* __restrict__ init_state,
        void* __restrict__ final_state,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ pair_prefix,
        const int* __restrict__ segment_prefix,
        int total_tiles, int total_pairs, int total_segments,
        int T_seq, int H, int NT) {
    constexpr int C = 16, BT = 32, D = 128, BV = 16, SD = D + 4;
    constexpr int NW = 4, KTW = 2;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const int bh = blockIdx.x, v0 = blockIdx.y * BV;

    int h, seq_len, np, ht_base, xp_base, t0_base, seg_base;
    if constexpr (VL) {
        const int seq = bh / H;
        h = bh % H;
        const int64_t bos = cu_seqlens[seq];
        seq_len = int(cu_seqlens[seq + 1] - bos);
        np = (seq_len + BT - 1) / BT;
        ht_base = h * total_tiles + tile_prefix[seq];
        xp_base = h * total_pairs + pair_prefix[seq];
        seg_base = h * total_segments + segment_prefix[seq];
        t0_base = int(bos);
    } else {
        const int b = bh / H;
        h = bh % H;
        seq_len = T_seq;
        np = (NT + 1) / 2;
        ht_base = bh * NT;
        xp_base = bh * np;
        seg_base = bh * ((NT + 3) / 4);
        t0_base = b * T_seq;
    }

    __shared__ __bf16 kd[BT * SD], kr[BT * D];
    __shared__ __bf16 rmat[BT * BV], umat[BT * BV];
    __shared__ __bf16 c00[C * C], c11[C * C], c10[C * C];
    __shared__ float gt0[D], gt1[D], gtp[D], beta[BT];
    __shared__ float partial[NW * BT * BV];

    float sreg[KTW][4];
    if (np == 0) return;
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

    // One-pair software pipeline.  Global reads for pair p+1 are issued into
    // VGPRs before pair p's MFMA work, then committed to LDS at the boundary.
    bf16x8 kd_r[2], kr_r[2], c_r;
    f32x4 gt0_r, gt1_r;
    __bf16 v_r[2];
    float beta_r;
    auto stage = [&](int p) {
        const int ht0 = ht_base + p * 2;
        const int xp = xp_base + p;
        const int t0 = t0_base + p * BT;
        const int alen = min(BT, seq_len - p * BT);
        const bool second = alen > C;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int vi = tid + j * 256;
            if (vi < (BT * D) / 8) {
                const int m = vi >> 4;
                kd_r[j] = m < alen
                    ? reinterpret_cast<const bf16x8*>(
                        ws_kd + int64_t(ht0) * C * D)[vi] : bf16x8{};
                kr_r[j] = m < alen
                    ? reinterpret_cast<const bf16x8*>(
                        ws_kr + int64_t(ht0) * C * D)[vi] : bf16x8{};
            }
            const int ei = tid + j * 256;
            if (ei < BT * BV) {
                const int m = ei / BV, vv = ei % BV;
                v_r[j] = m < alen
                    ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                    : (__bf16)0.0f;
            }
        }
        if (tid < 3 * (C * C) / 8) {
            if (tid < (C * C) / 8)
                c_r = reinterpret_cast<const bf16x8*>(
                    ws_inv + int64_t(ht0) * C * C)[tid];
            else if (tid < 2 * (C * C) / 8)
                c_r = second ? reinterpret_cast<const bf16x8*>(
                    ws_inv + int64_t(ht0 + 1) * C * C)[tid - (C * C) / 8]
                    : bf16x8{};
            else
                c_r = second ? reinterpret_cast<const bf16x8*>(
                    cross_inv10 + int64_t(xp) * C * C)[tid - 2 * (C * C) / 8]
                    : bf16x8{};
        }
        if (tid < D / 4) {
            gt0_r = reinterpret_cast<const f32x4*>(
                ws_gt + int64_t(ht0) * D)[tid];
            gt1_r = second ? reinterpret_cast<const f32x4*>(
                ws_gt + int64_t(ht0 + 1) * D)[tid] : f32x4{};
        }
        if (tid < BT)
            beta_r = tid < alen
                ? sigmoid_tanh(beta_g[int64_t(t0 + tid) * H + h]) : 0.0f;
    };
    auto commit_meta = [&]() {
        if (tid < D / 4) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int d = tid * 4 + i;
                gt0[d] = ex2(gt0_r[i]);
                gt1[d] = ex2(gt1_r[i]);
                gtp[d] = ex2(gt0_r[i] + gt1_r[i]);
            }
        }
        if (tid < (C * C) / 8)
            reinterpret_cast<bf16x8*>(c00)[tid] = c_r;
        else if (tid < 2 * (C * C) / 8)
            reinterpret_cast<bf16x8*>(c11)[tid - (C * C) / 8] = c_r;
        else if (tid < 3 * (C * C) / 8)
            reinterpret_cast<bf16x8*>(c10)[tid - 2 * (C * C) / 8] = c_r;
        if (tid < BT) beta[tid] = beta_r;
    };
    auto commit_data = [&](bool second) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int vi = tid + j * 256;
            if (vi < (BT * D) / 8) {
                const int m = vi >> 4, d0 = (vi & 15) * 8;
                bf16x8 xk = kd_r[j], xr = kr_r[j];
                if (m < C && second) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                        xr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * gt1[d0 + i]);
                }
                reinterpret_cast<bf16x8*>(kd + m * SD)[vi & 15] = xk;
                reinterpret_cast<bf16x8*>(kr)[vi] = xr;
            }
            const int ei = tid + j * 256;
            if (ei < BT * BV) rmat[ei] = v_r[j];
        }
    };

    stage(0);
    commit_meta();
    __syncthreads();
    commit_data(min(BT, seq_len) > C);
    __syncthreads();

    for (int p = 0; p < np; ++p) {
        const int ht0 = ht_base + p * 2;
        const int xp = xp_base + p;
        const int t0 = t0_base + p * BT;
        const int alen = min(BT, seq_len - p * BT);
        const bool second = alen > C;

        if ((p & 1) == 0) {
            const int ss = seg_base + p / 2;
            #pragma unroll
            for (int kt = 0; kt < KTW; ++kt) {
                const int vv = v0 + (lane & 15);
                const int kk = (wave * KTW + kt) * C + (lane >> 4) * 4;
                bf16x4 x;
                #pragma unroll
                for (int i = 0; i < 4; ++i) x[i] = f32_to_bf16(sreg[kt][i]);
                *reinterpret_cast<bf16x4*>(
                    cs_sin + (int64_t(ss) * D + vv) * D + kk) = x;
            }
        }
        const bool has_next = p + 1 < np;
        if (has_next) stage(p + 1);

        f32x4 p0 = gemm_regB<SD, KTW>(kd + wave * KTW * C, sreg, lane);
        f32x4 p1 = gemm_regB<SD, KTW>(kd + C * SD + wave * KTW * C, sreg, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = (lane >> 4) * 4 + i, vv = lane & 15;
            partial[(wave * BT + r) * BV + vv] = p0[i];
            partial[(wave * BT + C + r) * BV + vv] = p1[i];
        }
        __syncthreads();

        if (wave == 0) {
            #pragma unroll
            for (int half = 0; half < 2; ++half)
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = half * C + (lane >> 4) * 4 + i;
                    const int vv = lane & 15;
                    float sum = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < NW; ++w)
                        sum += partial[(w * BT + r) * BV + vv];
                    rmat[r * BV + vv] = f32_to_bf16(
                        (bf16_to_f32(rmat[r * BV + vv]) - sum) * beta[r]);
                }
        }
        __syncthreads();

        if (wave == 0) {
            f32x4 u = mm_std_16_tr(c00, rmat, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                const __bf16 x = f32_to_bf16(u[i]);
                umat[r * BV + vv] = x;
                cs_u[(int64_t(ht0) * C + r) * D + v0 + vv] = x;
            }
        } else if (wave == 1 && second) {
            f32x4 a = mm_std_16_tr(c10, rmat, lane);
            f32x4 b = mm_std_16_tr(c11, rmat + C * BV, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                const __bf16 x = f32_to_bf16(a[i] + b[i]);
                umat[(C + r) * BV + vv] = x;
                cs_u[(int64_t(ht0 + 1) * C + r) * D + v0 + vv] = x;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kt = 0; kt < KTW; ++kt) {
            const int gkt = wave * KTW + kt;
            f32x4 a = mm_cf_trB(kr, D, gkt * C, umat, lane);
            f32x4 b = second
                ? mm_cf_trB(kr + C * D, D, gkt * C, umat + C * BV, lane)
                : f32x4{0.f, 0.f, 0.f, 0.f};
            const int kb = gkt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = sreg[kt][i] * gtp[kb + i] + a[i] + b[i];
        }
        __syncthreads();
        if (has_next) {
            const int next_alen = min(BT, seq_len - (p + 1) * BT);
            commit_meta();
            __syncthreads();
            commit_data(next_alen > C);
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

// One BT64 output CTA: eight waves own V16 each and consume two BT32 pairs.
template <bool VL = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt32_segment_out_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_mqk,
        const __bf16* __restrict__ cross_mqk10,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ pair_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_pairs, int total_segments,
        int T_seq, int H, int NT) {
    constexpr int C = 16, BT = 32, D = 128, SD = D + 4, NKB = D / C;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const int v0 = wave * C;

    int h, seq_idx = 0, ht_base, xp_base, ss, t0_base, pairs, seq_len;
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
        seq_len = int(cu_seqlens[lo + 1] - bos);
        if (local_seg >= (seq_len + 63) / 64) return;
        const int local_pair = local_seg * 2;
        ht_base = h * total_tiles + tile_prefix[lo] + local_pair * 2;
        xp_base = h * total_pairs + pair_prefix[lo] + local_pair;
        ss = h * total_segments + gsi;
        t0_base = int(bos) + local_pair * BT;
        pairs = min(2, (seq_len + BT - 1) / BT - local_pair);
    } else {
        const int seg = blockIdx.x, bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        seq_len = T_seq;
        ht_base = bh * NT + seg * 4;
        xp_base = bh * ((NT + 1) / 2) + seg * 2;
        ss = bh * ((NT + 3) / 4) + seg;
        t0_base = b * T_seq + seg * 64;
        pairs = min(2, (NT + 1) / 2 - seg * 2);
    }

    __shared__ __bf16 qd[BT * SD], kr[BT * D], umat[BT * D];
    __shared__ __bf16 m00[C * C], m11[C * C], m10[C * C];
    __shared__ float gt0[D], gt1[D], gtp[D];

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

    for (int p = 0; p < pairs; ++p) {
        const int ht0 = ht_base + p * 2, xp = xp_base + p;
        const int t0 = t0_base + p * BT;
        const int alen = min(BT, seq_len - (t0 - (VL ? int(cu_seqlens[seq_idx])
                                                         : (t0_base - (blockIdx.x * 64)))));
        const bool second = alen > C;

        for (int d = tid; d < D; d += 512) {
            const float a = ws_gt[int64_t(ht0) * D + d];
            const float b = second ? ws_gt[int64_t(ht0 + 1) * D + d] : 0.0f;
            gt0[d] = ex2(a);
            gt1[d] = ex2(b);
            gtp[d] = ex2(a + b);
        }
        for (int i = tid; i < C * C; i += 512) {
            m00[i] = ws_mqk[int64_t(ht0) * C * C + i];
            m11[i] = second ? ws_mqk[int64_t(ht0 + 1) * C * C + i]
                            : (__bf16)0.0f;
            m10[i] = second ? cross_mqk10[int64_t(xp) * C * C + i]
                            : (__bf16)0.0f;
        }
        __syncthreads();

        for (int idx = tid; idx < BT * D; idx += 512) {
            const int m = idx / D, d = idx % D, cm = m & 15;
            if (m < alen) {
                __bf16 q = ws_qd[(int64_t(ht0 + (m >> 4)) * C + cm) * D + d];
                __bf16 k = ws_kr[(int64_t(ht0 + (m >> 4)) * C + cm) * D + d];
                if (m >= C)
                    q = f32_to_bf16(bf16_to_f32(q) * gt0[d]);
                else if (second)
                    k = f32_to_bf16(bf16_to_f32(k) * gt1[d]);
                qd[m * SD + d] = q;
                kr[m * D + d] = k;
                umat[m * D + d] = cs_u[
                    (int64_t(ht0 + (m >> 4)) * C + cm) * D + d];
            } else {
                qd[m * SD + d] = (__bf16)0.0f;
                kr[m * D + d] = (__bf16)0.0f;
                umat[m * D + d] = (__bf16)0.0f;
            }
        }
        __syncthreads();

        f32x4 o00 = gemm_regB<SD, NKB>(qd, sreg, lane);
        f32x4 o01 = mm_std_tile_bf16(m00, umat, v0, D, lane);
        f32x4 o10 = gemm_regB<SD, NKB>(qd + C * SD, sreg, lane);
        f32x4 o11a = mm_std_tile_bf16(m10, umat, v0, D, lane);
        f32x4 o11b = mm_std_tile_bf16(m11, umat + C * D, v0, D, lane);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = (lane >> 4) * 4 + i, vv = v0 + (lane & 15);
            if (r < alen) {
                const __bf16 a = f32_to_bf16(o00[i]);
                const __bf16 b = f32_to_bf16(o01[i]);
                out_g[(int64_t(t0 + r) * H + h) * D + vv] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
            if (C + r < alen) {
                const __bf16 a = f32_to_bf16(o10[i]);
                const __bf16 b = f32_to_bf16(o11a[i] + o11b[i]);
                out_g[(int64_t(t0 + C + r) * H + h) * D + vv] =
                    f32_to_bf16(bf16_to_f32(a) + bf16_to_f32(b));
            }
        }

        #pragma unroll
        for (int kt = 0; kt < NKB; ++kt) {
            f32x4 a = mm_contract_first_bf16(kr, umat, kt * C, v0, D, D, lane);
            f32x4 b = second
                ? mm_contract_first_bf16(kr + C * D, umat + C * D,
                                         kt * C, v0, D, D, lane)
                : f32x4{0.f, 0.f, 0.f, 0.f};
            const int kb = kt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = sreg[kt][i] * gtp[kb + i] + a[i] + b[i];
        }
        __syncthreads();
    }
}

}  // namespace flashkda_hip
