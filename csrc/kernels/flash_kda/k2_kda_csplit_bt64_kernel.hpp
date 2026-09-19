// Native BT64 recurrent scan for gfx942.
//
// One 8-wave CTA owns V16.  Each wave owns K16 of the fp32 recurrent state;
// waves 0..3 independently solve one 16-token row block of the 64x64
// lower-triangular system.  One recurrence step advances a full BT64 segment;
// output is produced later by the existing BT16 replay kernel.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

__device__ __forceinline__ constexpr int bt64_tri_tile(int r, int c) {
    return r * (r + 1) / 2 + c;
}

template <int NW = 4, bool HI = false, bool HO = false,
          bool SFP32 = false, bool VL = false>
__global__ void __launch_bounds__(NW * 64)
k2_kda_csplit_bt64_scan_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ cross32,
        const __bf16* __restrict__ cross64,
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
    constexpr int C = 16, BT = 64, D = 128, BV = 16, SD = D + 4;
    static_assert(NW == 4 || NW == 8);
    constexpr int KTW = 8 / NW;
    constexpr int NTH = NW * 64;
    constexpr int RW = ((BT * D) / 8 + NTH - 1) / NTH;
    constexpr int VR = (BT * BV + NTH - 1) / NTH;
    constexpr int CIR = ((10 * C * C) / 8 + NTH - 1) / NTH;
    const int tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
    const int bh = blockIdx.x, v0 = blockIdx.y * BV;

    int h, seq_len, ns, ht_base, xp_base, xs_base, t0_base;
    if constexpr (VL) {
        const int seq = bh / H;
        h = bh % H;
        const int64_t bos = cu_seqlens[seq];
        seq_len = int(cu_seqlens[seq + 1] - bos);
        ns = (seq_len + BT - 1) / BT;
        ht_base = h * total_tiles + tile_prefix[seq];
        xp_base = h * total_pairs + pair_prefix[seq];
        xs_base = h * total_segments + segment_prefix[seq];
        t0_base = int(bos);
    } else {
        const int b = bh / H;
        h = bh % H;
        seq_len = T_seq;
        ns = (NT + 3) / 4;
        ht_base = bh * NT;
        xp_base = bh * ((NT + 1) / 2);
        xs_base = bh * ns;
        t0_base = b * T_seq;
    }
    if (ns == 0) return;

    __shared__ __bf16 kd[BT * SD], kr[BT * D];
    __shared__ __bf16 rmat[BT * BV], umat[BT * BV];
    __shared__ __bf16 cinv[10 * C * C];
    __shared__ float decay[4 * D], beta[BT];
    // Both layouts reserve 16 KiB.  NW=4 keeps four K32 partials in FP32;
    // NW=8 keeps eight K16 partials in BF16 so all 64 rows can be staged and
    // reduced with one CTA barrier instead of synchronizing per row tile.
    __shared__ uint32_t partial_storage[4096];

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

    // Full-segment software pipeline.  LDS already limits this kernel to one
    // CTA/CU, so use the available VGPR budget to overlap segment n+1's HBM
    // reads with segment n's MFMA chain.
    bf16x8 kd_r[RW], kr_r[RW], ci_r[CIR];
    f32x4 gt_r;
    __bf16 v_r[VR];
    float beta_r;
    auto stage = [&](int s) {
        const int ht0 = ht_base + s * 4;
        const int xp0 = xp_base + s * 2;
        const int xs = xs_base + s;
        const int t0 = t0_base + s * BT;
        const int alen = min(BT, seq_len - s * BT);
        const int nch = (alen + C - 1) / C;
        #pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int vi = tid + j * NTH;
            if (vi < (BT * D) / 8) {
                const int m = vi >> 4;
                kd_r[j] = m < alen
                    ? reinterpret_cast<const bf16x8*>(
                        ws_kd + int64_t(ht0) * C * D)[vi] : bf16x8{};
                kr_r[j] = m < alen
                    ? reinterpret_cast<const bf16x8*>(
                        ws_kr + int64_t(ht0) * C * D)[vi] : bf16x8{};
            }
        }
        #pragma unroll
        for (int j = 0; j < VR; ++j) {
            const int ei = tid + j * NTH;
            if (ei < BT * BV) {
                const int m = ei / BV, vv = ei % BV;
                v_r[j] = m < alen
                    ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                    : (__bf16)0.0f;
            }
        }
        #pragma unroll
        for (int j = 0; j < CIR; ++j) {
            const int vi = tid + j * NTH;
            if (vi < (10 * C * C) / 8) {
                const int tile = vi / ((C * C) / 8);
                const int e8 = vi % ((C * C) / 8);
                bf16x8 x{};
                if (tile == 0)
                    x = reinterpret_cast<const bf16x8*>(
                        ws_inv + int64_t(ht0) * C * C)[e8];
                else if (tile == 1 && nch > 1)
                    x = reinterpret_cast<const bf16x8*>(
                        cross32 + int64_t(xp0) * C * C)[e8];
                else if (tile == 2 && nch > 1)
                    x = reinterpret_cast<const bf16x8*>(
                        ws_inv + int64_t(ht0 + 1) * C * C)[e8];
                else if ((tile == 3 || tile == 4) && nch > 2)
                    x = reinterpret_cast<const bf16x8*>(
                        cross64 + (int64_t(xs) * 4 + tile - 3) * C * C)[e8];
                else if (tile == 5 && nch > 2)
                    x = reinterpret_cast<const bf16x8*>(
                        ws_inv + int64_t(ht0 + 2) * C * C)[e8];
                else if ((tile == 6 || tile == 7) && nch > 3)
                    x = reinterpret_cast<const bf16x8*>(
                        cross64 + (int64_t(xs) * 4 + tile - 4) * C * C)[e8];
                else if (tile == 8 && nch > 3)
                    x = reinterpret_cast<const bf16x8*>(
                        cross32 + int64_t(xp0 + 1) * C * C)[e8];
                else if (tile == 9 && nch > 3)
                    x = reinterpret_cast<const bf16x8*>(
                        ws_inv + int64_t(ht0 + 3) * C * C)[e8];
                ci_r[j] = x;
            }
        }
        if (tid < D) {
            gt_r = f32x4{
                ws_gt[int64_t(ht0) * D + tid],
                nch > 1 ? ws_gt[int64_t(ht0 + 1) * D + tid] : 0.0f,
                nch > 2 ? ws_gt[int64_t(ht0 + 2) * D + tid] : 0.0f,
                nch > 3 ? ws_gt[int64_t(ht0 + 3) * D + tid] : 0.0f};
        }
        if (tid < BT)
            beta_r = tid < alen
                ? sigmoid_tanh(beta_g[int64_t(t0 + tid) * H + h]) : 0.0f;
    };
    auto commit_meta = [&]() {
        if (tid < D) {
            decay[tid]         = ex2(gt_r[0] + gt_r[1] + gt_r[2] + gt_r[3]);
            decay[D + tid]     = ex2(gt_r[1] + gt_r[2] + gt_r[3]);
            decay[2 * D + tid] = ex2(gt_r[2] + gt_r[3]);
            decay[3 * D + tid] = ex2(gt_r[3]);
        }
        #pragma unroll
        for (int j = 0; j < CIR; ++j) {
            const int vi = tid + j * NTH;
            if (vi < (10 * C * C) / 8)
                reinterpret_cast<bf16x8*>(cinv)[vi] = ci_r[j];
        }
        if (tid < BT) beta[tid] = beta_r;
    };
    auto commit_data = [&]() {
        #pragma unroll
        for (int j = 0; j < RW; ++j) {
            const int vi = tid + j * NTH;
            if (vi < (BT * D) / 8) {
                const int m = vi >> 4;
                reinterpret_cast<bf16x8*>(kd + m * SD)[vi & 15] = kd_r[j];
                reinterpret_cast<bf16x8*>(kr)[vi] = kr_r[j];
            }
        }
        #pragma unroll
        for (int j = 0; j < VR; ++j) {
            const int ei = tid + j * NTH;
            if (ei < BT * BV) rmat[ei] = v_r[j];
        }
    };

    stage(0);
    commit_meta();
    __syncthreads();
    commit_data();
    __syncthreads();

    for (int s = 0; s < ns; ++s) {
        const int ht0 = ht_base + s * 4;
        const int xp0 = xp_base + s * 2;
        const int xs = xs_base + s;
        const int t0 = t0_base + s * BT;
        const int alen = min(BT, seq_len - s * BT);
        const int nch = (alen + C - 1) / C;

        // Save the BT64 entry state for the chunk-parallel output pass.
        #pragma unroll
        for (int kt = 0; kt < KTW; ++kt) {
            const int vv = v0 + (lane & 15);
            const int kk = (wave * KTW + kt) * C + (lane >> 4) * 4;
            bf16x4 x;
            #pragma unroll
            for (int i = 0; i < 4; ++i) x[i] = f32_to_bf16(sreg[kt][i]);
            *reinterpret_cast<bf16x4*>(
                cs_sin + (int64_t(xs) * D + vv) * D + kk) = x;
        }

        const bool has_next = s + 1 < ns;
        if (has_next) stage(s + 1);

        if constexpr (NW == 4) {
            float* partial = reinterpret_cast<float*>(partial_storage);
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                f32x4 p = gemm_regB<SD, KTW>(
                    kd + rb * C * SD + wave * KTW * C, sreg, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                    partial[(wave * BT + rb * C + r) * BV + vv] = p[i];
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                float sum = 0.0f;
                #pragma unroll
                for (int w = 0; w < NW; ++w)
                    sum += partial[(w * BT + wave * C + r) * BV + vv];
                rmat[(wave * C + r) * BV + vv] = f32_to_bf16(
                    (bf16_to_f32(rmat[(wave * C + r) * BV + vv]) - sum)
                    * beta[wave * C + r]);
            }
            __syncthreads();
        } else {
            // K16/wave: stage all four token tiles, then reduce them at once.
            __bf16* partial = reinterpret_cast<__bf16*>(partial_storage);
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                f32x4 p = gemm_regB<SD, KTW>(
                    kd + rb * C * SD + wave * C, sreg, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                    partial[(wave * BT + rb * C + r) * BV + vv] =
                        f32_to_bf16(p[i]);
                }
            }
            __syncthreads();
            if (wave < 4) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                    float sum = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < NW; ++w)
                        sum += bf16_to_f32(
                            partial[(w * BT + wave * C + r) * BV + vv]);
                    rmat[(wave * C + r) * BV + vv] = f32_to_bf16(
                        (bf16_to_f32(rmat[(wave * C + r) * BV + vv]) - sum)
                        * beta[wave * C + r]);
                }
            }
            __syncthreads();
        }

        // Wave r solves output row block r against all preceding RHS blocks.
        if (wave < 4) {
            f32x4 u = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j <= wave) {
                    f32x4 x = mm_std_16_tr(
                        cinv + bt64_tri_tile(wave, j) * C * C,
                        rmat + j * C * BV, lane);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) u[i] += x[i];
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                const __bf16 x = f32_to_bf16(u[i]);
                umat[(wave * C + r) * BV + vv] = x;
                if (wave * C < alen)
                    cs_u[(int64_t(ht0 + wave) * C + r) * D + v0 + vv] = x;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kt = 0; kt < KTW; ++kt) {
            const int gkt = wave * KTW + kt;
            f32x4 c0 = mm_cf_trB(kr, D, gkt * C, umat, lane);
            f32x4 c1 = nch > 1
                ? mm_cf_trB(kr + C * D, D, gkt * C, umat + C * BV, lane)
                : f32x4{0.f, 0.f, 0.f, 0.f};
            f32x4 c2 = nch > 2
                ? mm_cf_trB(kr + 2 * C * D, D, gkt * C, umat + 2 * C * BV, lane)
                : f32x4{0.f, 0.f, 0.f, 0.f};
            f32x4 c3 = nch > 3
                ? mm_cf_trB(kr + 3 * C * D, D, gkt * C, umat + 3 * C * BV, lane)
                : f32x4{0.f, 0.f, 0.f, 0.f};
            const int kb = gkt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = sreg[kt][i] * decay[kb + i]
                    + c0[i] * decay[D + kb + i]
                    + c1[i] * decay[2 * D + kb + i]
                    + c2[i] * decay[3 * D + kb + i] + c3[i];
        }
        __syncthreads();
        if (has_next) {
            commit_meta();
            __syncthreads();
            commit_data();
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

}  // namespace flashkda_hip
