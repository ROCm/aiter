// Low-LDS BT64 recurrent scan for gfx942.
//
// The full-segment kernel keeps all 64 rows of kd/kr and all K-split partials
// in LDS (~60 KiB), limiting gfx942 to one CTA/CU.  This variant streams one
// 16-row tile through a shared buffer.  Its ~20 KiB LDS footprint permits
// multiple resident CTAs to hide the loss of the segment-ahead VGPR prefetch.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

template <bool HI = false, bool HO = false,
          bool SFP32 = false, bool VL = false>
__global__ void __launch_bounds__(256)
k2_kda_csplit_bt64_stream_kernel(
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
    constexpr int NW = 4, KTW = 2;
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

    __shared__ __bf16 km[C * SD];
    __shared__ __bf16 rmat[BT * BV], umat[BT * BV];
    __shared__ __bf16 cinv[10 * C * C];
    __shared__ float decay[4 * D], beta[BT];
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

    for (int s = 0; s < ns; ++s) {
        const int ht0 = ht_base + s * 4;
        const int xp0 = xp_base + s * 2;
        const int xs = xs_base + s;
        const int t0 = t0_base + s * BT;
        const int alen = min(BT, seq_len - s * BT);
        const int nch = (alen + C - 1) / C;

        // Save this segment's entry state for the parallel output replay.
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

        // Load the ten triangular inverse tiles directly into LDS.
        for (int vi = tid; vi < (10 * C * C) / 8; vi += 256) {
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
            reinterpret_cast<bf16x8*>(cinv)[vi] = x;
        }
        if (tid < D) {
            const float g0 = ws_gt[int64_t(ht0) * D + tid];
            const float g1 = nch > 1 ? ws_gt[int64_t(ht0 + 1) * D + tid] : 0.0f;
            const float g2 = nch > 2 ? ws_gt[int64_t(ht0 + 2) * D + tid] : 0.0f;
            const float g3 = nch > 3 ? ws_gt[int64_t(ht0 + 3) * D + tid] : 0.0f;
            decay[tid]         = ex2(g0 + g1 + g2 + g3);
            decay[D + tid]     = ex2(g1 + g2 + g3);
            decay[2 * D + tid] = ex2(g2 + g3);
            decay[3 * D + tid] = ex2(g3);
        }
        if (tid < BT)
            beta[tid] = tid < alen
                ? sigmoid_tanh(beta_g[int64_t(t0 + tid) * H + h]) : 0.0f;
        for (int ei = tid; ei < BT * BV; ei += 256) {
            const int m = ei / BV, vv = ei % BV;
            rmat[ei] = m < alen
                ? v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv]
                : (__bf16)0.0f;
        }
        __syncthreads();

        // Stream four kd tiles.  The 4 KiB K-split buffer is reused per tile.
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            const bool valid = rb < nch;
            const int vi = tid;
            const int m = vi >> 4;
            reinterpret_cast<bf16x8*>(km + m * SD)[vi & 15] = valid
                ? reinterpret_cast<const bf16x8*>(
                    ws_kd + int64_t(ht0 + rb) * C * D)[vi] : bf16x8{};
            __syncthreads();
            f32x4 p = gemm_regB<SD, KTW>(
                km + wave * KTW * C, sreg, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                partial[(wave * C + r) * BV + vv] = p[i];
            }
            __syncthreads();
            if (wave == 0) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int r = (lane >> 4) * 4 + i, vv = lane & 15;
                    float sum = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < NW; ++w)
                        sum += partial[(w * C + r) * BV + vv];
                    rmat[(rb * C + r) * BV + vv] = f32_to_bf16(
                        (bf16_to_f32(rmat[(rb * C + r) * BV + vv]) - sum)
                        * beta[rb * C + r]);
                }
            }
            __syncthreads();
        }

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
                if (wave < nch)
                    cs_u[(int64_t(ht0 + wave) * C + r) * D + v0 + vv] = x;
            }
        }
        __syncthreads();

        float carry[KTW][4] = {};
        // Reuse km for kr and accumulate the four suffix-scaled contributions.
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            const bool valid = rb < nch;
            const int vi = tid;
            const int m = vi >> 4;
            reinterpret_cast<bf16x8*>(km + m * SD)[vi & 15] = valid
                ? reinterpret_cast<const bf16x8*>(
                    ws_kr + int64_t(ht0 + rb) * C * D)[vi] : bf16x8{};
            __syncthreads();
            #pragma unroll
            for (int kt = 0; kt < KTW; ++kt) {
                const int gkt = wave * KTW + kt;
                f32x4 c = mm_cf_trB(
                    km, SD, gkt * C, umat + rb * C * BV, lane);
                const int kb = gkt * C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float suffix = rb < 3
                        ? decay[(rb + 1) * D + kb + i] : 1.0f;
                    carry[kt][i] += c[i] * suffix;
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for (int kt = 0; kt < KTW; ++kt) {
            const int gkt = wave * KTW + kt;
            const int kb = gkt * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                sreg[kt][i] = sreg[kt][i] * decay[kb + i] + carry[kt][i];
        }
        __syncthreads();
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
