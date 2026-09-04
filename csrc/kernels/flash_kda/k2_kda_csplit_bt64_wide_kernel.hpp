// RTP BT64 serial scan for gfx942: BV32, two K64 state slabs, four waves.
//
// This keeps the direct RTP algebra
//
//   R  = beta * (V - Kd @ H^T)
//   V' = C @ R,                        C = (I + L)^-1
//   H' = decay * H + Kr_suffix^T @ V'
//
// entirely on chip.  Unlike the earlier V16/K32 implementation, a wave owns
// a complete 16x32 output row tile and reads the published K128xV32 state.
// Consequently Kd@H needs no cross-wave partial buffer or reduction.
#pragma once
#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

__device__ __forceinline__ constexpr int bt64_wide_tri_tile(int r, int c) {
    return r * (r + 1) / 2 + c;
}

template <int Kd, int LDA, int LDB>
__device__ __forceinline__ f32x4 bt64_wide_mm_std_k(
        const __bf16* __restrict__ a,
        const __bf16* __restrict__ b,
        int n0, int lane) {
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
    f32x4 c = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int k0 = 0; k0 < Kd; k0 += 16) {
        bf16x4 af, bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            af[i] = a[row * LDA + k0 + kb + i];
            bf[i] = b[(k0 + kb + i) * LDB + n0 + row];
        }
        c = mfma_bf16(af, bf, c);
    }
    return c;
}

template <bool HI = false, bool HO = false,
          bool SFP32 = false, bool VL = false>
__global__ void __launch_bounds__(256)
k2_kda_csplit_bt64_wide_kernel(
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
    constexpr int C = 16, BT = 64, D = 128, BV = 32, SD = D + 4;
    constexpr int NK64 = 2, NVT = 2;
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

    // kd is reused for suffix-scaled kr after Kd@H.  The 8 KiB phase pool is
    // H^T[K,V] first, then aliases R[BT,BV] and V'[BT,BV].
    __shared__ __bf16 kd[BT * SD];
    __shared__ __bf16 phase[2 * BT * BV];
    __shared__ __bf16 cinv[10 * C * C];
    __shared__ float decay[4 * D], beta[BT];
    __bf16* h_t = phase;
    __bf16* rmat = phase;
    __bf16* umat = phase + BT * BV;

    // Each wave owns K16 in each K64 slab and both V16 tiles.
    float hs[NK64][NVT][4];
    const int64_t state_base = int64_t(bh) * D * D;
    #pragma unroll
    for (int bk = 0; bk < NK64; ++bk)
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + vt * C + (lane & 15);
                const int kk = bk * 64 + wave * C + (lane >> 4) * 4 + i;
                const int64_t idx = state_base + int64_t(vv) * D + kk;
                if constexpr (HI)
                    hs[bk][vt][i] = SFP32
                        ? reinterpret_cast<const float*>(init_state)[idx]
                        : bf16_to_f32(reinterpret_cast<const __bf16*>(init_state)[idx]);
                else
                    hs[bk][vt][i] = 0.0f;
            }

    for (int s = 0; s < ns; ++s) {
        const int ht0 = ht_base + s * 4;
        const int xp0 = xp_base + s * 2;
        const int xs = xs_base + s;
        const int t0 = t0_base + s * BT;
        const int alen = min(BT, seq_len - s * BT);
        const int nch = (alen + C - 1) / C;

        // Publish the pre-update state and its [K,V] BF16 MFMA operand.
        #pragma unroll
        for (int bk = 0; bk < NK64; ++bk)
            #pragma unroll
            for (int vt = 0; vt < NVT; ++vt)
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int lv = vt * C + (lane & 15);
                    const int kk = bk * 64 + wave * C + (lane >> 4) * 4 + i;
                    const __bf16 x = f32_to_bf16(hs[bk][vt][i]);
                    h_t[kk * BV + lv] = x;
                    cs_sin[(int64_t(xs) * D + v0 + lv) * D + kk] = x;
                }

        // Load all four BT16 Kd rows, the tiled C64 inverse, beta, and decay.
        for (int vi = tid; vi < (BT * D) / 8; vi += 256) {
            const int m = vi >> 4;
            const bf16x8 x = m < alen
                ? reinterpret_cast<const bf16x8*>(
                    ws_kd + int64_t(ht0) * C * D)[vi] : bf16x8{};
            reinterpret_cast<bf16x8*>(kd + m * SD)[vi & 15] = x;
        }
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
        __syncthreads();

        // A wave owns token rows wave*16..wave*16+15 and both V16 tiles.
        f32x4 kh[NVT];
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt)
            kh[vt] = bt64_wide_mm_std_k<D, SD, BV>(
                kd + wave * C * SD, h_t, vt * C, lane);
        // h_t and rmat deliberately alias the phase pool.  Do not let one
        // wave publish R until every wave has finished consuming H^T.
        __syncthreads();
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = wave * C + (lane >> 4) * 4 + i;
                const int lv = vt * C + (lane & 15);
                const float vv = m < alen
                    ? bf16_to_f32(v_g[(int64_t(t0 + m) * H + h) * D + v0 + lv])
                    : 0.0f;
                rmat[m * BV + lv] = f32_to_bf16((vv - kh[vt][i]) * beta[m]);
            }
        __syncthreads();

        // Apply the already-computed C=(I+L)^-1; this is a block GEMM, not a
        // second inverse or triangular solve.
        #pragma unroll
        for (int vt = 0; vt < NVT; ++vt) {
            f32x4 u = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j <= wave) {
                    f32x4 x = mm_std_tile_bf16(
                        cinv + bt64_wide_tri_tile(wave, j) * C * C,
                        rmat + j * C * BV, vt * C, BV, lane);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) u[i] += x[i];
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = wave * C + (lane >> 4) * 4 + i;
                const int lv = vt * C + (lane & 15);
                const __bf16 x = f32_to_bf16(u[i]);
                umat[m * BV + lv] = x;
                // K6 replays complete BT16 tiles.  Materialize the padded
                // rows of the final live tile as zeros instead of leaving
                // stale workspace data behind.
                if (wave < nch)
                    cs_u[(int64_t(ht0) * C + m) * D + v0 + lv] = x;
            }
        }
        __syncthreads();

        // Reuse kd storage for Kr with the later-chunk suffix decay folded in.
        for (int vi = tid; vi < (BT * D) / 8; vi += 256) {
            const int m = vi >> 4, d0 = (vi & 15) * 8, chunk = m >> 4;
            bf16x8 x{};
            if (m < alen) {
                x = reinterpret_cast<const bf16x8*>(
                    ws_kr + int64_t(ht0) * C * D)[vi];
                if (chunk < 3) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                        x[i] = f32_to_bf16(
                            bf16_to_f32(x[i]) * decay[(chunk + 1) * D + d0 + i]);
                }
            }
            reinterpret_cast<bf16x8*>(kd + m * SD)[vi & 15] = x;
        }
        __syncthreads();

        // Kr_suffix^T @ V': each wave owns K16 in both K64 slabs and V32.
        #pragma unroll
        for (int bk = 0; bk < NK64; ++bk)
            #pragma unroll
            for (int vt = 0; vt < NVT; ++vt) {
                f32x4 carry = {0.f, 0.f, 0.f, 0.f};
                #pragma unroll
                for (int rb = 0; rb < 4; ++rb) {
                    f32x4 x = mm_contract_first_bf16(
                        kd + rb * C * SD, umat + rb * C * BV,
                        bk * 64 + wave * C, vt * C, SD, BV, lane);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) carry[i] += x[i];
                }
                const int kb = bk * 64 + wave * C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    hs[bk][vt][i] = hs[bk][vt][i] * decay[kb + i] + carry[i];
            }
        __syncthreads();
    }

    if constexpr (HO) {
        #pragma unroll
        for (int bk = 0; bk < NK64; ++bk)
            #pragma unroll
            for (int vt = 0; vt < NVT; ++vt)
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int vv = v0 + vt * C + (lane & 15);
                    const int kk = bk * 64 + wave * C + (lane >> 4) * 4 + i;
                    const int64_t idx = state_base + int64_t(vv) * D + kk;
                    if constexpr (SFP32)
                        reinterpret_cast<float*>(final_state)[idx] = hs[bk][vt][i];
                    else
                        reinterpret_cast<__bf16*>(final_state)[idx] = f32_to_bf16(hs[bk][vt][i]);
                }
    }
}

}  // namespace flashkda_hip
