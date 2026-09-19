// Low-LDS RTP BT64 recurrent scan for gfx942: BV16, two K64 slabs, four waves.
//
// A wave owns one BT16 output row block and one K16 state block inside each
// K64 slab.  Publishing the four K16 state blocks as H^T lets every wave form
// a complete Kd@H result, so this mapping has no cross-wave partial reduction:
//
//   R  = beta * (V - Kd @ H^T)
//   V' = C @ R,                         C = (I + L)^-1
//   H' = decay * H + Kr_suffix^T @ V'
//
// R and V' stay transposed in vT[V16][BT64+4].  The phase pool is reused as
// H^T[V16][K64+4] + Kd[BT64][K64+4], the ten packed C tiles, and four
// wave-owned K16 rows of Kr^T[K64][BT64+4].  Static LDS is therefore exactly
// 15,360 bytes.
#pragma once

#include <hip/hip_runtime.h>

#include "mfma.hpp"

namespace flashkda_hip {

__device__ __forceinline__ constexpr int bt64_bv16_tri_tile(int r, int c) {
    return r * (r + 1) / 2 + c;
}

// D[m,n] = sum_k A[m,k] * B[n,k], with independent row pitches.  The second
// operand is deliberately stored transposed: this produces exactly the same
// MFMA fragments as mm_std_16_tr (C@R) and mm_cf_trB (Kr^T@V') without the
// strided-B LDS reads that are expensive on gfx942.
template <int Kd, int LDA, int LDB>
__device__ __forceinline__ f32x4 bt64_bv16_contract_last(
        const __bf16* __restrict__ a,
        const __bf16* __restrict__ b,
        int lane) {
    const int row = lane & 15;
    const int kb = (lane >> 4) * 4;
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll
    for (int k0 = 0; k0 < Kd; k0 += 16) {
        bf16x4 af, bf;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            af[i] = a[row * LDA + k0 + kb + i];
            bf[i] = b[row * LDB + k0 + kb + i];
        }
        acc = mfma_bf16(af, bf, acc);
    }
    return acc;
}

template <bool HI = false, bool HO = false,
          bool SFP32 = false, bool VL = false>
__global__ void __launch_bounds__(256)
k2_kda_csplit_bt64_bv16_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_src,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const float* __restrict__ ws_decay,
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
        int T_seq, int H, int NT, unsigned scan_flags) {
    constexpr int C = 16;
    constexpr int BT = 64;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int NK64 = 2;
    constexpr int SK = 64 + 4;
    constexpr int ST = BT + 4;
    constexpr int PHASE_ELEMS = (BV + BT) * SK;
    constexpr int VT_ELEMS = BV * ST;
    constexpr int META_ELEMS = 4 * D + BT;
    constexpr unsigned USE_DECAY_TABLE = 1u << 0;
    constexpr unsigned BETA_ACTIVATED = 1u << 1;
    static_assert(PHASE_ELEMS * sizeof(__bf16) +
                      VT_ELEMS * sizeof(__bf16) +
                      META_ELEMS * sizeof(float) == 15360,
                  "BT64/BV16 scan must remain within the 15 KiB LDS design");

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int bh = blockIdx.x;
    const int v0 = blockIdx.y * BV;
    const bool use_decay_table = (scan_flags & USE_DECAY_TABLE) != 0;
    const bool beta_is_activated = (scan_flags & BETA_ACTIVATED) != 0;

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

    // phase is retrieve{hT,kd}, C10, or krT.  vT and the scalar metadata live
    // across those aliases.  10*C*C and 64*ST are both smaller than phase.
    __shared__ __bf16 phase[PHASE_ELEMS];
    __shared__ __bf16 vT[VT_ELEMS];
    __shared__ float decay[4 * D];
    __shared__ float beta[BT];
    __bf16* const hT = phase;
    __bf16* const kd = phase + BV * SK;
    __bf16* const cinv = phase;
    __bf16* const krT = phase;

    // A wave owns K16 in both K64 slabs for this V16 tile.
    float hs[NK64][4];
    const int64_t state_base = int64_t(bh) * D * D;
    #pragma unroll
    for (int bk = 0; bk < NK64; ++bk)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = bk * 64 + wave * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_base + int64_t(vv) * D + kk;
            if constexpr (HI)
                hs[bk][i] = SFP32
                    ? reinterpret_cast<const float*>(init_state)[idx]
                    : bf16_to_f32(
                        reinterpret_cast<const __bf16*>(init_state)[idx]);
            else
                hs[bk][i] = 0.0f;
        }

    for (int s = 0; s < ns; ++s) {
        const int ht0 = ht_base + s * 4;
        const int xp0 = xp_base + s * 2;
        const int xs = xs_base + s;
        const int t0 = t0_base + s * BT;
        const int alen = min(BT, seq_len - s * BT);
        const int nch = (alen + C - 1) / C;

        // These metadata arrays are independent of the phase pool.  Missing
        // tail chunks contribute zero log-decay and beta=0.
        if (tid < D) {
            if (use_decay_table) {
                // RTP-K6 leaves prep's per-BT16 decay table in ws_mqk.  Form
                // the four BT64 suffixes with bounded products, eliminating
                // four repeated v_exp operations in each of the eight V CTAs.
                const float e0 = ws_decay[int64_t(ht0) * D + tid];
                const float e1 = nch > 1
                    ? ws_decay[int64_t(ht0 + 1) * D + tid] : 1.0f;
                const float e2 = nch > 2
                    ? ws_decay[int64_t(ht0 + 2) * D + tid] : 1.0f;
                const float e3 = nch > 3
                    ? ws_decay[int64_t(ht0 + 3) * D + tid] : 1.0f;
                const float e23 = e2 * e3;
                const float e123 = e1 * e23;
                decay[tid] = e0 * e123;
                decay[D + tid] = e123;
                decay[2 * D + tid] = e23;
                decay[3 * D + tid] = e3;
            } else {
                const float g0 = ws_gt[int64_t(ht0) * D + tid];
                const float g1 = nch > 1
                    ? ws_gt[int64_t(ht0 + 1) * D + tid] : 0.0f;
                const float g2 = nch > 2
                    ? ws_gt[int64_t(ht0 + 2) * D + tid] : 0.0f;
                const float g3 = nch > 3
                    ? ws_gt[int64_t(ht0 + 3) * D + tid] : 0.0f;
                decay[tid] = ex2(g0 + g1 + g2 + g3);
                decay[D + tid] = ex2(g1 + g2 + g3);
                decay[2 * D + tid] = ex2(g2 + g3);
                decay[3 * D + tid] = ex2(g3);
            }
        }
        if (tid < BT) {
            beta[tid] = tid < alen
                ? (beta_is_activated
                    ? beta_src[int64_t(xs) * BT + tid]
                    : sigmoid_tanh(beta_src[int64_t(t0 + tid) * H + h]))
                : 0.0f;
        }

        // Retrieve Kd@H one K64 slab at a time.  Every wave publishes its K16
        // state slice to H^T; every wave then owns a complete BT16 result.
        f32x4 kh = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int bk = 0; bk < NK64; ++bk) {
            const int lv = lane & 15;
            const int kl = wave * C + (lane >> 4) * 4;
            bf16x4 hpack;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                hpack[i] = f32_to_bf16(hs[bk][i]);
            *reinterpret_cast<bf16x4*>(hT + lv * SK + kl) = hpack;

            // Snapshot is [segment, V, K], stored directly from the fp32 carry
            // before this segment updates it.
            *reinterpret_cast<bf16x4*>(
                cs_sin +
                (int64_t(xs) * D + v0 + lv) * D + bk * 64 + kl) = hpack;

            // 64 rows x K64, padded in LDS.  Form the tile address in int64
            // before multiplying by C/D so large flattened workspaces are safe.
            for (int vi = tid; vi < (BT * 64) / 8; vi += 256) {
                const int m = vi / (64 / 8);
                const int k8 = (vi % (64 / 8)) * 8;
                bf16x8 x{};
                if (m < alen) {
                    const int rb = m >> 4;
                    const int r = m & 15;
                    x = *reinterpret_cast<const bf16x8*>(
                        ws_kd +
                        (int64_t(ht0 + rb) * C + r) * D + bk * 64 + k8);
                }
                *reinterpret_cast<bf16x8*>(kd + m * SK + k8) = x;
            }
            __syncthreads();

            const f32x4 p = gemm_contract_last<__bf16, 64, SK>(
                kd + wave * C * SK, hT, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) kh[i] += p[i];

            // hT/kd are overwritten by the next slab and then by packed C.
            __syncthreads();
        }

        // Publish R transposed while loading the ten packed lower-triangular C
        // tiles into the now-free phase pool.
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = (lane >> 4) * 4 + i;
            const int m = wave * C + r;
            const int vv = lane & 15;
            const float value = m < alen
                ? bf16_to_f32(
                    v_g[(int64_t(t0 + m) * H + h) * D + v0 + vv])
                : 0.0f;
            vT[vv * ST + m] = m < alen
                ? f32_to_bf16((value - kh[i]) * beta[m])
                : (__bf16)0.0f;
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
        __syncthreads();

        // Pull both Kr K64 slabs into registers before C@R.  The loads are
        // independent of the C and beta-R operands, so their VMEM latency can
        // overlap the MFMA work below.  Keep the wave-owned K16 mapping used
        // by Phase D: four packs per slab cover all BT64 rows for this wave.
        bf16x4 kr_prefetch[NK64][4];
        #pragma unroll
        for (int bk = 0; bk < NK64; ++bk)
            #pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
                const int m = tm * C + (lane >> 2);
                const int k4 = wave * C + (lane & 3) * 4;
                bf16x4 x{};
                if (m < alen) {
                    const int rb = m >> 4;
                    const int r = m & 15;
                    x = *reinterpret_cast<const bf16x4*>(
                        ws_kr +
                        (int64_t(ht0 + rb) * C + r) * D + bk * 64 + k4);
                }
                kr_prefetch[bk][tm] = x;
            }

        // Each wave computes one BT16 row block of C@R.  vT still contains all
        // four R blocks, so all waves must finish before any wave overwrites it.
        f32x4 u = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (j <= wave) {
                const f32x4 x = bt64_bv16_contract_last<C, C, ST>(
                    cinv + bt64_bv16_tri_tile(wave, j) * C * C,
                    vT + j * C, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) u[i] += x[i];
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = (lane >> 4) * 4 + i;
            const int m = wave * C + r;
            const int vv = lane & 15;
            const __bf16 x = m < alen ? f32_to_bf16(u[i]) : (__bf16)0.0f;
            vT[vv * ST + m] = x;

            // K6 performs unconditional full-tile loads.  For every live BT16
            // tile, materialize all 16 rows and explicitly zero its tail rows.
            if (wave < nch)
                cs_u[((int64_t(ht0 + wave) * C + r) * D) + v0 + vv] = x;
        }

        // Each wave stages its prefetched K16 rows, then consumes those rows
        // across all four BT16 panels.  K0 publication shares one CTA barrier
        // with the cross-wave V' publication above; the disjoint K1 reuse only
        // needs wave barriers.  Keep the established ordering: raw-bf16 Kr
        // MFMA first, then apply the suffix to the fp32 fragment.
        #pragma unroll
        for (int bk = 0; bk < NK64; ++bk) {
            #pragma unroll
            for (int tm = 0; tm < 4; ++tm) {
                const int m = tm * C + (lane >> 2);
                const int k4 = wave * C + (lane & 3) * 4;
                const bf16x4 x = kr_prefetch[bk][tm];
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    krT[(k4 + i) * ST + m] = x[i];
            }
            if (bk == 0)
                __syncthreads();
            else
                __syncwarp();

            float carry[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int rb = 0; rb < 4; ++rb) {
                const f32x4 c = bt64_bv16_contract_last<C, ST, ST>(
                    krT + wave * C * ST + rb * C,
                    vT + rb * C, lane);
                const int kb =
                    bk * 64 + wave * C + (lane >> 4) * 4;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float suffix = rb < 3
                        ? decay[(rb + 1) * D + kb + i] : 1.0f;
                    carry[i] += c[i] * suffix;
                }
            }

            const int kb = bk * 64 + wave * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                hs[bk][i] = hs[bk][i] * decay[kb + i] + carry[i];

            // The next slab reuses this wave's K16 LDS rows.
            if (bk + 1 < NK64)
                __syncwarp();
        }

        // Segment phases alias the same LDS.  Keep a CTA boundary here so no
        // wave starts publishing the next segment while another still reads
        // the current segment's Kr^T rows.
        __syncthreads();
    }

    if constexpr (HO) {
        #pragma unroll
        for (int bk = 0; bk < NK64; ++bk)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int vv = v0 + (lane & 15);
                const int kk =
                    bk * 64 + wave * C + (lane >> 4) * 4 + i;
                const int64_t idx = state_base + int64_t(vv) * D + kk;
                if constexpr (SFP32)
                    reinterpret_cast<float*>(final_state)[idx] = hs[bk][i];
                else
                    reinterpret_cast<__bf16*>(final_state)[idx] =
                        f32_to_bf16(hs[bk][i]);
            }
    }
}

}  // namespace flashkda_hip
