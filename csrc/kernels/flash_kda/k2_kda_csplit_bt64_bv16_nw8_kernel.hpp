// Eight-wave low-LDS RTP BT64 recurrent scan for gfx942.
//
// One CTA still owns a V16 tile, but eight waves split the complete K128
// recurrent state into wave-owned K16 slices.  A segment publishes H^T and Kd
// once at their full K128 width:
//
//   R  = beta * (V - Kd @ H^T)
//   V' = C @ R,                         C = (I + L)^-1
//   H' = decay * H + Kr_suffix^T @ V'
//
// Waves 0..3 own the four BT16 rows for Kd@H and C@R.  All eight waves then
// publish and consume their independent K16 Kr rows to update the full state
// without the two-slab loop used by the four-wave kernel.  Every CTA barrier is
// deliberately outside wave-conditional code.
#pragma once

#include <hip/hip_runtime.h>

#include "mfma.hpp"

namespace flashkda_hip {

__device__ __forceinline__ constexpr int bt64_bv16_nw8_tri_tile(
        int r, int c) {
    return r * (r + 1) / 2 + c;
}

// D[m,n] = sum_k A[m,k] * B[n,k].  The two operands may use independent row
// pitches; B is stored transposed so the MFMA fragments are contiguous in LDS.
template <int Kd, int LDA, int LDB>
__device__ __forceinline__ f32x4 bt64_bv16_nw8_contract_last(
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

template <bool HI, bool HO, bool STATE_IN_FP32, bool STATE_OUT_FP32,
          bool VL, bool SEGMENT_RANGE>
__device__ __forceinline__ void
k2_kda_csplit_bt64_bv16_nw8_body(
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
        int T_seq, int H, int NT, unsigned scan_flags,
        int segment_begin, int segment_count) {
    constexpr int C = 16;
    constexpr int BT = 64;
    constexpr int D = 128;
    constexpr int BV = 16;
    constexpr int SK = D + 4;
    constexpr int ST = BT + 4;
    constexpr int PHASE_ELEMS = (BV + BT) * SK;
    constexpr int VT_ELEMS = BV * ST;
    constexpr int CINV_ELEMS = 10 * C * C;
    constexpr int KRT_ELEMS = D * ST;
    constexpr int META_ELEMS = 4 * D + BT;
    constexpr unsigned USE_DECAY_TABLE = 1u << 0;
    constexpr unsigned BETA_ACTIVATED = 1u << 1;
    static_assert(VT_ELEMS + KRT_ELEMS <= PHASE_ELEMS,
                  "V' and K128 Kr^T must fit the NW8 phase pool");
    static_assert(PHASE_ELEMS * sizeof(__bf16) +
                      VT_ELEMS * sizeof(__bf16) +
                      CINV_ELEMS * sizeof(__bf16) +
                      META_ELEMS * sizeof(float) == 30720,
                  "BT64/BV16 NW8 scan must use exactly 30 KiB LDS");

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

    // Packed C and R remain live while phase aliases {H^T,Kd} with the
    // disjoint {V',Kr^T} publication used by the state update.
    __shared__ __bf16 phase[PHASE_ELEMS];
    __shared__ __bf16 vT[VT_ELEMS];
    __shared__ __bf16 cinv[CINV_ELEMS];
    __shared__ float decay[4 * D];
    __shared__ float beta[BT];
    __bf16* const hT = phase;
    __bf16* const kd = phase + BV * SK;
    __bf16* const uT = phase;
    __bf16* const krT = phase + VT_ELEMS;

    // Every wave owns K16 for this V16 tile.
    float hs[4];
    const int64_t state_base = int64_t(bh) * D * D;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int vv = v0 + (lane & 15);
        const int kk = wave * C + (lane >> 4) * 4 + i;
        const int64_t idx = state_base + int64_t(vv) * D + kk;
        if constexpr (HI)
            hs[i] = STATE_IN_FP32
                ? reinterpret_cast<const float*>(init_state)[idx]
                : bf16_to_f32(
                    reinterpret_cast<const __bf16*>(init_state)[idx]);
        else
            hs[i] = 0.0f;
    }

    const int scan_begin = SEGMENT_RANGE ? min(segment_begin, ns) : 0;
    const int scan_end = SEGMENT_RANGE
        ? min(scan_begin + segment_count, ns) : ns;
    for (int s = scan_begin; s < scan_end; ++s) {
        const int ht0 = ht_base + s * 4;
        const int xp0 = xp_base + s * 2;
        const int xs = xs_base + s;
        const int t0 = t0_base + s * BT;
        const int alen = min(BT, seq_len - s * BT);
        const int nch = (alen + C - 1) / C;

        // Missing tail chunks contribute identity decay.  RTP-K6 reuses the
        // prep decay table; other routes retain the established exponent path.
        if (tid < D) {
            if (use_decay_table) {
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

        // Publish the complete K128 entry state and Kd exactly once.  Each
        // thread moves two aligned Kd vectors; tail rows are explicit zeros.
        const int lv = lane & 15;
        const int kl = wave * C + (lane >> 4) * 4;
        bf16x4 hpack;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            hpack[i] = f32_to_bf16(hs[i]);
        *reinterpret_cast<bf16x4*>(hT + lv * SK + kl) = hpack;
        *reinterpret_cast<bf16x4*>(
            cs_sin + (int64_t(xs) * D + v0 + lv) * D + kl) = hpack;

        for (int vi = tid; vi < (BT * D) / 8; vi += 512) {
            const int m = vi / (D / 8);
            const int k8 = (vi % (D / 8)) * 8;
            bf16x8 x{};
            if (m < alen) {
                const int rb = m >> 4;
                const int r = m & 15;
                x = *reinterpret_cast<const bf16x8*>(
                    ws_kd + (int64_t(ht0 + rb) * C + r) * D + k8);
            }
            *reinterpret_cast<bf16x8*>(kd + m * SK + k8) = x;
        }
        __syncthreads();

        // Batch the four strided V scalars before consuming any of them so
        // they share one outstanding VMEM batch instead of four serialized
        // load/consume waits.  Invalid tail rows are never read globally.
        __bf16 v_prefetch0 = (__bf16)0.0f;
        __bf16 v_prefetch1 = (__bf16)0.0f;
        __bf16 v_prefetch2 = (__bf16)0.0f;
        __bf16 v_prefetch3 = (__bf16)0.0f;
        if (wave < 4) {
            const int r0 = (lane >> 4) * 4;
            const int m0 = wave * C + r0;
            const int vv = lane & 15;
            if (m0 < alen)
                v_prefetch0 = v_g[
                    (int64_t(t0 + m0) * H + h) * D + v0 + vv];
            if (m0 + 1 < alen)
                v_prefetch1 = v_g[
                    (int64_t(t0 + m0 + 1) * H + h) * D + v0 + vv];
            if (m0 + 2 < alen)
                v_prefetch2 = v_g[
                    (int64_t(t0 + m0 + 2) * H + h) * D + v0 + vv];
            if (m0 + 3 < alen)
                v_prefetch3 = v_g[
                    (int64_t(t0 + m0 + 3) * H + h) * D + v0 + vv];
        }

        // Only the four token-owner waves read Kd/H^T.  Keep the established
        // two-K64 FP32 grouping while issuing K tiles in global order
        // 0,16,...,112, so this A/B does not introduce a new reduction order.
        f32x4 kh = {0.f, 0.f, 0.f, 0.f};
        if (wave < 4) {
            const f32x4 p0 = gemm_contract_last<__bf16, 64, SK>(
                kd + wave * C * SK, hT, lane);
            const f32x4 p1 = gemm_contract_last<__bf16, 64, SK>(
                kd + wave * C * SK + 64, hT + 64, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                kh[i] = p0[i] + p1[i];
        }
        // Publish R from the four token-owner waves.  It remains in vT while
        // the old H^T/Kd allocation is reused for the C@R result and Kr^T.
        if (wave < 4) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = (lane >> 4) * 4 + i;
                const int m = wave * C + r;
                const int vv = lane & 15;
                const __bf16 v_prefetch = i == 0 ? v_prefetch0 :
                    (i == 1 ? v_prefetch1 :
                    (i == 2 ? v_prefetch2 : v_prefetch3));
                const float value = m < alen
                    ? bf16_to_f32(v_prefetch)
                    : 0.0f;
                vT[vv * ST + m] = m < alen
                    ? f32_to_bf16((value - kh[i]) * beta[m])
                    : (__bf16)0.0f;
            }
        }

        // While waves 0..3 form Kd@H^T and R, waves 4..7 cooperatively load
        // all 320 packed-C vectors.  Local threads 0..63 take a second vector.
        if (wave >= 4) {
            const int ctid = tid - 4 * 64;
            for (int vi = ctid; vi < CINV_ELEMS / 8; vi += 4 * 64) {
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
                        cross64 +
                        (int64_t(xs) * 4 + tile - 3) * C * C)[e8];
                else if (tile == 5 && nch > 2)
                    x = reinterpret_cast<const bf16x8*>(
                        ws_inv + int64_t(ht0 + 2) * C * C)[e8];
                else if ((tile == 6 || tile == 7) && nch > 3)
                    x = reinterpret_cast<const bf16x8*>(
                        cross64 +
                        (int64_t(xs) * 4 + tile - 4) * C * C)[e8];
                else if (tile == 8 && nch > 3)
                    x = reinterpret_cast<const bf16x8*>(
                        cross32 + int64_t(xp0 + 1) * C * C)[e8];
                else if (tile == 9 && nch > 3)
                    x = reinterpret_cast<const bf16x8*>(
                        ws_inv + int64_t(ht0 + 3) * C * C)[e8];
                reinterpret_cast<bf16x8*>(cinv)[vi] = x;
            }
        }

        // Every wave prefetches the BT64 rows for its K16 Kr slice.  The loads
        // are independent of C@R and remain live only until the Kr^T publish.
        bf16x4 kr_prefetch[4];
        #pragma unroll
        for (int tm = 0; tm < 4; ++tm) {
            const int m = tm * C + (lane >> 2);
            const int k4 = wave * C + (lane & 3) * 4;
            bf16x4 x{};
            if (m < alen) {
                const int rb = m >> 4;
                const int r = m & 15;
                x = *reinterpret_cast<const bf16x4*>(
                    ws_kr + (int64_t(ht0 + rb) * C + r) * D + k4);
            }
            kr_prefetch[tm] = x;
        }

        // All H^T/Kd consumers, R producers, and packed-C loaders rendezvous
        // before phase is repurposed, publishing both inputs to C@R.
        __syncthreads();

        // Only waves 0..3 consume R.  Waves 4..7 can immediately publish their
        // disjoint K16 Kr^T slices without touching either R or packed C.
        f32x4 u = {0.f, 0.f, 0.f, 0.f};
        if (wave < 4) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j <= wave) {
                    const f32x4 x = bt64_bv16_nw8_contract_last<C, C, ST>(
                        cinv + bt64_bv16_nw8_tri_tile(wave, j) * C * C,
                        vT + j * C, lane);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i)
                        u[i] += x[i];
                }
            }
        }

        if (wave < 4) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = (lane >> 4) * 4 + i;
                const int m = wave * C + r;
                const int vv = lane & 15;
                const __bf16 x = m < alen
                    ? f32_to_bf16(u[i]) : (__bf16)0.0f;
                uT[vv * ST + m] = x;
                if (wave < nch)
                    cs_u[((int64_t(ht0 + wave) * C + r) * D) +
                         v0 + vv] = x;
            }
        }

        #pragma unroll
        for (int tm = 0; tm < 4; ++tm) {
            const int m = tm * C + (lane >> 2);
            const int k4 = wave * C + (lane & 3) * 4;
            const bf16x4 x = kr_prefetch[tm];
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                krT[(k4 + i) * ST + m] = x[i];
        }
        // This single CTA boundary publishes disjoint V' and Kr^T regions of
        // the old phase allocation before any state update.
        __syncthreads();

        float carry[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            const f32x4 c = bt64_bv16_nw8_contract_last<C, ST, ST>(
                krT + wave * C * ST + rb * C,
                uT + rb * C, lane);
            const int kb = wave * C + (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float suffix = rb < 3
                    ? decay[(rb + 1) * D + kb + i] : 1.0f;
                carry[i] += c[i] * suffix;
            }
        }

        const int kb = wave * C + (lane >> 4) * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            hs[i] = hs[i] * decay[kb + i] + carry[i];

        // No wave may publish the next segment's H^T/Kd until all eight waves
        // finish the current V'/Kr^T reads (and current decay metadata reads).
        __syncthreads();

    }

    if constexpr (HO) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = wave * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_base + int64_t(vv) * D + kk;
            if constexpr (STATE_OUT_FP32)
                reinterpret_cast<float*>(final_state)[idx] = hs[i];
            else
                reinterpret_cast<__bf16*>(final_state)[idx] =
                    f32_to_bf16(hs[i]);
        }
    }
}

// Preserve the production kernel's original template and kernarg ABI.  The
// range pipeline below is a separate symbol so an opt-in experiment cannot
// perturb default launch metadata or add live scalar arguments.
template <bool HI = false, bool HO = false,
          bool SFP32 = false, bool VL = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_bv16_nw8_kernel(
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
    k2_kda_csplit_bt64_bv16_nw8_body<
        HI, HO, SFP32, SFP32, VL, false>(
            v_g, beta_src, ws_kd, ws_kr, ws_gt, ws_decay, ws_inv,
            cross32, cross64, cs_u, cs_sin, init_state, final_state,
            cu_seqlens, tile_prefix, pair_prefix, segment_prefix,
            total_tiles, total_pairs, total_segments, T_seq, H, NT,
            scan_flags, 0, 0);
}

// Segment-range scan used by the gfx942 P3/P4 batch pipeline.  The packed
// specialization is currently launched only for N=1, where segment_begin is
// both the sequence-local and global segment offset.  Intermediate state is
// always FP32, while the first input and final public output retain their
// independently selected public dtypes.
template <bool HI, bool HO, bool STATE_IN_FP32, bool STATE_OUT_FP32,
          bool VL>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_bv16_nw8_range_kernel(
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
        int T_seq, int H, int NT, unsigned scan_flags,
        int segment_begin, int segment_count) {
    k2_kda_csplit_bt64_bv16_nw8_body<
        HI, HO, STATE_IN_FP32, STATE_OUT_FP32, VL, true>(
            v_g, beta_src, ws_kd, ws_kr, ws_gt, ws_decay, ws_inv,
            cross32, cross64, cs_u, cs_sin, init_state, final_state,
            cu_seqlens, tile_prefix, pair_prefix, segment_prefix,
            total_tiles, total_pairs,
            total_segments, T_seq, H, NT, scan_flags,
            segment_begin, segment_count);
}

}  // namespace flashkda_hip
