// gfx942 fused BT64 recurrent scan and V16 output consumer.
//
// The recurrent half is the same eight-wave, fp32-register-state scan as the
// common BT64/BV16/NW8 kernel.  Output is evaluated while the segment-entry
// state and the just-computed V'=C@R are still on chip:
//
//   O = qd_seg @ H_entry^T + A_seg @ V'.
//
// Waves 0..3 retain their scan ownership (one BT16 row block each).  Waves
// 4..7 own the corresponding four output row blocks.  A producer preceding
// this kernel supplies the stable, causal A tiles in `segment_a`:
//
//   segment_a[xs, tri(r,c), i, j],  tri(r,c)=r*(r+1)/2+c, c<=r,
//
// as BF16 [10,16,16] tiles.  Diagonal tiles must already be lower triangular,
// tail rows/columns must be zero, and each tile must use the direct-RTP scale
// described by k2_kda_csplit_bt64_out_kernel.hpp.  In particular this kernel
// never consumes ws_mqk: on the production K6 route that allocation aliases
// the per-chunk decay table.
//
// `segment_a` may alias the beginning of the old cs_sin allocation.  Neither
// cs_sin nor cs_u is produced or consumed here, so the standalone output pass
// can be removed once the A producer is integrated.
#pragma once

#include <hip/hip_runtime.h>

#include "../mfma.hpp"

namespace flashkda_hip::gfx942 {

__device__ __forceinline__ constexpr int fused_bt64_tri_tile(int r, int c) {
    return r * (r + 1) / 2 + c;
}

// D[m,n] = sum_k A[m,k] * B[n,k].  B is stored transposed so both MFMA
// fragments are contiguous in their respective row layouts.
template <int Kd, int LDA, int LDB>
__device__ __forceinline__ f32x4 fused_bt64_contract_last(
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
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_fused_nw8_kernel(
        const __bf16* __restrict__ v_g,
        const float* __restrict__ beta_src,
        __bf16* __restrict__ out_g,
        const __bf16* __restrict__ ws_kd,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const float* __restrict__ ws_decay,
        const __bf16* __restrict__ ws_inv,
        const __bf16* __restrict__ cross32,
        const __bf16* __restrict__ cross64,
        const __bf16* __restrict__ segment_a,
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
    constexpr int SK = D + 4;
    constexpr int ST = BT + 4;
    constexpr int PHASE_ELEMS = (BV + BT) * SK;
    constexpr int VT_ELEMS = BV * ST;
    constexpr int CINV_ELEMS = 10 * C * C;
    constexpr int KRT_ELEMS = D * ST;
    constexpr int META_ELEMS = 4 * D + BT;
    // Row block zero has an exact unit prefix and needs no LDS entry.  Keeping
    // only rows 1..3 also avoids the exact-32-KiB allocation boundary, which
    // is an important occupancy A/B on gfx942.
    constexpr int QSCALE_ELEMS = 3 * D;
    constexpr unsigned USE_DECAY_TABLE = 1u << 0;
    constexpr unsigned BETA_ACTIVATED = 1u << 1;
    static_assert(VT_ELEMS + KRT_ELEMS <= PHASE_ELEMS,
                  "V' and K128 Kr^T must fit the fused NW8 phase pool");
    static_assert(PHASE_ELEMS * sizeof(__bf16) +
                      VT_ELEMS * sizeof(__bf16) +
                      CINV_ELEMS * sizeof(__bf16) +
                      (META_ELEMS + QSCALE_ELEMS) * sizeof(float) == 32256,
                  "unexpected gfx942 fused BT64/BV16 LDS footprint");

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
    if (ns == 0)
        return;

    // The extra q_scale table raises the original 30 KiB scan allocation to
    // exactly 32 KiB, retaining two resident workgroups in gfx942's 64 KiB LDS.
    __shared__ __bf16 phase[PHASE_ELEMS];
    __shared__ __bf16 vT[VT_ELEMS];
    __shared__ __bf16 cinv[CINV_ELEMS];
    __shared__ float decay[4 * D];
    __shared__ float beta[BT];
    __shared__ float q_scale[QSCALE_ELEMS];
    __bf16* const hT = phase;
    __bf16* const kd = phase + BV * SK;
    __bf16* const uT = phase;
    __bf16* const krT = phase + VT_ELEMS;

    // Each wave owns one K16 slice for the CTA's V16 state tile.
    float hs[4];
    const int64_t state_base = int64_t(bh) * D * D;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int vv = v0 + (lane & 15);
        const int kk = wave * C + (lane >> 4) * 4 + i;
        const int64_t idx = state_base + int64_t(vv) * D + kk;
        if constexpr (HI)
            hs[i] = SFP32
                ? reinterpret_cast<const float*>(init_state)[idx]
                : bf16_to_f32(
                    reinterpret_cast<const __bf16*>(init_state)[idx]);
        else
            hs[i] = 0.0f;
    }

    for (int s = 0; s < ns; ++s) {
        const int ht0 = ht_base + s * 4;
        const int xp0 = xp_base + s * 2;
        const int xs = xs_base + s;
        const int t0 = t0_base + s * BT;
        const int alen = min(BT, seq_len - s * BT);
        const int nch = (alen + C - 1) / C;

        // Preserve the scan's suffix-decay construction exactly.  Missing tail
        // chunks are identity transitions.
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
                // K6 preparation already materialized exp2(gt) per chunk.
                // Reuse those four values for the output prefix instead of
                // redundantly issuing 3*D exp2 operations in every V16 CTA.
                q_scale[tid] = e0;
                q_scale[D + tid] = e0 * e1;
                q_scale[2 * D + tid] = e0 * e1 * e2;
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
                q_scale[tid] = ex2(g0);
                q_scale[D + tid] = ex2(g0 + g1);
                q_scale[2 * D + tid] = ex2(g0 + g1 + g2);
            }
        }
        if (tid < BT) {
            beta[tid] = tid < alen
                ? (beta_is_activated
                    ? beta_src[int64_t(xs) * BT + tid]
                    : sigmoid_tanh(beta_src[int64_t(t0 + tid) * H + h]))
                : 0.0f;
        }

        // Publish the complete K128 entry state and Kd.  Unlike the split
        // pipeline there is deliberately no cs_sin state snapshot store.
        const int lv = lane & 15;
        const int kl = wave * C + (lane >> 4) * 4;
        bf16x4 hpack;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            hpack[i] = f32_to_bf16(hs[i]);
        *reinterpret_cast<bf16x4*>(hT + lv * SK + kl) = hpack;

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

        // Waves 0..3 form R; waves 4..7 independently form the output's
        // qd_seg@H_entry term from the same BF16 entry-state snapshot.
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

        f32x4 kh = {0.f, 0.f, 0.f, 0.f};
        if (wave < 4) {
            const f32x4 p0 = fused_bt64_contract_last<64, SK, SK>(
                kd + wave * C * SK, hT, lane);
            const f32x4 p1 = fused_bt64_contract_last<64, SK, SK>(
                kd + wave * C * SK + 64, hT + 64, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                kh[i] = p0[i] + p1[i];
        }
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
                    ? bf16_to_f32(v_prefetch) : 0.0f;
                vT[vv * ST + m] = m < alen
                    ? f32_to_bf16((value - kh[i]) * beta[m])
                    : (__bf16)0.0f;
            }
        }

        f32x4 o_cross = {0.f, 0.f, 0.f, 0.f};
        if (wave >= 4) {
            const int rb = wave - 4;
            const int qrow = lane & 15;
            const int kb = (lane >> 4) * 4;
            const int m = rb * C + qrow;
            if (rb < nch) {
                #pragma unroll
                for (int k0 = 0; k0 < D; k0 += C) {
                    const int d0 = k0 + kb;
                    bf16x4 qf{};
                    if (m < alen) {
                        qf = *reinterpret_cast<const bf16x4*>(
                            ws_qd +
                            (int64_t(ht0 + rb) * C + qrow) * D + d0);
                        const f32x4 prefix = rb == 0
                            ? f32x4{1.0f, 1.0f, 1.0f, 1.0f}
                            : *reinterpret_cast<const f32x4*>(
                                q_scale + (rb - 1) * D + d0);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i)
                            qf[i] = f32_to_bf16(
                                bf16_to_f32(qf[i]) * prefix[i]);
                    }
                    const bf16x4 hf = *reinterpret_cast<const bf16x4*>(
                        hT + qrow * SK + d0);
                    o_cross = mfma_bf16(qf, hf, o_cross);
                }
            }
        }

        // While scan waves consume Kd/H, output waves load packed C.
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

        // Every wave prefetches the BT64 rows for its K16 Kr state slice.
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

        // All H/Kd/output-cross consumers and all producers rendezvous before
        // phase is repurposed as the disjoint {V',Kr^T} publication.
        __syncthreads();

        f32x4 u = {0.f, 0.f, 0.f, 0.f};
        if (wave < 4) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (j <= wave) {
                    const f32x4 x = fused_bt64_contract_last<C, C, ST>(
                        cinv + fused_bt64_tri_tile(wave, j) * C * C,
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
                uT[vv * ST + m] = m < alen
                    ? f32_to_bf16(u[i]) : (__bf16)0.0f;
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
        __syncthreads();

        // Form the complete state carry, but deliberately defer applying it to
        // hs until O has consumed the segment-entry state semantics.
        float carry[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int rb = 0; rb < 4; ++rb) {
            const f32x4 c = fused_bt64_contract_last<C, ST, ST>(
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

        // Consumer-only causal term.  segment_a is already BF16 and masked,
        // so each causal block costs exactly one vector load plus one MFMA and
        // no CTA synchronization.  Loading on demand here avoids carrying four
        // A fragments across both CTA barriers, which is important to gfx942
        // VGPR occupancy.
        f32x4 o_intra = {0.f, 0.f, 0.f, 0.f};
        if (wave >= 4) {
            const int rb = wave - 4;
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (rb < nch && c <= rb) {
                    const int arow = lane & 15;
                    const int acol4 = (lane >> 4) * 4;
                    const bf16x4 af = *reinterpret_cast<const bf16x4*>(
                        segment_a +
                        (int64_t(xs) * 10 + fused_bt64_tri_tile(rb, c)) *
                            C * C +
                        arow * C + acol4);
                    const int urow = lane & 15;
                    const int tok4 = (lane >> 4) * 4;
                    const bf16x4 uf = *reinterpret_cast<const bf16x4*>(
                        uT + urow * ST + c * C + tok4);
                    o_intra = mfma_bf16(af, uf, o_intra);
                }
            }
        }

        // The four output waves cover [BT64,V16].  Keep the two terms in FP32
        // until the single final rounding, matching the standalone RTP output.
        if (wave >= 4) {
            const int rb = wave - 4;
            const int vv = v0 + (lane & 15);
            const int row4 = (lane >> 4) * 4;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int m = rb * C + row4 + i;
                if (m < alen) {
                    const int64_t out_idx =
                        (int64_t(t0 + m) * H + h) * D + vv;
                    out_g[out_idx] = f32_to_bf16(o_cross[i] + o_intra[i]);
                }
            }
        }

        // Apply the unchanged BT64 recurrence only after output has consumed
        // H_entry and V'.  No output operation mutates carry or decay.
        const int kb = wave * C + (lane >> 4) * 4;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            hs[i] = hs[i] * decay[kb + i] + carry[i];

        // Protect all phase/meta reads before the next segment republishes H/Kd.
        __syncthreads();
    }

    if constexpr (HO) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int vv = v0 + (lane & 15);
            const int kk = wave * C + (lane >> 4) * 4 + i;
            const int64_t idx = state_base + int64_t(vv) * D + kk;
            if constexpr (SFP32)
                reinterpret_cast<float*>(final_state)[idx] = hs[i];
            else
                reinterpret_cast<__bf16*>(final_state)[idx] =
                    f32_to_bf16(hs[i]);
        }
    }
}

}  // namespace flashkda_hip::gfx942
