// Low-LDS RTP BT64 chunk-parallel K6 for gfx942.
//
// This is an A/B companion to k2_kda_csplit_bt64_out_kernel.  It preserves
// the same direct RTP algebra and 8-wave output ownership, but contracts K in
// four BK32 panels and aliases the non-overlapping K-panel and final A@V'
// phases in one 30,208-byte LDS allocation.  The two phase layouts are:
//
//   K phase (stride 36):
//     scaled Q  [64,36] @     0 B       local Q [64,36] @  4,608 B
//     Ki        [64,36] @ 9,216 B       H       [128,36] @ 13,824 B
//     q_scale [4,128]f  @23,040 B       a_scale[10,128]f @ 25,088 B
//
//   Final phase (stride 68):
//     A         [64,68] @     0 B       V'^T    [128,68] @  8,704 B
//
// The final BK32 panel's CTA barrier is the phase-alias fence.  V' remains in
// registers until that fence, so no final-phase byte is live during K-panel
// contraction.  K chunks are accumulated in the original 0,16,...,112 order.
#pragma once

#include <hip/hip_runtime.h>

#include "k2_kda_csplit_bt64_out_kernel.hpp"

namespace flashkda_hip {

template <bool VL, bool SEGMENT_RANGE>
__device__ __forceinline__ void
k2_kda_csplit_bt64_out_bk32_body(
        const __bf16* __restrict__ cs_u,      // [tile, 16, V], V'=C@R
        const __bf16* __restrict__ cs_sin,   // [segment, V, K]
        const __bf16* __restrict__ ws_qd,    // [tile, 16, K]
        const __bf16* __restrict__ ws_kr,    // [tile, 16, K]
        const float* __restrict__ ws_gt,      // [tile, K]
        __bf16* __restrict__ out_g,           // [T_total, H, V]
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT,
        int segment_begin) {
    constexpr int C = 16;
    constexpr int BT = 64;
    constexpr int D = 128;
    constexpr int BK = 32;
    constexpr int NW = 8;
    constexpr int STRIDE_K = BK + 4;
    constexpr int STRIDE_T = BT + 4;
    constexpr int O_T_M = 4;
    constexpr int O_T_N = 2;
    constexpr int O_E_N = D / (C * O_T_N);
    constexpr int A_E_N = BT / (C * O_T_N);
    static_assert(O_T_M * O_T_N == NW);
    static_assert(D % BK == 0 && BK % C == 0);
    static_assert(sizeof(__bf16) == 2 && sizeof(float) == 4);

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;

    int h, ht_base, ss, t0_base, alen, chunks;
    if constexpr (VL) {
        const int gsi = blockIdx.x + (SEGMENT_RANGE ? segment_begin : 0);
        h = blockIdx.y;

        // segment_prefix is exact while gridDim.x may be the workspace upper
        // bound.  Gap CTAs resolve to the final sequence and return uniformly.
        int lo = 0, hi = N;
        while (hi - lo > 1) {
            const int mid = (lo + hi) >> 1;
            if (segment_prefix[mid] <= gsi)
                lo = mid;
            else
                hi = mid;
        }
        const int local_seg = gsi - segment_prefix[lo];
        const int64_t bos = cu_seqlens[lo];
        const int seq_len = int(cu_seqlens[lo + 1] - bos);
        const int nseg = (seq_len + BT - 1) / BT;
        if (local_seg >= nseg)
            return;

        const int local_chunk = local_seg * (BT / C);
        ht_base = h * total_tiles + tile_prefix[lo] + local_chunk;
        ss = h * total_segments + gsi;
        t0_base = int(bos) + local_seg * BT;
        alen = min(BT, seq_len - local_seg * BT);
        chunks = (alen + C - 1) / C;
    } else {
        const int seg = blockIdx.x + (SEGMENT_RANGE ? segment_begin : 0);
        const int bh = blockIdx.y;
        const int b = bh / H;
        h = bh % H;
        const int nseg = (NT + (BT / C) - 1) / (BT / C);
        if (seg >= nseg)
            return;

        ht_base = bh * NT + seg * (BT / C);
        ss = bh * nseg + seg;
        t0_base = b * T_seq + seg * BT;
        alen = min(BT, T_seq - seg * BT);
        chunks = (alen + C - 1) / C;
    }

    // One raw allocation makes the lifetime alias explicit to the compiler.
    // Explicit 16-byte alignment covers the bf16x4 and FP32 views.
    constexpr int Q_PANEL_BYTES = BT * STRIDE_K * sizeof(__bf16);
    constexpr int Q_PANEL_OFF = 0;
    constexpr int Q_LOCAL_OFF = Q_PANEL_OFF + Q_PANEL_BYTES;
    constexpr int KI_PANEL_OFF = Q_LOCAL_OFF + Q_PANEL_BYTES;
    constexpr int RHS_PANEL_OFF = KI_PANEL_OFF + Q_PANEL_BYTES;
    constexpr int RHS_PANEL_BYTES = D * STRIDE_K * sizeof(__bf16);
    constexpr int Q_SCALE_OFF = RHS_PANEL_OFF + RHS_PANEL_BYTES;
    constexpr int Q_SCALE_BYTES = (BT / C) * D * sizeof(float);
    constexpr int A_SCALE_OFF = Q_SCALE_OFF + Q_SCALE_BYTES;
    constexpr int A_SCALE_BYTES = 10 * D * sizeof(float);
    constexpr int K_PHASE_BYTES = A_SCALE_OFF + A_SCALE_BYTES;

    constexpr int FINAL_A_OFF = 0;
    constexpr int FINAL_A_BYTES = BT * STRIDE_T * sizeof(__bf16);
    constexpr int FINAL_VT_OFF = FINAL_A_OFF + FINAL_A_BYTES;
    constexpr int FINAL_VT_BYTES = D * STRIDE_T * sizeof(__bf16);
    constexpr int FINAL_PHASE_BYTES = FINAL_VT_OFF + FINAL_VT_BYTES;
    constexpr int SMEM_BYTES = K_PHASE_BYTES;

    static_assert(Q_PANEL_BYTES == 4608);
    static_assert(Q_LOCAL_OFF == 4608);
    static_assert(KI_PANEL_OFF == 9216);
    static_assert(RHS_PANEL_OFF == 13824);
    static_assert(Q_SCALE_OFF == 23040);
    static_assert(A_SCALE_OFF == 25088);
    static_assert(K_PHASE_BYTES == 30208);
    static_assert(FINAL_A_BYTES == 8704);
    static_assert(FINAL_VT_OFF == 8704);
    static_assert(FINAL_PHASE_BYTES == 26112);
    static_assert(FINAL_PHASE_BYTES <= SMEM_BYTES);
    static_assert(Q_SCALE_OFF % alignof(float) == 0);
    static_assert(A_SCALE_OFF % alignof(float) == 0);
    static_assert(SMEM_BYTES % 16 == 0);

    __shared__ __align__(16) unsigned char smem[SMEM_BYTES];

    auto* q_panel = reinterpret_cast<__bf16*>(smem + Q_PANEL_OFF);
    auto* q_local_panel = reinterpret_cast<__bf16*>(smem + Q_LOCAL_OFF);
    auto* ki_panel = reinterpret_cast<__bf16*>(smem + KI_PANEL_OFF);
    auto* rhs_panel = reinterpret_cast<__bf16*>(smem + RHS_PANEL_OFF);
    auto* q_scale = reinterpret_cast<float*>(smem + Q_SCALE_OFF);
    auto* a_scale = reinterpret_cast<float*>(smem + A_SCALE_OFF);

    // These final-phase views deliberately alias the K-phase views above.
    auto* final_a = reinterpret_cast<__bf16*>(smem + FINAL_A_OFF);
    auto* vprime_T = reinterpret_cast<__bf16*>(smem + FINAL_VT_OFF);

    // Compute exact segment-relative factors with the same 4-cohort mapping
    // as the BK64 A/B kernel.  Every q_scale and a_scale element has one writer.
    constexpr int FACTOR_GROUPS = BT / C;
    static_assert(NW * 64 == FACTOR_GROUPS * D);
    const int factor_group = tid / D;
    const int factor_d = tid - factor_group * D;

    float gt[FACTOR_GROUPS];
    #pragma unroll
    for (int r = 0; r < FACTOR_GROUPS; ++r)
        gt[r] = r < chunks
            ? ws_gt[(int64_t(ht_base + r) * D) + factor_d] : 0.0f;

    float q_factor = 1.0f;
    if (factor_group != 0) {
        float exponent = gt[0];
        #pragma unroll
        for (int u = 1; u < FACTOR_GROUPS - 1; ++u) {
            if (u < factor_group)
                exponent += gt[u];
        }
        q_factor = ex2(exponent);
    }
    q_scale[factor_group * D + factor_d] = q_factor;

    #pragma unroll
    for (int r = 0; r < FACTOR_GROUPS; ++r) {
        #pragma unroll
        for (int c = 0; c <= r; ++c) {
            const int pair = bt64_out_tri_tile(r, c);
            if (bt64_out_factor_owner(pair) == factor_group) {
                float exponent = 0.0f;
                if (r == c) {
                    exponent = -gt[c];
                } else {
                    #pragma unroll
                    for (int u = 0; u < FACTOR_GROUPS; ++u) {
                        if (u > c && u < r)
                            exponent += gt[u];
                    }
                }
                a_scale[pair * D + factor_d] = ex2(exponent);
            }
        }
    }

    // V'=C@R remains register-resident throughout all four BK32 panels.
    constexpr int VEC_PER_ROW = D / 4;
    constexpr int VN_ITERS = (BT * VEC_PER_ROW) / (NW * 64);
    static_assert((BT * VEC_PER_ROW) % (NW * 64) == 0);
    bf16x4 vprime_reg[VN_ITERS];
    #pragma unroll
    for (int it = 0; it < VN_ITERS; ++it) {
        const int i = tid + it * NW * 64;
        const int row = i / VEC_PER_ROW;
        const int col4 = (i % VEC_PER_ROW) * 4;
        if (row < alen) {
            vprime_reg[it] = *reinterpret_cast<const bf16x4*>(
                cs_u + int64_t(ht_base) * C * D + row * D + col4);
        } else {
            vprime_reg[it] = bf16x4{};
        }
    }

    // Publish both factor tables before panel loading consumes q_scale.
    __syncthreads();

    const int o_m_base = (wave / O_T_N) * C;
    const int o_n_base = (wave % O_T_N) * (O_E_N * C);
    const int a_n_base = (wave % O_T_N) * (A_E_N * C);

    f32x4 o_cross[O_E_N];
    f32x4 avec[A_E_N];
    #pragma unroll
    for (int i = 0; i < O_E_N; ++i)
        o_cross[i] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int i = 0; i < A_E_N; ++i)
        avec[i] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};

    // One BK32 panel needs one q/ki bf16x4 and two H bf16x4 values per thread.
    // Prefetching panel p+1 before computing p preserves the BK64 overlap while
    // halving its live staging registers.
    constexpr int QK_ITERS = (BT * (BK / 4)) / (NW * 64);
    constexpr int H_ITERS = (D * (BK / 4)) / (NW * 64);
    static_assert((BT * (BK / 4)) % (NW * 64) == 0);
    static_assert((D * (BK / 4)) % (NW * 64) == 0);
    static_assert(QK_ITERS == 1 && H_ITERS == 2);

    bf16x4 q_reg[QK_ITERS];
    bf16x4 q_local_reg[QK_ITERS];
    bf16x4 ki_reg[QK_ITERS];
    bf16x4 h_reg[H_ITERS];

    auto load_panel = [&](int panel) {
        #pragma unroll
        for (int it = 0; it < QK_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int row = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            const int block = row / C;
            const int local_row = row % C;
            const int d0 = panel * BK + col4;
            bf16x4 qv{}, kv{};
            if (row < alen) {
                const bf16x4 qsrc = *reinterpret_cast<const bf16x4*>(
                    ws_qd + (int64_t(ht_base + block) * C + local_row) * D + d0);
                const bf16x4 ksrc = *reinterpret_cast<const bf16x4*>(
                    ws_kr + (int64_t(ht_base + block) * C + local_row) * D + d0);
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    qv[p] = f32_to_bf16(
                        bf16_to_f32(qsrc[p]) * q_scale[block * D + d0 + p]);
                    kv[p] = ksrc[p];
                }
                q_local_reg[it] = qsrc;
            } else {
                q_local_reg[it] = bf16x4{};
            }
            q_reg[it] = qv;
            ki_reg[it] = kv;
        }
        #pragma unroll
        for (int it = 0; it < H_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int v = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            h_reg[it] = *reinterpret_cast<const bf16x4*>(
                cs_sin + (int64_t(ss) * D + v) * D + panel * BK + col4);
        }
    };

    auto commit_panel = [&]() {
        #pragma unroll
        for (int it = 0; it < QK_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int row = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            *reinterpret_cast<bf16x4*>(q_panel + row * STRIDE_K + col4) =
                q_reg[it];
            *reinterpret_cast<bf16x4*>(q_local_panel + row * STRIDE_K + col4) =
                q_local_reg[it];
            *reinterpret_cast<bf16x4*>(ki_panel + row * STRIDE_K + col4) =
                ki_reg[it];
        }
        #pragma unroll
        for (int it = 0; it < H_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int v = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            *reinterpret_cast<bf16x4*>(rhs_panel + v * STRIDE_K + col4) =
                h_reg[it];
        }
    };

    load_panel(0);
    commit_panel();
    __syncthreads();

    const int row_chunk = o_m_base / C;
    #pragma unroll
    for (int panel = 0; panel < D / BK; ++panel) {
        if (panel + 1 < D / BK)
            load_panel(panel + 1);

        bt64_out_tiled_gemm<O_E_N, BK / C>(
            o_cross, q_panel, o_m_base, STRIDE_K,
            rhs_panel, o_n_base, STRIDE_K, lane);

        // Stable lower-triangular qd@ki^T.  The global K contraction order is
        // panel-major then ek-major: 0,16,32,48,64,80,96,112.
        #pragma unroll
        for (int ek = 0; ek < BK / C; ++ek) {
            const bf16x4 af = bt64_out_load_mfma_fragment(
                q_local_panel, o_m_base, ek * C, STRIDE_K, lane);
            #pragma unroll
            for (int en = 0; en < A_E_N; ++en) {
                const int col_base = a_n_base + en * C;
                const int col_chunk = col_base / C;
                if (col_chunk <= row_chunk) {
                    bf16x4 bf = bt64_out_load_mfma_fragment(
                        ki_panel, col_base, ek * C, STRIDE_K, lane);
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const int d = panel * BK + ek * C
                            + ((lane >> 4) << 2) + p;
                        bf[p] = f32_to_bf16(
                            bf16_to_f32(bf[p]) *
                            a_scale[bt64_out_tri_tile(
                                row_chunk, col_chunk) * D + d]);
                    }
                    avec[en] = mfma_bf16(af, bf, avec[en]);
                }
            }
        }

        // For panels 0..2 this protects the single K-panel buffer before its
        // overwrite.  For panel 3 it is the K-phase -> Final-phase alias fence:
        // all q/ki/H/scale reads complete before A or V'^T overwrites raw LDS.
        __syncthreads();
        if (panel + 1 < D / BK) {
            commit_panel();
            __syncthreads();
        }
    }

    // Materialize the register-resident causal A in the final-phase view.
    #pragma unroll
    for (int en = 0; en < A_E_N; ++en) {
        const int col = a_n_base + en * C + (lane & 15);
        const int row4 = o_m_base + ((lane >> 4) << 2);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int row = row4 + p;
            const float x = (row < alen && col < alen && row >= col)
                ? avec[en][p] : 0.0f;
            final_a[row * STRIDE_T + col] = f32_to_bf16(x);
        }
    }

    // Publish the prefetched V' tile transposed as [V, BT].
    #pragma unroll
    for (int it = 0; it < VN_ITERS; ++it) {
        const int i = tid + it * NW * 64;
        const int row = i / VEC_PER_ROW;
        const int col4 = (i % VEC_PER_ROW) * 4;
        #pragma unroll
        for (int p = 0; p < 4; ++p)
            vprime_T[(col4 + p) * STRIDE_T + row] = vprime_reg[it][p];
    }
    __syncthreads();

    f32x4 o_intra[O_E_N];
    #pragma unroll
    for (int i = 0; i < O_E_N; ++i)
        o_intra[i] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};
    bt64_out_tiled_gemm<O_E_N, BT / C>(
        o_intra, final_a, o_m_base, STRIDE_T,
        vprime_T, o_n_base, STRIDE_T, lane);

    #pragma unroll
    for (int en = 0; en < O_E_N; ++en) {
        const int col = o_n_base + en * C + (lane & 15);
        const int row4 = o_m_base + ((lane >> 4) << 2);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int row = row4 + p;
            if (row < alen) {
                const int64_t out_idx =
                    (int64_t(t0_base + row) * H + h) * D + col;
                out_g[out_idx] = f32_to_bf16(o_cross[en][p] + o_intra[en][p]);
            }
        }
    }
}

// Preserve the production output kernel's original symbol and kernarg ABI.
template <bool VL = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_out_bk32_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT) {
    k2_kda_csplit_bt64_out_bk32_body<VL, false>(
        cs_u, cs_sin, ws_qd, ws_kr, ws_gt, out_g, cu_seqlens,
        tile_prefix, segment_prefix, N, total_tiles, total_segments,
        T_seq, H, NT, 0);
}

template <bool VL = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_out_bk32_range_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT, int segment_begin) {
    k2_kda_csplit_bt64_out_bk32_body<VL, true>(
        cs_u, cs_sin, ws_qd, ws_kr, ws_gt, out_g, cu_seqlens,
        tile_prefix, segment_prefix, N, total_tiles, total_segments,
        T_seq, H, NT, segment_begin);
}

}  // namespace flashkda_hip
