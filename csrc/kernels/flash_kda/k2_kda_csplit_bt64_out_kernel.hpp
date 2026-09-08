// RTP BT64 chunk-parallel K6 for gfx942.
//
// One 8-wave CTA owns one complete (sequence, head, BT64 segment) and all
// V=128 output columns.  The serial scan has already materialized
//
//   V'      = C @ R                    in cs_u, and
//   H_entry                              in cs_sin.
//
// K6 therefore evaluates the segment directly, without replaying any state
// transition and without consuming the legacy ws_mqk tile:
//
//   qd_seg[r] = ws_qd[r] * 2^prefix_r
//   O = qd_seg @ H_entry^T
//       + tril(qd_seg @ ki_seg^T) @ V'.
//
// qd_seg is safe to materialize because its segment prefix is non-positive.
// The inverse-decay ki_seg operand is not: a long negative gate can make it
// overflow before its product with qd_seg cancels.  The lower-triangular A is
// therefore evaluated in stable 16x16 tiles.  For row chunk r and key chunk c:
//
//   scale(r,c) = 2^-gt[c],                       r == c
//              = 2^sum(gt[c+1:r]),              r > c.
//
// The cross-chunk exponent is non-positive.  The positive diagonal exponent
// spans only one BT16 chunk and cancels the bounded suffix already carried by
// ws_kr, rather than ever constructing a segment-global inverse decay.  The
// two K64 panels reuse the same LDS operands.  FP32 o_cross[64,128] and
// A[64,64] fragments stay in registers across both panels; A is then rounded
// to BF16 in LDS and consumed by the final A@V' MFMA pass.
#pragma once

#include <hip/hip_runtime.h>
#include "mfma.hpp"

namespace flashkda_hip {

__device__ __forceinline__ constexpr int bt64_out_tri_tile(int r, int c) {
    return r * (r + 1) / 2 + c;
}

// Spread the ten lower-triangular chunk-pair factors over four 128-thread
// cohorts.  Together with q_scale, cohorts 0..3 issue 2, 2, 3, and 3
// non-trivial exp2 operations respectively.  Consecutive two-wave cohorts
// therefore give each gfx942 SIMD five exp2 operations instead of leaving six
// waves idle behind the original ten-operation dependency chain.
__device__ __forceinline__ constexpr int bt64_out_factor_owner(int pair) {
    return pair < 3 ? 0 :
           pair < 5 ? 1 :
           pair < 7 ? 2 :
           pair == 8 ? 2 : 3;
}

// Load one row-major 16x16 MFMA operand fragment.  For a B operand holding
// rows of B^T, this is also the fragment needed by A @ B^T.
__device__ __forceinline__ bf16x4 bt64_out_load_mfma_fragment(
        const __bf16* __restrict__ x, int row_base, int col_base,
        int stride, int lane) {
    const int row = row_base + (lane & 15);
    const int col4 = col_base + ((lane >> 4) << 2);
    bf16x4 v;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        v[i] = x[row * stride + col4 + i];
    return v;
}

// C[16, 16*EN] += A[16, 16*EK] @ B[16*EN, 16*EK]^T.
// Each wave owns all EN output fragments for a single 16-row tile.
template <int EN, int EK>
__device__ __forceinline__ void bt64_out_tiled_gemm(
        f32x4 (&acc)[EN],
        const __bf16* __restrict__ a, int a_row, int a_stride,
        const __bf16* __restrict__ b, int b_row, int b_stride,
        int lane) {
    #pragma unroll
    for (int ek = 0; ek < EK; ++ek) {
        const bf16x4 af = bt64_out_load_mfma_fragment(
            a, a_row, ek * 16, a_stride, lane);
        #pragma unroll
        for (int en = 0; en < EN; ++en) {
            const bf16x4 bf = bt64_out_load_mfma_fragment(
                b, b_row + en * 16, ek * 16, b_stride, lane);
            acc[en] = mfma_bf16(af, bf, acc[en]);
        }
    }
}

template <bool VL = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_out_kernel(
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
        int T_seq, int H, int NT) {
    constexpr int C = 16;
    constexpr int BT = 64;
    constexpr int D = 128;
    constexpr int BK = 64;
    constexpr int NW = 8;
    constexpr int STRIDE_K = BK + 4;
    constexpr int STRIDE_T = BT + 4;
    constexpr int O_T_M = 4;
    constexpr int O_T_N = 2;
    constexpr int O_E_N = D / (C * O_T_N);   // four 16-col fragments/wave
    constexpr int A_E_N = BT / (C * O_T_N);  // two 16-col fragments/wave
    static_assert(O_T_M * O_T_N == NW);

    const int tid = threadIdx.x;
    const int wave = tid >> 6;
    const int lane = tid & 63;

    int h, ht_base, ss, t0_base, alen, chunks;
    if constexpr (VL) {
        const int gsi = blockIdx.x;
        h = blockIdx.y;

        // segment_prefix is exact while gridDim.x may be the workspace upper
        // bound.  Gap CTAs resolve to the final sequence and are dropped by
        // the local-segment test below.
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
        const int seg = blockIdx.x;
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

    // Static LDS footprint (gfx942):
    //   q_panel       64x68 bf16 =  8,704 B
    //   rhs_panel    128x68 bf16 = 17,408 B (H_entry panel)
    //   vprime_T    128x68 bf16 = 17,408 B
    //   a_or_ki       64x68 bf16 =  8,704 B (ki panel -> causal A)
    //   q scales       4x128 fp32 =  2,048 B
    //   A scales      10x128 fp32 =  5,120 B
    //   total                         59,392 B
    __shared__ __bf16 q_panel[BT * STRIDE_K];
    __shared__ __bf16 rhs_panel[D * STRIDE_K];
    __shared__ __bf16 vprime_T[D * STRIDE_T];
    __shared__ __bf16 a_or_ki[BT * STRIDE_K];
    __shared__ float q_scale[(BT / C) * D];
    __shared__ float a_scale[10 * D];

    // Exact segment-relative block-boundary factors.  ws_gt is already in
    // base-2 log space, matching the ex2 convention used by K1/K2.  Four
    // 128-thread cohorts cover one q_scale row and a balanced subset of the
    // ten triangular factors each.  The repeated gate reads add only 6 KiB
    // of contiguous traffic per CTA and expose the formerly serial exp2 work
    // to all eight waves without changing any factor's summation order.
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

    // Compute the ten stable lower-triangular pair scales once per CTA.  All
    // sixteen lanes that share a feature fragment reuse these values in every
    // QK MFMA instead of redundantly evaluating exp2.
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

    // Prefetch the complete V'=C@R tile.  Four bf16x4 values per thread cover
    // 64x128 exactly; invalid tail rows are explicitly zeroed before the
    // transposed LDS store used by A@V'.
    constexpr int VEC_PER_ROW = D / 4;
    constexpr int VN_ITERS = (BT * VEC_PER_ROW) / (NW * 64);
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
    __syncthreads();

    // O_T_M=4/O_T_N=2: pairs of waves share a 16-row output tile and each
    // wave owns 64 output columns.  The same 4x2 split maps A[64,64], with two
    // 16-column A fragments per wave.
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

    // Register prefetch for one K64 panel.  q/ki each need two bf16x4 values
    // per thread and H_entry needs four.  Loading panel bk+1 before computing
    // bk overlaps its HBM latency with the eight MFMA chains below.
    constexpr int QK_ITERS = (BT * (BK / 4)) / (NW * 64);
    constexpr int H_ITERS = (D * (BK / 4)) / (NW * 64);
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
            // vprime_T is not materialized until after the K-panel loop, so
            // use its first 64 rows as a temporary stable local-qd panel.
            *reinterpret_cast<bf16x4*>(vprime_T + row * STRIDE_K + col4) =
                q_local_reg[it];
            *reinterpret_cast<bf16x4*>(a_or_ki + row * STRIDE_K + col4) =
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

    #pragma unroll
    for (int panel = 0; panel < D / BK; ++panel) {
        if (panel + 1 < D / BK)
            load_panel(panel + 1);

        bt64_out_tiled_gemm<O_E_N, BK / C>(
            o_cross, q_panel, o_m_base, STRIDE_K,
            rhs_panel, o_n_base, STRIDE_K, lane);
        // Stable lower-triangular qd@ki^T.  Each wave owns two candidate
        // 16-column tiles; upper tiles are skipped.  Scaling the restored key
        // fragment by only the diagonal or intervening chunk gates avoids the
        // overflowing segment-global inverse-decay operand.
        const int row_chunk = o_m_base / C;
        #pragma unroll
        for (int ek = 0; ek < BK / C; ++ek) {
            const bf16x4 af = bt64_out_load_mfma_fragment(
                vprime_T, o_m_base, ek * C, STRIDE_K, lane);
            #pragma unroll
            for (int en = 0; en < A_E_N; ++en) {
                const int col_base = a_n_base + en * C;
                const int col_chunk = col_base / C;
                if (col_chunk <= row_chunk) {
                    bf16x4 bf = bt64_out_load_mfma_fragment(
                        a_or_ki, col_base, ek * C, STRIDE_K, lane);
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

        // No wave may overwrite the current LDS panel until all eight waves
        // have issued their final MFMA operand reads.
        __syncthreads();
        if (panel + 1 < D / BK) {
            commit_panel();
            __syncthreads();
        }
    }

    // Convert the register-resident A to its inclusive-causal BF16 operand.
    // The same LDS allocation previously holding ki is now dead and reused.
    #pragma unroll
    for (int en = 0; en < A_E_N; ++en) {
        const int col = a_n_base + en * C + (lane & 15);
        const int row4 = o_m_base + ((lane >> 4) << 2);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int row = row4 + p;
            const float x = (row < alen && col < alen && row >= col)
                ? avec[en][p] : 0.0f;
            a_or_ki[row * STRIDE_T + col] = f32_to_bf16(x);
        }
    }

    // Materialize the prefetched V' tile transposed as [V, BT].
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
        o_intra, a_or_ki, o_m_base, STRIDE_T,
        vprime_T, o_n_base, STRIDE_T, lane);

    // Each output element is uniquely owned by one lane.  Keep both MFMA
    // paths in FP32 through the add and round only once at the final store.
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

}  // namespace flashkda_hip
