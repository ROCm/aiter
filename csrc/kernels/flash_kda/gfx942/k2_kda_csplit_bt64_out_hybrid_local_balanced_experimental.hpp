// gfx942 experimental load-balanced hybrid local-Mqk/cross-Aqk output.
//
// Four exact diagonal tiles arrive from fused prepare.  This kernel rebuilds
// only the six cross-chunk qd@ki^T tiles, then consumes the packed ten-tile A
// with the sparse lower-block A@V schedule.
//
// One 8-wave CTA owns one complete (sequence, head, BT64 segment) and all
// V=128 output columns, preserving the parallel launch geometry of the tuned
// out_bk32 P4.  Unlike that kernel, the V-independent causal matrix
//
//   A = tril(qd_segment @ ki_segment^T)
//
// is supplied as ten packed BF16 16x16 tiles by an earlier producer:
//
//   segment_a[segment, tri(r,c), 16, 16], c <= r.
//
// This removes P4's q_local/ki panels, ten FP32 A-scale vectors, eighty QK
// MFMAs, and the dense upper-zero A@V' work.  The remaining operations are
//
//   O = scaled_qd @ H_entry^T + A @ V'.
//
// K is contracted in ascending K16 order exactly like the production out_bk32
// kernel, including its BF16 q scaling.  BK is a compile-time tuning control:
// BK64 halves the number of destructive panel fences.  DIRECT_QK lets the six
// active cross-tile waves read their Q/K fragments directly from the packed
// workspace; this trades modest L2 duplication for removing 17 KiB of LDS and
// preserves two-CTA LDS residency for BK64.  A@V' visits only the ten
// lower-triangular tiles and aliases the K phase allocation.
#pragma once

#include <hip/hip_runtime.h>

#include "../k2_kda_csplit_bt64_out_kernel.hpp"

namespace flashkda_hip::gfx942 {

__device__ __forceinline__ constexpr int out_hybrid_balanced_tri_tile(int r, int c) {
    return r * (r + 1) / 2 + c;
}

__device__ __forceinline__ constexpr int out_hybrid_balanced_cross_tile(
        int r, int c) {
    return r * (r - 1) / 2 + c;
}

__device__ __forceinline__ constexpr int out_hybrid_cross_row(int pair) {
    return pair < 1 ? 1 : pair < 3 ? 2 : 3;
}

__device__ __forceinline__ constexpr int out_hybrid_cross_col(int pair) {
    const int r = out_hybrid_cross_row(pair);
    return pair - r * (r - 1) / 2;
}

template <bool VL, int BK, bool DIRECT_QK, bool SEGMENT_RANGE>
__device__ __forceinline__ void
k2_kda_csplit_bt64_out_hybrid_local_balanced_body(
        const __bf16* __restrict__ cs_u,       // [tile,16,V], V'=C@R
        const __bf16* __restrict__ cs_sin,    // [segment,V,K], H_entry
        const __bf16* __restrict__ ws_qd,     // [tile,16,K], chunk-local qd
        const __bf16* __restrict__ ws_kr,     // [tile,16,K], bounded suffix K
        const float* __restrict__ ws_gt,       // [tile,K], base-2 log decay
        const __bf16* __restrict__ local_mqk, // [segment,4,16,16]
        __bf16* __restrict__ out_g,            // [T_total,H,V]
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT,
        int segment_begin, int segment_count) {
    constexpr int C = 16;
    constexpr int BT = 64;
    constexpr int D = 128;
    constexpr int NW = 8;
    constexpr int NUM_PANELS = D / BK;
    constexpr int STRIDE_K = BK + 4;
    constexpr int STRIDE_T = BT + 4;
    constexpr int TRI_TILES = 10;
    constexpr int CROSS_TILES = 6;
    constexpr int TILE_ELEMS = C * C;
    constexpr int O_T_N = 2;
    constexpr int O_E_N = D / (C * O_T_N);
    static_assert(O_E_N == 4);
    static_assert(BK == 32 || BK == 64);
    static_assert(D % BK == 0 && BK % C == 0);

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;
    const int range_block = static_cast<int>(blockIdx.x);
    if constexpr (SEGMENT_RANGE) {
        // The range launcher already uses grid.x=segment_count.  Retaining the
        // explicit bound in the kernel keeps the standalone contract safe if
        // a caller deliberately overlaunches it.
        if (range_block >= segment_count)
            return;
    }

    int h, ht_base, ss, t0_base, alen, chunks;
    if constexpr (VL) {
        const int gsi = range_block +
            (SEGMENT_RANGE ? segment_begin : 0);
        h = static_cast<int>(blockIdx.y);

        int lo = 0;
        int hi = N;
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
        if (local_seg < 0 || local_seg >= nseg)
            return;

        ht_base = h * total_tiles + tile_prefix[lo] + local_seg * (BT / C);
        ss = h * total_segments + gsi;
        t0_base = int(bos) + local_seg * BT;
        alen = min(BT, seq_len - local_seg * BT);
        chunks = (alen + C - 1) / C;
    } else {
        const int seg = range_block +
            (SEGMENT_RANGE ? segment_begin : 0);
        const int bh = static_cast<int>(blockIdx.y);
        const int nseg = (NT + (BT / C) - 1) / (BT / C);
        if (seg >= nseg)
            return;

        const int b = bh / H;
        h = bh % H;
        ht_base = bh * NT + seg * (BT / C);
        ss = bh * nseg + seg;
        t0_base = b * T_seq + seg * BT;
        alen = min(BT, T_seq - seg * BT);
        chunks = (alen + C - 1) / C;
    }

    // K phase: scaled Q, optional staged local Q/bounded Kr, H panel, four Q
    // prefix scales, and six cross-pair scales.  The BK64 DIRECT_QK layout is
    // 31,232 B and therefore retains two workgroups in gfx942's 64 KiB LDS;
    // staged BK64 is kept as an A/B control despite its one-CTA residency.
    constexpr int Q_BYTES = BT * STRIDE_K * sizeof(__bf16);
    constexpr int QLOCAL_BYTES = DIRECT_QK ? 0 : Q_BYTES;
    constexpr int KI_BYTES = DIRECT_QK ? 0 : Q_BYTES;
    constexpr int H_BYTES = D * STRIDE_K * sizeof(__bf16);
    constexpr int QSCALE_BYTES = (BT / C) * D * sizeof(float);
    constexpr int ASCALE_BYTES = CROSS_TILES * D * sizeof(float);
    constexpr int Q_OFF = 0;
    constexpr int QLOCAL_OFF = Q_OFF + Q_BYTES;
    constexpr int KI_OFF = QLOCAL_OFF + QLOCAL_BYTES;
    constexpr int H_OFF = KI_OFF + KI_BYTES;
    constexpr int QSCALE_OFF = H_OFF + H_BYTES;
    constexpr int ASCALE_OFF = QSCALE_OFF + QSCALE_BYTES;
    constexpr int K_PHASE_BYTES = ASCALE_OFF + ASCALE_BYTES;
    constexpr int A_BYTES = TRI_TILES * TILE_ELEMS * sizeof(__bf16);
    constexpr int VT_BYTES = D * STRIDE_T * sizeof(__bf16);
    constexpr int FINAL_PHASE_BYTES = A_BYTES + VT_BYTES;
    constexpr int SMEM_BYTES =
        K_PHASE_BYTES > FINAL_PHASE_BYTES ? K_PHASE_BYTES : FINAL_PHASE_BYTES;
    static_assert(QSCALE_BYTES == 2048 && ASCALE_BYTES == 3072);
    static_assert((BK != 32 || DIRECT_QK || K_PHASE_BYTES == 28160));
    static_assert((BK != 64 || DIRECT_QK || K_PHASE_BYTES == 48640));
    static_assert((BK != 64 || !DIRECT_QK || K_PHASE_BYTES == 31232));
    static_assert(A_BYTES == 5120 && VT_BYTES == 17408);
    static_assert(FINAL_PHASE_BYTES == 22528);
    static_assert(FINAL_PHASE_BYTES <= SMEM_BYTES);
    static_assert(SMEM_BYTES <= 48 * 1024);

    __shared__ __align__(16) unsigned char smem[SMEM_BYTES];
    auto* q_panel = reinterpret_cast<__bf16*>(smem + Q_OFF);
    auto* q_local_panel = reinterpret_cast<__bf16*>(smem + QLOCAL_OFF);
    auto* ki_panel = reinterpret_cast<__bf16*>(smem + KI_OFF);
    auto* h_panel = reinterpret_cast<__bf16*>(smem + H_OFF);
    auto* q_scale = reinterpret_cast<float*>(smem + QSCALE_OFF);
    auto* a_scale = reinterpret_cast<float*>(smem + ASCALE_OFF);
    auto* a_panel = reinterpret_cast<__bf16*>(smem);
    auto* vprime_T = reinterpret_cast<__bf16*>(smem + A_BYTES);

    // Match out_bk32's segment-relative scaled-Q operand exactly.  Every
    // thread owns one [chunk,d] factor and stores it once for all q rows.
    constexpr int FACTOR_GROUPS = BT / C;
    static_assert(NW * 64 == FACTOR_GROUPS * D);
    const int factor_group = tid / D;
    const int factor_d = tid - factor_group * D;

    float gt[FACTOR_GROUPS];
    #pragma unroll
    for (int r = 0; r < FACTOR_GROUPS; ++r)
        gt[r] = r < chunks
            ? ws_gt[int64_t(ht_base + r) * D + factor_d] : 0.0f;

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

    // Only strict cross-chunk pairs are reconstructed.  Keep the production
    // P4 factor ownership and exponent addition order for bitwise identity.
#pragma unroll
    for (int r = 0; r < FACTOR_GROUPS; ++r) {
#pragma unroll
        for (int c = 0; c < r; ++c) {
            const int tri = bt64_out_tri_tile(r, c);
            if (bt64_out_factor_owner(tri) == factor_group) {
                float exponent = 0.0f;
#pragma unroll
                for (int u = 0; u < FACTOR_GROUPS; ++u) {
                    if (u > c && u < r)
                        exponent += gt[u];
                }
                a_scale[out_hybrid_balanced_cross_tile(r, c) * D + factor_d] =
                    ex2(exponent);
            }
        }
    }

    // Keep V' in registers across the K panels.  This overlaps its global
    // fetch with qH and avoids occupying LDS until the aliased final phase.
    constexpr int VEC_PER_ROW = D / 4;
    constexpr int VN_ITERS = (BT * VEC_PER_ROW) / (NW * 64);
    static_assert(VN_ITERS == 4);
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

    const int o_m_base = (wave / O_T_N) * C;
    const int o_n_base = (wave % O_T_N) * (O_E_N * C);
    const int row_chunk = o_m_base / C;
    const int cross_pair = wave;
    const int cross_row = out_hybrid_cross_row(cross_pair);
    const int cross_col = out_hybrid_cross_col(cross_pair);

    f32x4 o_cross[O_E_N];
    f32x4 avec = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int en = 0; en < O_E_N; ++en)
        o_cross[en] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};

    constexpr int Q_ITERS = (BT * (BK / 4)) / (NW * 64);
    constexpr int H_ITERS = (D * (BK / 4)) / (NW * 64);
    static_assert(Q_ITERS == BK / 32 && H_ITERS == BK / 16);
    bf16x4 q_reg[Q_ITERS];
    bf16x4 q_local_reg[Q_ITERS];
    bf16x4 ki_reg[Q_ITERS];
    bf16x4 h_reg[H_ITERS];

    auto load_panel = [&](int panel) {
        #pragma unroll
        for (int it = 0; it < Q_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int row = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            const int block = row / C;
            const int local_row = row % C;
            const int d0 = panel * BK + col4;
            bf16x4 qv{};
            bf16x4 qlocal{};
            bf16x4 kv{};
            if (row < alen) {
                const bf16x4 qsrc = *reinterpret_cast<const bf16x4*>(
                    ws_qd + (int64_t(ht_base + block) * C + local_row) * D + d0);
                const f32x4 scale = *reinterpret_cast<const f32x4*>(
                    q_scale + block * D + d0);
                if constexpr (!DIRECT_QK) {
                    kv = *reinterpret_cast<const bf16x4*>(
                        ws_kr + (int64_t(ht_base + block) * C + local_row) * D
                            + d0);
                    qlocal = qsrc;
                }
                #pragma unroll
                for (int p = 0; p < 4; ++p)
                    qv[p] = f32_to_bf16(bf16_to_f32(qsrc[p]) * scale[p]);
            }
            q_reg[it] = qv;
            q_local_reg[it] = qlocal;
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
        for (int it = 0; it < Q_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int row = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            *reinterpret_cast<bf16x4*>(q_panel + row * STRIDE_K + col4) =
                q_reg[it];
            if constexpr (!DIRECT_QK) {
                *reinterpret_cast<bf16x4*>(
                    q_local_panel + row * STRIDE_K + col4) = q_local_reg[it];
                *reinterpret_cast<bf16x4*>(
                    ki_panel + row * STRIDE_K + col4) = ki_reg[it];
            }
        }
        #pragma unroll
        for (int it = 0; it < H_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int v = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            *reinterpret_cast<bf16x4*>(h_panel + v * STRIDE_K + col4) =
                h_reg[it];
        }
    };

    load_panel(0);
    commit_panel();
    __syncthreads();

    #pragma unroll
    for (int panel = 0; panel < NUM_PANELS; ++panel) {
        if (panel + 1 < NUM_PANELS)
            load_panel(panel + 1);

        bt64_out_tiled_gemm<O_E_N, BK / C>(
            o_cross, q_panel, o_m_base, STRIDE_K,
            h_panel, o_n_base, STRIDE_K, lane);

        // Six strict cross tiles are assigned one-per-wave to waves 0..5.
        // This removes the original row/V ownership imbalance at every panel
        // barrier while preserving each tile's ascending K16 accumulation.
        if (cross_pair < CROSS_TILES) {
#pragma unroll
            for (int ek = 0; ek < BK / C; ++ek) {
                bf16x4 af{};
                bf16x4 bf{};
                if constexpr (DIRECT_QK) {
                    const int row4 = (lane >> 4) << 2;
                    const int q_row = cross_row * C + (lane & 15);
                    const int k_row = cross_col * C + (lane & 15);
                    const int d4 = panel * BK + ek * C + row4;
                    if (q_row < alen) {
                        af = *reinterpret_cast<const bf16x4*>(
                            ws_qd + (int64_t(ht_base + cross_row) * C
                                + (lane & 15)) * D + d4);
                    }
                    if (k_row < alen) {
                        bf = *reinterpret_cast<const bf16x4*>(
                            ws_kr + (int64_t(ht_base + cross_col) * C
                                + (lane & 15)) * D + d4);
                    }
                } else {
                    af = bt64_out_load_mfma_fragment(
                        q_local_panel, cross_row * C, ek * C, STRIDE_K, lane);
                    bf = bt64_out_load_mfma_fragment(
                        ki_panel, cross_col * C, ek * C, STRIDE_K, lane);
                }
                const int d0 = panel * BK + ek * C +
                    ((lane >> 4) << 2);
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    bf[p] = f32_to_bf16(
                        bf16_to_f32(bf[p]) *
                        a_scale[cross_pair * D + d0 + p]);
                }
                avec = mfma_bf16(af, bf, avec);
            }
        }

        __syncthreads();
        if (panel + 1 < NUM_PANELS) {
            commit_panel();
            __syncthreads();
        }
    }

    // K-phase data is dead.  Populate packed A with four exact local tiles
    // and six reconstructed cross tiles while publishing register V'.
    if (tid < 256) {
        const int chunk = tid >> 6;
        const int vec4 = tid & 63;
        const auto* src4 = reinterpret_cast<const bf16x4*>(
            local_mqk + (int64_t(ss) * 4 + chunk) * TILE_ELEMS);
        auto* dst4 = reinterpret_cast<bf16x4*>(
            a_panel + out_hybrid_balanced_tri_tile(chunk, chunk) * TILE_ELEMS);
        dst4[vec4] = src4[vec4];
    }

    if (cross_pair < CROSS_TILES) {
        const int tile = out_hybrid_balanced_tri_tile(cross_row, cross_col);
        const int n = lane & 15;
        const int m4 = (lane >> 4) << 2;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int m = m4 + p;
            const int global_m = cross_row * C + m;
            const int global_n = cross_col * C + n;
            const bool valid = global_m < alen && global_n < alen;
            a_panel[(tile * C + m) * C + n] = valid
                ? f32_to_bf16(avec[p]) : (__bf16)0.0f;
        }
    }

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

    // A is block-lower-triangular.  The production P4 materializes zero upper
    // blocks and performs 16 dense block products per V16 tile.  Visiting only
    // c<=r preserves the nonzero accumulation order and reduces it to ten.
    f32x4 o_intra[O_E_N];
    #pragma unroll
    for (int en = 0; en < O_E_N; ++en)
        o_intra[en] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll
    for (int c = 0; c < FACTOR_GROUPS; ++c) {
        if (c <= row_chunk) {
            const __bf16* tile =
                a_panel + out_hybrid_balanced_tri_tile(row_chunk, c) * TILE_ELEMS;
            const bf16x4 af = bt64_out_load_mfma_fragment(
                tile, 0, 0, C, lane);
            #pragma unroll
            for (int en = 0; en < O_E_N; ++en) {
                const bf16x4 bf = bt64_out_load_mfma_fragment(
                    vprime_T, o_n_base + en * C, c * C, STRIDE_T, lane);
                o_intra[en] = mfma_bf16(af, bf, o_intra[en]);
            }
        }
    }

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

// Preserve the original full-output symbol and kernarg ABI.  The range
// specialization is a separate symbol so the N=1 P3/P4 experiment cannot
// perturb resource allocation or routing for the established hybrid path.
template <bool VL = false, int BK = 32, bool DIRECT_QK = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_out_hybrid_local_balanced_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ local_mqk,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT) {
    k2_kda_csplit_bt64_out_hybrid_local_balanced_body<
        VL, BK, DIRECT_QK, false>(
            cs_u, cs_sin, ws_qd, ws_kr, ws_gt, local_mqk, out_g,
            cu_seqlens, tile_prefix, segment_prefix, N, total_tiles,
            total_segments, T_seq, H, NT, 0, 0);
}

template <bool VL = false, int BK = 32, bool DIRECT_QK = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_out_hybrid_local_balanced_range_kernel(
        const __bf16* __restrict__ cs_u,
        const __bf16* __restrict__ cs_sin,
        const __bf16* __restrict__ ws_qd,
        const __bf16* __restrict__ ws_kr,
        const float* __restrict__ ws_gt,
        const __bf16* __restrict__ local_mqk,
        __bf16* __restrict__ out_g,
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT,
        int segment_begin, int segment_count) {
    k2_kda_csplit_bt64_out_hybrid_local_balanced_body<
        VL, BK, DIRECT_QK, true>(
            cs_u, cs_sin, ws_qd, ws_kr, ws_gt, local_mqk, out_g,
            cu_seqlens, tile_prefix, segment_prefix, N, total_tiles,
            total_segments, T_seq, H, NT, segment_begin, segment_count);
}

}  // namespace flashkda_hip::gfx942
