// gfx942 compact-Aqk BT64 chunk-parallel output kernel.
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
// K is contracted in K16 order exactly like the production out_bk32 kernel,
// including its BF16 q scaling.  BK and DOUBLE_BUFFER are compile-time tuning
// controls used by the gfx942 microbenchmarks: production keeps BK32 single
// buffering by default, while BK64 reduces the number of panel fences and a
// BK32 ping-pong layout removes the destructive-overwrite fence.  A@V' visits
// only the ten lower-triangular tiles.  Both phases alias one raw LDS allocation
// and every supported layout retains at least two resident 512-thread
// workgroups in gfx942's 64 KiB LDS.
#pragma once

#include <hip/hip_runtime.h>

#include "../k2_kda_csplit_bt64_out_kernel.hpp"

namespace flashkda_hip::gfx942 {

__device__ __forceinline__ constexpr int out_aqk_v2_tri_tile(int r, int c) {
    return r * (r + 1) / 2 + c;
}

template <bool VL = false, int BK = 32, bool DOUBLE_BUFFER = false>
__global__ void __launch_bounds__(512)
k2_kda_csplit_bt64_out_aqk_v2_kernel(
        const __bf16* __restrict__ cs_u,       // [tile,16,V], V'=C@R
        const __bf16* __restrict__ cs_sin,    // [segment,V,K], H_entry
        const __bf16* __restrict__ ws_qd,     // [tile,16,K], chunk-local qd
        const float* __restrict__ ws_gt,       // [tile,K], base-2 log decay
        const __bf16* __restrict__ segment_a, // [segment,10,16,16]
        __bf16* __restrict__ out_g,            // [T_total,H,V]
        const int32_t* __restrict__ cu_seqlens,
        const int* __restrict__ tile_prefix,
        const int* __restrict__ segment_prefix,
        int N, int total_tiles, int total_segments,
        int T_seq, int H, int NT) {
    constexpr int C = 16;
    constexpr int BT = 64;
    constexpr int D = 128;
    constexpr int NW = 8;
    constexpr int PANEL_BUFFERS = DOUBLE_BUFFER ? 2 : 1;
    constexpr int NUM_PANELS = D / BK;
    constexpr int STRIDE_K = BK + 4;
    constexpr int STRIDE_T = BT + 4;
    constexpr int TRI_TILES = 10;
    constexpr int TILE_ELEMS = C * C;
    constexpr int O_T_N = 2;
    constexpr int O_E_N = D / (C * O_T_N);
    static_assert(O_E_N == 4);
    static_assert(BK == 32 || BK == 64);
    static_assert(!(BK == 64 && DOUBLE_BUFFER),
                  "BK64 already amortizes panel barriers; double buffering "
                  "would exceed the two-CTA LDS budget");
    static_assert(D % BK == 0 && BK % C == 0);

    const int tid = static_cast<int>(threadIdx.x);
    const int wave = tid >> 6;
    const int lane = tid & 63;

    int h, ht_base, ss, t0_base, alen, chunks;
    if constexpr (VL) {
        const int gsi = static_cast<int>(blockIdx.x);
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
        const int seg = static_cast<int>(blockIdx.x);
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

    // K phase, parameterized by BK and buffer count:
    //   scaled Q  PANEL_BUFFERS * [64,BK+4]
    //   H panel   PANEL_BUFFERS * [128,BK+4]
    //   q_scale   [4,128] f32
    // BK64/single-buffer consumes 28,160 B; BK32/ping-pong 29,696 B.
    // Final phase:
    //   compact A [10,16,16] bf16  5,120 B
    //   V'^T      [128,68] bf16   17,408 B
    //                                  22,528 B
    constexpr int Q_PANEL_ELEMS = BT * STRIDE_K;
    constexpr int H_PANEL_ELEMS = D * STRIDE_K;
    constexpr int Q_PANEL_BYTES = Q_PANEL_ELEMS * sizeof(__bf16);
    constexpr int H_PANEL_BYTES = H_PANEL_ELEMS * sizeof(__bf16);
    constexpr int Q_BYTES = PANEL_BUFFERS * Q_PANEL_BYTES;
    constexpr int H_BYTES = PANEL_BUFFERS * H_PANEL_BYTES;
    constexpr int QSCALE_BYTES = (BT / C) * D * sizeof(float);
    constexpr int K_PHASE_BYTES = Q_BYTES + H_BYTES + QSCALE_BYTES;
    constexpr int A_BYTES = TRI_TILES * TILE_ELEMS * sizeof(__bf16);
    constexpr int VT_BYTES = D * STRIDE_T * sizeof(__bf16);
    constexpr int FINAL_PHASE_BYTES = A_BYTES + VT_BYTES;
    constexpr int SMEM_BYTES =
        K_PHASE_BYTES > FINAL_PHASE_BYTES ? K_PHASE_BYTES : FINAL_PHASE_BYTES;
    static_assert(QSCALE_BYTES == 2048);
    static_assert(A_BYTES == 5120);
    static_assert(VT_BYTES == 17408);
    static_assert((BK != 64 || DOUBLE_BUFFER || K_PHASE_BYTES == 28160));
    static_assert((BK != 32 || !DOUBLE_BUFFER || K_PHASE_BYTES == 29696));
    static_assert((BK != 32 || DOUBLE_BUFFER || K_PHASE_BYTES == 15872));
    static_assert(SMEM_BYTES <= 30 * 1024);

    __shared__ __align__(16) unsigned char smem[SMEM_BYTES];
    auto* q_panel = reinterpret_cast<__bf16*>(smem);
    auto* h_panel = reinterpret_cast<__bf16*>(smem + Q_BYTES);
    auto* q_scale = reinterpret_cast<float*>(smem + Q_BYTES + H_BYTES);
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

    f32x4 o_cross[O_E_N];
    #pragma unroll
    for (int en = 0; en < O_E_N; ++en)
        o_cross[en] = f32x4{0.0f, 0.0f, 0.0f, 0.0f};

    constexpr int Q_ITERS = (BT * (BK / 4)) / (NW * 64);
    constexpr int H_ITERS = (D * (BK / 4)) / (NW * 64);
    static_assert(Q_ITERS == BK / 32 && H_ITERS == BK / 16);
    bf16x4 q_reg[Q_ITERS];
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
            if (row < alen) {
                const bf16x4 qsrc = *reinterpret_cast<const bf16x4*>(
                    ws_qd + (int64_t(ht_base + block) * C + local_row) * D + d0);
                const f32x4 scale = *reinterpret_cast<const f32x4*>(
                    q_scale + block * D + d0);
                #pragma unroll
                for (int p = 0; p < 4; ++p)
                    qv[p] = f32_to_bf16(bf16_to_f32(qsrc[p]) * scale[p]);
            }
            q_reg[it] = qv;
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

    auto commit_panel = [&](int buffer) {
        __bf16* const q_dst = q_panel + buffer * Q_PANEL_ELEMS;
        __bf16* const h_dst = h_panel + buffer * H_PANEL_ELEMS;
        #pragma unroll
        for (int it = 0; it < Q_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int row = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            *reinterpret_cast<bf16x4*>(q_dst + row * STRIDE_K + col4) =
                q_reg[it];
        }
        #pragma unroll
        for (int it = 0; it < H_ITERS; ++it) {
            const int i = tid + it * NW * 64;
            const int v = i / (BK / 4);
            const int col4 = (i % (BK / 4)) * 4;
            *reinterpret_cast<bf16x4*>(h_dst + v * STRIDE_K + col4) =
                h_reg[it];
        }
    };

    load_panel(0);
    commit_panel(0);
    __syncthreads();

    #pragma unroll
    for (int panel = 0; panel < NUM_PANELS; ++panel) {
        if (panel + 1 < NUM_PANELS)
            load_panel(panel + 1);

        const int read_buffer = panel % PANEL_BUFFERS;
        const __bf16* const q_read =
            q_panel + read_buffer * Q_PANEL_ELEMS;
        const __bf16* const h_read =
            h_panel + read_buffer * H_PANEL_ELEMS;

        bt64_out_tiled_gemm<O_E_N, BK / C>(
            o_cross, q_read, o_m_base, STRIDE_K,
            h_read, o_n_base, STRIDE_K, lane);

        if (panel + 1 < NUM_PANELS) {
            if constexpr (!DOUBLE_BUFFER)
                __syncthreads();
            commit_panel((panel + 1) % PANEL_BUFFERS);
            __syncthreads();
        } else {
            // K reads must complete before final A/V' overwrites aliased LDS.
            __syncthreads();
        }
    }

    // K-phase data is dead.  Load the compact ten-tile A once per CTA while
    // publishing register-resident V' in the disjoint final-phase region.
    constexpr int A_VEC4 = TRI_TILES * TILE_ELEMS / 4;
    const auto* a_src = reinterpret_cast<const bf16x4*>(
        segment_a + int64_t(ss) * TRI_TILES * TILE_ELEMS);
    auto* a_dst = reinterpret_cast<bf16x4*>(a_panel);
    for (int vi = tid; vi < A_VEC4; vi += NW * 64)
        a_dst[vi] = a_src[vi];

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

    const int row_chunk = o_m_base / C;
    #pragma unroll
    for (int c = 0; c < FACTOR_GROUPS; ++c) {
        if (c <= row_chunk) {
            const __bf16* tile =
                a_panel + out_aqk_v2_tri_tile(row_chunk, c) * TILE_ELEMS;
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

}  // namespace flashkda_hip::gfx942
