// GDN Prefill K2 Scan-Only Kernel — phases a/b/b'/d
// Produces h_snap (fp32) and v_new (bf16) for the parallel output kernel.
// Eliminates phases c (cross-chunk) and e (intra-chunk output) to:
//   1. Reduce per-chunk work (~55% fewer MFMA instructions, ~35% fewer syncs)
//   2. Remove q/s_q from LDS → enable OCC_HINT=2 on gfx942
//   3. Eliminate duplicate k reads (single pf_k buffer)
//
// Grid: (cdiv(V, BV), B*H)   Block: (BLOCK_SIZE)
// h state: register-resident fp32, same MFMA accumulator layout as K2
// Chunks: serial iteration within each workgroup
//
// h_snap layout: [B, NT, H, K, V] fp32 — K-major for coalesced VMEM writes
// v_new layout: [B, T, H, V] bf16 — standard row-major
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, 2)
gdn_k2_scan_kernel(gdn_k2_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    constexpr int BT     = T::BT;
    constexpr int BK_SUB = T::BK_SUB;
    constexpr int BV     = T::BV;
    constexpr int N_K    = T::N_K;
    constexpr int BS     = T::BLOCK_SIZE;
    constexpr int WS     = T::WARP_SIZE;
    constexpr int PAD    = T::SMEM_PAD;
    constexpr int W      = 16;

    // h-accumulate GEMM tiling (same as K2)
    constexpr int H_T_M = (T::NUM_WARPS < BK_SUB / W) ? T::NUM_WARPS : BK_SUB / W;
    constexpr int H_T_N = T::NUM_WARPS / H_T_M;
    constexpr int H_E_M = BK_SUB / (W * H_T_M);
    constexpr int H_E_N = BV / (W * H_T_N);
    constexpr int H_E_K = BT / W;
    static_assert(H_E_N > 0, "Too many warps for BV dimension (h state)");

    // Retrieve GEMM tiling: w_bar[BT, BK_SUB] × h[BK_SUB, BV] → [BT, BV]
    constexpr bool BT_LARGE = (BT >= 32);
    constexpr int O_T_M = BT_LARGE
        ? ((T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W) : 1;
    constexpr int O_T_N = T::NUM_WARPS / O_T_M;
    constexpr int O_E_M = BT / (W * O_T_M);
    constexpr int O_E_N = BV / (W * O_T_N);
    constexpr int O_E_K = BK_SUB / W;
    static_assert(O_E_N > 0, "Too many warps for BV dimension (retrieve)");

    // LDS strides
    constexpr int STRIDE_BK = BK_SUB + PAD;
    constexpr int STRIDE_BT = BT + PAD;

    // Thread identity
    const int i_v  = blockIdx.x;
    const int i_nh = blockIdx.y;
    const int i_n  = i_nh / kargs.H;
    const int i_h  = i_nh % kargs.H;
    const int tid  = threadIdx.x;
    const int warp_id = tid / WS;
    const int lane_id = tid % WS;

    const int v_off = i_v * BV;
    const int bos   = i_n * kargs.T;
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;
    const int NT = kargs.NT;

    const int stride_k = H * K;
    const int stride_v = H * V;
    const int stride_g = H;

    // MFMA warp base offsets
    const int h_m_base = (warp_id / H_T_N) * W;
    const int h_n_base = (warp_id % H_T_N) * (H_E_N * W);

    int o_m_base, o_n_base;
    if constexpr (BT_LARGE) {
        o_m_base = (warp_id / O_T_N) * (O_E_M * W);
        o_n_base = (warp_id % O_T_N) * (O_E_N * W);
    } else {
        o_m_base = 0;
        o_n_base = warp_id * W;
    }

    // Register-resident h state
    v4f32_t h1[H_E_N];
    v4f32_t h2[H_E_N];
    clear_v4f32<H_E_N>(h1);
    if constexpr (N_K >= 2) clear_v4f32<H_E_N>(h2);

    if (kargs.ptr_h0 != nullptr) {
        const D_ACC* h0 = reinterpret_cast<const D_ACC*>(kargs.ptr_h0)
                          + (i_n * H + i_h) * V * K;
        for (int en = 0; en < H_E_N; en++) {
            for (int p = 0; p < 4; p++) {
                int row = h_m_base + (lane_id >> 4) * 4 + p;
                int col = h_n_base + en * W + (lane_id & 15);
                if (v_off + col < V) {
                    h1[en][p] = h0[(v_off + col) * K + row];
                    if constexpr (N_K >= 2)
                        h2[en][p] = h0[(v_off + col) * K + BK_SUB + row];
                }
            }
        }
    }

    // Shared memory layout (no s_q — saves 17KB for occupancy 2)
    //   s_g[BT] fp32:              persistent gate cumsum
    //   s_v_T[BV, STRIDE_BT] bf16: v_new transposed (phases b'→d)
    //   pool[...]:                  aliased per phase
    constexpr int smem_g_bytes  = BT * (int)sizeof(D_ACC);
    constexpr int smem_vT_bytes = BV * STRIDE_BT * (int)sizeof(D_ATTN);

    extern __shared__ char smem_buf[];
    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ATTN* s_v_T  = reinterpret_cast<D_ATTN*>(smem_buf + smem_g_bytes);
    D_ATTN* s_pool = reinterpret_cast<D_ATTN*>(smem_buf + smem_g_bytes + smem_vT_bytes);

    // HBM base pointers (no q)
    const D_ATTN* k_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                          + (bos * H + i_h) * K;
    const D_ATTN* w_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_w_bar)
                          + (bos * H + i_h) * K;
    const D_ATTN* u_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_u_bar)
                          + (bos * H + i_h) * V;
    const D_ACC*  g_hbm = reinterpret_cast<const D_ACC*>(kargs.ptr_g_cumsum)
                          + bos * H + i_h;

    // h_snap output: [B, NT, H, K, V] fp32 — K-major for coalesced writes
    D_ACC* snap_hbm = reinterpret_cast<D_ACC*>(kargs.ptr_h_snap);
    // v_new output
    D_ATTN* vn_hbm = reinterpret_cast<D_ATTN*>(kargs.ptr_v_new)
                     + (bos * H + i_h) * V;

    // Prefetch buffers: pf_w (w_bar), pf_k (k — single buffer, no duplicate)
    constexpr int PF_VEC   = 4;
    constexpr int PF_NVEC  = BK_SUB / PF_VEC;
    constexpr int PF_ELEMS = BT * PF_NVEC;
    constexpr int PF_LOADS = (PF_ELEMS + BS - 1) / BS;

    v4bf16_t pf_w[PF_LOADS];

    // Prologue: prefetch chunk 0, bk=0 w_bar
    {
        #pragma unroll
        for (int li = 0; li < PF_LOADS; li++) {
            int i = tid + li * BS;
            pf_w[li] = {};
            if (i < PF_ELEMS) {
                int row = i / PF_NVEC;
                int col = (i % PF_NVEC) * PF_VEC;
                if (row < kargs.T)
                    pf_w[li] = *reinterpret_cast<const v4bf16_t*>(
                        &w_hbm[row * stride_k + col]);
            }
        }
    }

    // =====================================================================
    // Chunk-serial main loop
    // =====================================================================
    constexpr int C_ELEMS = O_E_M * O_E_N;

    for (int i_t = 0; i_t < NT; i_t++) {
        const int t0 = i_t * BT;
        const D_ATTN* k_ch = k_hbm + (int64_t)t0 * stride_k;
        const D_ATTN* w_ch = w_hbm + (int64_t)t0 * stride_k;
        const D_ATTN* u_ch = u_hbm + (int64_t)t0 * stride_v;
        D_ATTN* vn_ch = vn_hbm + (int64_t)t0 * stride_v;

        const int T_rem = kargs.T - t0;
        const bool full_chunk = (T_rem >= BT);
        const bool v_full = (v_off + BV <= V);

        // Load g_cumsum
        for (int i = tid; i < BT; i += BS)
            s_g[i] = (i < T_rem) ? g_hbm[(t0 + i) * stride_g] : 0.0f;
        __syncthreads();

        int last_valid = full_chunk ? (BT - 1) : (T_rem - 1);
        D_ACC g_last = s_g[last_valid];

        // =============================================================
        // (a) Store h_snap to HBM — [K, V] layout for coalesced writes
        // =============================================================
        if (snap_hbm != nullptr) {
            D_ACC* snap = snap_hbm
                + ((int64_t)i_n * NT * H + (int64_t)i_t * H + i_h) * K * V;
            for (int en = 0; en < H_E_N; en++) {
                for (int p = 0; p < 4; p++) {
                    int row = h_m_base + (lane_id >> 4) * 4 + p;
                    int col = h_n_base + en * W + (lane_id & 15);
                    if (v_off + col < V) {
                        snap[row * V + (v_off + col)] = h1[en][p];
                        if constexpr (N_K >= 2)
                            snap[(BK_SUB + row) * V + (v_off + col)] = h2[en][p];
                    }
                }
            }
        }

        // =============================================================
        // (b) Retrieve via MFMA: retrieve[BT,BV] = Σ_bk w_bar × h
        // No cross-chunk GEMM (phase c removed)
        // =============================================================
        v4f32_t r_retrieve[C_ELEMS];
        clear_v4f32<C_ELEMS>(r_retrieve);

        D_ATTN* s_h_T   = s_pool;
        D_ATTN* s_sub_w  = s_pool + BV * STRIDE_BK;

        for (int bk = 0; bk < N_K; bk++) {
            v4f32_t* h_cur = (bk == 0) ? h1 : h2;

            // Spill h[bk] → s_h_T transposed
            for (int en = 0; en < H_E_N; en++) {
                int row_T = h_n_base + en * W + (lane_id & 15);
                int col_T = h_m_base + (lane_id >> 4) * 4;
                v4bf16_t val;
                for (int p = 0; p < 4; p++)
                    val[p] = static_cast<D_ATTN>(h_cur[en][p]);
                *reinterpret_cast<v4bf16_t*>(&s_h_T[row_T * STRIDE_BK + col_T]) = val;
            }

            // Install pf_w → s_sub_w
            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < PF_ELEMS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    *reinterpret_cast<v4bf16_t*>(&s_sub_w[row * STRIDE_BK + col]) = pf_w[li];
                }
            }
            __syncthreads();

            // Prefetch next bk's w from HBM
            if (bk + 1 < N_K) {
                int k_off_next = (bk + 1) * BK_SUB;
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    pf_w[li] = {};
                    if (i < PF_ELEMS) {
                        int row = i / PF_NVEC;
                        int col = (i % PF_NVEC) * PF_VEC;
                        if (full_chunk || row < T_rem)
                            pf_w[li] = *reinterpret_cast<const v4bf16_t*>(
                                &w_ch[row * stride_k + k_off_next + col]);
                    }
                }
            }

            // Retrieve GEMM: r_retrieve += s_sub_w × s_h_T
            tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                r_retrieve, s_sub_w, o_m_base, STRIDE_BK,
                            s_h_T,   o_n_base, STRIDE_BK, lane_id);
            __syncthreads();
        }

        // =============================================================
        // (b') v_new = u_bar - retrieve
        // Store transposed to s_v_T and also write bf16 to HBM
        // =============================================================
        for (int i = 0; i < C_ELEMS; i++) {
            int en = i % O_E_N;
            int s_base, c;
            if constexpr (BT_LARGE) {
                s_base = o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4;
                c = o_n_base + en * W + (lane_id & 15);
            } else {
                s_base = (lane_id >> 4) * 4;
                c = o_n_base + en * W + (lane_id & 15);
            }
            v4bf16_t v_new_pack;
            for (int p = 0; p < 4; p++) {
                int s = s_base + p;
                D_ACC u_val = 0.0f;
                if ((full_chunk || s < T_rem) && (v_full || v_off + c < V))
                    u_val = static_cast<D_ACC>(
                        u_ch[s * stride_v + v_off + c]);
                D_ACC v_new_val = u_val - r_retrieve[i][p];
                D_ATTN v_new_bf16 = static_cast<D_ATTN>(v_new_val);
                v_new_pack[p] = v_new_bf16;

                // Always write v_new to HBM (needed by output kernel)
                if ((full_chunk || s < T_rem) && (v_full || v_off + c < V))
                    vn_ch[s * stride_v + v_off + c] = v_new_bf16;
            }
            *reinterpret_cast<v4bf16_t*>(&s_v_T[c * STRIDE_BT + s_base]) = v_new_pack;
        }

        // Prefetch k[bk=0] for phase d (single load, no duplicate)
        v4bf16_t pf_k[PF_LOADS];
        #pragma unroll
        for (int li = 0; li < PF_LOADS; li++) {
            int i = tid + li * BS;
            pf_k[li] = {};
            if (i < PF_ELEMS) {
                int row = i / PF_NVEC;
                int col = (i % PF_NVEC) * PF_VEC;
                if (full_chunk || row < T_rem)
                    pf_k[li] = *reinterpret_cast<const v4bf16_t*>(
                        &k_ch[row * stride_k + col]);
            }
        }

        // =============================================================
        // (d) Decay h + Accumulate h += k_gated^T × v_new
        // =============================================================
        D_ACC decay = fast_exp(g_last);
        for (int en = 0; en < H_E_N; en++)
            for (int p = 0; p < 4; p++) {
                h1[en][p] *= decay;
                if constexpr (N_K >= 2) h2[en][p] *= decay;
            }

        D_ATTN* s_k_T = s_pool;

        for (int bk = 0; bk < N_K; bk++) {
            v4f32_t* h_cur = (bk == 0) ? h1 : h2;

            // Install pf_k with gating → s_k_T transposed
            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                if (i < PF_ELEMS) {
                    int s = i / PF_NVEC;
                    int j = (i % PF_NVEC) * PF_VEC;
                    D_ACC gate = (full_chunk || s < T_rem)
                        ? fast_exp(g_last - s_g[s]) : 0.0f;
                    for (int vi = 0; vi < PF_VEC; vi++)
                        s_k_T[(j + vi) * STRIDE_BT + s] = static_cast<D_ATTN>(
                            static_cast<D_ACC>(pf_k[li][vi]) * gate);
                }
            }
            __syncthreads();

            // Prefetch k[bk+1]
            if (bk + 1 < N_K) {
                int k_off_next = (bk + 1) * BK_SUB;
                #pragma unroll
                for (int li = 0; li < PF_LOADS; li++) {
                    int i = tid + li * BS;
                    pf_k[li] = {};
                    if (i < PF_ELEMS) {
                        int s = i / PF_NVEC;
                        int j = (i % PF_NVEC) * PF_VEC;
                        if (full_chunk || s < T_rem)
                            pf_k[li] = *reinterpret_cast<const v4bf16_t*>(
                                &k_ch[s * stride_k + k_off_next + j]);
                    }
                }
            }

            // GEMM: h[bk] += k_gated^T × v_new^T
            tiled_gemm_mfma<H_E_M, H_E_N, H_E_K>(
                h_cur,
                s_k_T, h_m_base, STRIDE_BT,
                s_v_T, h_n_base, STRIDE_BT,
                lane_id);
            __syncthreads();
        }

        // Prefetch next chunk's w_bar[bk=0]
        if (i_t + 1 < NT) {
            int next_t0 = (i_t + 1) * BT;
            #pragma unroll
            for (int li = 0; li < PF_LOADS; li++) {
                int i = tid + li * BS;
                pf_w[li] = {};
                if (i < PF_ELEMS) {
                    int row = i / PF_NVEC;
                    int col = (i % PF_NVEC) * PF_VEC;
                    if (next_t0 + row < kargs.T)
                        pf_w[li] = *reinterpret_cast<const v4bf16_t*>(
                            &w_hbm[(int64_t)(next_t0 + row) * stride_k + col]);
                }
            }
        }

    }  // end chunk loop

    // Epilogue: store final h → ptr_ht
    if (kargs.ptr_ht != nullptr) {
        D_ACC* ht = reinterpret_cast<D_ACC*>(kargs.ptr_ht)
                    + (i_n * H + i_h) * V * K;
        for (int en = 0; en < H_E_N; en++) {
            for (int p = 0; p < 4; p++) {
                int row = h_m_base + (lane_id >> 4) * 4 + p;
                int col = h_n_base + en * W + (lane_id & 15);
                if (v_off + col < V) {
                    ht[(v_off + col) * K + row] = h1[en][p];
                    if constexpr (N_K >= 2)
                        ht[(v_off + col) * K + BK_SUB + row] = h2[en][p];
                }
            }
        }
    }
}
