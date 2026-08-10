// Standalone C-input fused GDN K2 prototype.
//
// The existing Opus K2 consumes materialized WY factors:
//     v_new = u_bar - w_bar @ H.
// This prototype instead consumes the chunk inverse C=(I+L)^-1 and constructs
// the identical value on chip:
//     R     = V - exp(g) * (K @ H)
//     v_new = C @ (beta * R).
//
// It is deliberately dense-only and fixed to the requested gfx942-friendly
// configuration (BT=64, K=V=128, BV=64, 4 waves).  C is staged in the existing
// K2 phase pool after the K@H/Q@H phase, then that storage is reused for the
// gated QK^T matrix.  Thus the only LDS growth over the existing 4-wave K2 is
// s_beta[BT] fp32.
#pragma once

#include "opus_gdn/gdn_k2_c_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

template<typename Traits>
__device__ void gdn_k2_c_kernel_impl(gdn_k2_c_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC = typename T::D_ACC;

    constexpr int BT = T::BT;
    constexpr int BK_SUB = T::BK_SUB;
    constexpr int BV = T::BV;
    constexpr int N_K = T::N_K;
    constexpr int BS = T::BLOCK_SIZE;
    constexpr int WS = T::WARP_SIZE;
    constexpr int PAD = T::SMEM_PAD;
    constexpr int W = 16;
    constexpr int STRIDE_BK = BK_SUB + PAD;
    constexpr int STRIDE_BT = BT + PAD;

    static_assert(BT == 64, "C-input prototype is intentionally BT=64 only");
    static_assert(T::K == 128 && T::V == 128,
                  "C-input prototype is intentionally K=V=128 only");
    static_assert((!T::SPLIT_SCAN && BV == 64) ||
                      (T::SPLIT_SCAN && (BV == 16 || BV == 32 || BV == 64)),
                  "fused K2-C requires BV=64; split scan supports BV=16/32/64");
    static_assert(T::NUM_WARPS == 4, "C-input prototype is intentionally 4 waves");
    static_assert(!T::WAVE_OWNED ||
                      (T::SPLIT_SCAN
                           ? !T::PERSIST_Q
                           : (!T::PERSIST_Q && T::RETAIN_LAST_K && T::DIRECT_AV)),
                  "wave-owned staging is specialized for the low-LDS direct path");
    static_assert(!T::FUSE_VD_K0 || T::WAVE_OWNED,
                  "merged Vd/K0 publication is validated on the wave-owned path");
    static_assert(T::PREFETCH_D_K0_PACKS >= 0 &&
                      T::PREFETCH_D_K0_PACKS <= 4,
                  "Phase-D K0 prefetch count must be in [0, 4]");
    static_assert(T::PREFETCH_D_K0_PACKS == 0 ||
                      (!T::PERSIST_K && (T::RETAIN_LAST_K || T::SPLIT_SCAN)),
                  "deferred K0 prefetch requires a non-persistent K path");
    // h update: C[BK_SUB, BV], four waves tile K and one wave tile V.
    constexpr int H_T_M = (T::NUM_WARPS < BK_SUB / W) ? T::NUM_WARPS : BK_SUB / W;
    constexpr int H_T_N = T::NUM_WARPS / H_T_M;
    constexpr int H_E_M = BK_SUB / (W * H_T_M);
    constexpr int H_E_N = BV / (W * H_T_N);
    constexpr int H_E_K = BT / W;

    // K@H, Q@H, C@R and AV share an output [BT, BV] tiling.
    constexpr int O_T_M = (T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W;
    constexpr int O_T_N = T::NUM_WARPS / O_T_M;
    constexpr int O_E_M = BT / (W * O_T_M);
    constexpr int O_E_N = BV / (W * O_T_N);
    constexpr int O_E_K = BK_SUB / W;
    constexpr int C_ELEMS = O_E_M * O_E_N;
    constexpr int CINV_E_K = BT / W;

    // QK^T has output [BT, BT].
    constexpr int QKT_T_M = (T::NUM_WARPS < BT / W) ? T::NUM_WARPS : BT / W;
    constexpr int QKT_T_N = T::NUM_WARPS / QKT_T_M;
    constexpr int QKT_E_M = BT / (W * QKT_T_M);
    constexpr int QKT_E_N = BT / (W * QKT_T_N);
    constexpr int QKT_E_K = BK_SUB / W;

    static_assert(H_E_M == 1 && H_E_N == BV / W &&
                      O_E_M == 1 && O_E_N == BV / W,
                  "the hand-written layout assumes the fixed 4-wave tiling");

    const int i_v = blockIdx.x;
    const int i_nh = blockIdx.y;
    const int i_n = i_nh / kargs.H;
    const int i_h = i_nh % kargs.H;
    const int tid = threadIdx.x;
    const int warp_id = tid / WS;
    const int lane_id = tid % WS;
    const int v_off = i_v * BV;

    const int K = kargs.K;
    const int V = kargs.V;
    const int H = kargs.H;
    const int NT = kargs.NT;
    const int stride_k = H * K;
    const int stride_v = H * V;
    const int stride_c = H * BT;
    const int stride_g = H;
    // Form the batch/token/head and state/head bases in 64-bit before any
    // dimension products.  Pointer addition cannot recover an offset that
    // has already overflowed a signed 32-bit intermediate.
    const int64_t token_head_base =
        (static_cast<int64_t>(i_n) * kargs.T) * H + i_h;
    const int64_t state_head_base = static_cast<int64_t>(i_n) * H + i_h;

    const int h_m_base = (warp_id / H_T_N) * W;
    const int h_n_base = (warp_id % H_T_N) * (H_E_N * W);
    const int o_m_base = (warp_id / O_T_N) * (O_E_M * W);
    const int o_n_base = (warp_id % O_T_N) * (O_E_N * W);
    const int qkt_m_base = (warp_id / QKT_T_N) * (QKT_E_M * W);
    const int qkt_n_base = (warp_id % QKT_T_N) * (QKT_E_N * W);

    // h is held as [K, V] fp32 MFMA accumulators.  The global ABI is V-first.
    v4f32_t h1[H_E_N];
    v4f32_t h2[H_E_N];
    clear_v4f32<H_E_N>(h1);
    clear_v4f32<H_E_N>(h2);
    if (kargs.ptr_h0 != nullptr) {
        const D_ACC* h0 = reinterpret_cast<const D_ACC*>(kargs.ptr_h0)
                          + state_head_base * V * K;
        for (int en = 0; en < H_E_N; ++en) {
            for (int p = 0; p < 4; ++p) {
                const int row = h_m_base + (lane_id >> 4) * 4 + p;
                const int col = h_n_base + en * W + (lane_id & 15);
                h1[en][p] = h0[(v_off + col) * K + row];
                h2[en][p] = h0[(v_off + col) * K + BK_SUB + row];
            }
        }
    }

    // Persistent LDS: gates, beta, corrected value / beta-R transpose, and Q.
    // The tail phase pool is deliberately aliased across all non-overlapping
    // phases; see gdn_k2_c_traits::pool_bytes.
    extern __shared__ char smem_buf[];
    D_ACC* s_g_base = reinterpret_cast<D_ACC*>(smem_buf);
    D_ACC* s_beta = reinterpret_cast<D_ACC*>(smem_buf + T::smem_g_bytes);
    D_ACC* s_exp_g = s_beta + BT;
    D_ACC* s_state_gate = s_exp_g + (T::CACHE_GATES ? BT : 0);
    D_ATTN* s_v_T = reinterpret_cast<D_ATTN*>(
        smem_buf + T::smem_g_bytes + T::smem_beta_bytes
        + T::smem_gate_cache_bytes);
    D_ATTN* s_q = s_v_T + BV * STRIDE_BT;
    D_ATTN* s_k = s_q + T::smem_q_bytes / sizeof(D_ATTN);
    D_ATTN* s_pool = s_k + T::smem_k_bytes / sizeof(D_ATTN);

    const D_ATTN* q_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_q)
                          + token_head_base * K;
    const D_ATTN* k_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                          + token_head_base * K;
    const D_ATTN* v_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                          + token_head_base * V;
    const D_ATTN* c_hbm = reinterpret_cast<const D_ATTN*>(kargs.ptr_c)
                          + token_head_base * BT;
    const D_ACC* beta_hbm = reinterpret_cast<const D_ACC*>(kargs.ptr_beta)
                             + token_head_base;
    const D_ACC* g_hbm = reinterpret_cast<const D_ACC*>(kargs.ptr_g)
                          + token_head_base;
    D_ATTN* o_hbm = reinterpret_cast<D_ATTN*>(kargs.ptr_o)
                    + token_head_base * V;
    D_ATTN* h_snap_hbm = T::SPLIT_SCAN
        ? reinterpret_cast<D_ATTN*>(kargs.ptr_h_snap)
        : nullptr;
    D_ATTN* v_new_hbm = T::SPLIT_SCAN
        ? reinterpret_cast<D_ATTN*>(kargs.ptr_v_new)
            + token_head_base * V
        : nullptr;

    constexpr int VEC = 4;
    constexpr int KV_NVEC = BK_SUB / VEC;

    // Each CTA owns one V tile and must scan chunks in order for its state tile.
    for (int i_t = 0; i_t < NT; ++i_t) {
        D_ACC* s_g = s_g_base
            + (T::RELAX_BARRIERS ? (i_t & 1) * BT : 0);
        const int t0 = i_t * BT;
        const D_ATTN* q_ch = q_hbm + static_cast<int64_t>(t0) * stride_k;
        const D_ATTN* k_ch = k_hbm + static_cast<int64_t>(t0) * stride_k;
        const D_ATTN* v_ch = v_hbm + static_cast<int64_t>(t0) * stride_v;
        const D_ATTN* c_ch = c_hbm + static_cast<int64_t>(t0) * stride_c;
        D_ATTN* o_ch = o_hbm + static_cast<int64_t>(t0) * stride_v;
        const D_ACC* beta_ch =
            beta_hbm + static_cast<int64_t>(t0) * stride_g;
        const D_ACC* g_ch = g_hbm + static_cast<int64_t>(t0) * stride_g;
        D_ATTN* v_new_ch = T::SPLIT_SCAN
            ? v_new_hbm + static_cast<int64_t>(t0) * stride_v
            : nullptr;

        for (int i = tid; i < BT; i += BS) {
            const D_ACC gate = g_ch[i * stride_g];
            s_g[i] = gate;
            s_beta[i] = beta_ch[i * stride_g];
            if constexpr (T::CACHE_GATES)
                s_exp_g[i] = fast_exp(gate);
        }
        __syncthreads();
        const D_ACC g_last = s_g[BT - 1];
        if constexpr (T::CACHE_GATES) {
            if (tid < BT)
                s_state_gate[tid] = fast_exp(g_last - s_g[tid]);
        }

        // -----------------------------------------------------------------
        // Phase B: U=K@H and O_cross=Q@H.  Q is retained in LDS for Phase E.
        // -----------------------------------------------------------------
        v4f32_t r_kh[C_ELEMS];
        v4f32_t r_o_cross[C_ELEMS];
        clear_v4f32<C_ELEMS>(r_kh);
        clear_v4f32<C_ELEMS>(r_o_cross);

        D_ATTN* s_h_T = s_pool;                              // [BV, BK+pad]
        D_ATTN* s_sub_k_scratch = s_pool + BV * STRIDE_BK;   // [BT, BK+pad]

        for (int bk = 0; bk < N_K; ++bk) {
            v4f32_t* h_cur = bk == 0 ? h1 : h2;
            for (int en = 0; en < H_E_N; ++en) {
                const int row_t = h_n_base + en * W + (lane_id & 15);
                const int col_t = h_m_base + (lane_id >> 4) * 4;
                v4bf16_t h_pack;
                for (int p = 0; p < 4; ++p)
                    h_pack[p] = fast_f32_to_bf16(h_cur[en][p]);
                *reinterpret_cast<v4bf16_t*>(&s_h_T[row_t * STRIDE_BK + col_t]) = h_pack;
            }

            if constexpr (T::SPLIT_SCAN) {
                // K5-style scan: publish only K for K@H.  Q@H belongs to the
                // chunk-parallel K6 and is deliberately absent here.
                constexpr int PACKS_PER_THREAD = BT * KV_NVEC / BS;
                #pragma unroll
                for (int pack = 0; pack < PACKS_PER_THREAD; ++pack) {
                    const int i = warp_id * W * KV_NVEC + lane_id + pack * WS;
                    const int row = i / KV_NVEC;
                    const int col = (i % KV_NVEC) * VEC;
                    const v4bf16_t k_pack = *reinterpret_cast<const v4bf16_t*>(
                        &k_ch[row * stride_k + bk * BK_SUB + col]);
                    *reinterpret_cast<v4bf16_t*>(
                        &s_sub_k_scratch[row * STRIDE_BK + col]) = k_pack;
                }
                __syncthreads();

                // K6 needs the pre-update state for every chunk.  s_h_T is
                // already in the desired [V,K] orientation, so copy it with
                // contiguous bf16x4 global stores while it feeds K@H.
                D_ATTN* h_snap_ch = h_snap_hbm
                    + ((static_cast<int64_t>(i_n) * NT + i_t) * H + i_h)
                        * V * K;
                constexpr int H_NVEC = BK_SUB / VEC;
                for (int i = tid; i < BV * H_NVEC; i += BS) {
                    const int row = i / H_NVEC;
                    const int col = (i % H_NVEC) * VEC;
                    const v4bf16_t h_pack = *reinterpret_cast<const v4bf16_t*>(
                        &s_h_T[row * STRIDE_BK + col]);
                    *reinterpret_cast<v4bf16_t*>(
                        &h_snap_ch[(v_off + row) * K + bk * BK_SUB + col]) = h_pack;
                }

                tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                    r_kh, s_sub_k_scratch, o_m_base, STRIDE_BK,
                          s_h_T, o_n_base, STRIDE_BK, lane_id);
                __syncthreads();
            } else if constexpr (T::PERSIST_Q) {
                D_ATTN* s_sub_k = T::PERSIST_K
                    ? s_k + bk * BT * STRIDE_BK
                    : s_sub_k_scratch;
                D_ATTN* s_q_bk = s_q + bk * BT * STRIDE_BK;
                for (int i = tid; i < BT * KV_NVEC; i += BS) {
                    const int row = i / KV_NVEC;
                    const int col = (i % KV_NVEC) * VEC;
                    const v4bf16_t k_pack = *reinterpret_cast<const v4bf16_t*>(
                        &k_ch[row * stride_k + bk * BK_SUB + col]);
                    const v4bf16_t q_pack = *reinterpret_cast<const v4bf16_t*>(
                        &q_ch[row * stride_k + bk * BK_SUB + col]);
                    *reinterpret_cast<v4bf16_t*>(
                        &s_sub_k[row * STRIDE_BK + col]) = k_pack;
                    *reinterpret_cast<v4bf16_t*>(
                        &s_q_bk[row * STRIDE_BK + col]) = q_pack;
                }
                __syncthreads();
                tiled_gemm_mfma_shared_b<O_E_M, O_E_N, O_E_K>(
                    r_kh, r_o_cross,
                    s_sub_k, o_m_base, STRIDE_BK,
                    s_q_bk,  o_m_base, STRIDE_BK,
                    s_h_T,   o_n_base, STRIDE_BK,
                    lane_id);
                __syncthreads();
            } else {
                constexpr int PACKS_PER_THREAD = BT * KV_NVEC / BS;
                static_assert(!T::RETAIN_LAST_K || T::PREFETCH_Q,
                              "last-K retention reuses the prefetch registers");
                const bool retain_this_k = T::RETAIN_LAST_K && bk + 1 == N_K;
                v4bf16_t operand_prefetch[PACKS_PER_THREAD];
                for (int pack = 0; pack < PACKS_PER_THREAD; ++pack) {
                    const int i = T::WAVE_OWNED
                        ? warp_id * W * KV_NVEC + lane_id + pack * WS
                        : tid + pack * BS;
                    const int row = i / KV_NVEC;
                    const int col = (i % KV_NVEC) * VEC;
                    if constexpr (T::PREFETCH_Q) {
                        operand_prefetch[pack] = retain_this_k
                            ? *reinterpret_cast<const v4bf16_t*>(
                                &k_ch[row * stride_k + bk * BK_SUB + col])
                            : *reinterpret_cast<const v4bf16_t*>(
                                &q_ch[row * stride_k + bk * BK_SUB + col]);
                    }
                    const v4bf16_t first_pack = retain_this_k
                        ? *reinterpret_cast<const v4bf16_t*>(
                            &q_ch[row * stride_k + bk * BK_SUB + col])
                        : *reinterpret_cast<const v4bf16_t*>(
                            &k_ch[row * stride_k + bk * BK_SUB + col]);
                    *reinterpret_cast<v4bf16_t*>(
                        &s_sub_k_scratch[row * STRIDE_BK + col]) = first_pack;
                }
                __syncthreads();
                if (retain_this_k) {
                    tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                        r_o_cross, s_sub_k_scratch, o_m_base, STRIDE_BK,
                                   s_h_T, o_n_base, STRIDE_BK, lane_id);
                } else {
                    tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                        r_kh, s_sub_k_scratch, o_m_base, STRIDE_BK,
                              s_h_T, o_n_base, STRIDE_BK, lane_id);
                }
                if constexpr (T::WAVE_OWNED)
                    __syncwarp();
                else
                    __syncthreads();
                for (int pack = 0; pack < PACKS_PER_THREAD; ++pack) {
                    const int i = T::WAVE_OWNED
                        ? warp_id * W * KV_NVEC + lane_id + pack * WS
                        : tid + pack * BS;
                    const int row = i / KV_NVEC;
                    const int col = (i % KV_NVEC) * VEC;
                    v4bf16_t second_pack;
                    if constexpr (T::PREFETCH_Q) {
                        second_pack = operand_prefetch[pack];
                    } else {
                        second_pack = retain_this_k
                            ? *reinterpret_cast<const v4bf16_t*>(
                                &k_ch[row * stride_k + bk * BK_SUB + col])
                            : *reinterpret_cast<const v4bf16_t*>(
                                &q_ch[row * stride_k + bk * BK_SUB + col]);
                    }
                    *reinterpret_cast<v4bf16_t*>(
                        &s_sub_k_scratch[row * STRIDE_BK + col]) = second_pack;
                }
                if constexpr (T::WAVE_OWNED)
                    __syncwarp();
                else
                    __syncthreads();
                if (retain_this_k) {
                    tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                        r_kh, s_sub_k_scratch, o_m_base, STRIDE_BK,
                              s_h_T, o_n_base, STRIDE_BK, lane_id);
                } else {
                    tiled_gemm_mfma<O_E_M, O_E_N, O_E_K>(
                        r_o_cross, s_sub_k_scratch, o_m_base, STRIDE_BK,
                                   s_h_T, o_n_base, STRIDE_BK, lane_id);
                }
                // On the final K slab the beta-R publication barrier below
                // already prevents any wave from reusing s_pool before all
                // Q@H LDS reads finish.  Earlier slabs still need protection
                // before the next H/K staging pass overwrites the pool.
                if constexpr (T::RELAX_BARRIERS) {
                    if (bk + 1 < N_K)
                        __syncthreads();
                } else {
                    __syncthreads();
                }
            }
        }

        // Q@H needs the current token's decay, while K@H is consumed below to
        // construct the RTP-style residual R.  Split scan leaves Q@H to K6.
        if constexpr (!T::SPLIT_SCAN) {
            for (int i = 0; i < C_ELEMS; ++i) {
                const int s_base = o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4;
                for (int p = 0; p < 4; ++p)
                    r_o_cross[i][p] *= T::CACHE_GATES
                        ? s_exp_g[s_base + p]
                        : fast_exp(s_g[s_base + p]);
            }
        }

        // -----------------------------------------------------------------
        // Phase C: beta-R is staged transposed, C is loaded into the freed
        // phase pool, then Vd=C@(beta*R) overwrites s_v_T after every thread
        // has completed its LDS reads.
        // -----------------------------------------------------------------
        for (int i = 0; i < C_ELEMS; ++i) {
            const int en = i % O_E_N;
            const int s_base = o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4;
            const int c = o_n_base + en * W + (lane_id & 15);
            v4bf16_t beta_r_pack;
            for (int p = 0; p < 4; ++p) {
                const int s = s_base + p;
                const D_ACC exp_gate = T::CACHE_GATES
                    ? s_exp_g[s]
                    : fast_exp(s_g[s]);
                const D_ACC residual = static_cast<D_ACC>(
                    v_ch[s * stride_v + v_off + c])
                    - exp_gate * r_kh[i][p];
                beta_r_pack[p] = fast_f32_to_bf16(s_beta[s] * residual);
            }
            *reinterpret_cast<v4bf16_t*>(&s_v_T[c * STRIDE_BT + s_base]) = beta_r_pack;
        }
        __syncthreads();

        D_ATTN* s_c = s_pool;                                // [BT, BT+pad]
        if constexpr (T::WAVE_OWNED) {
            for (int i = lane_id; i < W * BT; i += WS) {
                const int row = warp_id * W + i / BT;
                const int col = i % BT;
                s_c[row * STRIDE_BT + col] = c_ch[row * stride_c + col];
            }
        } else if constexpr (T::VECTOR_C) {
            constexpr int C_VEC = 4;
            constexpr int C_NVEC = BT / C_VEC;
            for (int i = tid; i < BT * C_NVEC; i += BS) {
                const int row = i / C_NVEC;
                const int col = (i % C_NVEC) * C_VEC;
                const v4bf16_t c_pack =
                    *reinterpret_cast<const v4bf16_t*>(
                        &c_ch[row * stride_c + col]);
                *reinterpret_cast<v4bf16_t*>(
                    &s_c[row * STRIDE_BT + col]) = c_pack;
            }
        } else {
            for (int i = tid; i < BT * BT; i += BS) {
                const int row = i / BT;
                const int col = i % BT;
                s_c[row * STRIDE_BT + col] = c_ch[row * stride_c + col];
            }
        }
        if constexpr (T::WAVE_OWNED)
            __syncwarp();
        else
            __syncthreads();

        // Issue the selected Phase-D K0 packs before C@R so their VMEM latency
        // overlaps the MFMA and Vd conversion below. Fused traits retain K1
        // in LDS; split traits reload K1 because their smaller-BV scratch may
        // alias the C tile.
        constexpr int D_PACKS_PER_THREAD = BT * KV_NVEC / BS;
        static_assert(D_PACKS_PER_THREAD * BS == BT * KV_NVEC,
                      "the deferred K0 prefetch assumes an exact pack split");
        v4bf16_t d_k0_prefetch[D_PACKS_PER_THREAD];
        if constexpr (T::PREFETCH_D_K0_PACKS > 0) {
            #pragma unroll
            for (int pack = 0; pack < T::PREFETCH_D_K0_PACKS; ++pack) {
                const int i = tid + pack * BS;
                const int row = i / KV_NVEC;
                const int col = (i % KV_NVEC) * VEC;
                d_k0_prefetch[pack] = *reinterpret_cast<const v4bf16_t*>(
                    &k_ch[row * stride_k + col]);
            }
        }
        // BV16/32 cannot retain K1 in the aliased phase pool: the following
        // C tile covers that scratch range.  These scan-only instances have
        // enough register headroom to carry K1 across C@R instead.
        v4bf16_t d_k1_prefetch[D_PACKS_PER_THREAD];
        if constexpr (T::SPLIT_SCAN && !T::RETAIN_LAST_K) {
            #pragma unroll
            for (int pack = 0; pack < D_PACKS_PER_THREAD; ++pack) {
                const int i = tid + pack * BS;
                const int row = i / KV_NVEC;
                const int col = (i % KV_NVEC) * VEC;
                d_k1_prefetch[pack] = *reinterpret_cast<const v4bf16_t*>(
                    &k_ch[row * stride_k + BK_SUB + col]);
            }
        }

        v4f32_t r_vd[C_ELEMS];
        clear_v4f32<C_ELEMS>(r_vd);
        tiled_gemm_mfma<O_E_M, O_E_N, CINV_E_K>(
            r_vd, s_c, o_m_base, STRIDE_BT,
                  s_v_T, o_n_base, STRIDE_BT,
                  lane_id);

        // No thread may overwrite beta-R until every wave has consumed it.
        __syncthreads();
        for (int i = 0; i < C_ELEMS; ++i) {
            const int en = i % O_E_N;
            const int s_base = o_m_base
                + (i / O_E_N) * W + (lane_id >> 4) * 4;
            const int c = o_n_base + en * W + (lane_id & 15);
            v4bf16_t vd_pack;
            for (int p = 0; p < 4; ++p) {
                vd_pack[p] = fast_f32_to_bf16(r_vd[i][p]);
                if constexpr (T::SPLIT_SCAN) {
                    v_new_ch[(s_base + p) * stride_v + v_off + c]
                        = vd_pack[p];
                }
            }
            *reinterpret_cast<v4bf16_t*>(
                &s_v_T[c * STRIDE_BT + s_base]) = vd_pack;
        }
        // The first Phase-D K transpose is independent of Vd staging.  The
        // K publication barrier can publish both regions together.
        if constexpr (!T::FUSE_VD_K0)
            __syncthreads();

        // -----------------------------------------------------------------
        // Phase D: H <- exp(g_last) H + K^T @ (exp(g_last-g) * Vd).
        // C is no longer needed, so its phase-pool storage becomes K_gated^T.
        // -----------------------------------------------------------------
        const D_ACC decay = T::CACHE_GATES
            ? s_exp_g[BT - 1]
            : fast_exp(g_last);
        for (int en = 0; en < H_E_N; ++en) {
            for (int p = 0; p < 4; ++p) {
                h1[en][p] *= decay;
                h2[en][p] *= decay;
            }
        }

        D_ATTN* s_k_T = s_pool;                              // [BK, BT+pad]
        for (int bk = 0; bk < N_K; ++bk) {
            v4f32_t* h_cur = bk == 0 ? h1 : h2;
            if constexpr (T::PREFETCH_D_K0_PACKS > 0 || T::UNROLL_D_PACKS) {
                #pragma unroll
                for (int pack = 0; pack < D_PACKS_PER_THREAD; ++pack) {
                    const int i = tid + pack * BS;
                    const int s = i / KV_NVEC;
                    const int j = (i % KV_NVEC) * VEC;
                    const D_ACC gate = T::CACHE_GATES
                        ? s_state_gate[s]
                        : fast_exp(g_last - s_g[s]);
                    v4bf16_t k_pack;
                    if constexpr (T::PREFETCH_D_K0_PACKS > 0) {
                        if constexpr (T::SPLIT_SCAN && !T::RETAIN_LAST_K) {
                            if (bk == 1) {
                                k_pack = d_k1_prefetch[pack];
                            } else if (pack < T::PREFETCH_D_K0_PACKS) {
                                k_pack = d_k0_prefetch[pack];
                            } else {
                                k_pack = *reinterpret_cast<const v4bf16_t*>(
                                    &k_ch[s * stride_k + j]);
                            }
                        } else if (bk == 0 && pack < T::PREFETCH_D_K0_PACKS) {
                            k_pack = d_k0_prefetch[pack];
                        } else if constexpr (T::RETAIN_LAST_K) {
                            k_pack = bk + 1 == N_K
                                ? *reinterpret_cast<const v4bf16_t*>(
                                    &s_sub_k_scratch[s * STRIDE_BK + j])
                                : *reinterpret_cast<const v4bf16_t*>(
                                    &k_ch[s * stride_k + bk * BK_SUB + j]);
                        } else {
                            k_pack = *reinterpret_cast<const v4bf16_t*>(
                                &k_ch[s * stride_k + bk * BK_SUB + j]);
                        }
                    } else if constexpr (T::PERSIST_K) {
                        k_pack = *reinterpret_cast<const v4bf16_t*>(
                            &s_k[bk * BT * STRIDE_BK + s * STRIDE_BK + j]);
                    } else if constexpr (T::RETAIN_LAST_K) {
                        k_pack = bk + 1 == N_K
                            ? *reinterpret_cast<const v4bf16_t*>(
                                &s_sub_k_scratch[s * STRIDE_BK + j])
                            : *reinterpret_cast<const v4bf16_t*>(
                                &k_ch[s * stride_k + bk * BK_SUB + j]);
                    } else {
                        k_pack = *reinterpret_cast<const v4bf16_t*>(
                            &k_ch[s * stride_k + bk * BK_SUB + j]);
                    }
                    for (int vi = 0; vi < VEC; ++vi) {
                        s_k_T[(j + vi) * STRIDE_BT + s] = fast_f32_to_bf16(
                            static_cast<D_ACC>(k_pack[vi]) * gate);
                    }
                }
            } else {
                // Keep the original runtime loop as the exact variant-12
                // control.  Merely changing this to a fixed pack loop alters
                // code generation and register allocation on gfx942.
                for (int i = tid; i < BT * KV_NVEC; i += BS) {
                    const int s = i / KV_NVEC;
                    const int j = (i % KV_NVEC) * VEC;
                    const D_ACC gate = T::CACHE_GATES
                        ? s_state_gate[s]
                        : fast_exp(g_last - s_g[s]);
                    v4bf16_t k_pack;
                    if constexpr (T::PERSIST_K) {
                        k_pack = *reinterpret_cast<const v4bf16_t*>(
                            &s_k[bk * BT * STRIDE_BK + s * STRIDE_BK + j]);
                    } else if constexpr (T::RETAIN_LAST_K) {
                        k_pack = bk + 1 == N_K
                            ? *reinterpret_cast<const v4bf16_t*>(
                                &s_sub_k_scratch[s * STRIDE_BK + j])
                            : *reinterpret_cast<const v4bf16_t*>(
                                &k_ch[s * stride_k + bk * BK_SUB + j]);
                    } else {
                        k_pack = *reinterpret_cast<const v4bf16_t*>(
                            &k_ch[s * stride_k + bk * BK_SUB + j]);
                    }
                    for (int vi = 0; vi < VEC; ++vi) {
                        s_k_T[(j + vi) * STRIDE_BT + s] = fast_f32_to_bf16(
                            static_cast<D_ACC>(k_pack[vi]) * gate);
                    }
                }
            }
            __syncthreads();

            tiled_gemm_mfma<H_E_M, H_E_N, H_E_K>(
                h_cur, s_k_T, h_m_base, STRIDE_BT,
                       s_v_T, h_n_base, STRIDE_BT,
                       lane_id);
            if constexpr (T::WAVE_OWNED) {
                if (bk + 1 < N_K)
                    __syncthreads();
                else
                    __syncwarp();
            } else {
                __syncthreads();
            }
        }

        // Fused variants enter Phase E, whose publication/reuse protocol
        // separates the final state-update reads from the next chunk.  A
        // scan-only CTA skips Phase E, so it needs this explicit cross-wave
        // boundary before the next chunk overwrites s_pool/s_v_T.
        if constexpr (T::SPLIT_SCAN)
            __syncthreads();

        if constexpr (!T::SPLIT_SCAN) {
            // -------------------------------------------------------------
            // Phase E: causal gated QK^T @ Vd.  Persistent Q from Phase B is
            // reused; the phase pool aliases K then the attention matrix.
            // Split scan materializes H/Vd and delegates this entire phase
            // to the chunk-parallel K6 kernel.
            // -------------------------------------------------------------
            v4f32_t r_a[QKT_E_M * QKT_E_N];
            clear_v4f32<QKT_E_M * QKT_E_N>(r_a);
            D_ATTN* s_a = s_pool;                            // [BT, BT+pad]

            if constexpr (T::PERSIST_K) {
            for (int bk = 0; bk < N_K; ++bk) {
                if constexpr (T::DIRECT_AV) {
                    tiled_gemm_mfma<QKT_E_N, QKT_E_M, QKT_E_K>(
                        r_a, s_k + bk * BT * STRIDE_BK,
                             qkt_n_base, STRIDE_BK,
                             s_q + bk * BT * STRIDE_BK,
                             qkt_m_base, STRIDE_BK,
                             lane_id);
                } else {
                    tiled_gemm_mfma<QKT_E_M, QKT_E_N, QKT_E_K>(
                        r_a, s_q + bk * BT * STRIDE_BK,
                             qkt_m_base, STRIDE_BK,
                             s_k + bk * BT * STRIDE_BK,
                             qkt_n_base, STRIDE_BK,
                             lane_id);
                }
            }
            } else {
            for (int bk = 0; bk < N_K; ++bk) {
                const int qk_bk = T::RETAIN_LAST_K ? N_K - 1 - bk : bk;
                D_ATTN* s_q4 = T::PERSIST_Q
                    ? s_q + qk_bk * BT * STRIDE_BK
                    : s_pool;
                D_ATTN* s_k4 = T::PERSIST_Q
                    ? s_pool
                    : s_pool + BT * STRIDE_BK;
                constexpr int PACKS_PER_THREAD = BT * KV_NVEC / BS;
                for (int pack = 0; pack < PACKS_PER_THREAD; ++pack) {
                    const int i = T::WAVE_OWNED
                        ? warp_id * W * KV_NVEC + lane_id + pack * WS
                        : tid + pack * BS;
                    const int row = i / KV_NVEC;
                    const int col = (i % KV_NVEC) * VEC;
                    if constexpr (!T::PERSIST_Q) {
                        const v4bf16_t q_pack =
                            *reinterpret_cast<const v4bf16_t*>(
                                &q_ch[row * stride_k + qk_bk * BK_SUB + col]);
                        *reinterpret_cast<v4bf16_t*>(
                            &s_q4[row * STRIDE_BK + col]) = q_pack;
                    }
                    if (!T::RETAIN_LAST_K || qk_bk + 1 != N_K) {
                        const v4bf16_t k_pack =
                            *reinterpret_cast<const v4bf16_t*>(
                                &k_ch[row * stride_k + qk_bk * BK_SUB + col]);
                        *reinterpret_cast<v4bf16_t*>(
                            &s_k4[row * STRIDE_BK + col]) = k_pack;
                    }
                }
                if constexpr (T::WAVE_OWNED) {
                    if (qk_bk + 1 == N_K)
                        __syncwarp();
                    else
                        __syncthreads();
                } else {
                    __syncthreads();
                }
                if constexpr (T::DIRECT_AV) {
                    // Compute A^T=KQ^T.  The MFMA accumulator lane layout is
                    // exactly the source-fragment layout for A, so the gated
                    // fragments can feed A@Vd without an LDS round trip.
                    tiled_gemm_mfma<QKT_E_N, QKT_E_M, QKT_E_K>(
                        r_a, s_k4, qkt_n_base, STRIDE_BK,
                             s_q4, qkt_m_base, STRIDE_BK,
                             lane_id);
                    if (bk + 1 < N_K)
                        __syncthreads();
                } else {
                    tiled_gemm_mfma<QKT_E_M, QKT_E_N, QKT_E_K>(
                        r_a, s_q4, qkt_m_base, STRIDE_BK,
                             s_k4, qkt_n_base, STRIDE_BK,
                             lane_id);
                    __syncthreads();
                }
            }
            }

            v4f32_t r_o_intra[C_ELEMS];
            clear_v4f32<C_ELEMS>(r_o_intra);
            if constexpr (T::DIRECT_AV) {
            static_assert(QKT_E_M == 1 && O_E_M == 1 &&
                              QKT_E_N == CINV_E_K && QKT_E_N == 4,
                          "direct A-to-Vd handoff assumes the fixed tile map");
            for (int ek = 0; ek < CINV_E_K; ++ek) {
                const int row = qkt_m_base + (lane_id & 15);
                const int col_base = qkt_n_base + ek * W
                    + (lane_id >> 4) * 4;
                v4bf16_t a_tile;
                for (int p = 0; p < 4; ++p) {
                    const int col = col_base + p;
                    const D_ACC a = row >= col
                        ? r_a[ek][p] * fast_exp(s_g[row] - s_g[col])
                        : 0.0f;
                    a_tile[p] = fast_f32_to_bf16(a);
                }
                v4bf16_t b_tiles[O_E_N];
                for (int en = 0; en < O_E_N; ++en) {
                    b_tiles[en] = load_mfma_tile(
                        s_v_T, o_n_base + en * W, ek * W,
                        STRIDE_BT, lane_id);
                }
                for (int en = 0; en < O_E_N; ++en) {
                    r_o_intra[en] = mfma_f32_16x16x16_bf16(
                        a_tile, b_tiles[en], r_o_intra[en]);
                }
            }
            } else {
            for (int i = 0; i < QKT_E_M * QKT_E_N; ++i) {
                const int en = i % QKT_E_N;
                const int row_base = qkt_m_base
                    + (i / QKT_E_N) * W + (lane_id >> 4) * 4;
                const int col = qkt_n_base + en * W + (lane_id & 15);
                for (int p = 0; p < 4; ++p) {
                    const int row = row_base + p;
                    const D_ACC a = row >= col
                        ? r_a[i][p] * fast_exp(s_g[row] - s_g[col])
                        : 0.0f;
                    s_a[row * STRIDE_BT + col] = fast_f32_to_bf16(a);
                }
            }
            if constexpr (T::RELAX_BARRIERS) {
                // QK and AV use the same four-way row partition: a wave
                // writes and consumes only its own 16 rows of s_a.
                opus::s_waitcnt_lgkmcnt(opus::number<0>{});
            } else {
                __syncthreads();
            }
            tiled_gemm_mfma<O_E_M, O_E_N, CINV_E_K>(
                r_o_intra, s_a, o_m_base, STRIDE_BT,
                           s_v_T, o_n_base, STRIDE_BT,
                           lane_id);
            }

            for (int i = 0; i < C_ELEMS; ++i) {
            const int en = i % O_E_N;
            const int s_base = o_m_base + (i / O_E_N) * W + (lane_id >> 4) * 4;
            const int c = o_n_base + en * W + (lane_id & 15);
            for (int p = 0; p < 4; ++p) {
                const D_ACC out = kargs.scale * (r_o_cross[i][p] + r_o_intra[i][p]);
                o_ch[(s_base + p) * stride_v + v_off + c] = fast_f32_to_bf16(out);
            }
            }

            // The next chunk first touches only g/beta and then executes a CTA
            // barrier before reusing s_pool, so that entry barrier also
            // protects the final AV reads.  The last chunk never reuses it.
            if constexpr (!T::RELAX_BARRIERS)
                __syncthreads();
        }
    }

    if (kargs.ptr_ht != nullptr) {
        D_ACC* ht = reinterpret_cast<D_ACC*>(kargs.ptr_ht)
                    + state_head_base * V * K;
        for (int en = 0; en < H_E_N; ++en) {
            for (int p = 0; p < 4; ++p) {
                const int row = h_m_base + (lane_id >> 4) * 4 + p;
                const int col = h_n_base + en * W + (lane_id & 15);
                ht[(v_off + col) * K + row] = h1[en][p];
                ht[(v_off + col) * K + BK_SUB + row] = h2[en][p];
            }
        }
    }
}

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_BLOCKS)
gdn_k2_c_kernel(gdn_k2_c_kargs kargs) {
    gdn_k2_c_kernel_impl<Traits>(kargs);
}
