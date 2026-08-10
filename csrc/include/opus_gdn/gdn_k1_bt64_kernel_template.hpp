// GDN Prefill K1 Kernel — BT=64, MFMA bf16 16×16×16 optimized
// Step 1: g_cumsum + KKT Gram matrix (MFMA)
// Step 2: Triangular inverse (I+A)^{-1} via 4×16x16 forward sub + Schur merge (scalar)
//         + WY factor assembly w_bar, u_bar (MFMA)
//
// Grid: (NT, B*H)   Block: (BLOCK_SIZE = 256)
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16×16×16
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, 2)
gdn_k1_kernel(gdn_k1_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    static_assert(T::BT == 64, "This template is for BT=64 only");

    const int i_t  = blockIdx.x;
    const int i_bh = blockIdx.y;
    const int i_b  = i_bh / kargs.H;
    const int i_h  = i_bh % kargs.H;

    const int tid  = threadIdx.x;
    const int warp_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;

    constexpr int BT = T::BT;
    constexpr int BS = T::BLOCK_SIZE;
    constexpr int PAD = T::SMEM_PAD;
    constexpr int K_STRIDE = T::K_STRIDE;
    constexpr int BK_SUB = T::BK_SUB;
    constexpr int A_STRIDE = T::A_STRIDE;
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;

    const int chunk_start = i_t * BT;
    const int bos = i_b * kargs.T;

    // =====================================================================
    // Shared memory allocation
    // =====================================================================
    extern __shared__ char smem_buf[];

    // Phase 1 layout: g[BT] + beta[BT] + k[BT×K_STRIDE] (padded for MFMA)
    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ACC*  s_beta = s_g + BT;
    D_ATTN* s_k    = reinterpret_cast<D_ATTN*>(s_beta + BT);

    // Phase 2 layout (s_A aliases s_k region):
    D_ACC*  s_A    = reinterpret_cast<D_ACC*>(s_k);

    // =====================================================================
    // Phase 1a: Load g and beta, compute prefix sum
    // =====================================================================
    const D_ACC* g_base    = reinterpret_cast<const D_ACC*>(kargs.ptr_g)
                             + (bos + chunk_start) * H + i_h;
    const D_ACC* beta_base = reinterpret_cast<const D_ACC*>(kargs.ptr_beta)
                             + (bos + chunk_start) * H + i_h;

    for (int i = tid; i < BT; i += BS) {
        int global_t = chunk_start + i;
        if (global_t < kargs.T) {
            s_g[i]    = g_base[i * H];
            s_beta[i] = beta_base[i * H];
        } else {
            s_g[i]    = 0.0f;
            s_beta[i] = 0.0f;
        }
    }
    __syncthreads();

    // Prefix sum (Hillis-Steele)
    for (int stride = 1; stride < BT; stride <<= 1) {
        for (int i = tid; i < BT; i += BS) {
            if (i >= stride)
                s_g[i] += s_g[i - stride];
        }
        __syncthreads();
    }

    // Write g_cumsum to HBM
    D_ACC* g_cumsum_base = reinterpret_cast<D_ACC*>(kargs.ptr_g_cumsum)
                           + (bos + chunk_start) * H + i_h;
    for (int i = tid; i < BT; i += BS) {
        int global_t = chunk_start + i;
        if (global_t < kargs.T)
            g_cumsum_base[i * H] = s_g[i];
    }
    __syncthreads();

    // =====================================================================
    // Phase 1b: Load k[BT, K] into LDS with padding (stride = K_STRIDE)
    // =====================================================================
    const D_ATTN* k_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                           + ((bos + chunk_start) * H + i_h) * K;

    {
        // v8 (128-bit) k load — half the VMEM loads of v4 (BT=32 idea).
        using v8bf16_t = __bf16 __attribute__((ext_vector_type(8)));
        constexpr int K_VEC = T::K / 8;
        for (int i = tid; i < BT * K_VEC; i += BS) {
            int row = i / K_VEC;
            int col8 = (i % K_VEC) * 8;
            int global_t = chunk_start + row;
            v8bf16_t val{};
            if (global_t < kargs.T)
                val = *reinterpret_cast<const v8bf16_t*>(
                    &k_base[row * H * K + col8]);
            *reinterpret_cast<v8bf16_t*>(&s_k[row * K_STRIDE + col8]) = val;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 1c+1d: KKT GEMM via MFMA — k × k^T self-matmul
    //
    // C[BT, BT] = k[BT, K] × k^T[K, BT]
    // Both A and B read from s_k (same buffer, different row ranges per warp).
    // s_A aliases s_k — must finish ALL MFMA reads before writing to s_A.
    // =====================================================================

    constexpr int KKT_E_M = 1;
    constexpr int KKT_E_N = BT / 16;       // 4
    constexpr int KKT_E_K = T::K / 16;     // 8

    v4f32_t kkt_c[KKT_E_M * KKT_E_N];
    clear_v4f32<KKT_E_M * KKT_E_N>(kkt_c);

    tiled_gemm_mfma<KKT_E_M, KKT_E_N, KKT_E_K>(
        kkt_c, s_k, warp_id * 16, K_STRIDE,
               s_k, 0,            K_STRIDE, lane_id);

    // All warps must finish reading s_k before we write to s_A (aliases s_k)
    __syncthreads();

    // Post-MFMA: gate-scale lower triangle, zero upper+diagonal, write fp32 to s_A
    for (int en = 0; en < KKT_E_N; en++) {
        for (int p = 0; p < 4; p++) {
            int s = warp_id * 16 + (lane_id >> 4) * 4 + p;
            int r = en * 16 + (lane_id & 15);
            float val = 0.0f;
            if (s > r)
                val = kkt_c[en][p] * s_beta[s] * __expf(s_g[s] - s_g[r]);
            s_A[s * A_STRIDE + r] = val;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 2a: Triangular inverse — 4 × 16×16 forward substitution
    //
    // Each warp handles one 16×16 diagonal block independently.
    // No cross-warp reads during the inner loop → no __syncthreads needed.
    // =====================================================================

    int blk = warp_id;
    int blk_row_start = blk * 16;

    for (int i = 1; i < 16; i++) {
        for (int c = lane_id; c < 16; c += T::WARP_SIZE) {
            if (c >= i) continue;

            int global_row = blk_row_start + i;
            int global_col = blk_row_start + c;
            if (global_row >= BT) continue;

            D_ACC neg_a = -s_A[global_row * A_STRIDE + global_col];

            D_ACC acc = neg_a;
            for (int j = c + 1; j < i; j++) {
                int j_global = blk_row_start + j;
                acc += (-s_A[global_row * A_STRIDE + j_global]) *
                       s_A[j_global * A_STRIDE + global_col];
            }
            s_A[global_row * A_STRIDE + global_col] = acc;
        }
    }

    // Add identity
    for (int i = lane_id; i < 16; i += T::WARP_SIZE) {
        int r = blk_row_start + i;
        if (r < BT)
            s_A[r * A_STRIDE + r] += 1.0f;
    }
    // Zero upper triangle within block
    for (int idx = lane_id; idx < 16 * 16; idx += T::WARP_SIZE) {
        int r = idx / 16;
        int c = idx % 16;
        int gr = blk_row_start + r;
        int gc_col = blk_row_start + c;
        if (gr < BT && gc_col < BT && r < c)
            s_A[gr * A_STRIDE + gc_col] = 0.0f;
    }
    __syncthreads();

    // =====================================================================
    // Phase 2b: Schur complement merge — MFMA bf16 16×16×16
    //
    // Dependency DAG enables warp-level parallelism (3 levels, 3 barriers):
    //   Level 1: C_21, C_32, C_43  (3 warps, independent)
    //   Level 2: C_31, C_42        (2 warps, independent)
    //   Level 3: C_41              (1 warp)
    //
    // Register chaining: MFMA result → bf16 → next MFMA source B,
    // giving A_next × D_prev without LDS round-trip.
    // =====================================================================

    constexpr v4f32_t z4 = {0.f, 0.f, 0.f, 0.f};

    // Pre-save L blocks overwritten in Level 1 (L_32 by C_32, L_43 by C_43)
    v4bf16_t sav_L32, sav_L43, sav_L42;
    if (warp_id == 0) {
        sav_L32 = load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id);
        sav_L43 = load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
    } else if (warp_id == 1) {
        sav_L43 = load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
    }

    v4f32_t kept_c21 = z4, kept_c32 = z4, kept_c31 = z4;

    // --- Level 1: C_21, C_32, C_43 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 16, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        kept_c21 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 16, 16, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c21[p] = -kept_c21[p];
        store_fp32_tile(s_A, 16, 0, A_STRIDE, kept_c21, lane_id);
    } else if (warp_id == 1) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 16, 16, A_STRIDE, lane_id), z4);
        kept_c32 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 32, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c32[p] = -kept_c32[p];
        store_fp32_tile(s_A, 32, 16, A_STRIDE, kept_c32, lane_id);
    } else if (warp_id == 2) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 32, 32, A_STRIDE, lane_id), z4);
        v4f32_t c43 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c43[p] = -c43[p];
        store_fp32_tile(s_A, 48, 32, A_STRIDE, c43, lane_id);
    }
    __syncthreads();

    // Save L_42 (still valid, overwritten by C_42 in Level 2)
    if (warp_id == 0)
        sav_L42 = load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id);

    // --- Level 2: C_31, C_42 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L32, accum_to_src(kept_c21), t);
        kept_c31 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 32, 32, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) kept_c31[p] = -kept_c31[p];
        store_fp32_tile(s_A, 32, 0, A_STRIDE, kept_c31, lane_id);
    } else if (warp_id == 1) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 16, 16, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L43, accum_to_src(kept_c32), t);
        v4f32_t c42 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c42[p] = -c42[p];
        store_fp32_tile(s_A, 48, 16, A_STRIDE, c42, lane_id);
    }
    __syncthreads();

    // --- Level 3: C_41 ---
    if (warp_id == 0) {
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 0, A_STRIDE, lane_id),
            load_fp32_tile_T(s_A, 0, 0, A_STRIDE, lane_id), z4);
        t = mfma_f32_16x16x16_bf16(sav_L42, accum_to_src(kept_c21), t);
        t = mfma_f32_16x16x16_bf16(sav_L43, accum_to_src(kept_c31), t);
        v4f32_t c41 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_A, 48, 48, A_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c41[p] = -c41[p];
        store_fp32_tile(s_A, 48, 0, A_STRIDE, c41, lane_id);
    }
    __syncthreads();

    // s_A now contains C = (I + A)^{-1}

    // =====================================================================
    // Phase 2c: WY factor GEMMs via MFMA
    //
    // u_bar = C @ (v * beta)
    // w_bar = C @ (k * beta * exp(g_cumsum))
    //
    // Step 1: Convert C (fp32, s_A) → C_bf16 (placed after s_A in LDS)
    // Step 2: For each subtile, load pre-scaled v/k transposed → MFMA
    // =====================================================================

    // Convert C_inv to bf16 with padding, placed AFTER s_A
    constexpr int C_STRIDE = BT + PAD;  // 68
    D_ATTN* s_C_bf16 = reinterpret_cast<D_ATTN*>(
        smem_buf + BT * 2 * sizeof(D_ACC) + BT * A_STRIDE * sizeof(D_ACC));

    for (int i = tid; i < BT * BT; i += BS) {
        int s = i / BT;
        int j = i % BT;
        s_C_bf16[s * C_STRIDE + j] = static_cast<D_ATTN>(s_A[s * A_STRIDE + j]);
    }
    __syncthreads();

    // s_A region freed — reuse for v_scaled_T / k_scaled_T
    constexpr int VT_STRIDE = BT + PAD;  // 68
    D_ATTN* s_vT = s_k;

    constexpr int WY_EM = 1;
    constexpr int WY_EN = BK_SUB / 16;   // 4
    constexpr int WY_EK = BT / 16;       // 4

    D_ATTN* w_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_w_bar)
                         + ((bos + chunk_start) * H + i_h) * K;
    D_ATTN* u_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_u_bar)
                         + ((bos + chunk_start) * H + i_h) * V;

    // --- u_bar = C @ (v * beta) ---
    const D_ATTN* v_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                           + ((bos + chunk_start) * H + i_h) * V;

    for (int iv = 0; iv < T::N_V_ITERS; iv++) {
        int v_offset = iv * BK_SUB;

        // v_scaled_T[vi, j] = bf16(v[j, vi] * beta[j]), v8 (128-bit) HBM load
        {
            using v8bf16_t = __bf16 __attribute__((ext_vector_type(8)));
            constexpr int VEC = 8;
            constexpr int NVEC = BK_SUB / VEC;
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int vi = (i % NVEC) * VEC;
                v8bf16_t vals{};
                if (chunk_start + j < kargs.T)
                    vals = *reinterpret_cast<const v8bf16_t*>(
                        &v_base[j * H * V + v_offset + vi]);
                D_ACC beta_j = s_beta[j];
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(vi + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * beta_j);
            }
        }
        __syncthreads();

        v4f32_t wy_c[WY_EM * WY_EN];
        clear_v4f32<WY_EM * WY_EN>(wy_c);
        tiled_gemm_mfma<WY_EM, WY_EN, WY_EK>(
            wy_c, s_C_bf16, warp_id * 16, C_STRIDE,
                  s_vT,     0,             VT_STRIDE, lane_id);

        for (int en = 0; en < WY_EN; en++) {
            for (int p = 0; p < 4; p++) {
                int s  = warp_id * 16 + (lane_id >> 4) * 4 + p;
                int vi = en * 16 + (lane_id & 15);
                if (chunk_start + s < kargs.T)
                    u_bar_base[s * H * V + v_offset + vi] =
                        static_cast<D_ATTN>(wy_c[en][p]);
            }
        }
        __syncthreads();
    }

    // --- w_bar = C @ (k * beta * exp(g_cumsum)) ---
    for (int ik = 0; ik < T::N_K_ITERS; ik++) {
        int k_offset = ik * BK_SUB;

        // k_scaled_T[ki, j] = bf16(k[j, ki] * beta[j] * exp(gc[j])), v8 HBM load
        {
            using v8bf16_t = __bf16 __attribute__((ext_vector_type(8)));
            constexpr int VEC = 8;
            constexpr int NVEC = BK_SUB / VEC;
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int ki = (i % NVEC) * VEC;
                v8bf16_t vals{};
                if (chunk_start + j < kargs.T)
                    vals = *reinterpret_cast<const v8bf16_t*>(
                        &k_base[j * H * K + k_offset + ki]);
                D_ACC scale_j = s_beta[j] * __expf(s_g[j]);
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(ki + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * scale_j);
            }
        }
        __syncthreads();

        v4f32_t wy_c[WY_EM * WY_EN];
        clear_v4f32<WY_EM * WY_EN>(wy_c);
        tiled_gemm_mfma<WY_EM, WY_EN, WY_EK>(
            wy_c, s_C_bf16, warp_id * 16, C_STRIDE,
                  s_vT,     0,             VT_STRIDE, lane_id);

        for (int en = 0; en < WY_EN; en++) {
            for (int p = 0; p < 4; p++) {
                int s  = warp_id * 16 + (lane_id >> 4) * 4 + p;
                int ki = en * 16 + (lane_id & 15);
                if (chunk_start + s < kargs.T)
                    w_bar_base[s * H * K + k_offset + ki] =
                        static_cast<D_ATTN>(wy_c[en][p]);
            }
        }
        __syncthreads();
    }

    if (kargs.ptr_k1_done) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (tid == 0)
            __atomic_store_n(kargs.ptr_k1_done + i_t * (kargs.B * kargs.H) + i_bh,
                             1u, __ATOMIC_RELAXED);
    }
}
