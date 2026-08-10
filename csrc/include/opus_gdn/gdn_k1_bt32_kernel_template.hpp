// GDN Prefill K1 Kernel — BT=32 specialization, fully MFMA-accelerated
//
// Phase 1a: g_cumsum via Hillis-Steele prefix sum
// Phase 1b: k[BT,K] load with v8bf16_t vectorized HBM reads
// Phase 1c: KKT Gram matrix via MFMA tiled_gemm (k × k^T)
// Phase 2a: Triangular inverse via MFMA Horner Neumann (15 iterations/block)
// Phase 2a': Schur complement merge via 2 MFMA instructions
// Phase 2c: WY factors (u_bar, w_bar) via transposed MFMA GEMM
//           with v4bf16_t vectorized HBM loads and stores
//
// Optimizations vs generic K1:
//   - MFMA KKT replaces scalar dot product
//   - MFMA Horner Neumann replaces shuffle-based forward substitution
//   - MFMA Schur complement replaces 8192-op scalar computation
//   - s_C fp32 stride padded to BT+1 (33) to eliminate LDS bank conflicts (32/64-bank)
//   - Transposed WY GEMM enables v4bf16_t vectorized output stores (4x fewer VMEM)
//   - v4bf16_t vectorized HBM loads for v/k data
//
// Grid: (NT, B*H)   Block: (BLOCK_SIZE = 256)
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16×16×16
#pragma once

#include "opus_gdn/gdn_defs.h"
#include "opus_gdn/gdn_mfma_utils.h"

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, 2)
gdn_k1_bt32_kernel(gdn_k1_kargs kargs) {
    using namespace gdn_mfma;
    using T = Traits;
    using D_ATTN = typename T::D_ATTN;
    using D_ACC  = typename T::D_ACC;

    static_assert(T::BT == 32, "This template is for BT=32 only");

    const int i_t  = blockIdx.x;
    const int i_bh = blockIdx.y;
    const int i_b  = i_bh / kargs.H;
    const int i_h  = i_bh % kargs.H;

    const int tid     = threadIdx.x;
    const int warp_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;

    constexpr int BT = T::BT;       // 32
    constexpr int BS = T::BLOCK_SIZE; // 256
    constexpr int PAD = T::SMEM_PAD;
    constexpr int BK_SUB = T::BK_SUB;
    constexpr int K_STRIDE = T::K_STRIDE;
    constexpr int C_FP32_STRIDE = BT + 1; // 33: avoids 32-bank LDS conflict on fp32 C_inv
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;

    const int chunk_start = i_t * BT;
    const int bos = i_b * kargs.T;

    // =====================================================================
    // Shared memory
    // =====================================================================
    extern __shared__ char smem_buf[];

    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ACC*  s_beta = s_g + BT;
    D_ATTN* s_k    = reinterpret_cast<D_ATTN*>(s_beta + BT);

    // Phase 2: C[32×32] fp32 (aliases s_k)
    D_ACC*  s_C = reinterpret_cast<D_ACC*>(s_k);

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

    for (int stride = 1; stride < BT; stride <<= 1) {
        for (int i = tid; i < BT; i += BS) {
            if (i >= stride)
                s_g[i] += s_g[i - stride];
        }
        __syncthreads();
    }

    D_ACC* g_cumsum_base = reinterpret_cast<D_ACC*>(kargs.ptr_g_cumsum)
                           + (bos + chunk_start) * H + i_h;
    for (int i = tid; i < BT; i += BS) {
        if (chunk_start + i < kargs.T)
            g_cumsum_base[i * H] = s_g[i];
    }
    __syncthreads();

    // =====================================================================
    // Phase 1b: Load k[BT, K] into LDS with padding
    // =====================================================================
    const D_ATTN* k_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                           + ((bos + chunk_start) * H + i_h) * K;

    constexpr int K_VEC = T::K / 8;
    using v8bf16_t = __bf16 __attribute__((ext_vector_type(8)));
    for (int i = tid; i < BT * K_VEC; i += BS) {
        int row = i / K_VEC;
        int col8 = (i % K_VEC) * 8;
        v8bf16_t v{};
        if (chunk_start + row < kargs.T)
            v = *reinterpret_cast<const v8bf16_t*>(&k_base[row * H * K + col8]);
        *reinterpret_cast<v8bf16_t*>(&s_k[row * K_STRIDE + col8]) = v;
    }
    for (int i = tid; i < BT * PAD; i += BS) {
        int row = i / PAD;
        int p   = i % PAD;
        s_k[row * K_STRIDE + K + p] = static_cast<D_ATTN>(0);
    }
    __syncthreads();

    // =====================================================================
    // Phase 1c+1d: KKT GEMM via MFMA — k[BT,K] × k^T[K,BT] → [BT,BT]
    // Same approach as BT=64: MFMA reads all k data before any writes,
    // so s_C/s_k aliasing is safe without the register-staging workaround.
    // =====================================================================
    constexpr int KKT_E_N = BT / 16;       // 2
    constexpr int KKT_E_K = T::K / 16;     // 8

    v4f32_t kkt_c[KKT_E_N];
    clear_v4f32<KKT_E_N>(kkt_c);

    if (warp_id < BT / 16) {
        tiled_gemm_mfma<1, KKT_E_N, KKT_E_K>(
            kkt_c, s_k, warp_id * 16, K_STRIDE,
                   s_k, 0,            K_STRIDE, lane_id);
    }
    __syncthreads();

    for (int en = 0; en < KKT_E_N; en++) {
        for (int p = 0; p < 4; p++) {
            int s = warp_id * 16 + (lane_id >> 4) * 4 + p;
            int r = en * 16 + (lane_id & 15);
            float val = 0.0f;
            if (s < BT && s > r)
                val = kkt_c[en][p] * s_beta[s] * __expf(s_g[s] - s_g[r]);
            if (s < BT)
                s_C[s * C_FP32_STRIDE + r] = val;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 2a: MFMA Horner Neumann — same approach as BT=64
    // (I+A)^{-1} = Σ(-A)^n via Horner: I + (-A)(I + (-A)(I + ...))
    // 15 MFMA iterations per 16×16 block, register chaining (accum_to_src).
    // Warps 0,1 handle the two diagonal blocks independently.
    // =====================================================================
    if (warp_id < 2) {
        int br = warp_id * 16;

        v4bf16_t neg_A_tile;
        {
            int base = (br + (lane_id & 15)) * C_FP32_STRIDE + br + ((lane_id >> 4) << 2);
            neg_A_tile = v4bf16_t{
                static_cast<__bf16>(-s_C[base]),
                static_cast<__bf16>(-s_C[base + 1]),
                static_cast<__bf16>(-s_C[base + 2]),
                static_cast<__bf16>(-s_C[base + 3])};
        }

        v4f32_t I_accum;
        {
            int n = lane_id & 15;
            int m_base = (lane_id >> 4) * 4;
            I_accum = v4f32_t{
                (m_base == n) ? 1.0f : 0.0f,
                ((m_base + 1) == n) ? 1.0f : 0.0f,
                ((m_base + 2) == n) ? 1.0f : 0.0f,
                ((m_base + 3) == n) ? 1.0f : 0.0f};
        }

        v4f32_t C_accum = I_accum;
        for (int iter = 0; iter < 15; iter++) {
            C_accum = mfma_f32_16x16x16_bf16(
                neg_A_tile, accum_to_src(C_accum), I_accum);
        }

        store_fp32_tile(s_C, br, br, C_FP32_STRIDE, C_accum, lane_id);
        for (int idx = lane_id; idx < 16 * 16; idx += T::WARP_SIZE) {
            int r = idx / 16, c = idx % 16;
            if (r < c)
                s_C[(br + r) * C_FP32_STRIDE + br + c] = 0.0f;
        }
    }
    __syncthreads();

    // =====================================================================
    // Phase 2a': Schur complement merge via MFMA
    // C_10 = -(C_11_inv × L_10 × C_00_inv)
    // Two MFMA instructions: temp = L_10 × C_00_inv, C_10 = -(C_11_inv × temp)
    // =====================================================================
    if (warp_id == 0) {
        constexpr v4f32_t z4 = {0.f, 0.f, 0.f, 0.f};
        v4f32_t t = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_C, 16, 0, C_FP32_STRIDE, lane_id),
            load_fp32_tile_T(s_C, 0, 0, C_FP32_STRIDE, lane_id), z4);
        v4f32_t c10 = mfma_f32_16x16x16_bf16(
            load_fp32_tile(s_C, 16, 16, C_FP32_STRIDE, lane_id),
            accum_to_src(t), z4);
        for (int p = 0; p < 4; p++) c10[p] = -c10[p];
        store_fp32_tile(s_C, 16, 0, C_FP32_STRIDE, c10, lane_id);
    }
    __syncthreads();

    // =====================================================================
    // Phase 2c: WY factor GEMMs via MFMA
    //   u_bar = C_inv @ (v * beta)
    //   w_bar = C_inv @ (k * beta * exp(g_cumsum))
    //
    // Tiling: T_M=1, T_N=4 — warps along N.
    // Each warp: WY_E_M=2 M-tiles × 1 N-tile × WY_E_K=2 K-iterations.
    // =====================================================================

    constexpr int C_STRIDE = BT + PAD;    // 36
    constexpr int VT_STRIDE = BT + PAD;   // 36

    D_ATTN* s_C_bf16 = reinterpret_cast<D_ATTN*>(
        reinterpret_cast<char*>(s_k) + BK_SUB * C_STRIDE * (int)sizeof(D_ATTN));
    D_ATTN* s_vT = s_k;

    // Convert C_inv to bf16 with padding
    for (int i = tid; i < BT * BT; i += BS) {
        int s = i / BT;
        int j = i % BT;
        s_C_bf16[s * C_STRIDE + j] = static_cast<D_ATTN>(s_C[s * C_FP32_STRIDE + j]);
    }
    for (int i = tid; i < BT * PAD; i += BS) {
        int s = i / PAD;
        int p = i % PAD;
        s_C_bf16[s * C_STRIDE + BT + p] = static_cast<D_ATTN>(0);
    }
    __syncthreads();

    D_ATTN* w_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_w_bar)
                         + ((bos + chunk_start) * H + i_h) * K;
    D_ATTN* u_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_u_bar)
                         + ((bos + chunk_start) * H + i_h) * V;

    // --- u_bar = C_inv @ (v * beta) ---
    const D_ATTN* v_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                           + ((bos + chunk_start) * H + i_h) * V;

    for (int iv = 0; iv < T::N_V_ITERS; iv++) {
        int v_offset = iv * BK_SUB;

        {
            constexpr int VEC = 4;
            constexpr int NVEC = BK_SUB / VEC;
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int vi = (i % NVEC) * VEC;
                v4bf16_t vals = {};
                if (chunk_start + j < kargs.T)
                    vals = *reinterpret_cast<const v4bf16_t*>(
                        &v_base[j * H * V + v_offset + vi]);
                D_ACC beta_j = s_beta[j];
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(vi + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * beta_j);
            }
        }
        __syncthreads();

        v4f32_t wy_c[2];
        clear_v4f32<2>(wy_c);
        tiled_gemm_mfma<1, 2, 2>(
            wy_c, s_vT, warp_id * 16, VT_STRIDE,
                  s_C_bf16, 0, C_STRIDE, lane_id);

        for (int en = 0; en < 2; en++) {
            int s = en * 16 + (lane_id & 15);
            int vi_base = warp_id * 16 + (lane_id >> 4) * 4;
            if (chunk_start + s < kargs.T) {
                *reinterpret_cast<v4bf16_t*>(
                    &u_bar_base[s * H * V + v_offset + vi_base]) = v4bf16_t{
                    static_cast<__bf16>(wy_c[en][0]),
                    static_cast<__bf16>(wy_c[en][1]),
                    static_cast<__bf16>(wy_c[en][2]),
                    static_cast<__bf16>(wy_c[en][3])};
            }
        }
        __syncthreads();
    }

    // --- w_bar = C_inv @ (k * beta * exp(g_cumsum)) ---
    for (int ik = 0; ik < T::N_K_ITERS; ik++) {
        int k_offset = ik * BK_SUB;

        {
            constexpr int VEC = 4;
            constexpr int NVEC = BK_SUB / VEC;
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int ki = (i % NVEC) * VEC;
                v4bf16_t vals = {};
                if (chunk_start + j < kargs.T)
                    vals = *reinterpret_cast<const v4bf16_t*>(
                        &k_base[j * H * K + k_offset + ki]);
                D_ACC scale_j = s_beta[j] * __expf(s_g[j]);
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(ki + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * scale_j);
            }
        }
        __syncthreads();

        v4f32_t wy_c[2];
        clear_v4f32<2>(wy_c);
        tiled_gemm_mfma<1, 2, 2>(
            wy_c, s_vT, warp_id * 16, VT_STRIDE,
                  s_C_bf16, 0, C_STRIDE, lane_id);

        for (int en = 0; en < 2; en++) {
            int s = en * 16 + (lane_id & 15);
            int ki_base = warp_id * 16 + (lane_id >> 4) * 4;
            if (chunk_start + s < kargs.T) {
                *reinterpret_cast<v4bf16_t*>(
                    &w_bar_base[s * H * K + k_offset + ki_base]) = v4bf16_t{
                    static_cast<__bf16>(wy_c[en][0]),
                    static_cast<__bf16>(wy_c[en][1]),
                    static_cast<__bf16>(wy_c[en][2]),
                    static_cast<__bf16>(wy_c[en][3])};
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
