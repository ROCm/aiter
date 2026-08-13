// GDN Prefill K1 Kernel — BT=16, Neumann Series + MFMA WY
// Step 1: g_cumsum + KKT Gram matrix (scalar)
// Step 2: Triangular inverse (I+A)^{-1} via Neumann series (scalar)
//         + WY factor assembly w_bar, u_bar (MFMA)
//
// Grid: (NT, B*H)   Block: (BLOCK_SIZE = 256)
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16×16×16
//
// A is 16×16 strictly lower triangular → nilpotent of order 16.
// Neumann series: (I+A)^{-1} = I - A + A² - A³ + … + (-A)^{15}
// WY GEMMs tiled with warps along N (T_M=1, T_N=4).
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

    static_assert(T::BT == 16, "This template is for BT=16 only");

    const int i_t  = blockIdx.x;   // chunk index
    const int i_bh = blockIdx.y;   // batch*H index
    const int i_b  = i_bh / kargs.H;
    const int i_h  = i_bh % kargs.H;

    const int tid     = threadIdx.x;
    const int warp_id = tid / T::WARP_SIZE;
    const int lane_id = tid % T::WARP_SIZE;

    constexpr int BT = T::BT;  // 16
    constexpr int BS = T::BLOCK_SIZE;
    constexpr int PAD = T::SMEM_PAD;
    constexpr int BK_SUB = T::BK_SUB;
    const int K  = kargs.K;
    const int V  = kargs.V;
    const int H  = kargs.H;

    const int chunk_start = i_t * BT;
    const int bos = i_b * kargs.T;

    // =====================================================================
    // Shared memory
    // =====================================================================
    extern __shared__ char smem_buf[];

    // Phase 1: g[BT] + beta[BT] + k[BT×K]
    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);        // 16 fp32
    D_ACC*  s_beta = s_g + BT;                                  // 16 fp32
    D_ATTN* s_k    = reinterpret_cast<D_ATTN*>(s_beta + BT);   // 16×K bf16

    // Phase 2: C[BT×BT] fp32 = 1KB (where s_k was — s_k is only 16*128*2=4KB)
    D_ACC*  s_C = reinterpret_cast<D_ACC*>(s_k);  // 16×16 fp32

    // Neumann temp: neg_A_pow[BT×BT] fp32 = 1KB (beyond s_C)
    D_ACC*  s_neg_A_pow = s_C + BT * BT;          // 16×16 fp32

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

    // Hillis-Steele prefix sum (4 steps for BT=16)
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
        if (chunk_start + i < kargs.T)
            g_cumsum_base[i * H] = s_g[i];
    }
    __syncthreads();

    // =====================================================================
    // Phase 1b: Load k[BT, K] into LDS
    // =====================================================================
    const D_ATTN* k_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                           + ((bos + chunk_start) * H + i_h) * K;

    for (int i = tid; i < BT * K; i += BS) {
        int row = i / K;
        int col = i % K;
        if (chunk_start + row < kargs.T)
            s_k[i] = k_base[row * H * K + col];
        else
            s_k[i] = static_cast<D_ATTN>(0);
    }
    __syncthreads();

    // =====================================================================
    // Phase 1c+1d: Scalar KKT + gate scaling
    // A[s,r] = beta[s] * exp(gc[s]-gc[r]) * (k_s . k_r) for s>r, else 0
    //
    // BT=16: only one 16×16 output tile. Scalar uses all 256 threads
    // (BT*BT=256=BS), each computing one element with K=128 MACs.
    // =====================================================================
    {
        int row = tid / BT;
        int col = tid % BT;
        D_ACC val = 0.0f;
        if (row > col) {
            D_ACC dot = 0.0f;
            for (int ki = 0; ki < K; ki++)
                dot += (D_ACC)s_k[row * K + ki] * (D_ACC)s_k[col * K + ki];
            D_ACC g_diff = s_g[row] - s_g[col];
            val = s_beta[row] * __expf(g_diff) * dot;
        }
        __syncthreads();
        s_C[tid] = val;
    }
    __syncthreads();

    // =====================================================================
    // Phase 2a: Neumann series — C = (I + A)^{-1} = Σ_{n=0}^{15} (-A)^n
    //
    // A is 16×16 strictly lower triangular → nilpotent, series is exact.
    // Kept scalar for fp32 precision in the inverse.
    // =====================================================================

    // Negate A in place: s_C now holds -A (strict lower tri part) + zeros elsewhere
    for (int i = tid; i < BT * BT; i += BS) {
        int r = i / BT;
        int c = i % BT;
        if (r > c)
            s_C[i] = -s_C[i];
    }
    __syncthreads();

    // s_C = -A. Accumulate C_result = I + (-A) + (-A)^2 + ... in registers.
    // BT*BT = 256 = BS, so each thread owns exactly one element of C.
    int my_r = tid / BT;
    int my_c = tid % BT;

    D_ACC c_accum = (my_r == my_c) ? 1.0f : 0.0f;  // n=0 term: I

    // Initialize neg_A_pow = I
    s_neg_A_pow[tid] = (my_r == my_c) ? 1.0f : 0.0f;
    __syncthreads();

    for (int n = 1; n < BT; n++) {
        D_ACC new_val = 0.0f;
        for (int j = 0; j < BT; j++)
            new_val += s_neg_A_pow[my_r * BT + j] * s_C[j * BT + my_c];
        __syncthreads();
        s_neg_A_pow[tid] = new_val;
        __syncthreads();

        c_accum += new_val;
    }

    // Write C_result to s_C (overwriting -A, which is no longer needed)
    s_C[tid] = c_accum;
    __syncthreads();

    // =====================================================================
    // Phase 2c: WY factor GEMMs via MFMA
    //   u_bar = C @ (v * beta)
    //   w_bar = C @ (k * beta * exp(g_cumsum))
    //
    // Tiling: T_M=1, T_N=4 — warps along N (each warp handles 16×16 tile).
    // C[16,16] fp32 → C_bf16[16, 16+PAD] for MFMA A operand.
    // v/k pre-scaled + transposed → s_vT[64, 16+PAD] for MFMA B operand.
    // =====================================================================

    constexpr int C_STRIDE = BT + PAD;   // 20
    constexpr int VT_STRIDE = BT + PAD;  // 20

    // Convert C to bf16 — placed after s_vT region to avoid overlap during conversion
    D_ATTN* s_C_bf16 = reinterpret_cast<D_ATTN*>(
        reinterpret_cast<char*>(s_k) + BK_SUB * C_STRIDE * (int)sizeof(D_ATTN));

    for (int i = tid; i < BT * BT; i += BS) {
        int s = i / BT;
        int j = i % BT;
        s_C_bf16[s * C_STRIDE + j] = static_cast<D_ATTN>(s_C[i]);
    }
    for (int i = tid; i < BT * PAD; i += BS) {
        int s = i / PAD;
        int p = i % PAD;
        s_C_bf16[s * C_STRIDE + BT + p] = static_cast<D_ATTN>(0);
    }
    __syncthreads();

    // s_C/s_neg_A_pow region freed — reuse as s_vT for transposed operand
    D_ATTN* s_vT = s_k;

    D_ATTN* w_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_w_bar)
                         + ((bos + chunk_start) * H + i_h) * K;
    D_ATTN* u_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_u_bar)
                         + ((bos + chunk_start) * H + i_h) * V;

    // --- u_bar = C @ (v * beta) ---
    const D_ATTN* v_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                           + ((bos + chunk_start) * H + i_h) * V;

    for (int iv = 0; iv < T::N_V_ITERS; iv++) {
        int v_offset = iv * BK_SUB;

        // v_scaled_T[vi, j] = bf16(v[j, vi] * beta[j])
        for (int i = tid; i < BT * BK_SUB; i += BS) {
            int j  = i / BK_SUB;
            int vi = i % BK_SUB;
            D_ATTN v_val = (chunk_start + j < kargs.T)
                ? v_base[j * H * V + v_offset + vi]
                : static_cast<D_ATTN>(0);
            float scaled = static_cast<D_ACC>(v_val) * s_beta[j];
            s_vT[vi * VT_STRIDE + j] = static_cast<D_ATTN>(scaled);
        }
        for (int i = tid; i < BK_SUB * PAD; i += BS) {
            int vi = i / PAD;
            int p  = i % PAD;
            s_vT[vi * VT_STRIDE + BT + p] = static_cast<D_ATTN>(0);
        }
        __syncthreads();

        // MFMA: each warp computes 16×16 tile at different N offset
        v4f32_t wy_c[1];
        clear_v4f32<1>(wy_c);
        tiled_gemm_mfma<1, 1, 1>(
            wy_c, s_C_bf16, 0, C_STRIDE,
                  s_vT, warp_id * 16, VT_STRIDE, lane_id);

        for (int p = 0; p < 4; p++) {
            int s  = (lane_id >> 4) * 4 + p;
            int vi = warp_id * 16 + (lane_id & 15);
            if (chunk_start + s < kargs.T)
                u_bar_base[s * H * V + v_offset + vi] =
                    static_cast<D_ATTN>(wy_c[0][p]);
        }
        __syncthreads();
    }

    // --- w_bar = C @ (k * beta * exp(g_cumsum)) ---
    for (int ik = 0; ik < T::N_K_ITERS; ik++) {
        int k_offset = ik * BK_SUB;

        // k_scaled_T[ki, j] = bf16(k[j, ki] * beta[j] * exp(gc[j]))
        for (int i = tid; i < BT * BK_SUB; i += BS) {
            int j  = i / BK_SUB;
            int ki = i % BK_SUB;
            D_ATTN k_val = (chunk_start + j < kargs.T)
                ? k_base[j * H * K + k_offset + ki]
                : static_cast<D_ATTN>(0);
            float scaled = static_cast<D_ACC>(k_val) * s_beta[j] * __expf(s_g[j]);
            s_vT[ki * VT_STRIDE + j] = static_cast<D_ATTN>(scaled);
        }
        for (int i = tid; i < BK_SUB * PAD; i += BS) {
            int ki = i / PAD;
            int p  = i % PAD;
            s_vT[ki * VT_STRIDE + BT + p] = static_cast<D_ATTN>(0);
        }
        __syncthreads();

        v4f32_t wy_c[1];
        clear_v4f32<1>(wy_c);
        tiled_gemm_mfma<1, 1, 1>(
            wy_c, s_C_bf16, 0, C_STRIDE,
                  s_vT, warp_id * 16, VT_STRIDE, lane_id);

        for (int p = 0; p < 4; p++) {
            int s  = (lane_id >> 4) * 4 + p;
            int ki = warp_id * 16 + (lane_id & 15);
            if (chunk_start + s < kargs.T)
                w_bar_base[s * H * K + k_offset + ki] =
                    static_cast<D_ATTN>(wy_c[0][p]);
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
