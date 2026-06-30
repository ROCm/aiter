// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Gated DeltaNet (GDN) chunk-prepare kernel — shared types, kargs, traits.
//
// `gdn_chunk_prepare` fuses the FOUR intra-chunk FLA prefill kernels that must
// run before the h-state recurrence:
//   1. chunk_local_cumsum        — inclusive prefix sum of the log-gate g
//   2. chunk_scaled_dot_kkt_fwd  — gated KKT Gram  A = tril(K Kᵀ ⊙ decay)
//   3. solve_tril                — triangular inverse  C = (I + A)⁻¹
//   4. recompute_w_u_fwd         — WY factors  w_bar = C·(k·β·e^g), u_bar = C·(v·β)
// into a single token-parallel HIP kernel.
//
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA bf16 16×16×16.
#pragma once

#ifdef __HIP_DEVICE_COMPILE__
using bf16_t = __bf16;
#else
using bf16_t = unsigned short;
#endif

// --------------------------------------------------------------------------
// Kernel arguments
//   inputs : k[B,T,H,K] bf16, v[B,T,H,V] bf16, g[B,T,H] fp32, beta[B,T,H] fp32
//   outputs: g_cumsum[B,T,H] fp32, w_bar[B,T,H,K] bf16, u_bar[B,T,H,V] bf16
// --------------------------------------------------------------------------
struct gdn_chunk_prepare_kargs {
    const void* __restrict__ ptr_k;        // [B, T, H, K]   bf16
    const void* __restrict__ ptr_v;        // [B, T, H, V]   bf16
    const void* __restrict__ ptr_beta;     // [B, T, H]      fp32
    const void* __restrict__ ptr_g;        // [B, T, H]      fp32
    void* __restrict__ ptr_w_bar;          // [B, T, H, K]   bf16  output
    void* __restrict__ ptr_u_bar;          // [B, T, H, V]   bf16  output
    void* __restrict__ ptr_g_cumsum;       // [B, T, H]      fp32  output
    int B;
    int T;
    int H;
    int K;
    int V;
};

// --------------------------------------------------------------------------
// Traits
//
// Layout: [B, T, H, K] with stride_t = H*K, stride_h = K, stride_k = 1
// MFMA: bf16 16×16×16 uniformly on gfx942/gfx950
// --------------------------------------------------------------------------
template<int BT_,
         int K_  = 128,
         int V_  = 128,
         int NUM_WARPS_ = 4>
struct gdn_chunk_prepare_traits {
    static constexpr int BT = BT_;
    static constexpr int K  = K_;
    static constexpr int V  = V_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = bf16_t;
    using D_ACC  = float;

    // MFMA tile: 16×16×16 bf16 → fp32
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 16;

    // KKT GEMM: [BT, K] × [K, BT] = [BT, BT]
    static constexpr int KKT_E_M = BT / W_M;          // 4 (BT=64)
    static constexpr int KKT_E_N = BT / W_N;          // 4
    static constexpr int KKT_E_K = K / W_K;           // 8

    // w_bar/u_bar GEMMs processed in 64-wide subtiles (BK_sub=BV_sub=64)
    static constexpr int BK_SUB = 64;
    static constexpr int BV_SUB = 64;
    static constexpr int N_K_ITERS = K / BK_SUB;      // 2
    static constexpr int N_V_ITERS = V / BV_SUB;      // 2

    static constexpr int WY_E_M = BT / W_M;           // 4
    static constexpr int WY_E_N = BK_SUB / W_N;       // 4
    static constexpr int WY_E_K = BT / W_K;           // 4

    static constexpr int VEC_KV = 8;                  // bf16x8 = 16 bytes

    // LDS padding: +4 bf16/row to avoid bank conflicts (32-bank gfx942 / 64-bank gfx950)
    static constexpr int SMEM_PAD = 4;
    static constexpr int K_STRIDE = K + SMEM_PAD;
    static constexpr int A_STRIDE = BT + 1;           // fp32 A padded to BT+1

    // LDS layout (bytes)
    static constexpr int smem_k_padded_bytes = BT * K_STRIDE * (int)sizeof(D_ATTN);
    static constexpr int smem_A_bytes        = BT * A_STRIDE * (int)sizeof(D_ACC);
    static constexpr int smem_scalar_bytes   = BT * 2 * (int)sizeof(D_ACC);

    // OCC=3 size (C⁻¹ register-cached, no s_C_bf16). Used on gfx942 where the
    // smaller LDS lifts occupancy 2→3.
    static constexpr size_t smem_size_bytes() {
        int phase1   = smem_k_padded_bytes + smem_scalar_bytes;
        int phase2ab = smem_A_bytes + 16 * 16 * (int)sizeof(D_ACC) + smem_scalar_bytes;
        return phase1 > phase2ab ? phase1 : phase2ab;
    }

    // OCC=2 size (C⁻¹ staged in LDS as s_C_bf16). Used on gfx950, whose 160KB
    // LDS is not the occupancy limiter — register-caching gains nothing there.
    static constexpr size_t smem_size_bytes_cinv_lds() {
        int base = (int)smem_size_bytes();
        int c_bf16_bytes = BT * (BT + SMEM_PAD) * (int)sizeof(D_ATTN);
        int phase2c = smem_A_bytes + c_bf16_bytes + smem_scalar_bytes;
        return (phase2c > base) ? phase2c : base;
    }
};
