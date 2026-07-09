// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// GDN chunk-prepare kernel — unified template, arch-specialized (gfx942/gfx950).
//
// Fuses the FOUR intra-chunk FLA prefill kernels into one token-parallel kernel:
//   1. chunk_local_cumsum        — inclusive prefix sum of log-gate g
//   2. chunk_scaled_dot_kkt_fwd  — gated KKT Gram  A = tril(K Kᵀ ⊙ decay)
//   3. solve_tril                — triangular inverse  C = (I + A)⁻¹  (Neumann squaring + Schur)
//   4. recompute_w_u_fwd         — WY factors  w_bar = C·(k·β·e^g), u_bar = C·(v·β)
//
// Grid: (NT, B*H)   Block: (BLOCK_SIZE = 256, 4 warps)   MFMA bf16 16×16×16.
//
// Arch偏特化 (best-only per arch):
//   gfx950 (MI350): OCC=2, C⁻¹ staged in LDS as bf16 (s_C_bf16). 160KB LDS is
//                   not the occupancy limiter, so register-caching gains nothing.
//   gfx942 (MI300): OCC=3, C⁻¹ register-cached (no s_C_bf16) → smaller LDS lifts
//                   occupancy 2→3 (-30% on the LDS-bound MI300).
//
// opus.hpp usage: all global<->LDS traffic goes through opus layout descriptors
// (make_gmem / make_smem / make_layout) for vectorized b128 load/store dispatch;
// MFMA via opus::mfma<bf16,bf16,fp32,16,16,16>; LDS MFMA-tile access via
// opus::make_smem. The fp32 triangular-inverse epilogue keeps its explicit
// matrix-core fragment→coord mapping (numerically proven; an opus fragment-layout
// rewrite there would risk silent bf16 corruption for no perf gain).
#pragma once

#include <hip/hip_runtime.h>
#include "opus/opus.hpp"
#include "opus_gdn/gdn_chunk_prepare_defs.h"

// ===========================================================================
// MFMA bf16 16×16×16 helpers (opus-backed). Kept local to this kernel so the
// chunk-prepare op is self-contained.
// ===========================================================================
namespace gdn_cp {

using v4bf16_t = opus::bf16x4_t;
using v4f32_t  = opus::fp32x4_t;

__device__ inline v4f32_t mfma_f32_16x16x16_bf16(v4bf16_t a, v4bf16_t b, v4f32_t c) {
    return opus::mfma<opus::bf16_t, opus::bf16_t, opus::fp32_t, 16, 16, 16>{}(a, b, c);
}

__device__ inline v4bf16_t load_mfma_tile(
        const opus::bf16_t* __restrict__ lds, int row_base, int col_base,
        int stride, int lane_id) {
    int addr = (row_base + (lane_id & 15)) * stride + col_base + ((lane_id >> 4) << 2);
    return opus::make_smem<opus::bf16_t>(const_cast<opus::bf16_t*>(lds)).template load<4>(addr);
}

template<int E_M, int E_N, int E_K>
__device__ void tiled_gemm_mfma(
        v4f32_t* __restrict__ c,
        const opus::bf16_t* __restrict__ lds_a, int m_base, int stride_a,
        const opus::bf16_t* __restrict__ lds_b, int n_base, int stride_b,
        int lane_id) {
    for (int ek = 0; ek < E_K; ek++) {
        v4bf16_t a_tiles[E_M];
        for (int em = 0; em < E_M; em++)
            a_tiles[em] = load_mfma_tile(lds_a, m_base + em * 16, ek * 16, stride_a, lane_id);
        v4bf16_t b_tiles[E_N];
        for (int en = 0; en < E_N; en++)
            b_tiles[en] = load_mfma_tile(lds_b, n_base + en * 16, ek * 16, stride_b, lane_id);
        for (int em = 0; em < E_M; em++)
            for (int en = 0; en < E_N; en++)
                c[em * E_N + en] = mfma_f32_16x16x16_bf16(
                    a_tiles[em], b_tiles[en], c[em * E_N + en]);
    }
}

template<int N>
__device__ inline void clear_v4f32(v4f32_t* c) {
    for (int i = 0; i < N; i++) opus::clear(c[i]);
}

__device__ __forceinline__ opus::bf16_t fast_f32_to_bf16(float f) {
    unsigned u = __builtin_bit_cast(unsigned, f);
    u += 0x7FFF + ((u >> 16) & 1);
    return __builtin_bit_cast(opus::bf16_t, static_cast<unsigned short>(u >> 16));
}

__device__ inline v4bf16_t load_fp32_tile(
        const float* __restrict__ s, int row_base, int col_base,
        int stride, int lane_id) {
    int base = (row_base + (lane_id & 15)) * stride + col_base + ((lane_id >> 4) << 2);
    return v4bf16_t{
        fast_f32_to_bf16(s[base]),
        fast_f32_to_bf16(s[base + 1]),
        fast_f32_to_bf16(s[base + 2]),
        fast_f32_to_bf16(s[base + 3])};
}

__device__ inline v4bf16_t load_fp32_tile_T(
        const float* __restrict__ s, int row_base, int col_base,
        int stride, int lane_id) {
    int n = lane_id & 15;
    int col = col_base + n;
    int kb4 = (lane_id >> 4) << 2;
    return v4bf16_t{
        fast_f32_to_bf16(s[(row_base + kb4) * stride + col]),
        fast_f32_to_bf16(s[(row_base + kb4 + 1) * stride + col]),
        fast_f32_to_bf16(s[(row_base + kb4 + 2) * stride + col]),
        fast_f32_to_bf16(s[(row_base + kb4 + 3) * stride + col])};
}

__device__ inline v4bf16_t accum_to_src(v4f32_t d) {
    return v4bf16_t{
        fast_f32_to_bf16(d[0]), fast_f32_to_bf16(d[1]),
        fast_f32_to_bf16(d[2]), fast_f32_to_bf16(d[3])};
}

__device__ inline void store_fp32_tile(
        float* __restrict__ s, int row_base, int col_base,
        int stride, v4f32_t d, int lane_id) {
    int n = lane_id & 15;
    int mb4 = (lane_id >> 4) << 2;
    s[(row_base + mb4) * stride + col_base + n] = d[0];
    s[(row_base + mb4 + 1) * stride + col_base + n] = d[1];
    s[(row_base + mb4 + 2) * stride + col_base + n] = d[2];
    s[(row_base + mb4 + 3) * stride + col_base + n] = d[3];
}

}  // namespace gdn_cp

// ===========================================================================
// Arch-specialized occupancy: gfx942 register-caches C⁻¹ (OCC=3); gfx950 keeps
// the simpler LDS path (OCC=2).
// ===========================================================================
#if defined(__gfx950__)
#define GDN_CP_OCC 2
#else
#define GDN_CP_OCC 3
#endif

template<typename Traits>
__global__ void __launch_bounds__(Traits::BLOCK_SIZE, GDN_CP_OCC)
gdn_chunk_prepare_kernel(gdn_chunk_prepare_kargs kargs) {
    using namespace gdn_cp;
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

    // ---- Shared memory ----
    extern __shared__ char smem_buf[];
    D_ACC*  s_g    = reinterpret_cast<D_ACC*>(smem_buf);
    D_ACC*  s_beta = s_g + BT;
    D_ATTN* s_k    = reinterpret_cast<D_ATTN*>(s_beta + BT);
    D_ACC*  s_A    = reinterpret_cast<D_ACC*>(s_k);

    // =====================================================================
    // Phase 1 — chunk_local_cumsum: load g/beta, inclusive prefix sum of g
    // =====================================================================
    const D_ACC* g_base    = reinterpret_cast<const D_ACC*>(kargs.ptr_g)
                             + (bos + chunk_start) * H + i_h;
    const D_ACC* beta_base = reinterpret_cast<const D_ACC*>(kargs.ptr_beta)
                             + (bos + chunk_start) * H + i_h;

    for (int i = tid; i < BT; i += BS) {
        int global_t = chunk_start + i;
        s_beta[i] = (global_t < kargs.T) ? beta_base[i * H] : 0.0f;
    }

    // Single-warp __shfl_up prefix sum (BT == WARP_SIZE == 64): warp-lockstep
    // register exchange — no LDS round-trip, no __syncthreads. Bit-identical to
    // the block Hillis-Steele but drops ~6 of its barriers.
    static_assert(BT == T::WARP_SIZE, "single-warp scan requires BT == WARP_SIZE");
    if (warp_id == 0) {
        int global_t = chunk_start + lane_id;
        float val = (global_t < kargs.T) ? g_base[lane_id * H] : 0.0f;
        #pragma unroll
        for (int off = 1; off < BT; off <<= 1) {
            float up = __shfl_up(val, off, BT);
            if (lane_id >= off) val += up;
        }
        s_g[lane_id] = val;
    }
    __syncthreads();

    D_ACC* g_cumsum_base = reinterpret_cast<D_ACC*>(kargs.ptr_g_cumsum)
                           + (bos + chunk_start) * H + i_h;
    for (int i = tid; i < BT; i += BS) {
        int global_t = chunk_start + i;
        if (global_t < kargs.T)
            g_cumsum_base[i * H] = s_g[i];
    }
    __syncthreads();

    // =====================================================================
    // Phase 2 — chunk_scaled_dot_kkt_fwd: load k into LDS, KKT GEMM, gate-scale
    // =====================================================================
    const D_ATTN* k_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_k)
                           + ((bos + chunk_start) * H + i_h) * K;
    {
        // opus vectorized gmem->smem load (b128). Layouts describe the [BT,K]
        // tile: gmem row stride = H*K (token-major), smem row stride = K_STRIDE.
        constexpr int VEC = T::VEC_KV;        // 8 bf16 = 16 B
        constexpr int K_VEC = T::K / VEC;
        auto g_k   = opus::make_gmem(reinterpret_cast<const opus::bf16_t*>(k_base));
        auto sm_k  = opus::make_smem(reinterpret_cast<opus::bf16_t*>(s_k));
        auto lay_g = opus::make_layout(opus::make_tuple(BT, K), opus::make_tuple(H * K, 1));
        auto lay_s = opus::make_layout(opus::make_tuple(BT, K), opus::make_tuple(K_STRIDE, 1));
        for (int i = tid; i < BT * K_VEC; i += BS) {
            int row = i / K_VEC;
            int col = (i % K_VEC) * VEC;
            auto val = (chunk_start + row < kargs.T)
                           ? g_k.template load<VEC>(lay_g(row, col))
                           : opus::vector_t<opus::bf16_t, VEC>{};
            sm_k.template store<VEC>(val, lay_s(row, col));
        }
    }
    __syncthreads();

    constexpr int KKT_E_M = 1;
    constexpr int KKT_E_N = BT / 16;       // 4
    constexpr int KKT_E_K = T::K / 16;     // 8

    v4f32_t kkt_c[KKT_E_M * KKT_E_N];
    clear_v4f32<KKT_E_M * KKT_E_N>(kkt_c);
    tiled_gemm_mfma<KKT_E_M, KKT_E_N, KKT_E_K>(
        kkt_c, s_k, warp_id * 16, K_STRIDE,
               s_k, 0,            K_STRIDE, lane_id);
    __syncthreads();

    // gate-scale lower triangle, zero upper+diagonal, write fp32 to s_A
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
    // Phase 3 — solve_tril: triangular inverse C = (I+A)⁻¹
    //   3a. per-warp 16×16 diagonal block inverse via Neumann squaring
    //   3b. Schur-complement merge (3-level warp-parallel DAG)
    // =====================================================================
    constexpr v4f32_t z4 = {0.f, 0.f, 0.f, 0.f};

    {
        int br = warp_id * 16;
        v4bf16_t neg_A_tile;
        {
            int base = (br + (lane_id & 15)) * A_STRIDE + br + ((lane_id >> 4) << 2);
            neg_A_tile = v4bf16_t{
                static_cast<__bf16>(-s_A[base]),
                static_cast<__bf16>(-s_A[base + 1]),
                static_cast<__bf16>(-s_A[base + 2]),
                static_cast<__bf16>(-s_A[base + 3])};
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
        // (I+A)⁻¹ = Σ_{n=0}^{15} Bⁿ = (I+B)(I+B²)(I+B⁴)(I+B⁸), B = -A nilpotent.
        // 6 MFMAs (3 squarings + 3 products) vs 15 Horner iterations.
        v4f32_t  b2   = mfma_f32_16x16x16_bf16(neg_A_tile, neg_A_tile, z4);
        v4bf16_t b2_o = accum_to_src(b2);
        v4f32_t  b4   = mfma_f32_16x16x16_bf16(b2_o, b2_o, z4);
        v4bf16_t b4_o = accum_to_src(b4);
        v4f32_t  b8   = mfma_f32_16x16x16_bf16(b4_o, b4_o, z4);
        v4bf16_t b8_o = accum_to_src(b8);
        v4f32_t C_accum;
        for (int n = 0; n < 4; n++) C_accum[n] = b8[n] + I_accum[n];
        C_accum = mfma_f32_16x16x16_bf16(b4_o, accum_to_src(C_accum), C_accum);
        C_accum = mfma_f32_16x16x16_bf16(b2_o, accum_to_src(C_accum), C_accum);
        C_accum = mfma_f32_16x16x16_bf16(neg_A_tile, accum_to_src(C_accum), C_accum);

        store_fp32_tile(s_A, br, br, A_STRIDE, C_accum, lane_id);
        for (int idx = lane_id; idx < 16 * 16; idx += T::WARP_SIZE) {
            int r = idx / 16, c = idx % 16;
            if (r < c)
                s_A[(br + r) * A_STRIDE + br + c] = 0.0f;
        }
    }
    __syncthreads();

    // Schur complement merge — 3-level warp-parallel DAG (3 barriers):
    //   L1: C_21, C_32, C_43   L2: C_31, C_42   L3: C_41
    v4bf16_t sav_L32, sav_L43, sav_L42;
    if (warp_id == 0) {
        sav_L32 = load_fp32_tile(s_A, 32, 16, A_STRIDE, lane_id);
        sav_L43 = load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
    } else if (warp_id == 1) {
        sav_L43 = load_fp32_tile(s_A, 48, 32, A_STRIDE, lane_id);
    }

    v4f32_t kept_c21 = z4, kept_c32 = z4, kept_c31 = z4;

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

    if (warp_id == 0)
        sav_L42 = load_fp32_tile(s_A, 48, 16, A_STRIDE, lane_id);

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
    // s_A now holds C = (I + A)⁻¹

    // =====================================================================
    // Phase 4 — recompute_w_u_fwd: WY factor GEMMs
    //   u_bar = C @ (v * beta)
    //   w_bar = C @ (k * beta * exp(g_cumsum))
    // =====================================================================
#if defined(__gfx950__)
    // gfx950 (OCC=2): stage C⁻¹ in LDS as bf16 (placed after s_A).
    constexpr int C_STRIDE = BT + PAD;  // 68
    D_ATTN* s_C_bf16 = reinterpret_cast<D_ATTN*>(
        smem_buf + BT * 2 * sizeof(D_ACC) + BT * A_STRIDE * sizeof(D_ACC));
    for (int i = tid; i < BT * BT; i += BS) {
        int s = i / BT;
        int j = i % BT;
        s_C_bf16[s * C_STRIDE + j] = static_cast<D_ATTN>(s_A[s * A_STRIDE + j]);
    }
    __syncthreads();
    constexpr int WY_EM = 1;
#else
    // gfx942 (OCC=3): cache C⁻¹ tiles in registers (no s_C_bf16 in LDS → OCC 2→3).
    constexpr int WY_EK_C = BT / 16;  // 4
    v4bf16_t cached_C[WY_EK_C];
    for (int ek = 0; ek < WY_EK_C; ek++)
        cached_C[ek] = load_fp32_tile(s_A, warp_id * 16, ek * 16, A_STRIDE, lane_id);
#endif

    constexpr int VT_STRIDE = BT + PAD;  // 68
    D_ATTN* s_vT = s_k;                   // reuse freed s_A/s_k region

    constexpr int WY_EN = BK_SUB / 16;   // 4
    constexpr int WY_EK = BT / 16;       // 4

    D_ATTN* w_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_w_bar)
                         + ((bos + chunk_start) * H + i_h) * K;
    D_ATTN* u_bar_base = reinterpret_cast<D_ATTN*>(kargs.ptr_u_bar)
                         + ((bos + chunk_start) * H + i_h) * V;
    const D_ATTN* v_base = reinterpret_cast<const D_ATTN*>(kargs.ptr_v)
                           + ((bos + chunk_start) * H + i_h) * V;

    // --- u_bar = C @ (v * beta) ---
    for (int iv = 0; iv < T::N_V_ITERS; iv++) {
        int v_offset = iv * BK_SUB;
        {
            // opus vectorized gmem load; transposed β-scaled store stays scalar.
            constexpr int VEC = T::VEC_KV;
            constexpr int NVEC = BK_SUB / VEC;
            auto g_v   = opus::make_gmem(reinterpret_cast<const opus::bf16_t*>(v_base));
            auto lay_v = opus::make_layout(opus::make_tuple(BT, V), opus::make_tuple(H * V, 1));
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int vi = (i % NVEC) * VEC;
                auto vals = (chunk_start + j < kargs.T)
                                ? g_v.template load<VEC>(lay_v(j, v_offset + vi))
                                : opus::vector_t<opus::bf16_t, VEC>{};
                D_ACC beta_j = s_beta[j];
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(vi + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * beta_j);
            }
        }
        __syncthreads();

#if defined(__gfx950__)
        v4f32_t wy_c[WY_EM * WY_EN];
        clear_v4f32<WY_EM * WY_EN>(wy_c);
        tiled_gemm_mfma<WY_EM, WY_EN, WY_EK>(
            wy_c, s_C_bf16, warp_id * 16, C_STRIDE,
                  s_vT,     0,            VT_STRIDE, lane_id);
#else
        v4f32_t wy_c[WY_EN];
        clear_v4f32<WY_EN>(wy_c);
        for (int ek = 0; ek < WY_EK; ek++) {
            v4bf16_t b_tiles[WY_EN];
            for (int en = 0; en < WY_EN; en++)
                b_tiles[en] = load_mfma_tile(s_vT, en * 16, ek * 16, VT_STRIDE, lane_id);
            for (int en = 0; en < WY_EN; en++)
                wy_c[en] = mfma_f32_16x16x16_bf16(cached_C[ek], b_tiles[en], wy_c[en]);
        }
#endif

        for (int en = 0; en < WY_EN; en++) {
            for (int p = 0; p < 4; p++) {
                int s  = warp_id * 16 + (lane_id >> 4) * 4 + p;
                int vi = en * 16 + (lane_id & 15);
                if (chunk_start + s < kargs.T)
                    u_bar_base[s * H * V + v_offset + vi] = static_cast<D_ATTN>(wy_c[en][p]);
            }
        }
        __syncthreads();
    }

    // --- w_bar = C @ (k * beta * exp(g_cumsum)) ---
    for (int ik = 0; ik < T::N_K_ITERS; ik++) {
        int k_offset = ik * BK_SUB;
        {
            // opus vectorized gmem load; transposed (β·e^g)-scaled store stays scalar.
            constexpr int VEC = T::VEC_KV;
            constexpr int NVEC = BK_SUB / VEC;
            auto g_kk  = opus::make_gmem(reinterpret_cast<const opus::bf16_t*>(k_base));
            auto lay_k = opus::make_layout(opus::make_tuple(BT, K), opus::make_tuple(H * K, 1));
            for (int i = tid; i < BT * NVEC; i += BS) {
                int j  = i / NVEC;
                int ki = (i % NVEC) * VEC;
                auto vals = (chunk_start + j < kargs.T)
                                ? g_kk.template load<VEC>(lay_k(j, k_offset + ki))
                                : opus::vector_t<opus::bf16_t, VEC>{};
                D_ACC scale_j = s_beta[j] * __expf(s_g[j]);
                for (int vv = 0; vv < VEC; vv++)
                    s_vT[(ki + vv) * VT_STRIDE + j] = static_cast<D_ATTN>(
                        static_cast<D_ACC>(vals[vv]) * scale_j);
            }
        }
        __syncthreads();

#if defined(__gfx950__)
        v4f32_t wy_c[WY_EM * WY_EN];
        clear_v4f32<WY_EM * WY_EN>(wy_c);
        tiled_gemm_mfma<WY_EM, WY_EN, WY_EK>(
            wy_c, s_C_bf16, warp_id * 16, C_STRIDE,
                  s_vT,     0,            VT_STRIDE, lane_id);
#else
        v4f32_t wy_c[WY_EN];
        clear_v4f32<WY_EN>(wy_c);
        for (int ek = 0; ek < WY_EK; ek++) {
            v4bf16_t b_tiles[WY_EN];
            for (int en = 0; en < WY_EN; en++)
                b_tiles[en] = load_mfma_tile(s_vT, en * 16, ek * 16, VT_STRIDE, lane_id);
            for (int en = 0; en < WY_EN; en++)
                wy_c[en] = mfma_f32_16x16x16_bf16(cached_C[ek], b_tiles[en], wy_c[en]);
        }
#endif

        for (int en = 0; en < WY_EN; en++) {
            for (int p = 0; p < 4; p++) {
                int s  = warp_id * 16 + (lane_id >> 4) * 4 + p;
                int ki = en * 16 + (lane_id & 15);
                if (chunk_start + s < kargs.T)
                    w_bar_base[s * H * K + k_offset + ki] = static_cast<D_ATTN>(wy_c[en][p]);
            }
        }
        __syncthreads();
    }
}
