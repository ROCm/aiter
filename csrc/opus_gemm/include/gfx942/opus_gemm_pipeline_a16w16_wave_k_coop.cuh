// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx942 wave-K-cooperative pipeline (kid 10300).
//
// Target small-shape tiles: one WG owns a small (M, N) output tile; all waves
// within the WG split across K (T_K=BLOCK_SIZE/64) and accumulate into
// wave-local fp32 partials, then LDS-reduce those partials and store bf16 Y.
// No splitK across WGs, no atomic, no separate reduce kernel, no cross-WG sync.
//
// Geometry: B=(B_M, B_N, B_K), T=(1, 1, BLOCK_SIZE/64), W=(16, 16, 16),
// LDS_DEPTH=1. Current tuned point is 512x16x16x64 (T_K=8).
//   E_M = B_M / W_M, E_N = B_N / W_N, E_K = B_K / W_K
//   Each wave runs K_LOOPS = K / (B_K * T_K) K-tiles and emits E_M*E_N
//   fp32 acc groups (4 fp32 per lane per group).
//
// Determinism: each wave's accumulator is a fixed K-loop order; LDS reduce
// of T_K partials uses fixed wave-id order.
#pragma once

#include "opus_gemm_traits_a16w16.cuh"

#ifdef __HIP_DEVICE_COMPILE__

namespace opus_wkc_gfx942 {

using opus::operator""_I;

// Lane layout for v_mfma_f32_16x16x16_bf16:
//   A operand: lane (m, k4) where m = lane%16, k4 = lane/16
//              short4 per lane = A[m, k4*4 + 0..3]
//   B operand: same shape -- short4 per lane = B[n, k4*4 + 0..3]
//   C accumulator: 4 fp32 per lane, lane (m_sub, n) where m_sub = lane/16, n = lane%16
//                  acc[i] = C[m_sub*4 + i, n]

template<typename T>
struct wkc_layout {
    static_assert(T::T_M >= 1 && T::T_N >= 1 && T::T_K >= 1,
                  "wave-K-coop: T_M/T_N/T_K must be positive");
    static_assert(T::T_M == 1 && T::T_N == 1,
                  "wave-K-coop: all waves must split K (T_M=T_N=1)");
    static_assert(T::T_M * T::T_N * T::T_K ==
                  T::BLOCK_SIZE / opus::get_warp_size(),
                  "wave-K-coop: T_M*T_N*T_K must equal waves per WG");
    static_assert(T::BLOCK_SIZE == 64 || T::BLOCK_SIZE == 256 ||
                  T::BLOCK_SIZE == 512 || T::BLOCK_SIZE == 1024,
                  "wave-K-coop: supported BS is 64, 256, 512, or 1024");
    static_assert(T::W_M == 16 && T::W_N == 16 && T::W_K == 16,
                  "wave-K-coop: mfma 16x16x16 only");
    static_assert(T::VEC_A == T::VEC_B, "wave-K-coop: VEC_A==VEC_B");
    static_assert(T::B_M % T::W_M == 0, "wave-K-coop: B_M divisible by W_M");
    static_assert(T::B_N % T::W_N == 0, "wave-K-coop: B_N divisible by W_N");
    static_assert(T::B_M % T::T_M == 0, "wave-K-coop: B_M divisible by T_M");
    static_assert(T::B_N % T::T_N == 0, "wave-K-coop: B_N divisible by T_N");
    static_assert(T::B_K % T::W_K == 0,
                  "wave-K-coop: B_K must cleanly divide W_K");

    static constexpr int WAVES_PER_WG = T::BLOCK_SIZE / opus::get_warp_size();
    static constexpr int TILE_M = T::B_M / T::T_M;     // M rows owned by one wave group
    static constexpr int TILE_N = T::B_N / T::T_N;     // N rows owned by one wave group
    static constexpr int E_M = TILE_M / T::W_M;        // M groups per wave
    static constexpr int E_N = TILE_N / T::W_N;        // N groups per wave
    static constexpr int E_K = T::B_K / T::W_K;        // K subgroups per K-tile (=4 with B_K=64)
    static constexpr int N_SUB = E_M * E_N;            // mfma groups per wave per K-tile
};

// ---- LDS layout ----
// A staging:  B_M rows x (B_K * T_K) cols (full K-tile for all waves in one slab)
//             stride = B_K * T_K + pad
// B staging:  B_N rows x (B_K * T_K) cols
//             stride = B_K * T_K + pad
// Reduce stage: T_K x (B_M * B_N) fp32 partials.
//
// Keep the partial stage separate from A/B LDS. This costs 8 KiB for the tuned
// 512x16x16x64 point, but removes the pre-reuse WG barrier after the K-loop.

template<typename T>
struct wkc_smem {
    static constexpr int WAVE_SKEW_BYTES = 0;
    static constexpr int PARTIAL_SKEW_FLOATS = 0;
    static constexpr int A_ROWS = wkc_layout<T>::TILE_M;
    static constexpr int A_COLS = T::B_K;          // per-wave K-segment width (one K-tile)
    // For bf16 LDS row reads, bank step is (stride * 2B) / 4B.  B_K+8 gives
    // 36 mod 32 and aliases rows every 8; B_K+2 gives 33 mod 32 and spreads
    // the MFMA operand rows across banks.
    static constexpr int A_PAD  = 4;
    static constexpr int A_STRIDE = A_COLS + A_PAD;
    static constexpr int A_SLICE_BYTES = A_ROWS * A_STRIDE * sizeof(typename T::D_A);
    static constexpr int A_WAVE_BYTES = A_SLICE_BYTES + WAVE_SKEW_BYTES;
    static constexpr int A_SLABS = T::T_K;
    static constexpr int A_BYTES = A_WAVE_BYTES * A_SLABS;
    static constexpr int B_ROWS = wkc_layout<T>::TILE_N;
    static constexpr int B_COLS = T::B_K;
    static constexpr int B_PAD  = 4;
    static constexpr int B_STRIDE = B_COLS + B_PAD;
    static constexpr int B_SLICE_BYTES = B_ROWS * B_STRIDE * sizeof(typename T::D_B);
    static constexpr int B_WAVE_BYTES = B_SLICE_BYTES + WAVE_SKEW_BYTES;
    static constexpr int B_SLABS = T::T_K;
    static constexpr int B_BYTES = B_WAVE_BYTES * B_SLABS;

    static constexpr int PARTIAL_FLOATS = T::B_M * T::B_N;
    static constexpr int PARTIAL_STRIDE_FLOATS = PARTIAL_FLOATS + PARTIAL_SKEW_FLOATS;
    static constexpr int PARTIAL_BYTES = PARTIAL_STRIDE_FLOATS * sizeof(float);
    static constexpr int REDUCE_STAGE_BYTES = T::T_K * PARTIAL_BYTES;
    static constexpr int AB_BYTES = A_BYTES + B_BYTES;
    static constexpr bool ALIAS_PARTIAL =
        (AB_BYTES + REDUCE_STAGE_BYTES > 64 * 1024) &&
        (AB_BYTES <= 64 * 1024);

    static constexpr int LDS_BYTES =
        AB_BYTES + (ALIAS_PARTIAL ? 0 : REDUCE_STAGE_BYTES);
    static_assert(LDS_BYTES <= 64 * 1024,
                  "wave-K-coop: LDS budget exceeded (need <= 64 KiB per WG)");
};

// ---- Gmem load (per wave, into VGPR) ----
//
// A tile per wave: (B_M, B_K) at K offset (wave_id_k * B_K) inside the current
// K-tile group. Wave-local K layout: lane (m, k_off) covers (B_M, B_K).
//
// Load layout: 256 thread WG but each wave (64 lanes) does its own load.
//   threads_k = B_K / VEC_A     (e.g. 64 / 8 = 8)
//   threads_m = warp_size / threads_k = 64 / 8 = 8
//   m_per_load = lane / threads_k   (0..7)
//   k_off       = (lane % threads_k) * VEC_A   (0, 8, ..., 56)
//   n_loads = B_M / threads_m = e.g. 32 / 8 = 4

template<typename T, typename G>
OPUS_D inline opus::vector_t<typename T::D_A,
        T::VEC_A * (wkc_layout<T>::TILE_M * T::B_K /
                    (opus::get_warp_size() * T::VEC_A))>
load_a_wave_tile(G& g_a, int wave_m_offset, int wave_k_tile_offset,
                 int lane_id, int stride_a)
{
    constexpr int VEC = T::VEC_A;
    constexpr int LANES = opus::get_warp_size();
    constexpr int THREADS_K = T::B_K / VEC;
    constexpr int THREADS_M = LANES / THREADS_K;
    constexpr int N_LOADS = wkc_layout<T>::TILE_M / THREADS_M;
    static_assert(THREADS_K * THREADS_M == LANES,
                  "wkc: B_K/VEC_A must evenly divide warp_size");
    static_assert(wkc_layout<T>::TILE_M % THREADS_M == 0,
                  "wkc: wave-local M tile must divide by THREADS_M");

    opus::vector_t<typename T::D_A, VEC * N_LOADS> out;
    int m_in_lane = lane_id / THREADS_K;
    int k_off     = (lane_id % THREADS_K) * VEC;
    #pragma unroll
    for (int i = 0; i < N_LOADS; ++i) {
        int m = wave_m_offset + i * THREADS_M + m_in_lane;
        int g_off = m * stride_a + wave_k_tile_offset + k_off;
        auto v = g_a.template load<VEC>(g_off);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) out[i * VEC + j] = v[j];
    }
    return out;
}

template<typename T, typename G>
OPUS_D inline opus::vector_t<typename T::D_B,
        T::VEC_B * (wkc_layout<T>::TILE_N * T::B_K /
                    (opus::get_warp_size() * T::VEC_B))>
load_b_wave_tile(G& g_b, int wave_n_offset, int wave_k_tile_offset,
                 int lane_id, int stride_b)
{
    constexpr int VEC = T::VEC_B;
    constexpr int LANES = opus::get_warp_size();
    constexpr int THREADS_K = T::B_K / VEC;
    constexpr int THREADS_N = LANES / THREADS_K;
    constexpr int N_LOADS = wkc_layout<T>::TILE_N / THREADS_N;
    static_assert(wkc_layout<T>::TILE_N % THREADS_N == 0,
                  "wkc: wave-local N tile must divide by THREADS_N");

    opus::vector_t<typename T::D_B, VEC * N_LOADS> out;
    int n_in_lane = lane_id / THREADS_K;
    int k_off     = (lane_id % THREADS_K) * VEC;
    #pragma unroll
    for (int i = 0; i < N_LOADS; ++i) {
        int n = wave_n_offset + i * THREADS_N + n_in_lane;
        int g_off = n * stride_b + wave_k_tile_offset + k_off;
        auto v = g_b.template load<VEC>(g_off);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) out[i * VEC + j] = v[j];
    }
    return out;
}

// ---- LDS staging: vgpr -> LDS (one wave's slab) ----
//
// Wave i writes its A slab into LDS region [i * A_BYTES_PER_WAVE .. ),
// preserving load layout (same m/k mapping). Likewise B.

template<typename T, typename V>
OPUS_D inline void store_a_wave_to_lds(char* smem_a_base, int wave_slab_id,
                                       const V& va, int lane_id)
{
    constexpr int VEC = T::VEC_A;
    constexpr int LANES = opus::get_warp_size();
    constexpr int THREADS_K = T::B_K / VEC;
    constexpr int THREADS_M = LANES / THREADS_K;
    constexpr int N_LOADS = wkc_layout<T>::TILE_M / THREADS_M;

    typename T::D_A* s = reinterpret_cast<typename T::D_A*>(
        smem_a_base + wave_slab_id * wkc_smem<T>::A_WAVE_BYTES);
    int m_in_lane = lane_id / THREADS_K;
    int k_off     = (lane_id % THREADS_K) * VEC;
    #pragma unroll
    for (int i = 0; i < N_LOADS; ++i) {
        int m = i * THREADS_M + m_in_lane;
        int s_off = m * wkc_smem<T>::A_STRIDE + k_off;
        opus::vector_t<typename T::D_A, VEC> v;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) v[j] = va[i * VEC + j];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) s[s_off + j] = v[j];
    }
}

template<typename T, typename V>
OPUS_D inline void store_b_wave_to_lds(char* smem_b_base, int wave_slab_id,
                                       const V& vb, int lane_id)
{
    constexpr int VEC = T::VEC_B;
    constexpr int LANES = opus::get_warp_size();
    constexpr int THREADS_K = T::B_K / VEC;
    constexpr int THREADS_N = LANES / THREADS_K;
    constexpr int N_LOADS = wkc_layout<T>::TILE_N / THREADS_N;

    typename T::D_B* s = reinterpret_cast<typename T::D_B*>(
        smem_b_base + wave_slab_id * wkc_smem<T>::B_WAVE_BYTES);
    int n_in_lane = lane_id / THREADS_K;
    int k_off     = (lane_id % THREADS_K) * VEC;
    #pragma unroll
    for (int i = 0; i < N_LOADS; ++i) {
        int n = i * THREADS_N + n_in_lane;
        int s_off = n * wkc_smem<T>::B_STRIDE + k_off;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) s[s_off + j] = vb[i * VEC + j];
    }
}

// ---- LDS -> mfma operand ----
//
// Per K-tile, per wave:
//   For each (e_m, e_k) read short4_ab (4 bf16) of A[e_m*W_M + (lane%16),
//                                              e_k*W_K + (lane/16)*4 + 0..3]
//   For each (e_n, e_k) read short4_ab of B[e_n*W_N + (lane%16),
//                                          e_k*W_K + (lane/16)*4 + 0..3]
// Then for each (e_m, e_n) accumulate: acc[e_m*E_N + e_n] += A * B
// over k subgroups e_k = 0..E_K-1

using short4_ab = opus::vector_t<__bf16, 4>;
using float4_acc = float __attribute__((ext_vector_type(4)));

template<typename T>
OPUS_D inline short4_ab read_a_pair_from_lds(const char* smem_a_base, int wave_slab_id,
                                              int e_m, int e_k, int lane_id)
{
    const typename T::D_A* s = reinterpret_cast<const typename T::D_A*>(
        smem_a_base + wave_slab_id * wkc_smem<T>::A_WAVE_BYTES);
    int m = e_m * T::W_M + (lane_id % 16);
    int k = e_k * T::W_K + (lane_id / 16) * 4;
    short4_ab r;
    #pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = s[m * wkc_smem<T>::A_STRIDE + k + j];
    return r;
}

template<typename T>
OPUS_D inline short4_ab read_b_pair_from_lds(const char* smem_b_base, int wave_slab_id,
                                              int e_n, int e_k, int lane_id)
{
    const typename T::D_B* s = reinterpret_cast<const typename T::D_B*>(
        smem_b_base + wave_slab_id * wkc_smem<T>::B_WAVE_BYTES);
    int n = e_n * T::W_N + (lane_id % 16);
    int k = e_k * T::W_K + (lane_id / 16) * 4;
    short4_ab r;
    #pragma unroll
    for (int j = 0; j < 4; ++j) r[j] = s[n * wkc_smem<T>::B_STRIDE + k + j];
    return r;
}

OPUS_D inline void mfma_16x16x16_bf16(short4_ab a, short4_ab b, float4_acc& acc)
{
    // Standard mfma: dst = A @ B + dst. Asm operand order: dst, srcA, srcB, srcC.
    asm volatile(
        "v_mfma_f32_16x16x16_bf16 %[c], %[a], %[b], %[c]\n"
        : [c] "+v" (acc)
        : [a] "v" (a), [b] "v" (b)
        : "memory");
}

// Compute one K-tile per wave: walk E_K K-subgroups, for each compute E_M*E_N mfma.
template<typename T>
OPUS_D inline void wave_compute_one_k_tile(const char* smem_a_base,
                                            const char* smem_b_base,
                                            int a_slab_id, int b_slab_id,
                                            int lane_id,
                                            float4_acc* acc)
{
    constexpr int E_M = wkc_layout<T>::E_M;
    constexpr int E_N = wkc_layout<T>::E_N;
    constexpr int E_K = wkc_layout<T>::E_K;

    if constexpr (E_M == 1 && E_N == 1 && E_K == 4) {
        short4_ab a0 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 0, 0, lane_id);
        short4_ab b0 = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, 0, 0, lane_id);
        short4_ab a1 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 0, 1, lane_id);
        short4_ab b1 = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, 0, 1, lane_id);
        short4_ab a2 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 0, 2, lane_id);
        short4_ab b2 = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, 0, 2, lane_id);
        short4_ab a3 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 0, 3, lane_id);
        short4_ab b3 = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, 0, 3, lane_id);

        s_waitcnt_lgkmcnt(6_I);
        mfma_16x16x16_bf16(a0, b0, acc[0]);

        s_waitcnt_lgkmcnt(4_I);
        mfma_16x16x16_bf16(a1, b1, acc[0]);

        s_waitcnt_lgkmcnt(2_I);
        mfma_16x16x16_bf16(a2, b2, acc[0]);

        s_waitcnt_lgkmcnt(0_I);
        mfma_16x16x16_bf16(a3, b3, acc[0]);
    } else if constexpr (E_M == 2 && E_N == 1 && E_K == 4) {
        short4_ab a00 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 0, 0, lane_id);
        short4_ab a10 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 1, 0, lane_id);
        short4_ab b0  = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, 0, 0, lane_id);
        short4_ab a01 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 0, 1, lane_id);
        short4_ab a11 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 1, 1, lane_id);
        short4_ab b1  = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, 0, 1, lane_id);
        short4_ab a02 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 0, 2, lane_id);
        short4_ab a12 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 1, 2, lane_id);
        short4_ab b2  = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, 0, 2, lane_id);
        short4_ab a03 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 0, 3, lane_id);
        short4_ab a13 = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, 1, 3, lane_id);
        short4_ab b3  = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, 0, 3, lane_id);

        s_waitcnt_lgkmcnt(9_I);
        mfma_16x16x16_bf16(a00, b0, acc[0]);
        mfma_16x16x16_bf16(a10, b0, acc[1]);

        s_waitcnt_lgkmcnt(6_I);
        mfma_16x16x16_bf16(a01, b1, acc[0]);
        mfma_16x16x16_bf16(a11, b1, acc[1]);

        s_waitcnt_lgkmcnt(3_I);
        mfma_16x16x16_bf16(a02, b2, acc[0]);
        mfma_16x16x16_bf16(a12, b2, acc[1]);

        s_waitcnt_lgkmcnt(0_I);
        mfma_16x16x16_bf16(a03, b3, acc[0]);
        mfma_16x16x16_bf16(a13, b3, acc[1]);
    } else if constexpr (E_M == 2 && E_N == 2 && E_K == 4) {
        short4_ab a0 = read_a_pair_from_lds<T>(
            smem_a_base, a_slab_id, 0, 0, lane_id);
        short4_ab b0 = read_b_pair_from_lds<T>(
            smem_b_base, b_slab_id, 0, 0, lane_id);
        short4_ab a1 = read_a_pair_from_lds<T>(
            smem_a_base, a_slab_id, 1, 0, lane_id);
        short4_ab b1 = read_b_pair_from_lds<T>(
            smem_b_base, b_slab_id, 1, 0, lane_id);

#define OPUS_WKC_E2N2_STEP(K_NEXT)                                                   \
        s_waitcnt_lgkmcnt(2_I);                                                       \
        mfma_16x16x16_bf16(a0, b0, acc[0]);                                           \
        s_waitcnt_lgkmcnt(1_I);                                                       \
        mfma_16x16x16_bf16(a1, b0, acc[2]);                                           \
        s_waitcnt_lgkmcnt(0_I);                                                       \
        {                                                                             \
            short4_ab na0 = read_a_pair_from_lds<T>(                                  \
                smem_a_base, a_slab_id, 0, K_NEXT, lane_id);                           \
            short4_ab nb0 = read_b_pair_from_lds<T>(                                  \
                smem_b_base, b_slab_id, 0, K_NEXT, lane_id);                           \
            short4_ab na1 = read_a_pair_from_lds<T>(                                  \
                smem_a_base, a_slab_id, 1, K_NEXT, lane_id);                           \
            short4_ab nb1 = read_b_pair_from_lds<T>(                                  \
                smem_b_base, b_slab_id, 1, K_NEXT, lane_id);                           \
            mfma_16x16x16_bf16(a0, b1, acc[1]);                                       \
            mfma_16x16x16_bf16(a1, b1, acc[3]);                                       \
            a0 = na0; b0 = nb0; a1 = na1; b1 = nb1;                                   \
        }

        OPUS_WKC_E2N2_STEP(1)
        OPUS_WKC_E2N2_STEP(2)
        OPUS_WKC_E2N2_STEP(3)

        s_waitcnt_lgkmcnt(2_I);
        mfma_16x16x16_bf16(a0, b0, acc[0]);
        s_waitcnt_lgkmcnt(1_I);
        mfma_16x16x16_bf16(a1, b0, acc[2]);
        s_waitcnt_lgkmcnt(0_I);
        mfma_16x16x16_bf16(a0, b1, acc[1]);
        mfma_16x16x16_bf16(a1, b1, acc[3]);

#undef OPUS_WKC_E2N2_STEP
    } else {
        #pragma unroll
        for (int e_k = 0; e_k < E_K; ++e_k) {
            short4_ab a_buf[E_M];
            short4_ab b_buf[E_N];
            #pragma unroll
            for (int e_m = 0; e_m < E_M; ++e_m)
                a_buf[e_m] = read_a_pair_from_lds<T>(smem_a_base, a_slab_id, e_m, e_k, lane_id);
            #pragma unroll
            for (int e_n = 0; e_n < E_N; ++e_n)
                b_buf[e_n] = read_b_pair_from_lds<T>(smem_b_base, b_slab_id, e_n, e_k, lane_id);
            #pragma unroll
            for (int e_m = 0; e_m < E_M; ++e_m) {
                #pragma unroll
                for (int e_n = 0; e_n < E_N; ++e_n) {
                    mfma_16x16x16_bf16(a_buf[e_m], b_buf[e_n],
                                        acc[e_m * E_N + e_n]);
                }
            }
        }
    }
}

// ---- Wave-LDS reduce + bf16 store ----
//
// After K-loop: each wave holds E_M*E_N float4_acc (= 4 fp32 per lane per group)
// representing its slice's contribution to the full (B_M, B_N) output.
//
// We need to sum T_K wave partials and write a single bf16 Y[m, n].
//
// Step 1: each wave writes its acc to LDS partial buffer at offset wave_id * partial_bytes.
//   acc lane (m_sub = lane/16, n = lane%16); 4 fp32 per lane occupy
//   C[m_sub*4 + 0..3, n] for the W_M*W_N=256 cell tile of group (e_m, e_n).
//   Full output cell = (e_m*W_M + m_sub*4 + i, e_n*W_N + n) for i in 0..3
//
// Step 2: barrier
//
// Step 3: the whole WG reads all T_K partials from LDS, sums them in fixed
//         wave-id order, casts bf16, stores Y.

template<typename T>
OPUS_D inline void store_wave_acc_to_lds_partial(char* lds_partial_base, int wave_id_k,
                                                  const float4_acc* acc, int lane_id)
{
    constexpr int E_M = wkc_layout<T>::E_M;
    constexpr int E_N = wkc_layout<T>::E_N;
    float* s = reinterpret_cast<float*>(
        lds_partial_base + wave_id_k * wkc_smem<T>::PARTIAL_BYTES);

    int m_sub = lane_id / 16;
    int n     = lane_id % 16;
    #pragma unroll
    for (int e_m = 0; e_m < E_M; ++e_m) {
        #pragma unroll
        for (int e_n = 0; e_n < E_N; ++e_n) {
            int m_base = e_m * T::W_M + m_sub * 4;
            int n_pos  = e_n * T::W_N + n;
            const float4_acc& a = acc[e_m * E_N + e_n];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                s[(m_base + i) * T::B_N + n_pos] = a[i];
            }
        }
    }
}

template<typename T, typename Y_T>
OPUS_D inline void reduce_partials_and_store_y(const char* lds_partial_base,
                                                Y_T* ptr_y, int row, int col,
                                                int m_total, int n_total,
                                                int stride_y, int tid_in_wg)
{
    constexpr int TILE_CELLS = T::B_M * T::B_N;
    const bool full_tile = (row + T::B_M <= m_total) && (col + T::B_N <= n_total);
    for (int idx = tid_in_wg; idx < TILE_CELLS; idx += T::BLOCK_SIZE) {
        int m = idx / T::B_N;
        int n = idx - m * T::B_N;
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < T::T_K; ++w) {
            const float* p = reinterpret_cast<const float*>(
                lds_partial_base + w * wkc_smem<T>::PARTIAL_BYTES);
            sum += p[idx];
        }
        int gm = row + m;
        int gn = col + n;
        if (full_tile || (gm < m_total && gn < n_total)) {
            ptr_y[gm * stride_y + gn] = static_cast<Y_T>(sum);
        }
    }
}

} // namespace opus_wkc_gfx942

#endif // __HIP_DEVICE_COMPILE__

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a16w16_wave_k_coop_kernel(opus_gemm_noscale_kargs kargs)
{
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx942__)
    using namespace opus_wkc_gfx942;
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;

    static_assert(T::T_M * T::T_N * T::T_K ==
                  T::BLOCK_SIZE / opus::get_warp_size(),
                  "wave-K-coop: T_M*T_N*T_K must equal waves per WG");
    static_assert(T::BLOCK_SIZE == 64 || T::BLOCK_SIZE == 256 ||
                  T::BLOCK_SIZE == 512 || T::BLOCK_SIZE == 1024,
                  "wave-K-coop: supported BS is 64, 256, 512, or 1024");
    static_assert(T::LDS_DEPTH == 1, "wave-K-coop: single-buffer LDS");

    constexpr int N_SUB = wkc_layout<T>::N_SUB;
    constexpr int K_TILE_FULL = T::B_K * T::T_K;  // full WG K-tile = T_K wave-tiles

    const int tile_n = opus::block_id_x();
    const int tile_m = opus::block_id_y();
    const int batch_id = opus::block_id_z();
    const int row = tile_m * T::B_M;
    const int col = tile_n * T::B_N;

    const int tid = opus::thread_id_x();
    const int wave_id = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wave_id_k = wave_id;
    const int lane_id = tid % 64;

    const D_A* ptr_a = reinterpret_cast<const D_A*>(kargs.ptr_a)
                       + batch_id * kargs.stride_a_batch + row * kargs.stride_a;
    const D_B* ptr_b = reinterpret_cast<const D_B*>(kargs.ptr_b)
                       + batch_id * kargs.stride_b_batch + col * kargs.stride_b;
    D_C* ptr_y = reinterpret_cast<D_C*>(kargs.ptr_c)
                 + batch_id * kargs.stride_c_batch;

    auto g_a = [&]() {
        return make_gmem(ptr_a, ((kargs.m - row) * kargs.stride_a) * sizeof(D_A));
    }();
    auto g_b = [&]() {
        return make_gmem(ptr_b, ((kargs.n - col) * kargs.stride_b) * sizeof(D_B));
    }();

    // LDS buffer (reused for A/B during K-loop, partials after)
    __shared__ char smem[wkc_smem<T>::LDS_BYTES];
    char* smem_a = smem;
    char* smem_b = smem + wkc_smem<T>::A_BYTES;
    char* smem_partial = smem + wkc_smem<T>::A_BYTES + wkc_smem<T>::B_BYTES;
    if constexpr (wkc_smem<T>::ALIAS_PARTIAL) {
        smem_partial = smem;
    }

    // Per-wave accumulators
    float4_acc acc[N_SUB] = {};

    const int wg_k_loops = ceil_div(kargs.k, K_TILE_FULL);
    const int wave_m_offset = 0;
    const int wave_n_offset = 0;
    const int wave_k_offset = wave_id_k * T::B_K;
    const int a_slab_id = wave_id_k;
    const int b_slab_id = wave_id_k;

    auto va = load_a_wave_tile<T>(
        g_a, wave_m_offset, wave_k_offset, lane_id, kargs.stride_a);
    auto vb = load_b_wave_tile<T>(
        g_b, wave_n_offset, wave_k_offset, lane_id, kargs.stride_b);

    #pragma unroll 4
    for (int t = 0; t < wg_k_loops; ++t) {
        store_b_wave_to_lds<T>(smem_b, b_slab_id, vb, lane_id);
        store_a_wave_to_lds<T>(smem_a, a_slab_id, va, lane_id);
        const bool has_next = (t + 1) < wg_k_loops;
        auto va_next = va;
        auto vb_next = vb;
        if (has_next) {
            int next_k_base = (t + 1) * K_TILE_FULL;
            vb_next = load_b_wave_tile<T>(
                g_b, wave_n_offset, next_k_base + wave_k_offset,
                lane_id, kargs.stride_b);
            va_next = load_a_wave_tile<T>(
                g_a, wave_m_offset, next_k_base + wave_k_offset,
                lane_id, kargs.stride_a);
        }
        wave_compute_one_k_tile<T>(
            smem_a, smem_b, a_slab_id, b_slab_id, lane_id,
            acc);
        va = va_next;
        vb = vb_next;
    }

    // K loop done. Write wave partials to a separate LDS stage and reduce.
    if constexpr (wkc_smem<T>::ALIAS_PARTIAL) {
        __builtin_amdgcn_s_barrier();
    }
    store_wave_acc_to_lds_partial<T>(
        smem_partial, wave_id_k, acc, lane_id);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    reduce_partials_and_store_y<T, D_C>(
        smem_partial, ptr_y, row, col, kargs.m, kargs.n, kargs.stride_c, tid);

#endif // __gfx942__
#endif // __HIP_DEVICE_COMPILE__
}
