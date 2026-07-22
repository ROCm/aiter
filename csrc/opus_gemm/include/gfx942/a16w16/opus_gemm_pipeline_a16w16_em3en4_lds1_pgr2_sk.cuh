// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// gfx942 EM3EN4 LDS1/PGR2 splitK pipeline (kid 10204).
// Device geometry is 96x128x128; host tile is 128M x 96N via A/B swap.
#pragma once

#include "opus_gemm_traits_a16w16.cuh"
#include "opus_gemm_helpers_a16w16.cuh"
#include "splitk_reduce_gfx942.cuh"

#ifdef __HIP_DEVICE_COMPILE__

#include "opus_gemm_mfma16x16x16_gfx942.cuh"

namespace opus_em3en4_gfx942 {

using opus::operator""_I;

// EM3EN4 runs physical 96x128x{64|128} 16x16x16 BF16 MFMA with single-buffer LDS.
template<typename T>
struct em3en4_traits_requirements {
    static_assert(T::LDS_DEPTH == 1, "EM3EN4 requires single-buffer LDS (LDS_DEPTH=1)");
    static_assert(T::B_M == 96 && T::B_N == 128 && (T::B_K == 64 || T::B_K == 128),
                  "EM3EN4 requires traits BLOCK=(96, 128, {64|128})");
    static_assert(T::BLOCK_SIZE == 256 && T::T_M == 2 && T::T_N == 2,
                  "EM3EN4 requires BLOCK_SIZE=256, T_M=T_N=2");
    static_assert(T::W_M == 16 && T::W_N == 16 && T::W_K == 16,
                  "EM3EN4 requires 16x16x16 MFMA");
    static_assert(T::VEC_A == T::VEC_B, "EM3EN4 requires VEC_A == VEC_B");
    static_assert(T::smem_linear_wave % T::B_K == 0,
                  "EM3EN4 requires smem_linear_wave % B_K == 0");
};

template<typename T>
struct em3en4_smem_layout {
    // Enforce EM3EN4-only traits configuration.
    using _check = em3en4_traits_requirements<T>;
    // Row-major LDS with 16-element pad to avoid bank/stride conflicts.
    static constexpr int smem_padding = 16;
    static constexpr int row_stride = T::B_K + smem_padding;
    static constexpr int a_bytes = T::B_M * row_stride * sizeof(typename T::D_A);
    static constexpr int b_bytes = T::B_N * row_stride * sizeof(typename T::D_B);
    static_assert(a_bytes + b_bytes <= 64 * 1024);
};

template<int N_SUB>
OPUS_D inline auto em3en4_acc_to_vgpr(const float4_acc* acc)
{
    using VC = float __attribute__((ext_vector_type(N_SUB * 4)));
    VC result;
    float* out = reinterpret_cast<float*>(&result);
    #pragma unroll
    for (int i = 0; i < N_SUB; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) out[i * 4 + j] = acc[i][j];
    }
    return result;
}

template<int OFFSET>
OPUS_D inline void lds_write_u128_offset(unsigned addr, __uint128_t v)
{
    using lds_u128_ptr = __uint128_t __attribute__((address_space(3)))*;
    *reinterpret_cast<lds_u128_ptr>(addr + OFFSET) = v;
}

// gmem load: A and B share the layout (vecs_per_row, m=vec/vpr, k=rem*VEC).
template<typename T, int I, typename G, int VEC>
OPUS_D inline auto em3en4_load_gmem_chunk(G& g, int tile, int tid, int stride)
{
    constexpr int vecs_per_row = T::B_K / VEC;
    int vec_id = I * T::BLOCK_SIZE + tid;
    int m = vec_id / vecs_per_row;
    int k = (vec_id - m * vecs_per_row) * VEC;
    return g.template load<VEC>(m * stride + tile * T::B_K + k);
}

// smem store via ds_write_b128 offset; A/B share the row-major path.
template<typename T, int I, typename S, int VEC, typename V>
OPUS_D inline void em3en4_store_smem_chunk(S& s, const V& v, int tid)
{
    constexpr int vecs_per_row = T::B_K / VEC;
    constexpr int row_stride = em3en4_smem_layout<T>::row_stride;
    constexpr int rows_per_chunk = T::BLOCK_SIZE / vecs_per_row;
    constexpr int chunk_bytes = rows_per_chunk * row_stride * sizeof(typename S::scalar_type);
    int m = tid / vecs_per_row;
    int k = (tid - m * vecs_per_row) * VEC;
    unsigned addr = static_cast<unsigned>(reinterpret_cast<__UINTPTR_TYPE__>(s.ptr)) +
                    static_cast<unsigned>((m * row_stride + k) * sizeof(typename S::scalar_type));
    lds_write_u128_offset<I * chunk_bytes>(addr, __builtin_bit_cast(__uint128_t, v));
}

template<typename V>
struct em3en4_a_chunk_pack {
    V c0, c1, c2, c3, c4, c5;
};

template<typename V>
struct em3en4_b_chunk_pack {
    V c0, c1, c2, c3, c4, c5, c6, c7;
};

template<typename T, typename G>
OPUS_D inline auto em3en4_load_a_chunks(G& g, int tile, int tid, int stride)
{
    using chunk_t = decltype(em3en4_load_gmem_chunk<T, 0, G, T::VEC_A>(g, tile, tid, stride));
    return em3en4_a_chunk_pack<chunk_t>{
        em3en4_load_gmem_chunk<T, 0, G, T::VEC_A>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 1, G, T::VEC_A>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 2, G, T::VEC_A>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 3, G, T::VEC_A>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 4, G, T::VEC_A>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 5, G, T::VEC_A>(g, tile, tid, stride),
    };
}

template<typename T, typename G>
OPUS_D inline auto em3en4_load_b_chunks(G& g, int tile, int tid, int stride)
{
    using chunk_t = decltype(em3en4_load_gmem_chunk<T, 0, G, T::VEC_B>(g, tile, tid, stride));
    return em3en4_b_chunk_pack<chunk_t>{
        em3en4_load_gmem_chunk<T, 0, G, T::VEC_B>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 1, G, T::VEC_B>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 2, G, T::VEC_B>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 3, G, T::VEC_B>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 4, G, T::VEC_B>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 5, G, T::VEC_B>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 6, G, T::VEC_B>(g, tile, tid, stride),
        em3en4_load_gmem_chunk<T, 7, G, T::VEC_B>(g, tile, tid, stride),
    };
}

template<typename T, typename S, typename V>
OPUS_D inline void em3en4_store_a_chunks(S& s, const em3en4_a_chunk_pack<V>& p, int tid)
{
    em3en4_store_smem_chunk<T, 0, S, T::VEC_A>(s, p.c0, tid);
    em3en4_store_smem_chunk<T, 1, S, T::VEC_A>(s, p.c1, tid);
    em3en4_store_smem_chunk<T, 2, S, T::VEC_A>(s, p.c2, tid);
    em3en4_store_smem_chunk<T, 3, S, T::VEC_A>(s, p.c3, tid);
    em3en4_store_smem_chunk<T, 4, S, T::VEC_A>(s, p.c4, tid);
    em3en4_store_smem_chunk<T, 5, S, T::VEC_A>(s, p.c5, tid);
}

template<typename T, typename S, typename V>
OPUS_D inline void em3en4_store_b_chunks(S& s, const em3en4_b_chunk_pack<V>& p, int tid)
{
    em3en4_store_smem_chunk<T, 0, S, T::VEC_B>(s, p.c0, tid);
    em3en4_store_smem_chunk<T, 1, S, T::VEC_B>(s, p.c1, tid);
    em3en4_store_smem_chunk<T, 2, S, T::VEC_B>(s, p.c2, tid);
    em3en4_store_smem_chunk<T, 3, S, T::VEC_B>(s, p.c3, tid);
    em3en4_store_smem_chunk<T, 4, S, T::VEC_B>(s, p.c4, tid);
    em3en4_store_smem_chunk<T, 5, S, T::VEC_B>(s, p.c5, tid);
    em3en4_store_smem_chunk<T, 6, S, T::VEC_B>(s, p.c6, tid);
    em3en4_store_smem_chunk<T, 7, S, T::VEC_B>(s, p.c7, tid);
}

template<int OFFSET>
OPUS_D inline i32x4_t lds_read_i32x4_offset(unsigned addr)
{
    using lds_u128_ptr = const __uint128_t __attribute__((address_space(3)))*;
    auto value = *reinterpret_cast<lds_u128_ptr>(addr + OFFSET);
    return __builtin_bit_cast(i32x4_t, value);
}

OPUS_D inline short4_ab em3en4_pair_lo(const i32x4_t& v)
{
    return reinterpret_cast<const short4_ab*>(&v)[0];
}

OPUS_D inline short4_ab em3en4_pair_hi(const i32x4_t& v)
{
    return reinterpret_cast<const short4_ab*>(&v)[1];
}

OPUS_D inline void em3en4_mfma(short4_ab b, short4_ab a, float4_acc& acc)
{
    acc = opus::mfma_f32_16x16x16_bf16{}(b, a, acc);
}

template<typename T, typename S, int WAVE_DIM, int VEC>
OPUS_D inline unsigned em3en4_lds_base_b128(S& s, int wave_id_dim, int lane_id)
{
    constexpr int row_stride = em3en4_smem_layout<T>::row_stride;
    const int lane_in_wave = lane_id % WAVE_DIM;
    const int lane_k = lane_id / WAVE_DIM;
    return static_cast<unsigned>(reinterpret_cast<__UINTPTR_TYPE__>(s.ptr)) +
           static_cast<unsigned>(((wave_id_dim * WAVE_DIM + lane_in_wave) * row_stride +
               lane_k * VEC) * sizeof(typename S::scalar_type));
}

// ds_read offset = I_DIM * (T_DIM * WAVE_DIM * row_stride_bytes) + I_PAIR * 32 * dtype_bytes
template<typename T, int I_PAIR, int I_DIM, int T_DIM, int WAVE_DIM, typename D>
OPUS_D inline i32x4_t em3en4_read_pair_b128_base(unsigned base)
{
    constexpr int row_stride = em3en4_smem_layout<T>::row_stride;
    constexpr int row_stride_bytes = row_stride * sizeof(D);
    constexpr int wave_row_stride_bytes = T_DIM * WAVE_DIM * row_stride_bytes;
    constexpr int pair_stride_bytes = 32 * sizeof(D);
    constexpr int row_offset = I_DIM * wave_row_stride_bytes + I_PAIR * pair_stride_bytes;
    return lds_read_i32x4_offset<row_offset>(base);
}

// Drain 4 mfma (lo head) of the just-loaded next pair while still in the active K-tile.
template<typename T>
OPUS_D inline void em3en4_compute_pair_lo_head_packed_b128(
    const i32x4_t& a0p, const i32x4_t& a1p,
    const i32x4_t& b0p, const i32x4_t& b1p,
    float4_acc* acc)
{
    auto a0 = em3en4_pair_lo(a0p);
    auto a1 = em3en4_pair_lo(a1p);
    auto b0 = em3en4_pair_lo(b0p);
    auto b1 = em3en4_pair_lo(b1p);
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b0, a0, acc[0]);
    em3en4_mfma(b0, a1, acc[4]);
    em3en4_mfma(b1, a0, acc[1]);
    em3en4_mfma(b1, a1, acc[5]);
    __builtin_amdgcn_sched_barrier(0);
}

// Dense single-pair: lo (12 mfma) + hi (12 mfma), no prefetch, for the last tile.
template<typename T>
OPUS_D inline void em3en4_compute_pair_both_packed_b128(
    const i32x4_t& a0p, const i32x4_t& a1p, const i32x4_t& a2p,
    const i32x4_t& b0p, const i32x4_t& b1p, const i32x4_t& b2p, const i32x4_t& b3p,
    float4_acc* acc)
{
    auto a0lo = em3en4_pair_lo(a0p); auto a1lo = em3en4_pair_lo(a1p); auto a2lo = em3en4_pair_lo(a2p);
    auto b0lo = em3en4_pair_lo(b0p); auto b1lo = em3en4_pair_lo(b1p);
    auto b2lo = em3en4_pair_lo(b2p); auto b3lo = em3en4_pair_lo(b3p);
    auto a0hi = em3en4_pair_hi(a0p); auto a1hi = em3en4_pair_hi(a1p); auto a2hi = em3en4_pair_hi(a2p);
    auto b0hi = em3en4_pair_hi(b0p); auto b1hi = em3en4_pair_hi(b1p);
    auto b2hi = em3en4_pair_hi(b2p); auto b3hi = em3en4_pair_hi(b3p);
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b0lo, a0lo, acc[0]);  em3en4_mfma(b0lo, a1lo, acc[4]);
    em3en4_mfma(b1lo, a0lo, acc[1]);  em3en4_mfma(b1lo, a1lo, acc[5]);
    em3en4_mfma(b0lo, a2lo, acc[8]);  em3en4_mfma(b1lo, a2lo, acc[9]);
    em3en4_mfma(b2lo, a0lo, acc[2]);  em3en4_mfma(b2lo, a1lo, acc[6]);
    em3en4_mfma(b2lo, a2lo, acc[10]); em3en4_mfma(b3lo, a0lo, acc[3]);
    em3en4_mfma(b3lo, a1lo, acc[7]);  em3en4_mfma(b3lo, a2lo, acc[11]);
    em3en4_mfma(b0hi, a0hi, acc[0]);  em3en4_mfma(b0hi, a1hi, acc[4]);
    em3en4_mfma(b0hi, a2hi, acc[8]);  em3en4_mfma(b1hi, a0hi, acc[1]);
    em3en4_mfma(b1hi, a1hi, acc[5]);  em3en4_mfma(b1hi, a2hi, acc[9]);
    em3en4_mfma(b2hi, a0hi, acc[2]);  em3en4_mfma(b2hi, a1hi, acc[6]);
    em3en4_mfma(b2hi, a2hi, acc[10]); em3en4_mfma(b3hi, a0hi, acc[3]);
    em3en4_mfma(b3hi, a1hi, acc[7]);  em3en4_mfma(b3hi, a2hi, acc[11]);
    __builtin_amdgcn_sched_barrier(0);
}

// Dense pair compute plus NEXT_PAIR LDS prefetch.
template<typename T, int NEXT_PAIR>
OPUS_D inline void em3en4_compute_pair_and_prefetch_next_packed_b128(
    unsigned abase, unsigned bbase,
    const i32x4_t& a0p, const i32x4_t& a1p, const i32x4_t& a2p,
    const i32x4_t& b0p, const i32x4_t& b1p, const i32x4_t& b2p, const i32x4_t& b3p,
    i32x4_t& na0p, i32x4_t& na1p, i32x4_t& na2p,
    i32x4_t& nb0p, i32x4_t& nb1p, i32x4_t& nb2p, i32x4_t& nb3p,
    float4_acc* acc)
{
    static_assert(NEXT_PAIR > 0 && NEXT_PAIR < 4);
    static_assert(T::E_M == 3 && T::E_N == 4 && T::E_K == 8);
    static_assert(std::is_same_v<typename T::D_A, typename T::D_B>);

    constexpr int row_stride = em3en4_smem_layout<T>::row_stride;
    constexpr int row_stride_bytes = row_stride * sizeof(typename T::D_A);
    constexpr int a_wave_row_stride_bytes = T::T_M * T::W_M * row_stride_bytes;
    constexpr int b_wave_row_stride_bytes = T::T_N * T::W_N * row_stride_bytes;
    constexpr int pair_offset = NEXT_PAIR * 32 * sizeof(typename T::D_A);

    auto a0lo = em3en4_pair_lo(a0p); auto a1lo = em3en4_pair_lo(a1p); auto a2lo = em3en4_pair_lo(a2p);
    auto b0lo = em3en4_pair_lo(b0p); auto b1lo = em3en4_pair_lo(b1p);
    auto b2lo = em3en4_pair_lo(b2p); auto b3lo = em3en4_pair_lo(b3p);
    auto a0hi = em3en4_pair_hi(a0p); auto a1hi = em3en4_pair_hi(a1p); auto a2hi = em3en4_pair_hi(a2p);
    auto b0hi = em3en4_pair_hi(b0p); auto b1hi = em3en4_pair_hi(b1p);
    auto b2hi = em3en4_pair_hi(b2p); auto b3hi = em3en4_pair_hi(b3p);

    opus::s_waitcnt_lgkmcnt(opus::number<3>{});
    em3en4_mfma(b0lo, a0lo, acc[0]);
    __builtin_amdgcn_sched_barrier(0);
    na0p = lds_read_i32x4_offset<pair_offset>(abase);
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b0lo, a1lo, acc[4]);
    __builtin_amdgcn_sched_barrier(0);
    nb0p = lds_read_i32x4_offset<pair_offset>(bbase);
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b1lo, a0lo, acc[1]);
    __builtin_amdgcn_sched_barrier(0);
    na1p = lds_read_i32x4_offset<a_wave_row_stride_bytes + pair_offset>(abase);
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b1lo, a1lo, acc[5]);
    __builtin_amdgcn_sched_barrier(0);
    nb1p = lds_read_i32x4_offset<b_wave_row_stride_bytes + pair_offset>(bbase);
    __builtin_amdgcn_sched_barrier(0);
    opus::s_waitcnt_lgkmcnt(opus::number<4>{});
    em3en4_mfma(b0lo, a2lo, acc[8]);
    em3en4_mfma(b1lo, a2lo, acc[9]);
    __builtin_amdgcn_sched_barrier(0);
    na2p = lds_read_i32x4_offset<2 * a_wave_row_stride_bytes + pair_offset>(abase);
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b2lo, a0lo, acc[2]);
    em3en4_mfma(b2lo, a1lo, acc[6]);
    __builtin_amdgcn_sched_barrier(0);
    nb2p = lds_read_i32x4_offset<2 * b_wave_row_stride_bytes + pair_offset>(bbase);
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b2lo, a2lo, acc[10]);
    em3en4_mfma(b3lo, a0lo, acc[3]);
    __builtin_amdgcn_sched_barrier(0);
    nb3p = lds_read_i32x4_offset<3 * b_wave_row_stride_bytes + pair_offset>(bbase);
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b3lo, a1lo, acc[7]);  em3en4_mfma(b3lo, a2lo, acc[11]);
    em3en4_mfma(b0hi, a0hi, acc[0]);  em3en4_mfma(b0hi, a1hi, acc[4]);
    em3en4_mfma(b0hi, a2hi, acc[8]);  em3en4_mfma(b1hi, a0hi, acc[1]);
    em3en4_mfma(b1hi, a1hi, acc[5]);  em3en4_mfma(b1hi, a2hi, acc[9]);
    em3en4_mfma(b2hi, a0hi, acc[2]);  em3en4_mfma(b2hi, a1hi, acc[6]);
    em3en4_mfma(b2hi, a2hi, acc[10]); em3en4_mfma(b3hi, a0hi, acc[3]);
    em3en4_mfma(b3hi, a1hi, acc[7]);  em3en4_mfma(b3hi, a2hi, acc[11]);
    __builtin_amdgcn_sched_barrier(0);
}

// Half-pair 2-mfma drain while PGR2 stores run.
template<bool HI, int AC_LO, int AC_HI>
OPUS_D inline void em3en4_half_2mfma(
    const i32x4_t& bop0, const i32x4_t& bop1, const i32x4_t& aop, float4_acc* acc)
{
    short4_ab b0, b1, a0;
    if constexpr (HI) {
        b0 = em3en4_pair_hi(bop0); b1 = em3en4_pair_hi(bop1); a0 = em3en4_pair_hi(aop);
    } else {
        b0 = em3en4_pair_lo(bop0); b1 = em3en4_pair_lo(bop1); a0 = em3en4_pair_lo(aop);
    }
    __builtin_amdgcn_sched_barrier(0);
    em3en4_mfma(b0, a0, acc[AC_LO]);
    em3en4_mfma(b1, a0, acc[AC_HI]);
    __builtin_amdgcn_sched_barrier(0);
}

// pgr2 store + reload chunk: waits vmem with HAS_FUTURE-aware drain count.
template<typename T, int I, bool HAS_FUTURE, int DRAIN, typename S, typename G, int VEC, typename V>
OPUS_D inline void em3en4_pgr2_store_reload_chunk(
    S& s, G& g, V& v, int future_tile, int tid, int stride)
{
    if constexpr (HAS_FUTURE) s_waitcnt_vmcnt(13_I);
    else                       s_waitcnt_vmcnt(opus::number<DRAIN>{});
    em3en4_store_smem_chunk<T, I, S, VEC>(s, v, tid);
    if constexpr (HAS_FUTURE) v = em3en4_load_gmem_chunk<T, I, G, VEC>(g, future_tile, tid, stride);
}

// Finish last mfma pair while PGR2 refills the next tile.
template<typename T, bool HAS_FUTURE,
         typename SA, typename SB, typename GA, typename GB, typename VA, typename VB>
OPUS_D inline void em3en4_finish_last_pair_with_pgr2_chunks(
    SA& s_a, SB& s_b,
    const i32x4_t& a0p, const i32x4_t& a1p, const i32x4_t& a2p,
    const i32x4_t& b0p, const i32x4_t& b1p, const i32x4_t& b2p, const i32x4_t& b3p,
    float4_acc* acc,
    em3en4_a_chunk_pack<VA>& pfa, em3en4_b_chunk_pack<VB>& pfb,
    GA& g_a, GB& g_b, int future_tile, int tid, int stride_a, int stride_b)
{
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    em3en4_pgr2_store_reload_chunk<T, 0, HAS_FUTURE, 13, SA, GA, T::VEC_A>(s_a, g_a, pfa.c0, future_tile, tid, stride_a);
    em3en4_half_2mfma<false, 2, 3>(b2p, b3p, a0p, acc);
    em3en4_pgr2_store_reload_chunk<T, 1, HAS_FUTURE, 12, SA, GA, T::VEC_A>(s_a, g_a, pfa.c1, future_tile, tid, stride_a);
    em3en4_half_2mfma<false, 6, 7>(b2p, b3p, a1p, acc);
    em3en4_pgr2_store_reload_chunk<T, 0, HAS_FUTURE, 11, SB, GB, T::VEC_B>(s_b, g_b, pfb.c0, future_tile, tid, stride_b);
    em3en4_half_2mfma<false, 8, 9>(b0p, b1p, a2p, acc);
    em3en4_pgr2_store_reload_chunk<T, 1, HAS_FUTURE, 10, SB, GB, T::VEC_B>(s_b, g_b, pfb.c1, future_tile, tid, stride_b);
    em3en4_half_2mfma<false, 10, 11>(b2p, b3p, a2p, acc);
    em3en4_pgr2_store_reload_chunk<T, 2, HAS_FUTURE,  9, SA, GA, T::VEC_A>(s_a, g_a, pfa.c2, future_tile, tid, stride_a);
    em3en4_half_2mfma<true, 0, 1>(b0p, b1p, a0p, acc);
    em3en4_pgr2_store_reload_chunk<T, 3, HAS_FUTURE,  8, SA, GA, T::VEC_A>(s_a, g_a, pfa.c3, future_tile, tid, stride_a);
    em3en4_half_2mfma<true, 4, 5>(b0p, b1p, a1p, acc);
    em3en4_pgr2_store_reload_chunk<T, 2, HAS_FUTURE,  7, SB, GB, T::VEC_B>(s_b, g_b, pfb.c2, future_tile, tid, stride_b);
    em3en4_half_2mfma<true, 8, 9>(b0p, b1p, a2p, acc);
    em3en4_pgr2_store_reload_chunk<T, 3, HAS_FUTURE,  6, SB, GB, T::VEC_B>(s_b, g_b, pfb.c3, future_tile, tid, stride_b);
    em3en4_half_2mfma<true, 2, 3>(b2p, b3p, a0p, acc);
    em3en4_pgr2_store_reload_chunk<T, 4, HAS_FUTURE,  5, SA, GA, T::VEC_A>(s_a, g_a, pfa.c4, future_tile, tid, stride_a);
    em3en4_half_2mfma<true, 6, 7>(b2p, b3p, a1p, acc);
    em3en4_pgr2_store_reload_chunk<T, 5, HAS_FUTURE,  4, SA, GA, T::VEC_A>(s_a, g_a, pfa.c5, future_tile, tid, stride_a);
    em3en4_half_2mfma<true, 10, 11>(b2p, b3p, a2p, acc);
    em3en4_pgr2_store_reload_chunk<T, 4, HAS_FUTURE,  3, SB, GB, T::VEC_B>(s_b, g_b, pfb.c4, future_tile, tid, stride_b);
    em3en4_pgr2_store_reload_chunk<T, 5, HAS_FUTURE,  2, SB, GB, T::VEC_B>(s_b, g_b, pfb.c5, future_tile, tid, stride_b);
    em3en4_pgr2_store_reload_chunk<T, 6, HAS_FUTURE,  1, SB, GB, T::VEC_B>(s_b, g_b, pfb.c6, future_tile, tid, stride_b);
    em3en4_pgr2_store_reload_chunk<T, 7, HAS_FUTURE,  0, SB, GB, T::VEC_B>(s_b, g_b, pfb.c7, future_tile, tid, stride_b);

    s_waitcnt_lgkmcnt(0_I);
}

// Compute K-tile (3 pair-and-prefetch + 1 lo-head + finish_last_pair_with_pgr2_chunks).
template<typename T, bool HAS_FUTURE,
         typename SA, typename SB, typename GA, typename GB, typename VA, typename VB>
OPUS_D inline void em3en4_compute_smem_packed_b128_pgr2_chunks_fine(
    SA& s_a, SB& s_b,
    int wave_id_m, int wave_id_n, int lane_id, float4_acc* acc,
    em3en4_a_chunk_pack<VA>& pfa, em3en4_b_chunk_pack<VB>& pfb,
    GA& g_a, GB& g_b, int future_tile, int tid, int stride_a, int stride_b)
{
    static_assert(T::E_M == 3 && T::E_N == 4 && T::E_K == 8);
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;

    const unsigned abase = em3en4_lds_base_b128<T, SA, T::W_M, T::VEC_A>(s_a, wave_id_m, lane_id);
    const unsigned bbase = em3en4_lds_base_b128<T, SB, T::W_N, T::VEC_B>(s_b, wave_id_n, lane_id);

    i32x4_t a0p = em3en4_read_pair_b128_base<T, 0, 0, T::T_M, T::W_M, D_A>(abase);
    i32x4_t b0p = em3en4_read_pair_b128_base<T, 0, 0, T::T_N, T::W_N, D_B>(bbase);
    i32x4_t a1p = em3en4_read_pair_b128_base<T, 0, 1, T::T_M, T::W_M, D_A>(abase);
    i32x4_t b1p = em3en4_read_pair_b128_base<T, 0, 1, T::T_N, T::W_N, D_B>(bbase);
    i32x4_t a2p = em3en4_read_pair_b128_base<T, 0, 2, T::T_M, T::W_M, D_A>(abase);
    i32x4_t b2p = em3en4_read_pair_b128_base<T, 0, 2, T::T_N, T::W_N, D_B>(bbase);
    i32x4_t b3p = em3en4_read_pair_b128_base<T, 0, 3, T::T_N, T::W_N, D_B>(bbase);
    i32x4_t na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p;

    em3en4_compute_pair_and_prefetch_next_packed_b128<T, 1>(
        abase, bbase, a0p, a1p, a2p, b0p, b1p, b2p, b3p,
        na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p, acc);
    em3en4_compute_pair_and_prefetch_next_packed_b128<T, 2>(
        abase, bbase, na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p,
        a0p, a1p, a2p, b0p, b1p, b2p, b3p, acc);
    em3en4_compute_pair_and_prefetch_next_packed_b128<T, 3>(
        abase, bbase, a0p, a1p, a2p, b0p, b1p, b2p, b3p,
        na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p, acc);

    s_waitcnt_lgkmcnt(3_I);
    em3en4_compute_pair_lo_head_packed_b128<T>(na0p, na1p, nb0p, nb1p, acc);

    em3en4_finish_last_pair_with_pgr2_chunks<T, HAS_FUTURE>(
        s_a, s_b, na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p, acc,
        pfa, pfb, g_a, g_b, future_tile, tid, stride_a, stride_b);
}

// Dense final K-tile: 4 pairs all with LDS reads from the same tile (no pgr2).
template<typename T, typename SA, typename SB>
OPUS_D inline void em3en4_compute_smem_packed_b128_dense_final(
    SA& s_a, SB& s_b, int wave_id_m, int wave_id_n, int lane_id, float4_acc* acc)
{
    static_assert(T::E_M == 3 && T::E_N == 4 && T::E_K == 8);
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;

    const unsigned abase = em3en4_lds_base_b128<T, SA, T::W_M, T::VEC_A>(s_a, wave_id_m, lane_id);
    const unsigned bbase = em3en4_lds_base_b128<T, SB, T::W_N, T::VEC_B>(s_b, wave_id_n, lane_id);

    i32x4_t a0p = em3en4_read_pair_b128_base<T, 0, 0, T::T_M, T::W_M, D_A>(abase);
    i32x4_t b0p = em3en4_read_pair_b128_base<T, 0, 0, T::T_N, T::W_N, D_B>(bbase);
    i32x4_t a1p = em3en4_read_pair_b128_base<T, 0, 1, T::T_M, T::W_M, D_A>(abase);
    i32x4_t b1p = em3en4_read_pair_b128_base<T, 0, 1, T::T_N, T::W_N, D_B>(bbase);
    i32x4_t a2p = em3en4_read_pair_b128_base<T, 0, 2, T::T_M, T::W_M, D_A>(abase);
    i32x4_t b2p = em3en4_read_pair_b128_base<T, 0, 2, T::T_N, T::W_N, D_B>(bbase);
    i32x4_t b3p = em3en4_read_pair_b128_base<T, 0, 3, T::T_N, T::W_N, D_B>(bbase);
    i32x4_t na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p;

    em3en4_compute_pair_and_prefetch_next_packed_b128<T, 1>(
        abase, bbase, a0p, a1p, a2p, b0p, b1p, b2p, b3p,
        na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p, acc);
    em3en4_compute_pair_and_prefetch_next_packed_b128<T, 2>(
        abase, bbase, na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p,
        a0p, a1p, a2p, b0p, b1p, b2p, b3p, acc);
    em3en4_compute_pair_and_prefetch_next_packed_b128<T, 3>(
        abase, bbase, a0p, a1p, a2p, b0p, b1p, b2p, b3p,
        na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p, acc);

    s_waitcnt_lgkmcnt(3_I);
    em3en4_compute_pair_both_packed_b128<T>(
        na0p, na1p, na2p, nb0p, nb1p, nb2p, nb3p, acc);
}

template<typename T, typename Mma, typename GC, typename VC>
OPUS_D inline void store_workspace_full_tile_xpose(
    Mma& mma, GC& g_c, VC& v_c, int stride_ws,
    int wave_id_m, int wave_id_n, int lane_id)
{
    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c,
                                      wave_id_n, lane_id / mma.grpn_c);
    auto u_gc = opus::partition_layout_c<1>(mma,
                    opus::make_tuple(1_I, stride_ws), p_coord_c);
    opus::store<1>(g_c, v_c, u_gc, 0, opus::number<3>{});
}

// Initial prologue: load+store all A/B chunks for tile 0, optionally prefetch tile 1.
template<typename T, typename SA, typename SB, typename GA, typename GB>
OPUS_D inline void em3en4_run_single_lds_pgr2_segment(
    SA& s_a, SB& s_b, GA& g_a, GB& g_b,
    int loops, int tid, int lane_id, int wave_id_m, int wave_id_n,
    int stride_a, int stride_b, float4_acc* acc)
{
    static_assert(T::B_M == 96 && T::B_N == 128 && T::B_K == 128);
    static_assert(T::E_M == 3 && T::E_N == 4 && T::E_K == 8);
    static_assert(T::T_M == 2 && T::T_N == 2 && T::BLOCK_SIZE == 256);
    if (loops <= 0) return;

    using a_chunk_t = decltype(em3en4_load_gmem_chunk<T, 0, GA, T::VEC_A>(g_a, 0, tid, stride_a));
    using b_chunk_t = decltype(em3en4_load_gmem_chunk<T, 0, GB, T::VEC_B>(g_b, 0, tid, stride_b));

    auto init_a = em3en4_load_a_chunks<T>(g_a, 0, tid, stride_a);
    auto init_b = em3en4_load_b_chunks<T>(g_b, 0, tid, stride_b);
    s_waitcnt_vmcnt(0_I);
    em3en4_store_a_chunks<T>(s_a, init_a, tid);
    em3en4_store_b_chunks<T>(s_b, init_b, tid);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    em3en4_a_chunk_pack<a_chunk_t> pfa{};
    em3en4_b_chunk_pack<b_chunk_t> pfb{};
    if (loops > 1) {
        pfa = em3en4_load_a_chunks<T>(g_a, 1, tid, stride_a);
        pfb = em3en4_load_b_chunks<T>(g_b, 1, tid, stride_b);
    }

    int tile = 0;
    for (; tile + 2 < loops; ++tile) {
        em3en4_compute_smem_packed_b128_pgr2_chunks_fine<T, true>(
            s_a, s_b, wave_id_m, wave_id_n, lane_id, acc,
            pfa, pfb, g_a, g_b, tile + 2, tid, stride_a, stride_b);
    }
    if (tile + 1 < loops) {
        em3en4_compute_smem_packed_b128_pgr2_chunks_fine<T, false>(
            s_a, s_b, wave_id_m, wave_id_n, lane_id, acc,
            pfa, pfb, g_a, g_b, tile + 2, tid, stride_a, stride_b);
        ++tile;
    }
    if (tile < loops) {
        em3en4_compute_smem_packed_b128_dense_final<T>(
            s_a, s_b, wave_id_m, wave_id_n, lane_id, acc);
    }
}

} // namespace opus_em3en4_gfx942

#endif // __HIP_DEVICE_COMPILE__

template<typename Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a16w16_em3en4_lds1_pgr2_sk_kernel(opus_gemm_splitk_kargs kargs) {
#ifdef __HIP_DEVICE_COMPILE__
#if defined(__gfx942__)
    using namespace opus;
    using namespace opus_em3en4_gfx942;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = typename T::D_A;
    using D_B = typename T::D_B;
    using D_C = typename T::D_C;
    using D_ACC = typename T::D_ACC;

    static_assert(T::B_M == 96 && T::B_N == 128 && T::B_K == 128);
    static_assert(T::T_M == 2 && T::T_N == 2 && T::BLOCK_SIZE == 256);
    static_assert(T::E_M == 3 && T::E_N == 4 && T::E_K == 8);
    static_assert(std::is_same_v<D_C, D_ACC>,
                  "EM3EN4 LDS1/PGR2 splitK main kernel writes fp32 workspace");

    int wgid_full = opus::block_id_x();
    int split_id = wgid_full % kargs.split_k;
    int wgid = wgid_full / kargs.split_k;

    const int total_iters = ceil_div(kargs.k, T::B_K);
    const int iters_full = ceil_div(total_iters, kargs.split_k);
    int loops = (split_id < kargs.split_k - 1)
                    ? iters_full
                    : (total_iters - (kargs.split_k - 1) * iters_full);
    if (loops <= 0) return;
    int k_start = split_id * iters_full * T::B_K;

    const int num_tiles_n = ceil_div_constexpr(kargs.n, T::B_M);
    int row = (wgid / num_tiles_n) * T::B_N;
    int col = (wgid % num_tiles_n) * T::B_M;

    int batch_id = opus::block_id_z();
    int wave_id = __builtin_amdgcn_readfirstlane(opus::thread_id_x() / get_warp_size());
    int lane_id = opus::thread_id_x() % get_warp_size();
    int wave_id_m = wave_id / T::T_N;
    int wave_id_n = wave_id % T::T_N;

    // host kid is 128M x 96N; we swap A/B on entry so device traits stay 96 x 128.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_b)
                         + batch_id * kargs.stride_b_batch + col * kargs.stride_b + k_start,
                         ((kargs.n - col) * kargs.stride_b - k_start) * sizeof(D_A));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_a)
                         + batch_id * kargs.stride_a_batch + row * kargs.stride_a + k_start,
                         ((kargs.m - row) * kargs.stride_a - k_start) * sizeof(D_B));
    auto g_c = make_gmem(opus_splitk_ws_ptr<D_C>(kargs.ws_handle)
                         + (size_t)split_id * kargs.batch * kargs.stride_ws_batch
                         + (size_t)batch_id * kargs.stride_ws_batch
                         + (size_t)row * kargs.stride_ws
                         + (size_t)col);

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});

    constexpr int N_SUB = T::E_M * T::E_N;
    float4_acc acc[N_SUB] = {};

    const int tid = opus::thread_id_x();

    constexpr int smem_a_byte = em3en4_smem_layout<T>::a_bytes;
    constexpr int smem_b_byte = em3en4_smem_layout<T>::b_bytes;
    __shared__ char smem_a[smem_a_byte];
    __shared__ char smem_b[smem_b_byte];
    smem<D_A> s_a = make_smem<D_A>(reinterpret_cast<D_A*>(smem_a));
    smem<D_B> s_b = make_smem<D_B>(reinterpret_cast<D_B*>(smem_b));

    em3en4_run_single_lds_pgr2_segment<T>(
        s_a, s_b, g_a, g_b, loops, tid, lane_id, wave_id_m, wave_id_n,
        kargs.stride_b, kargs.stride_a, acc);

    auto v_c = em3en4_acc_to_vgpr<N_SUB>(acc);
    store_workspace_full_tile_xpose<T>(mma, g_c, v_c, kargs.stride_ws,
                                       wave_id_m, wave_id_n, lane_id);
#endif // __gfx942__
#endif // __HIP_DEVICE_COMPILE__
}
