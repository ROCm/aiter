// SPDX-License-Identifier: MIT
// LG-QSDMA-r1 Opus layouts for the FP8 quad-subtile GEMM (UNCOMPILED source; Evidence builds).
// Derivation (large-amend-r1.md §2.2):
//  * A: row-major global, reference ga/sa geometry (byte-identical to BF16 B_K=64: 128 B per row).
//    LDS chunk c = rep*8+wave (1024 B + 32 B pad), 8 rows/chunk. ra: reference make_layout_ra with E_K -> SEG=2
//    (two 16-B K segments 64 K apart); logical row for (em, wave_m, lane%16) = em*32 + wave_m*16 + lane%16.
//  * B: stays in aiter shuffle(16,16) PHYSICAL order. Per n16 band and 128-K tile the bytes are contiguous
//    (4 k32 blocks at stride 512 = 2048 B). LDS chunk = band*2+seg holds 1024 B in lane order
//    (lane = kgrp*16 + n); a fragment read is lane*16 within the chunk, i.e. n = lane%16, kg = lane/16,
//    K = seg*64 + kg*16 (== M4 preshuffled_weight_view strides (16K,512,256,16,1), first-axis-fast decode).
#pragma once
#include <opus/opus.hpp>
#include "quad_fp8_traits.hpp"
#include "quad_fp8_fieldmap.hpp"

namespace lgq {
using opus::operator""_I;

// ---- A global/LDS/fragment-read (reference structure, fp8 constants) ----
template<typename T = Q>
inline __device__ auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;                       // 8
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;     // 64
    constexpr int threads_m_per_wave = T::WAVE / threads_k;            // 8
    constexpr auto shape = opus::make_tuple(
        opus::number<ceil_div_c(T::HALF_B_M, threads_m_per_block)>{}, opus::number<T::T_M>{},
        opus::number<threads_m_per_wave>{}, opus::number<T::T_N>{},
        opus::number<threads_k>{}, opus::number<T::VEC_A>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, lane_id / threads_k, wave_id_n, lane_id % threads_k}));
}
template<typename T = Q>
inline __device__ auto make_layout_sa(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WAVE;
    constexpr auto shape = opus::make_tuple(opus::number<ceil_div_c(T::smem_m_rep, num_waves)>{},
        opus::number<T::T_M>{}, opus::number<T::T_N>{}, opus::number<T::VEC_A>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}), opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::chunk_stride>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n}));
}
// u_ra via the partition shape of the A operand: first-axis-fast lane decode (lane%16 row, lane/16 K group),
// two 16-B segments 64 K apart (SEG). Equivalent to partition_layout_a's lane factorisation, spelled with
// explicit strides so the LDS row stride includes the 32-B pad.
template<typename T = Q>
inline __device__ auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr auto shape = opus::make_tuple(opus::number<T::E_M>{}, opus::number<T::T_N>{},
        opus::number<T::T_M>{}, opus::number<T::W_M / T::T_N>{}, opus::number<T::SEG>{},
        opus::number<T::WAVE / T::W_M>{}, opus::number<T::VEC_A>{});
    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_m = lane_id % T::W_M;
    return opus::make_layout(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::chunk_stride>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{lane_m % T::T_N, wave_id_m, lane_m / T::T_N, lane_id / T::W_M}));
}

// ---- B physical (shuffled) global/LDS/fragment-read ----
// Global (element offset from half-tile band0*16K + tile*2048): chunk = rep*8+wave -> band = rep*4+wave/2,
// seg = wave%2 ; offset = band*16K + seg*1024 + lane*16 + v. Row axis stride = 16K, col axis stride 1.
template<typename T = Q>
inline __device__ auto make_layout_gb_phys(int lane_id, int wave_id, int stride_band /* 16*K */) {
    constexpr auto shape = opus::make_tuple(opus::number<2>{} /*rep*/, opus::number<4>{} /*wave/2*/,
        opus::number<2>{} /*wave%2 = seg*/, opus::number<T::WAVE>{}, opus::number<T::VEC_B>{});
    constexpr auto dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{stride_band, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id / 2, wave_id % 2, lane_id}));
}
template<typename T = Q>   // LDS write: chunk = rep*8 + wave (wave = wave_m*4 + wave_n)
inline __device__ auto make_layout_sb_phys(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WAVE;
    constexpr auto shape = opus::make_tuple(opus::number<T::smem_n_rep / num_waves>{},
        opus::number<T::T_M>{}, opus::number<T::T_N>{}, opus::number<T::VEC_B>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}), opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::chunk_stride>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n}));
}
// Fragment read: chunk = (en*4 + wave_n)*2 + seg ; in-chunk offset = lane*16 + v (lane = kg*16 + n).
template<typename T = Q>
inline __device__ auto make_layout_rb_phys(int lane_id, int wave_id_n) {
    constexpr auto shape = opus::make_tuple(opus::number<T::E_N>{}, opus::number<T::T_N>{},
        opus::number<T::SEG>{}, opus::number<T::WAVE>{}, opus::number<T::VEC_B>{});
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout(shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{opus::number<T::chunk_stride>{}, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, lane_id}));
}

// ---- tile -> (destination, field segment) table consumers ----
// Output slab address layout over [4][M][F]; offsets built from an Opus layout, not hand arithmetic.
inline __device__ auto make_layout_slab(int M) {
    return opus::make_layout<-1>(opus::make_tuple(kNumDest, M, kF),
        opus::make_tuple(static_cast<long long>(M) * kF, static_cast<long long>(kF), 1LL));
}
} // namespace lgq
