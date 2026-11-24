// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "opus_gemm_traits_a8w8_scale_gfx950.cuh"

// ============================================================================
// Layout functions for A/B matrix global/shared/register data movement
// Guarded: these are __device__ functions only needed on the device pass.
// ============================================================================

#ifdef __HIP_DEVICE_COMPILE__

template<typename T>
inline __device__ auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_M / threads_m_per_block>{},
        opus::number<T::T_N>{},
        opus::number<threads_m_per_wave>{},
        opus::number<T::T_M>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_A>{});

    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
}

template<typename T>
inline __device__ auto make_layout_sa(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_A>{});

    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        sa_block_shape,
        opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template<typename T>
inline __device__ auto make_layout_ra(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M / T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::W_M / T::T_M>{},
        opus::number<T::E_K>{},
        opus::number<T::W_M * T::W_K / opus::get_warp_size() / T::VEC_A>{},
        opus::number<opus::get_warp_size() / T::W_M>{},
        opus::number<T::VEC_A>{});

    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_m = lane_id % T::W_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m / T::T_N, lane_id_m % T::T_M, wave_id_m % T::T_N, lane_id_m / T::T_M, lane_id / T::W_M}));
}

template<typename T>
inline __device__ auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_n_per_wave = opus::get_warp_size() / threads_k;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_N / threads_n_per_block>{},
        opus::number<T::T_N>{},
        opus::number<threads_n_per_wave>{},
        opus::number<T::T_M>{},
        opus::number<threads_k>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m, lane_id % threads_k}));
}

// Producer pair for a shuffle_weight(w, layout=(16,16)) B buffer.
//
// The preshuffled buffer is a row of 16-column blocks, one per 16 columns and
// 16*stride_b bytes apart, each holding its whole K as 256-byte units of
// [n(16)][k byte(16)] -- so n is the *inner* index of a unit, the opposite of
// row-major [N, K] where k is.
//
// That inversion is why the plain make_layout_gb cannot just be re-pointed at a
// preshuffled buffer: its n puts wave_m innermost, so the 16 columns of one unit
// are split across four waves and every wave reads 16 of each 64 bytes. Measured
// at 2.20x the L1 traffic of the plain path, which cost ~6% wall time.
//
// So the thread mapping is inverted with it: n_in = lane%16 and the k chunk is
// lane/16, making a wave's issue exactly the 1024 contiguous bytes at lane*16
// inside one block's 64-k half. Each wave owns one block (nb) and takes its two
// halves as the two y steps.
//
// The tile then lands in LDS in that same preshuffled order rather than the
// plain path's byte order -- the async copy cannot permute it back, see
// make_layout_sb_preshuffle below -- so the consumer reads it with
// make_layout_rb_preshuffle. All three layouts must be changed together.
template<typename T>
inline __device__ auto make_layout_gb_preshuffle(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int loads = T::HALF_B_N / threads_n_per_block;
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    static_assert(T::VEC_B == 16,
                  "a preshuffle unit stores 16 contiguous K bytes per column");
    static_assert(T::HALF_B_N / 16 == num_waves,
                  "this mapping gives each wave exactly one 16-column block");
    static_assert(loads * opus::get_warp_size() * T::VEC_B == 16 * T::B_K,
                  "a wave's y steps must cover its block's whole K tile");

    // The wave's 16-column block. Blocks are 16*stride_b apart, so this is the
    // only term that needs the runtime row stride.
    const int nb = wave_id_n * T::T_M + wave_id_m;

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_N / 16>{},
        opus::number<loads>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{16 * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{nb, lane_id}));
}

// LDS write for the preshuffled producer.
//
// The async path is buffer_load_lds, whose LDS destination is one wave-uniform
// base per issue with the lanes landing at base + lane*VEC -- the copy cannot
// permute. So LDS cannot be made to hold the plain path's byte order while the
// global read stays coalesced; the tile is stored in its native preshuffled
// order instead, one 16-column block per B_K*16 bytes, and the consumer reads
// that order back (make_layout_rb_preshuffle).
template<typename T>
inline __device__ auto make_layout_sb_preshuffle(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr int threads_k = T::B_K / T::VEC_B;
    constexpr int loads = T::HALF_B_N / (T::BLOCK_SIZE / threads_k);

    static_assert((T::HALF_B_N / 16) * T::B_K * 16
                      <= T::smem_n_rep * (T::smem_linear_wave + T::smem_padding),
                  "preshuffled B tile must fit the existing per-half B LDS buffer");

    const int nb = wave_id_n * T::T_M + wave_id_m;

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_N / 16>{},
        opus::number<loads>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape,
            opus::tuple{T::B_K * 16, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{nb, lane_id}));
}

// Consumer read of the preshuffled LDS tile.
//
// The plain make_layout_rb resolves to n = e_n*(T_N*W_N) + wave_n*W_N + lane%W_N
// and k = rep*(W_K/2) + (lane/W_N)*VEC_B + byte, i.e. the wave's column inside a
// 16-column group is already lane%16 and its k already runs VEC_B per lane --
// exactly how the preshuffle orders a unit. So against a preshuffled tile the
// whole read collapses to a lane-linear one: the wave owns blocks wave_n +
// e_n*T_N, and each (e_n, k rep) issue is 1024 contiguous bytes at lane*16.
template<typename T>
inline __device__ auto make_layout_rb_preshuffle(int lane_id, int wave_id_n) {
    constexpr int k_halves = T::W_N * T::W_K / (opus::get_warp_size() * T::VEC_B);

    static_assert(T::W_N == 16, "a preshuffle block is 16 columns wide");

    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_N>{},
        opus::number<T::E_K>{},
        opus::number<k_halves>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape,
            opus::tuple{T::B_K * 16, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

template<typename T>
inline __device__ auto make_layout_sb(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / opus::get_warp_size();

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / num_waves>{},
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<opus::get_warp_size()>{},
        opus::number<T::VEC_B>{});

    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        sb_block_shape,
        opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template<typename T>
inline __device__ auto make_layout_rb(int lane_id, int wave_id_n) {
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},
        opus::number<T::T_M>{},
        opus::number<T::T_N>{},
        opus::number<T::W_N / T::T_M>{},
        opus::number<T::E_K>{},
        opus::number<T::W_N * T::W_K / opus::get_warp_size() / T::VEC_B>{},
        opus::number<opus::get_warp_size() / T::W_N>{},
        opus::number<T::VEC_B>{});

    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    auto lane_id_n = lane_id % T::W_N;

    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{T::smem_linear_wave + T::smem_padding, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{lane_id_n % T::T_M, wave_id_n, lane_id_n / T::T_M, lane_id / T::W_N}));
}

template<typename T>
inline __device__ auto make_layout_sfa(int lane_id, int wave_id_m, int stride_sfa) {
    constexpr auto sfa_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::W_M>{},
        opus::number<T::B_K / T::GROUP_K>{});

    constexpr auto sfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout(
        sfa_block_shape,
        opus::unfold_x_stride(sfa_block_dim, sfa_block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(sfa_block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
#endif // __HIP_DEVICE_COMPILE__
