// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// a8w8_mxfp8 pipeline: MXFP8 x MXFP8 GEMM with 1x32 UE8M0 scales on gfx950, 4 waves (2x2), one 256x256x128 tile per
// block, C bf16 [M, N].  Scales are row-major x_scale [M, K/32] / w_scale [N, K/32] e8m0 in every direction.
// Traits A_T / B_T pick the operand storage (N = row-major A [M, K] or B [N, K]; T = A [K, M] or B [K, N]):
//   NT (0, 0): A [M, K],  B [N, K];      NN (0, 1): A [M, K],  B [K, N];
//   TT (1, 0): A [K, M],  B [N, K];      TN (1, 1): A [K, M],  B [K, N].
// B_SHUF (NT only): B is shuffle_weight([N, K], (16, 16)) -- aiter gemm_a8w8_mxfp8_bpreshuffle.  A plain B is copied
// and read as the mirror of A (wave_m <-> wave_n): B [N, K] as A [M, K], B [K, N] as A [K, M].
// Tiles B_M / B_N in {256, 192, 128}, MODE persistent / one tile / split-K and BOUNDARY in every direction.
// A transposed operand is copied into LDS k-major and read back with ds_read_tr8.  A / B fragments are pinned with
// [[clang::amdgpu_pin_agpr/vgpr]], so only a pin toolchain compiles it: it ships as pre-compiled .co images
// (gen_co/build_co.py --arch gfx950).
#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_traits_a8w8_mxfp8_gfx950.cuh"

using opus::operator""_I;

// ===========================================================================
// A: global -> LDS -> VGPR
// ===========================================================================
template<class T>
__device__ inline auto make_layout_ga(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K_BYTES / T::VEC_A;                 // 8
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;     // 32
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;       // 8
    static_assert(threads_k == T::B_K / T::VEC_A_ELEMS);               // 128 fp8 / 16 fp8 per lane

    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_M / threads_m_per_block>{},             // 4  y issue
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<threads_m_per_wave>{},                            // 8  p lane / 8
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<threads_k>{},                                     // 8  p lane % 8
        opus::number<T::VEC_A>{});                                     // 16 y bytes
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        // swizzle: the lane writing LDS slot (lane % 8) of row x = lane / 8 (chunk parity wave_m) fetches K slot
        // (lane % 8) ^ x ^ (wave_m << 2)
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,
                             (lane_id % threads_k) ^ (((lane_id / threads_k) ^ (wave_id_m << 2)) & 7)}));
}

template<class T>
__device__ inline auto make_layout_sa(int wave_id_m, int wave_id_n) {
    static_assert(T::smem_sub_a == T::WARP_SIZE / (T::B_K_BYTES / T::VEC_A), "one wave issue = one chunk");

    constexpr auto sa_block_shape = opus::make_tuple(
        opus::number<T::smem_m_rep / T::NUM_WAVES>{},                  // IA y issue (4 for 256 rows)
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::VEC_A>{});                                     // 16 y bytes
    constexpr auto sa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    constexpr int IA = T::a_buffer_load_insts;

    return opus::make_layout(
        sa_block_shape,
        // chunk = (wave_n * 2 + wave_m) * IA + issue: a wave's IA issues are adjacent (MUBUF imm offsets)
        opus::make_tuple(opus::number<T::smem_linear_wave_a>{}, opus::number<2 * IA * T::smem_linear_wave_a>{}, opus::number<IA * T::smem_linear_wave_a>{}, 1_I),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline auto make_layout_ra(int lane_id, int wave_id_m) {
    // MFMA operand of lane (row lane % 16, group g = lane / 16): E_C = 2 chunks of 16 bytes at
    // bytes 64 c + 16 g of the 128-byte row; chunk c -> + c * B_K_BYTES / E_C (coordinate fixed to 0).
    static_assert(T::smem_sub_a * T::T_M == T::W_M && T::T_M == 2);
    static_assert(T::WARP_SIZE / T::W_M * T::E_C * T::VEC_A == T::B_K_BYTES);
    constexpr auto shape = opus::make_tuple(
        opus::number<T::E_M>{},                                        // y repeat
        opus::number<T::T_M>{},                                        // p wave
        opus::number<T::T_M>{},                                        // p r % 2
        opus::number<T::W_M / T::T_M>{},                               // p r / 2
        opus::number<T::E_C>{},                                        // p chunk (64 K bytes), fixed to 0
        opus::number<T::WARP_SIZE / T::W_M>{},                         // p g = lane / 16 (16 K bytes)
        opus::number<T::VEC_A>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_r = lane_id % T::W_M;
    constexpr int IA = T::a_buffer_load_insts;
    return opus::make_layout<T::VEC_A>(
        shape,
        // chunk (e, wave, r % 2) = (wave * 2 + r % 2) * IA + e; inside: r / 2 -> 128 B, c -> 64 B, slot -> 16 B
        opus::make_tuple(opus::number<T::smem_linear_wave_a>{}, opus::number<2 * IA * T::smem_linear_wave_a>{}, opus::number<IA * T::smem_linear_wave_a>{},
                         opus::number<T::B_K_BYTES>{}, opus::number<T::B_K_BYTES / T::E_C>{}, opus::number<T::VEC_A>{}, 1_I),
        // swizzle (see make_layout_ga): K slot g of row x = r / 2, chunk parity r % 2, sits at slot g ^ x ^ ((r % 2) << 2);
        // chunk c = 1 (K slot 4 + g) is at slot (that ^ 4): load_a adds a_c1_delta.
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, lane_r % T::T_M, lane_r / T::T_M, 0_I,
                             (lane_id / T::W_M) ^ (((lane_r / T::T_M) ^ ((lane_r % T::T_M) << 2)) & 7)}));
}

template<class T>
__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    if constexpr (!T::B_SHUF) {
    // Plain B [N, K]: make_layout_ga with wave_m <-> wave_n.
    constexpr int threads_k = T::B_K_BYTES / T::VEC_B;                 // 8
    constexpr int threads_n_per_block = T::BLOCK_SIZE / threads_k;     // 32
    constexpr int threads_n_per_wave = T::WARP_SIZE / threads_k;       // 8
    static_assert(threads_k == T::B_K / T::VEC_B_ELEMS);               // 128 fp8 / 16 fp8 per lane

    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::HALF_B_N / threads_n_per_block>{},             // 4  y issue
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<threads_n_per_wave>{},                            // 8  p lane / 8
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<threads_k>{},                                     // 8  p lane % 8
        opus::number<T::VEC_B>{});                                     // 16 y bytes
    constexpr auto gb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{stride_b, 1_I}),
        // swizzle: the lane writing LDS slot (lane % 8) of row x = lane / 8 (chunk parity wave_n) fetches K slot
        // (lane % 8) ^ x ^ (wave_n << 2)
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_m, lane_id / threads_k, wave_id_n,
                             (lane_id % threads_k) ^ (((lane_id / threads_k) ^ (wave_id_n << 2)) & 7)}));
    } else {
    // B is the (16, 16) shuffle_weight of [N, K]: the 16 bytes (n, k .. k + 15), k % 16 == 0, sit at
    //   (n / 16) * 16 K + (k / 32) * 512 + ((k % 32) / 16) * 256 + (n % 16) * 16,
    // so 16 rows x one K tile (128) is one contiguous 2 KB block ("group").  One wave issue copies
    // 1 KB of it verbatim (lane l -> + 16 l): group G = issue * 2 + wave_m (rows 16 G .. 16 G + 15
    // of the half tile), K half = wave_n (k / 64).  LDS chunk 4 issue + 2 wave_m + wave_n (make_layout_sb)
    // = 2 G + K half, i.e. the LDS image of a group is its global image (+ padding between halves).
    static_assert(T::T_M == 2 && T::T_N == 2 && T::B_K_BYTES * 16 == 2 * T::smem_linear_wave_b);
    constexpr int IB = T::b_buffer_load_insts;
    static_assert(T::HALF_B_N / 16 == IB * T::T_M, "IB issues x 2 wave_m groups per half");
    const int s16 = 16 * stride_b;                                     // one 16-row block of shuffled B
    constexpr auto shape = opus::make_tuple(
        opus::number<IB>{},                                            // y issue        (G = issue * 2 + wave_m)
        opus::number<T::T_M>{},                                        // p wave_m
        opus::number<T::T_N>{},                                        // p wave_n = K half (1 KB)
        opus::number<T::WARP_SIZE>{},                                  // p lane         (16 bytes each)
        opus::number<T::VEC_B>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        shape,
        opus::make_tuple(2 * s16, s16, opus::number<T::smem_linear_wave_b>{}, opus::number<T::VEC_B>{}, 1_I),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
    }
}

template<class T>
__device__ inline auto make_layout_sb(int wave_id_m, int wave_id_n) {
    static_assert(T::smem_sub_b == T::WARP_SIZE / (T::B_K_BYTES / T::VEC_B), "one wave issue = one chunk");

    constexpr auto sb_block_shape = opus::make_tuple(
        opus::number<T::smem_n_rep / T::NUM_WAVES>{},                  // IB y issue (4 for 256 rows)
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::VEC_B>{});                                     // 16 y bytes
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    constexpr int IB = T::b_buffer_load_insts;

    return opus::make_layout(
        sb_block_shape,
        // chunk = (wave_m * 2 + wave_n) * IB + issue
        opus::make_tuple(opus::number<T::smem_linear_wave_b>{}, opus::number<2 * IB * T::smem_linear_wave_b>{}, opus::number<IB * T::smem_linear_wave_b>{}, 1_I),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    constexpr int IB = T::b_buffer_load_insts;
    if constexpr (!T::B_SHUF) {
    // Plain B: make_layout_ra with wave_n (chunk (e, wave, r % 2) = (wave * 2 + r % 2) * IB + e, the same swizzle;
    // chunk c = 1 is at + a_c1_delta in load_b).
    static_assert(T::smem_sub_b * T::T_N == T::W_N && T::T_N == 2);
    static_assert(T::WARP_SIZE / T::W_N * T::E_C * T::VEC_B == T::B_K_BYTES);
    constexpr auto shape = opus::make_tuple(
        opus::number<T::E_N>{},                                        // y repeat
        opus::number<T::T_N>{},                                        // p wave
        opus::number<T::T_N>{},                                        // p r % 2
        opus::number<T::W_N / T::T_N>{},                               // p r / 2
        opus::number<T::E_C>{},                                        // p chunk (64 K bytes), fixed to 0
        opus::number<T::WARP_SIZE / T::W_N>{},                         // p g = lane / 16 (16 K bytes)
        opus::number<T::VEC_B>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    const int lane_r = lane_id % T::W_N;
    return opus::make_layout<T::VEC_B>(
        shape,
        opus::make_tuple(opus::number<T::smem_linear_wave_b>{}, opus::number<2 * IB * T::smem_linear_wave_b>{}, opus::number<IB * T::smem_linear_wave_b>{},
                         opus::number<T::B_K_BYTES>{}, opus::number<T::B_K_BYTES / T::E_C>{}, opus::number<T::VEC_B>{}, 1_I),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, lane_r % T::T_N, lane_r / T::T_N, 0_I,
                             (lane_id / T::W_N) ^ (((lane_r / T::T_N) ^ ((lane_r % T::T_N) << 2)) & 7)}));
    } else {
    // Shuffled B in LDS: chunk (= 1 KB + padding) 4 e + 2 wave_n + c; lane l reads + 16 l.
    // Chunk c -> + c * (smem_linear_wave_b + smem_padding) in load_b (coordinate fixed to 0).
    static_assert(T::T_N == 2 && T::W_N == 16 && T::WARP_SIZE * T::VEC_B == T::smem_linear_wave_b);
    constexpr int chunk = T::smem_linear_wave_b + T::smem_padding;
    constexpr auto shape = opus::make_tuple(
        opus::number<T::E_N>{},                                        // y repeat e
        opus::number<T::T_N>{},                                        // p wave_n
        opus::number<T::E_C>{},                                        // p chunk, fixed to 0
        opus::number<T::WARP_SIZE>{},                                  // p lane
        opus::number<T::VEC_B>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        shape,
        opus::make_tuple(opus::number<chunk>{}, opus::number<2 * IB * chunk>{}, opus::number<IB * chunk>{}, opus::number<T::VEC_B>{}, 1_I),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_n, 0_I, lane_id}));
    }
}

template<class T>
__device__ inline auto make_layout_gsfa(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto gsfa_block_shape = opus::make_tuple(
        opus::number<2>{},                                             // 2  y issue (128 rows)
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::WARP_SIZE / 2>{},                              // 32 p lane / 2 = row
        opus::number<2>{},                                             // 2  p lane % 2 = 16-byte half
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto gsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        gsfa_block_shape,
        opus::unfold_x_stride(gsfa_block_dim, gsfa_block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(gsfa_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / 2, lane_id % 2}));
}

template<class T>
__device__ inline auto make_layout_ssfa(int wave_id_m, int wave_id_n) {
    constexpr int wave_bytes = T::WARP_SIZE * T::VEC_GLOBAL_SCALE;    // 1024
    constexpr auto ssfa_block_shape = opus::make_tuple(
        opus::number<2>{},                                        // 1  y M repeat
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto ssfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        ssfa_block_shape,
        opus::unfold_x_stride(ssfa_block_dim, ssfa_block_shape, opus::tuple{opus::number<wave_bytes>{}, 1_I}),
        opus::unfold_p_coord(ssfa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template<class T>
__device__ inline auto make_layout_gsfb(int lane_id, int wave_id_m, int wave_id_n, int stride_sfb) {
    constexpr auto gsfb_block_shape = opus::make_tuple(
        opus::number<2>{},                                             // 2  y issue (128 rows)
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::WARP_SIZE / 2>{},                              // 32 p lane / 2 = row
        opus::number<2>{},                                             // 2  p lane % 2 = 16-byte half
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto gsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        gsfb_block_shape,
        opus::unfold_x_stride(gsfb_block_dim, gsfb_block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(gsfb_block_dim, opus::tuple{wave_id_m, wave_id_n, lane_id / 2, lane_id % 2}));
}

template<class T>
__device__ inline auto make_layout_ssfb(int wave_id_m, int wave_id_n) {
    constexpr int wave_bytes = T::WARP_SIZE * T::VEC_GLOBAL_SCALE;    // 1024
    constexpr auto ssfb_block_shape = opus::make_tuple(
        opus::number<2>{},                                        // 1  y N repeat
        opus::number<T::T_M>{},                                        // 2  p wave_m
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::VEC_GLOBAL_SCALE>{});                          // 16 y bytes
    constexpr auto ssfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        ssfb_block_shape,
        opus::unfold_x_stride(ssfb_block_dim, ssfb_block_shape, opus::tuple{opus::number<wave_bytes>{}, 1_I}),
        opus::unfold_p_coord(ssfb_block_dim, opus::tuple{wave_id_m, wave_id_n}));
}

template<class T>
__device__ inline auto make_layout_rsfa(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::B_M / T::HALF_B_M>{},                          // 2  y half h   (128 rows)
        opus::number<T::WARP_SIZE / T::W_M>{},                         // 4  p g as e   (32 rows)
        opus::number<T::T_M>{},                                        // 2  p wave_m   (16 rows)
        opus::number<T::W_M>{},                                        // 16 p r        (1 row)
        opus::number<T::VEC_LDS_SCALE>{});                             // 8  y bytes
    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_LDS_SCALE>(
        rsfa_block_shape,
        // rows h * HALF_B_M + 32 g + 16 wave + r (g < E: repeat g of half h; the other lane groups read unused rows)
        opus::make_tuple(opus::number<T::HALF_B_M * T::SCALE_PANEL_GROUPS>{}, opus::number<32 * T::SCALE_PANEL_GROUPS>{},
                         opus::number<16 * T::SCALE_PANEL_GROUPS>{}, opus::number<T::SCALE_PANEL_GROUPS>{}, 1_I),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{lane_id / T::W_M, wave_id_m, lane_id % T::W_M}));
}

template<class T>
__device__ inline auto make_layout_rsfa16(int lane_id, int wave_id_m) {
    constexpr auto rsfa_block_shape = opus::make_tuple(
        opus::number<T::B_M / T::HALF_B_M>{},                          // 2  y half h   (128 rows)
        opus::number<T::WARP_SIZE / T::W_M>{},                         // 4  p g as e   (32 rows)
        opus::number<T::T_M>{},                                        // 2  p wave_m   (16 rows)
        opus::number<T::W_M>{},                                        // 16 p r        (1 row)
        opus::number<T::VEC_GLOBAL_SCALE>{});                             // 8  y bytes
    constexpr auto rsfa_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        rsfa_block_shape,
        // rows h * HALF_B_M + 32 g + 16 wave + r (g < E: repeat g of half h; the other lane groups read unused rows)
        opus::make_tuple(opus::number<T::HALF_B_M * T::SCALE_PANEL_GROUPS>{}, opus::number<32 * T::SCALE_PANEL_GROUPS>{},
                         opus::number<16 * T::SCALE_PANEL_GROUPS>{}, opus::number<T::SCALE_PANEL_GROUPS>{}, 1_I),
        opus::unfold_p_coord(rsfa_block_dim, opus::tuple{lane_id / T::W_M, wave_id_m, lane_id % T::W_M}));
}

template<class T>
__device__ inline auto make_layout_rsfb(int lane_id, int wave_id_n) {
    constexpr auto rsfb_block_shape = opus::make_tuple(
        opus::number<T::B_N / T::HALF_B_N>{},                          // 2  y half h   (128 rows)
        opus::number<T::WARP_SIZE / T::W_N>{},                         // 4  p g as e   (32 rows)
        opus::number<T::T_N>{},                                        // 2  p wave_n   (16 rows)
        opus::number<T::W_N>{},                                        // 16 p r        (1 row)
        opus::number<T::VEC_LDS_SCALE>{});                             // 8  y bytes
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_LDS_SCALE>(
        rsfb_block_shape,
        // rows h * HALF_B_N + 32 g + 16 wave + r (g < E: repeat g of half h; the other lane groups read unused rows)
        opus::make_tuple(opus::number<T::HALF_B_N * T::SCALE_PANEL_GROUPS>{}, opus::number<32 * T::SCALE_PANEL_GROUPS>{},
                         opus::number<16 * T::SCALE_PANEL_GROUPS>{}, opus::number<T::SCALE_PANEL_GROUPS>{}, 1_I),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{lane_id / T::W_N, wave_id_n, lane_id % T::W_N}));
}

template<class T>
__device__ inline auto make_layout_rsfb16(int lane_id, int wave_id_n) {
    constexpr auto rsfb_block_shape = opus::make_tuple(
        opus::number<T::B_N / T::HALF_B_N>{},                          // 2  y half h   (128 rows)
        opus::number<T::WARP_SIZE / T::W_N>{},                         // 4  p g as e   (32 rows)
        opus::number<T::T_N>{},                                        // 2  p wave_n   (16 rows)
        opus::number<T::W_N>{},                                        // 16 p r        (1 row)
        opus::number<T::VEC_GLOBAL_SCALE>{});                             // 8  y bytes
    constexpr auto rsfb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_GLOBAL_SCALE>(
        rsfb_block_shape,
        // rows h * HALF_B_N + 32 g + 16 wave + r (g < E: repeat g of half h; the other lane groups read unused rows)
        opus::make_tuple(opus::number<T::HALF_B_N * T::SCALE_PANEL_GROUPS>{}, opus::number<32 * T::SCALE_PANEL_GROUPS>{},
                         opus::number<16 * T::SCALE_PANEL_GROUPS>{}, opus::number<T::SCALE_PANEL_GROUPS>{}, 1_I),
        opus::unfold_p_coord(rsfb_block_dim, opus::tuple{lane_id / T::W_N, wave_id_n, lane_id % T::W_N}));
}

__device__ inline unsigned int transpose_scale_dword_4x4(unsigned int packed, int scale_call) {
    const auto rows01_or_23 = __builtin_amdgcn_permlane16_swap(packed, packed, false, true);
    const unsigned int select_q01 = 0x06020400u + (scale_call & 1) * 0x01010101u;
    const unsigned int q01_or_q23 = __builtin_amdgcn_perm(rows01_or_23[1], rows01_or_23[0], select_q01);
    const auto rows0123 = __builtin_amdgcn_permlane32_swap(q01_or_q23, q01_or_q23, false, true);
    const unsigned int select_q = 0x05040100u + (scale_call >> 1) * 0x02020202u;
    return __builtin_amdgcn_perm(rows0123[1], rows0123[0], select_q);
}

// LDS-DMA of all issues of a layout with one m0: the issue's LDS offset goes into the MUBUF immediate
// (added to the global address too, so it is subtracted from the per-issue voffset).  From fp4 candidate_m5.
template<int VEC, class G, class LG, class LS>
__device__ inline void async_load_imm(G& g, void* smem_base, const LG& u_g, const LS& u_s, int s_os) {
    const auto gofs = opus::layout_to_offsets<VEC>(u_g);
    const auto sofs = opus::layout_to_offsets<VEC>(u_s);
    constexpr auto imm = opus::layout_imm_offsets_v<opus::remove_cvref_t<LS>, VEC>;
    constexpr int n = static_cast<int>(sizeof(imm) / sizeof(imm[0]));
    char* const base = reinterpret_cast<char*>(smem_base) + sofs[0];
    opus::static_for<n>([&](auto i) {
        constexpr int io = imm[decltype(i)::value] - imm[0];
        static_assert(io >= 0 && io <= 4095, "LDS issue stride must fit the 12-bit MUBUF offset");
        g.template async_load<VEC>(base, gofs[decltype(i)::value] - io, s_os, opus::number<io>{});
    });
}
// sched_group_barrier masks (LLVM AMDGPU)
constexpr int SCHED_VALU = 0x002;
constexpr int SCHED_MFMA = 0x008;
constexpr int SCHED_VMEM_READ = 0x820;   // 0x020 VMEM read | 0x800 LDS DMA (newer LLVM classifies buffer_load ... lds as LDS DMA)
constexpr int SCHED_DS_READ = 0x100;

template<int REPS, int MFMAS, int OTHER_MASK, int OTHERS, int GROUP = 0>
__device__ inline void sched_pairs() {
    opus::static_for<REPS>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, MFMAS, GROUP);
        __builtin_amdgcn_sched_group_barrier(OTHER_MASK, OTHERS, GROUP);
    });
}

// One MFMA, then DS reads and VALU.
template<int DS, int VALU>
__device__ inline void sched_m() {
    __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
    if constexpr (DS > 0) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, DS, 0);
    if constexpr (VALU > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, VALU, 0);
}
// 16 MFMAs, DS reads front-loaded (at most CAP per MFMA), VALU per MFMA, one VMEM every VM_EVERY MFMAs.
template<int DS_TOTAL, int CAP, int VALU, int VM_EVERY>
__device__ inline void sched_gen16() {
    opus::static_for<16>([&](auto j_) {
        constexpr int j = decltype(j_)::value;
        constexpr int left = DS_TOTAL - CAP * j;
        constexpr int ds = left <= 0 ? 0 : (left > CAP ? CAP : left);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (ds > 0) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, ds, 0);
        if constexpr (VM_EVERY > 0) { if constexpr (j % VM_EVERY == 0) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0); }
        if constexpr (VALU > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, VALU, 0);
    });
}
// Tiles other than 256 x 256 (NT): Q = E_M * E_N MFMAs per quadrant with DS / VMEM / VALU spread evenly over them.
template<int Q, int DS, int VMEM, int VALU>
__device__ inline void sched_spread() {
    opus::static_for<Q>([&](auto j_) {
        constexpr int j = decltype(j_)::value;
        constexpr int ds = DS * (j + 1) / Q - DS * j / Q;
        constexpr int vm = VMEM * (j + 1) / Q - VMEM * j / Q;
        constexpr int va = VALU * (j + 1) / Q - VALU * j / Q;
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (ds > 0) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, ds, 0);
        if constexpr (vm > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, vm, 0);
        if constexpr (va > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, va, 0);
    });
}
// Per-quadrant instruction counts of the generic schedule (per wave): A / B ds_read_b128 per half tile (2 chunks per
// operand), A / B LDS-DMA issues per half tile, scale transposes per half.
template<class T>
struct opus_mxfp8_counts {
    static constexpr bool HAND = T::B_M == 256 && T::B_N == 256;   // hand-tuned schedules (NT and NN / TT / TN)
    static constexpr bool HAND_TRANS = HAND && (T::A_T || T::B_T);
    static constexpr int Q = T::E_M * T::E_N;
    static constexpr int RA = (T::A_T ? 4 : 2) * T::E_M, RB = (T::B_T ? 4 : 2) * T::E_N;   // a TR8 operand: 4 DS reads
    static constexpr int QA = T::a_copy_insts, QB = T::b_copy_insts;
    static constexpr int SV = 8;
};
// sched_gen16 <DS_TOTAL, CAP> of the transposed directions (a ds_read_tr8 operand is 4 DS reads instead of 2) for
// C11, C00, the prologue and the main-loop / last-tile 16-MFMA groups.
template<bool A_T, bool B_T> struct opus_mxfp8_sched;
template<> struct opus_mxfp8_sched<false, true> {   // NN
    static constexpr int C11_DS = 8, C11_CAP = 1, C00_DS = 24, C00_CAP = 2, PRO_DS = 24, PRO_CAP = 2, MAIN_DS = 16, MAIN_CAP = 2;
};
template<> struct opus_mxfp8_sched<true, false> {   // TT
    static constexpr int C11_DS = 16, C11_CAP = 2, C00_DS = 24, C00_CAP = 2, PRO_DS = 24, PRO_CAP = 2, MAIN_DS = 8, MAIN_CAP = 1;
};
template<> struct opus_mxfp8_sched<true, true> {    // TN
    static constexpr int C11_DS = 16, C11_CAP = 2, C00_DS = 32, C00_CAP = 3, PRO_DS = 32, PRO_CAP = 2, MAIN_DS = 16, MAIN_CAP = 1;
};
// C11 (16 fp8 MFMAs): 8 DS (A0 of t+1: 4 repeats x 2 chunks) + scale transposes of half 0.
template<class T>
__device__ inline void sched_c11() {
    if constexpr (opus_mxfp8_counts<T>::HAND_TRANS) {
        using S = opus_mxfp8_sched<T::A_T, T::B_T>;
        sched_gen16<S::C11_DS, S::C11_CAP, 1, 0>();
    } else if constexpr (!opus_mxfp8_counts<T>::HAND) {
        using C = opus_mxfp8_counts<T>;
        sched_spread<C::Q, C::RA, 0, C::SV>();
    } else {
        opus::static_for<8>([&](auto) { sched_m<1, 1>(); sched_m<0, 1>(); });
    }
}
// C00 (16 fp8 MFMAs): 16 DS (A1 / B1 of t+1), one per MFMA, + scale transposes of half 1.
template<class T>
__device__ inline void sched_c00() {   // DS reads front-loaded: 2 per MFMA in the first 8, so they land before B_a
    if constexpr (opus_mxfp8_counts<T>::HAND_TRANS) {
        using S = opus_mxfp8_sched<T::A_T, T::B_T>;
        sched_gen16<S::C00_DS, S::C00_CAP, 1, 0>();
    } else if constexpr (!opus_mxfp8_counts<T>::HAND) {
        using C = opus_mxfp8_counts<T>;
        sched_spread<C::Q, C::RA + C::RB, 0, C::SV>();
    } else {
        opus::static_for<8>([&](auto) { sched_m<2, 1>(); });
        opus::static_for<8>([&](auto) { sched_m<0, 1>(); });
    }
}

template<int REPS, int DS, int VMEM, int VALU>
__device__ inline void sched_x4() {
    static_assert(DS <= 2 && VMEM <= 1);
    // fp8: each group is 2 MFMAs (64 cycles, as 4 fp4 MFMAs) carrying the fp4 group's DS / VMEM.
    opus::static_for<REPS>([&](auto) {
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (DS >= 1) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        if constexpr (VMEM >= 1) __builtin_amdgcn_sched_group_barrier(SCHED_VMEM_READ, 1, 0);
        __builtin_amdgcn_sched_group_barrier(SCHED_MFMA, 1, 0);
        if constexpr (DS >= 2) __builtin_amdgcn_sched_group_barrier(SCHED_DS_READ, 1, 0);
        if constexpr (VALU > 0) __builtin_amdgcn_sched_group_barrier(SCHED_VALU, VALU, 0);
    });
}

// The prologue / main-loop / last-tile 16-MFMA groups (VMEM: one every 2 MFMAs in the transposed directions).
template<class T>
__device__ inline void sched_pro() {
    if constexpr (opus_mxfp8_counts<T>::HAND_TRANS) {
        using S = opus_mxfp8_sched<T::A_T, T::B_T>;
        sched_gen16<S::PRO_DS, S::PRO_CAP, 0, 0>();
    } else if constexpr (!opus_mxfp8_counts<T>::HAND) {
        using C = opus_mxfp8_counts<T>;
        sched_spread<C::Q, C::RA + C::RB, 0, 0>();
    } else {
        sched_x4<8, 2, 0, 0>();
    }
}
template<class T>
__device__ inline void sched_main() {
    if constexpr (opus_mxfp8_counts<T>::HAND_TRANS) {
        using S = opus_mxfp8_sched<T::A_T, T::B_T>;
        sched_gen16<S::MAIN_DS, S::MAIN_CAP, 0, 2>();
    } else if constexpr (!opus_mxfp8_counts<T>::HAND) {
        using C = opus_mxfp8_counts<T>;
        sched_spread<C::Q, C::RB + 2, C::QA + C::QB, 0>();
    } else {
        sched_x4<4, 2, 1, 0>();
        sched_x4<4, 1, 1, 0>();
    }
}
template<class T>
__device__ inline void sched_last() {
    if constexpr (opus_mxfp8_counts<T>::HAND_TRANS) {
        using S = opus_mxfp8_sched<T::A_T, T::B_T>;
        sched_gen16<S::MAIN_DS, S::MAIN_CAP, 0, 0>();
    } else if constexpr (!opus_mxfp8_counts<T>::HAND) {
        using C = opus_mxfp8_counts<T>;
        sched_spread<C::Q, C::RB + 2, 0, 0>();
    } else {
        sched_x4<4, 2, 0, 0>();
        sched_x4<4, 1, 0, 0>();
    }
}
// C10 (copies of tile t + 2) and the copy-free quadrants of the last tiles.
template<class T>
__device__ inline void sched_c10() {
    if constexpr (!opus_mxfp8_counts<T>::HAND) {
        using C = opus_mxfp8_counts<T>;
        sched_spread<C::Q, 0, C::QA + C::QB, 0>();
    } else {
        sched_x4<8, 0, 1, 0>();
    }
}
template<class T>
__device__ inline void sched_plain() {
    if constexpr (!opus_mxfp8_counts<T>::HAND) {
        sched_spread<opus_mxfp8_counts<T>::Q, 0, 0, 0>();
    } else {
        sched_x4<8, 0, 0, 0>();
    }
}

// fp8 MFMA operand per lane: 32 fp8 = 8 registers, loaded as E_C = 2 chunks of 16 bytes and
// pinned as ONE 8-register value (pinning the chunks and concatenating them in the MFMA call
// leaves the concatenation unpinned: the compiler then copies every operand AGPR -> VGPR).
using chunk16_t = opus::vector_t<opus::i32_t, 4>;
using op8_t = opus::vector_t<opus::i32_t, 8>;

template<class T, int M_REPEAT, int N_REPEAT, int C_REG, class MMA, class CF>
__device__ inline void mma_scale_one(
    MMA& mma,
    const op8_t& a8,
    const op8_t& b8,
    CF& c,
    unsigned int v_sfa,
    unsigned int v_sfb) {
    (void)mma;
    using base_mma = typename opus::remove_cvref_t<MMA>::MMA;
    using i32x8_t = opus::vector_t<opus::i32_t, 8>;

    if constexpr (C_REG >= 0) {
        [[clang::amdgpu_pin_agpr(C_REG)]] c = base_mma{}(a8, b8, c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    } else if constexpr (C_REG <= -2) {
        [[clang::amdgpu_pin_vgpr(-C_REG - 2)]] c = base_mma{}(a8, b8, c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    } else {
        c = base_mma{}(a8, b8, c, static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<M_REPEAT>{}, opus::number<N_REPEAT>{});
    }
}

template<class T, int HM, int HN, int M, int N>
constexpr int c_reg() {
    constexpr int frag = (M * T::E_N + N) * 4;
    if constexpr (HN == 0) return 128 + HM * 64 + frag;           // AGPR
    else return -(128 + HM * 64 + frag) - 2;                        // VGPR
}

// ===========================================================================
// Kernel
// ===========================================================================
// Kernel body as a device function: it backs the template __global__ below and the extern "C" entry points of the
// pre-compiled .co images (gen_co/build_co.py --arch gfx950).
template<class Traits>
__device__ __forceinline__ void gemm_a8w8_mxfp8_body(opus_gemm_mxfp8_kargs_gfx950 kargs) {
#if defined(__gfx950__)
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    constexpr bool TRANS = T::A_T || T::B_T;   // NN / TT / TN
    using D_A = opus::fp8_t;    // math types of the MMA (OCP e4m3 on gfx950)
    using D_B = opus::fp8_t;
    using D_ACC = opus::fp32_t;
    using D_C = opus::bf16_t;
    using D_AB = opus::u8_t;
    using D_SF = opus::e8m0_t;  // UE8M0 scale, 1 byte (load<16> = 16 bytes)

    const int num_tiles_n = opus_mxfp8_ceil_div(kargs.n, T::B_N);
    // MODE 0 (persistent): block b runs tiles b, b + grid_blocks, ... of the batch-major tile list.  Otherwise one tile
    // (block_id_x, batch block_id_z): the loop runs once and folds away.
    const int tiles_per_batch = T::PERSIST || T::SPLITK ? opus_mxfp8_ceil_div(kargs.m, T::B_M) * num_tiles_n : 1;
    // MODE 2 (split-K): block b computes K segment b % split_k of tile b / split_k (batch-major tile list).
    const int split_k = T::SPLITK ? kargs.split_k : 1;
    const int tile_end = T::PERSIST ? tiles_per_batch * kargs.batch : 1;
    auto run_tile = [&](const int tile_id) __attribute__((always_inline)) {
    constexpr bool TILE_LIST = T::PERSIST || T::SPLITK;
    const int tile_in_batch = TILE_LIST ? tile_id % tiles_per_batch : static_cast<int>(block_id_x());
    const int block_m = tile_in_batch / num_tiles_n;
    const int block_n = tile_in_batch % num_tiles_n;
    const int loops = (T::SPLITK ? kargs.k / split_k : kargs.k) / T::B_K;
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = TILE_LIST ? tile_id / tiles_per_batch : static_cast<int>(block_id_z());
    // Split-K: this block's K segment starts at K tile kt0; the A / B / scale bases below are moved there.
    const int k_split = T::SPLITK ? static_cast<int>(block_id_x()) % split_k : 0;
    const int kt0 = k_split * loops;
    // A_T / B_T: K tile kt0 starts at k row kt0 * B_K.
    const int a_k0 = !T::SPLITK || T::A_T ? 0 : kt0 * T::B_K_BYTES;   // A_T / B_T: kt0 goes into the K tile base (below)
    const int b_k0 = !T::SPLITK || T::B_T ? 0 : kt0 * T::B_K_BYTES * (T::B_SHUF ? 16 : 1);
    const int sf_k0 = T::SPLITK ? kt0 * T::NUM_KGROUPS : 0;
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    // Persistent: lane id from mbcnt over an opaque all-ones mask, so the lane-dependent offsets are recomputed per tile
    // instead of being hoisted out of the tile loop and kept live (they spill).
    unsigned int lane_mask = ~0u;
    if constexpr (T::PERSIST) asm volatile("" : "+s"(lane_mask));
    const int lane_id = T::PERSIST ? static_cast<int>(__builtin_amdgcn_mbcnt_hi(lane_mask, __builtin_amdgcn_mbcnt_lo(lane_mask, 0u)))
                                    : static_cast<int>(thread_id_x() % T::WARP_SIZE);
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;

    // A_T / B_T: A / B is stored [K, M] / [K, N] (stride_a = M / stride_b = N); the tile starts at column row / col.
    // BOUNDARY: rows of A / B / C past the matrix are outside the buffer range (loads return 0, stores are dropped); C
    // columns past N are masked per 16-byte store.  A_T / B_T: the columns past M / N of a k row read the next k row (only
    // C rows / columns past M / N use them, and those are not stored); the range ends with the matrix.
    const int rows_in = T::BOUNDARY ? kargs.m - row : T::B_M, cols_in = T::BOUNDARY ? kargs.n - col : T::B_N;
    // A / B are based in 64 bits so that the 32-bit buffer offsets cover one tile whatever the size of the operand (as C
    // below).  Row-major A [M, K] / B [N, K] (shuffled B too): the base is the tile's first row and the offsets span its
    // B_M / B_N rows of K; the range is what is left of the operand, saturated to 32 bits (only a last, ragged tile ends
    // inside it).  A_T / B_T (stored [K, M] / [K, N]): a tile's columns are spread over all K rows, so the base is moved
    // to each K tile (a_k_base / b_k_base below) and the offsets span its B_K rows of M / N.
    auto range_u32 = [](long bytes) { return static_cast<unsigned int>(bytes < 0 ? 0 : bytes > 0xffffffffl ? 0xffffffffl : bytes); };
    const D_AB* const a_base = reinterpret_cast<const D_AB*>(kargs.ptr_a) + static_cast<long>(batch_id) * kargs.stride_a_batch +
                               (T::A_T ? static_cast<long>(row) : static_cast<long>(row) * kargs.stride_a) + a_k0;
    const D_AB* const b_base = reinterpret_cast<const D_AB*>(kargs.ptr_b) + static_cast<long>(batch_id) * kargs.stride_b_batch +
                               (T::B_T ? static_cast<long>(col) : static_cast<long>(col) * kargs.stride_b) + b_k0;
    // B_SHUF: 16 rows of the shuffled [N, K] are one block of 16 K bytes, so row col still starts at col * K.
    auto g_a = make_gmem(a_base, T::BOUNDARY ? range_u32(static_cast<long>(rows_in) * kargs.stride_a - a_k0) : 0xffffffffu);
    auto g_b = make_gmem(b_base, T::BOUNDARY ? range_u32(static_cast<long>(cols_in) * kargs.stride_b - b_k0) : 0xffffffffu);
    // A_T / B_T buffer of the K tile being copied: from its first k row, column row / col, to the end of the matrix.  The
    // copies run through the K tiles of this block's K segment in order (prefetch_a / prefetch_b: half 0 then half 1 of
    // tiles 0, 1, 2, ...), so the base and what is left of the matrix are stepped by one K tile after each half 1 --
    // no 64-bit multiply in the loop.  BOUNDARY: the columns past M / N of a k row read the next k row (only C rows /
    // columns past M / N use them, and those are not stored), and past the last k row nothing.
    const long a_k_step = static_cast<long>(T::B_K) * kargs.stride_a, b_k_step = static_cast<long>(T::B_K) * kargs.stride_b;
    const D_AB* a_k_base = a_base + kt0 * a_k_step;
    const D_AB* b_k_base = b_base + kt0 * b_k_step;
    long a_k_left = (kargs.k / T::B_K - kt0) * a_k_step - row, b_k_left = (kargs.k / T::B_K - kt0) * b_k_step - col;
    // C: 64-bit tile base and a buffer range bounded to this output tile (as the blockscale 4wave_pin_agpr WIDE_OUTPUT), so
    // the 32-bit range and offsets cover one tile whatever the size of C; the host keeps one tile under 2 GiB.  BOUNDARY:
    // the range ends at the last valid element of the tile (rows past M start beyond it since stride_c >= cols_in).
    D_C* const c_tile = reinterpret_cast<D_C*>(kargs.ptr_c) + static_cast<long>(batch_id) * kargs.stride_c_batch +
                        static_cast<long>(row) * kargs.stride_c + col;
    auto g_c = make_gmem(c_tile, T::BOUNDARY ? (static_cast<unsigned int>((rows_in < T::B_M ? rows_in : T::B_M) - 1) *
                                                    static_cast<unsigned int>(kargs.stride_c) +
                                                static_cast<unsigned int>(cols_in < T::B_N ? cols_in : T::B_N)) * 2u
                                             : 0xffffffffu);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row * kargs.stride_sfa + sf_k0,
                           static_cast<unsigned int>((kargs.m - row) * kargs.stride_sfa - sf_k0));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + col * kargs.stride_sfb + sf_k0,
                           static_cast<unsigned int>((kargs.n - col) * kargs.stride_sfb - sf_k0));

    const auto u_ga = make_layout_ga<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_gb = make_layout_gb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sa = make_layout_sa<T>(wave_id_m, wave_id_n);
    const auto u_sb = make_layout_sb<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra<T>(lane_id, wave_id_m);
    const auto u_rb = make_layout_rb<T>(lane_id, wave_id_n);
    const auto u_gsfa = make_layout_gsfa<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = make_layout_ssfa<T>(wave_id_m, wave_id_n);
    const auto u_gsfb = make_layout_gsfb<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfb);
    const auto u_ssfb = make_layout_ssfb<T>(wave_id_m, wave_id_n);
    const auto u_rsfa = make_layout_rsfa<T>(lane_id, wave_id_m);
    const auto u_rsfb = make_layout_rsfb<T>(lane_id, wave_id_n);
    const auto u_rsfa16 = make_layout_rsfa16<T>(lane_id, wave_id_m);
    const auto u_rsfb16 = make_layout_rsfb16<T>(lane_id, wave_id_n);

    constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave_a + T::smem_padding);
    constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave_b + T::smem_padding);
    constexpr int smem_a_bytes = smem_a_elem * 4 * sizeof(D_A);
    constexpr int smem_b_bytes = smem_b_elem * 4 * sizeof(D_B);
    // One array: A | B | sfa panels | sfb panels (160 KB); the C staging of the epilogue reuses it from the start.
    constexpr int smem_all_bytes = smem_a_bytes + smem_b_bytes + T::SCALE_PANEL_BUFFERS * (T::SFA_PANEL_BYTES + T::SFB_PANEL_BYTES);
    static_assert(smem_all_bytes == T::LDS_BYTES);
    __shared__ __attribute__((aligned(16))) char smem_all[smem_all_bytes];
    char* const smem_ab = smem_all;
    char* const smem_a = smem_ab;
    char* const smem_b = smem_ab + smem_a_bytes;
    char* const smem_sfa = smem_ab + smem_a_bytes + smem_b_bytes;
    char* const smem_sfb = smem_sfa + T::SCALE_PANEL_BUFFERS * T::SFA_PANEL_BYTES;
    auto s_a = make_smem(reinterpret_cast<D_AB*>(smem_a));
    auto s_b = make_smem(reinterpret_cast<D_AB*>(smem_b));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));

    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, 1>{},          // K: one MFMA (W_K = 128 = the K tile)
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using MMA = decltype(mma);
    op8_t v_a[2][T::E_M];   // [half][repeat]: E_M fp8 operands, 32 B each
    op8_t v_b[2][T::E_N];
    using CFrag = opus::vector_t<D_ACC, MMA::mma_c_len>;   // one 16x16 C fragment, 4 fp32 per lane
    CFrag v_c[2][2][T::E_M * T::E_N];
    opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
        clear(v_c[hm.value][hn.value][f.value]);
    }); }); });

    using D_SF_PACK = opus::u32_t;
    D_SF_PACK v_sfa[2][T::E_K];
    D_SF_PACK v_sfb[2][T::E_K];

    // ---- offsets ----
    auto ga_offset = [&](int half_tile_m, int tile_k) {
        if constexpr (T::A_T) return half_tile_m * T::HALF_B_M;   // the K tile is in the buffer base (a_k_base)
        else return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K_BYTES;
    };
    // Shuffled B: a K tile is 128 / 32 = 4 blocks of 512 bytes inside each 16-row block (16 K bytes per row).
    auto gb_offset = [&](int half_tile_n, int tile_k) {
        if constexpr (T::B_T) return half_tile_n * T::HALF_B_N;   // the K tile is in the buffer base (b_k_base)
        else return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K_BYTES * (T::B_SHUF ? 16 : 1);
    };
    auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
    auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * T::NUM_KGROUPS; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin * T::NUM_KGROUPS; };
    auto ssfa_offset = [&](int stage, int tile_in_panel) { return stage * T::SFA_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS; };
    auto ssfb_offset = [&](int stage, int tile_in_panel) { return stage * T::SFB_PANEL_BYTES + tile_in_panel * T::NUM_KGROUPS; };

    // ---- global -> LDS ----
    // A_T copy: half tile = 128 k rows x 128 B (m); LDS row k at k * 128, wave w = wave_n * 2 + wave_m owns
    // rows 32 w .. 32 w + 31 (4 issues of 8 rows = LDS chunks w * 4 + i, MUBUF imm offsets).  The 16-byte
    // slot s of row k holds m-group s ^ f(k), f(k) = ((k % 8) / 2) ^ 4 ((k / 16) & 1), so the 8 rows x 16 B
    // of a TR8 group and the 2 groups of a pass (k-blocks 2j, 2j+1) hit 16 different 16-byte bank slots.
    // B_M < 256: the HALF_B_M / 16 m-groups fill the low slots' m-groups; a slot of m-group >= HALF_B_M / 16 is never
    // read and copies m-group (that - (8 - HALF_B_M / 16)) of the tile instead (same cache lines, no extra traffic).
    const int tt_w = wave_id_n * 2 + wave_id_m;
    auto prefetch_a = [&](int stage, int tile, int half_tile_m) {
        if constexpr (T::A_T) {
            char* const base = reinterpret_cast<char*>(s_a.ptr) + sa_offset(stage, half_tile_m) + tt_w * 4 * T::smem_linear_wave_a;
            const int kr = lane_id / 8, slot = lane_id % 8;
            auto g_at = make_gmem(a_k_base, T::BOUNDARY ? range_u32(a_k_left) : 0xffffffffu);
            opus::static_for<4>([&](auto i_) {
                constexpr int i = decltype(i_)::value;
                constexpr int io = i * T::smem_linear_wave_a;
                const int f = ((kr >> 1) ^ (((i >> 1) & 1) * 4)) & 7;
                constexpr int MG = T::HALF_B_M / 16;
                const int mg = slot ^ f;
                const int gofs = (tt_w * 32 + i * 8 + kr) * kargs.stride_a + (MG == 8 || mg < MG ? mg : mg - (8 - MG)) * T::VEC_A;
                g_at.template async_load<T::VEC_A>(base, gofs - io, ga_offset(half_tile_m, tile), opus::number<io>{});
            });
            if (half_tile_m == 1) { a_k_base += a_k_step; a_k_left -= a_k_step; }   // next K tile
        } else {
            async_load_imm<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage, half_tile_m), ga_offset(half_tile_m, tile));
        }
    };
    // B_T copy: prefetch_a's A_T copy with wave w = wave_m * 2 + wave_n and n-groups of HALF_B_N / 16.
    const int nn_w = wave_id_m * 2 + wave_id_n;
    auto prefetch_b = [&](int stage, int tile, int half_tile_n) {
        if constexpr (T::B_T) {
            char* const base = reinterpret_cast<char*>(s_b.ptr) + sb_offset(stage, half_tile_n) + nn_w * 4 * T::smem_linear_wave_b;
            const int kr = lane_id / 8, slot = lane_id % 8;
            auto g_bt = make_gmem(b_k_base, T::BOUNDARY ? range_u32(b_k_left) : 0xffffffffu);
            opus::static_for<4>([&](auto i_) {
                constexpr int i = decltype(i_)::value;
                constexpr int io = i * T::smem_linear_wave_b;
                const int f = ((kr >> 1) ^ (((i >> 1) & 1) * 4)) & 7;
                constexpr int NG = T::HALF_B_N / 16;
                const int ng = slot ^ f;
                const int gofs = (nn_w * 32 + i * 8 + kr) * kargs.stride_b + (NG == 8 || ng < NG ? ng : ng - (8 - NG)) * T::VEC_B;
                g_bt.template async_load<T::VEC_B>(base, gofs - io, gb_offset(half_tile_n, tile), opus::number<io>{});
            });
            if (half_tile_n == 1) { b_k_base += b_k_step; b_k_left -= b_k_step; }   // next K tile
        } else {
            async_load_imm<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage, half_tile_n), gb_offset(half_tile_n, tile));
        }
    };
    auto prefetch_sfa = [&](int panel_k_begin, int buf) {
        async_load<T::VEC_GLOBAL_SCALE>(g_sfa, s_sfa.ptr, u_gsfa, u_ssfa + ssfa_offset(buf, 0), gsfa_offset(panel_k_begin));
    };
    auto prefetch_sfb = [&](int panel_k_begin, int buf) {
        async_load<T::VEC_GLOBAL_SCALE>(g_sfb, s_sfb.ptr, u_gsfb, u_ssfb + ssfb_offset(buf, 0), gsfb_offset(panel_k_begin));
    };

    // ---- LDS -> VGPR ----
    const auto ra_offs = opus::layout_to_offsets<T::VEC_A>(u_ra);   // [repeat]
    const int a_slot0 = ((lane_id / T::W_M) ^ (((lane_id % T::W_M) / 2) ^ (((lane_id % T::W_M) % 2) << 2))) & 7;
    const int a_c1_delta = ((a_slot0 ^ 4) - a_slot0) * T::VEC_A;     // K slot 4 + g
    const auto rb_offs = opus::layout_to_offsets<T::VEC_B>(u_rb);
    // A_T read (TR8): operand e (m16 group 2e + wave_m), chunk c = k-block 4c + g, rows k 8h + b (b = r / 2):
    //   LDS (4c + g) * 2 KB + h * 1 KB + b * 128 + ((2e + wave_m) ^ (r / 4) ^ 4 (g & 1)) * 16 + (r % 2) * 8.
    const int tt_g = lane_id / T::W_M, tt_r = lane_id % T::W_M;
    int tt_ra[T::E_M];
    opus::static_for<T::E_M>([&](auto e) {
        tt_ra[e.value] = tt_g * 2048 + (tt_r >> 1) * 128 + (tt_r & 1) * 8
                       + (((2 * e.value + wave_id_m) ^ (tt_r >> 2) ^ ((tt_g & 1) * 4)) & 7) * 16;
    });
    // B_T read (TR8): the A_T read with operand e = n16 group 2e + wave_n.  2e only moves bits 1-2 of the slot, so
    // operand e reads at nn_rb ^ (32 e) (one live offset instead of E_N: 256x256 NN / TN spill otherwise).
    const int nn_g = lane_id / T::W_N, nn_r = lane_id % T::W_N;
    const int nn_rb = nn_g * 2048 + (nn_r >> 1) * 128 + (nn_r & 1) * 8 + ((wave_id_n ^ (nn_r >> 2) ^ ((nn_g & 1) * 4)) & 7) * 16;
    using i32x2_t = opus::vector_t<opus::i32_t, 2>;
    auto tr8 = [&](auto& sm_, int byte_ofs) {
        // pointer arithmetic on the view's (restrict) LDS pointer: no integer round trip, so the compiler
        // keeps the no-alias fact and does not wait for the LDS-DMA copies in flight (vmcnt(0))
        return __builtin_amdgcn_ds_read_tr8_b64_v2i32(reinterpret_cast<OPUS_LDS_ADDR i32x2_t*>(sm_.ptr + byte_ofs));
    };
    auto load_a = [&](auto& sa_, auto half_) {
        constexpr int half = decltype(half_)::value;
        static_assert(T::E_C == 2 && sizeof(D_AB) == 1);
        opus::static_for<T::E_M>([&](auto e) {
            constexpr int agpr = 0 + half * 32 + e.value * 8;   // operand e = 8 consecutive AGPRs
            if constexpr (T::A_T) {
                // 4 TR8 reads (2 registers each) -> one 8-register operand
                constexpr int o = half * (T::smem_m_rep * T::smem_linear_wave_a);   // sa_offset(0, half) in bytes
                const int b = tt_ra[e.value] + o;
                const i32x2_t q0 = tr8(sa_, b), q1 = tr8(sa_, b + 1024), q2 = tr8(sa_, b + 8192), q3 = tr8(sa_, b + 9216);
                const chunk16_t c0 = __builtin_shufflevector(q0, q1, 0, 1, 2, 3), c1 = __builtin_shufflevector(q2, q3, 0, 1, 2, 3);
                [[clang::amdgpu_pin_agpr(agpr)]] v_a[half][e.value] = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            } else {
                const auto c0 = __builtin_bit_cast(chunk16_t, load<T::VEC_A>(sa_, ra_offs[e.value] + sa_offset(0, half)));
                const auto c1 = __builtin_bit_cast(chunk16_t, load<T::VEC_A>(sa_, ra_offs[e.value] + sa_offset(0, half) + a_c1_delta));
                [[clang::amdgpu_pin_agpr(agpr)]] v_a[half][e.value] = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            }
        });
    };
    auto load_b = [&](auto& sb_, auto half_) {
        constexpr int half = decltype(half_)::value;
        static_assert(T::E_C == 2 && sizeof(D_AB) == 1);
        opus::static_for<T::E_N>([&](auto e) {
            constexpr int agpr = 64 + half * 32 + e.value * 8;   // operand e = 8 consecutive AGPRs
            if constexpr (T::B_T) {
                // 4 TR8 reads (2 registers each) -> one 8-register operand
                constexpr int o = half * (T::smem_n_rep * T::smem_linear_wave_b);   // sb_offset(0, half) in bytes
                const int b = (nn_rb ^ (32 * e.value)) + o;
                const i32x2_t q0 = tr8(sb_, b), q1 = tr8(sb_, b + 1024), q2 = tr8(sb_, b + 8192), q3 = tr8(sb_, b + 9216);
                const chunk16_t c0 = __builtin_shufflevector(q0, q1, 0, 1, 2, 3), c1 = __builtin_shufflevector(q2, q3, 0, 1, 2, 3);
                [[clang::amdgpu_pin_agpr(agpr)]] v_b[half][e.value] = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            } else {
                const auto c0 = __builtin_bit_cast(chunk16_t, load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half)));
                // chunk c = 1: shuffled + IB chunks, plain the swizzled K slot 4 + g (as A)
                const auto c1 = __builtin_bit_cast(chunk16_t, load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + (T::B_SHUF ? T::b_buffer_load_insts * T::smem_linear_wave_b : a_c1_delta)));
                [[clang::amdgpu_pin_agpr(agpr)]] v_b[half][e.value] = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            }
        });
    };
    opus::vector_t<D_SF, 2 * T::VEC_LDS_SCALE> v_sfa_raw, v_sfb_raw;
    auto load_sfa = [&](auto& ssfa_, int buf, int tile_in_panel) { v_sfa_raw = load<T::VEC_LDS_SCALE>(ssfa_, u_rsfa + ssfa_offset(buf, tile_in_panel)); };
    // 4-tile main loop: one 16-byte read per row and half per panel, dword (tile % 4) picked at compile time.
    using SF8 = opus::vector_t<opus::u32_t, 8>;                    // [h * 4 + tile % 4]
    using SF2 = opus::vector_t<opus::u32_t, 2>;
    SF8 r16a, r16b;
    auto load_sf16 = [&](auto& ssfa_, auto& ssfb_, int buf, int tip) {
        r16a = __builtin_bit_cast(SF8, load<T::VEC_GLOBAL_SCALE>(ssfa_, u_rsfa16 + ssfa_offset(buf, tip)));
        r16b = __builtin_bit_cast(SF8, load<T::VEC_GLOBAL_SCALE>(ssfb_, u_rsfb16 + ssfb_offset(buf, tip)));
    };
    auto pick_sf16 = [&](auto j_) {
        constexpr int j = decltype(j_)::value;
        v_sfa_raw = __builtin_bit_cast(decltype(v_sfa_raw), SF2{r16a[j], r16a[4 + j]});
        v_sfb_raw = __builtin_bit_cast(decltype(v_sfb_raw), SF2{r16b[j], r16b[4 + j]});
    };
    auto load_sfb = [&](auto& ssfb_, int buf, int tile_in_panel) { v_sfb_raw = load<T::VEC_LDS_SCALE>(ssfb_, u_rsfb + ssfb_offset(buf, tile_in_panel)); };

    //
    // fp8: load<4> x 2 halves = 8 e8m0 per lane (only dwords [0] and [1] below exist: E_K = 1).
    // Before (load<8> x 2 halves = 16 e8m0, row-major; lane 16g+r holds row g*32+r and row 128+g*32+r):
    //            dword0 [0][0]        dword1 [0][1]        dword2 [1][0]                dword3 [1][1]
    // lane 0     M0k0  M0k1  M0k2  M0k3   M0k4  M0k5  M0k6  M0k7   M128k0 M128k1 M128k2 M128k3   M128k4 M128k5 M128k6 M128k7
    // lane 16    M32k0 M32k1 M32k2 M32k3  M32k4 M32k5 M32k6 M32k7  M160k0 M160k1 M160k2 M160k3   M160k4 M160k5 M160k6 M160k7
    // lane 32    M64k0 M64k1 M64k2 M64k3  M64k4 M64k5 M64k6 M64k7  M192k0 M192k1 M192k2 M192k3   M192k4 M192k5 M192k6 M192k7
    // lane 48    M96k0 M96k1 M96k2 M96k3  M96k4 M96k5 M96k6 M96k7  M224k0 M224k1 M224k2 M224k3   M224k4 M224k5 M224k6 M224k7
    //
    // After (byte e = repeat e, picked by op_sel; lane group g = K group g of the MFMA):
    //            v_sfa[0][0]           v_sfa[0][1]           v_sfa[1][0]                 v_sfa[1][1]
    // lane 0     M0k0 M32k0 M64k0 M96k0  M0k4 M32k4 M64k4 M96k4  M128k0 M160k0 M192k0 M224k0  M128k4 M160k4 M192k4 M224k4
    // lane 16    M0k1 M32k1 M64k1 M96k1  M0k5 M32k5 M64k5 M96k5  M128k1 M160k1 M192k1 M224k1  M128k5 M160k5 M192k5 M224k5
    // lane 32    M0k2 M32k2 M64k2 M96k2  M0k6 M32k6 M64k6 M96k6  M128k2 M160k2 M192k2 M224k2  M128k6 M160k6 M192k6 M224k6
    // lane 48    M0k3 M32k3 M64k3 M96k3  M0k7 M32k7 M64k7 M96k7  M128k3 M160k3 M192k3 M224k3  M128k7 M160k7 M192k7 M224k7
    //
    using SF_DW = opus::vector_t<D_SF_PACK, 2 * T::E_K>;
    // Half h (rows h*128 + ...) of the scales read last (v_sfa_raw / v_sfb_raw) -> v_sfa[h] / v_sfb[h].
    // In the loop half 0 is done during C11 (v_sfa[0] / v_sfb[0] are no longer used by tile t)
    // and half 1 during C00(t+1) (which only uses half 0).
    auto transpose_sf = [&](auto h_) {
        constexpr int h = decltype(h_)::value;
        const auto dwa = __builtin_bit_cast(SF_DW, v_sfa_raw);
        const auto dwb = __builtin_bit_cast(SF_DW, v_sfb_raw);
        opus::static_for<T::E_K>([&](auto kk) {
            v_sfa[h][kk.value] = transpose_scale_dword_4x4(dwa[h * T::E_K + kk.value], lane_id / T::W_M);
            v_sfb[h][kk.value] = transpose_scale_dword_4x4(dwb[h * T::E_K + kk.value], lane_id / T::W_N);
        });
    };

    // LDS reads go through __restrict__ copies of the views, passed as function parameters
    // (clang ignores __restrict__ on locals): without them the compiler cannot tell the
    // reads from the in-flight LDS-DMA copies and puts a vmcnt(0) before the reads.
    auto views = [&](OPUS_LDS_ADDR char* __restrict__ read_a, OPUS_LDS_ADDR char* __restrict__ read_b,
                     OPUS_LDS_ADDR char* __restrict__ read_sfa, OPUS_LDS_ADDR char* __restrict__ read_sfb, auto&& body) {
        auto rd_a = s_a;     rd_a.ptr = read_a;
        auto rd_b = s_b;     rd_b.ptr = read_b;
        auto rd_sfa = s_sfa; rd_sfa.ptr = read_sfa;
        auto rd_sfb = s_sfb; rd_sfb.ptr = read_sfb;
        body(rd_a, rd_b, rd_sfa, rd_sfb);
    };

    // ===== Prologue =====
    constexpr int frags = T::E_M * T::E_N;   // 16 (m, n) steps per quadrant, 2 MFMAs each
    // Issue tile 0 and scale panel 0, then tile 1 and panel 1 (tile 2, 3); wait only for
    // tile 0 + panel 0 (counted vmcnt), read A0 / B0 and the scales of tile 0, and compute
    // C00 of tile 0 while A1 / B1 of tile 0 are read.  The main loop then starts with
    // "C00 of tile t is done".
    prefetch_a(0, 0, 0);
    prefetch_b(0, 0, 0);
    prefetch_a(0, 0, 1);
    prefetch_b(0, 0, 1);
    prefetch_sfa(0, 0);
    prefetch_sfb(0, 0);
    if (loops > 1) {
        prefetch_a(1, 1, 0);
        prefetch_b(1, 1, 0);
        prefetch_a(1, 1, 1);
        prefetch_b(1, 1, 1);
    }
    if (loops > 1) {
        s_waitcnt_vmcnt(number<2 * (T::a_copy_insts + T::b_copy_insts)>{});          // tile 1 may fly
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    // Tile 1 and panel 1 may still be in flight: read through the views.
    views(s_a.ptr, s_b.ptr, s_sfa.ptr, s_sfb.ptr, [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
        load_a(rd_a, number<0>{});
        load_b(rd_b, number<0>{});
        load_sf16(rd_sfa, rd_sfb, 0, 0);
        pick_sf16(number<0>{});
        transpose_sf(number<0>{});
        transpose_sf(number<1>{});
        __builtin_amdgcn_sched_barrier(0);
        // C00 of tile 0 + A1, B1 of tile 0.
        load_a(rd_a, number<1>{});
        load_b(rd_b, number<1>{});
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
        });
        sched_pro<T>();
        __builtin_amdgcn_sched_barrier(0);
    });

    // ===== Main loop =====
    int tile = 0;
    // Scale panel buffers tracked instead of (k / 2) % 3: sf_rd holds the panel of tile t+1,
    // sf_wr = sf_rd + 1 (mod 3) receives the panel copied in iteration t (t+3 for odd t,
    // t+2 for even t -- both one panel after tile t+1's); after an even t, sf_rd advances.
    int sf_rd = 0;
    // Two tiles per trip: stage / next_stage and the panel parity are compile-time in each
    // copy; A / B / C are pinned, so both copies use the same registers.
    // 4 tiles per trip (tile % 4 == 0 on entry): the scale dword of each tile is compile-time.
#pragma clang loop unroll(disable)
    for (; tile + 5 < loops; tile += 4) {
        {   // even tile
            constexpr int stage = 0;
            constexpr int next_stage = 1;
            const int sf_wr = sf_rd == 0 ? T::SCALE_PANEL_BUFFERS - 1 : sf_rd - 1;   // panel P + 2 (buffer of P - 1)
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
                });
                sched_c10<T>();
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<T::a_copy_insts + T::b_copy_insts>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                pick_sf16(number<1>{});          // tile t + 1 = 4P + 1
                load_b(rd_b, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
                });
                sched_main<T>();
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                if ((tile & 7) == 0 && tile + 8 < loops) {     // t = 4P: sfb of panel P + 2
                    prefetch_sfb(tile + 8, sf_wr);
                    asm volatile("; panel copy: sfb" ::: "memory");
                }
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                transpose_sf(number<0>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
                });
                sched_c11<T>();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                transpose_sf(number<1>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
                });
                sched_c00<T>();
                __builtin_amdgcn_sched_barrier(0);
            });
            if ((tile & 7) == 6) sf_rd = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;   // panel of tile t + 1 changes
        }
        {   // odd tile
            const int tile_odd = tile + 1;
            const int tile = tile_odd;
            constexpr int stage = 1;
            constexpr int next_stage = 0;
            const int sf_wr = sf_rd == 0 ? T::SCALE_PANEL_BUFFERS - 1 : sf_rd - 1;   // panel P + 2 (buffer of P - 1)
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
                });
                sched_c10<T>();
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<T::a_copy_insts + T::b_copy_insts>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                pick_sf16(number<2>{});          // tile t + 1 = 4P + 2
                load_b(rd_b, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
                });
                sched_main<T>();
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                // odd tile: sfa of the panel starting at tile + 3; past the last panel it only
                // overwrites the buffer of a consumed panel (harmless), so no branch.
                if ((tile & 7) == 1 && tile + 7 < loops) {     // t = 4P + 1: sfa of panel P + 2
                    prefetch_sfa(tile + 7, sf_wr);
                    asm volatile("; panel copy: sfa" ::: "memory");
                }
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                transpose_sf(number<0>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
                });
                sched_c11<T>();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                transpose_sf(number<1>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
                });
                sched_c00<T>();
                __builtin_amdgcn_sched_barrier(0);
            });
        }
            {   // tiles t + 2, t + 3
        const int tile_q = tile + 2;
        const int tile = tile_q;
        {   // even tile
            constexpr int stage = 0;
            constexpr int next_stage = 1;
            const int sf_wr = sf_rd == 0 ? T::SCALE_PANEL_BUFFERS - 1 : sf_rd - 1;   // panel P + 2 (buffer of P - 1)
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
                });
                sched_c10<T>();
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<T::a_copy_insts + T::b_copy_insts>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                pick_sf16(number<3>{});          // tile t + 1 = 4P + 3
                load_b(rd_b, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
                });
                sched_main<T>();
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                if ((tile & 7) == 0 && tile + 8 < loops) {     // t = 4P: sfb of panel P + 2
                    prefetch_sfb(tile + 8, sf_wr);
                    asm volatile("; panel copy: sfb" ::: "memory");
                }
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                transpose_sf(number<0>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
                });
                sched_c11<T>();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                transpose_sf(number<1>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
                });
                sched_c00<T>();
                __builtin_amdgcn_sched_barrier(0);
            });
            if ((tile & 7) == 6) sf_rd = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;   // panel of tile t + 1 changes
        }
        {   // odd tile
            const int tile_odd = tile + 1;
            const int tile = tile_odd;
            constexpr int stage = 1;
            constexpr int next_stage = 0;
            const int sf_wr = sf_rd == 0 ? T::SCALE_PANEL_BUFFERS - 1 : sf_rd - 1;   // panel P + 2 (buffer of P - 1)
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
                });
                sched_c10<T>();
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<T::a_copy_insts + T::b_copy_insts>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                load_sf16(rd_sfa, rd_sfb, sf_rd, (tile + 1) & 4);   // tile t + 1 starts a panel
                pick_sf16(number<0>{});
                load_b(rd_b, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
                });
                sched_main<T>();
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                // odd tile: sfa of the panel starting at tile + 3; past the last panel it only
                // overwrites the buffer of a consumed panel (harmless), so no branch.
                if ((tile & 7) == 1 && tile + 7 < loops) {     // t = 4P + 1: sfa of panel P + 2
                    prefetch_sfa(tile + 7, sf_wr);
                    asm volatile("; panel copy: sfa" ::: "memory");
                }
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                transpose_sf(number<0>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
                });
                sched_c11<T>();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                transpose_sf(number<1>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
                });
                sched_c00<T>();
                __builtin_amdgcn_sched_barrier(0);
            });
        }
            }
    }
#pragma clang loop unroll(disable)
    for (int once = 0; (!TRANS || once < 1) && tile + 3 < loops; ++once, tile += 2) {   // transposed: at most once after the 4-tile loop
        {   // even tile
            constexpr int stage = 0;
            constexpr int next_stage = 1;
            const int sf_wr = sf_rd == 0 ? T::SCALE_PANEL_BUFFERS - 1 : sf_rd - 1;   // panel P + 2 (buffer of P - 1)
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
                });
                sched_c10<T>();
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<T::a_copy_insts + T::b_copy_insts>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                load_sfa(rd_sfa, sf_rd, (tile + 1) & 7);
                load_sfb(rd_sfb, sf_rd, (tile + 1) & 7);
                load_b(rd_b, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
                });
                sched_main<T>();
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                if ((tile & 7) == 0 && tile + 8 < loops) {     // t = 4P: sfb of panel P + 2
                    prefetch_sfb(tile + 8, sf_wr);
                    asm volatile("; panel copy: sfb" ::: "memory");
                }
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                transpose_sf(number<0>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
                });
                sched_c11<T>();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                transpose_sf(number<1>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
                });
                sched_c00<T>();
                __builtin_amdgcn_sched_barrier(0);
            });
            if ((tile & 7) == 6) sf_rd = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;   // panel of tile t + 1 changes
        }
        {   // odd tile
            const int tile_odd = tile + 1;
            const int tile = tile_odd;
            constexpr int stage = 1;
            constexpr int next_stage = 0;
            const int sf_wr = sf_rd == 0 ? T::SCALE_PANEL_BUFFERS - 1 : sf_rd - 1;   // panel P + 2 (buffer of P - 1)
            views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
                  [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
                // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
                // tile t+2 may be copied into it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
                prefetch_a(stage, tile + 2, 0);
                prefetch_b(stage, tile + 2, 0);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
                });
                sched_c10<T>();
                __builtin_amdgcn_sched_barrier(0);
                // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
                s_waitcnt_vmcnt(number<T::a_copy_insts + T::b_copy_insts>{});
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);

                // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
                load_sfa(rd_sfa, sf_rd, (tile + 1) & 7);
                load_sfb(rd_sfb, sf_rd, (tile + 1) & 7);
                load_b(rd_b, number<0>{});
                prefetch_a(stage, tile + 2, 1);
                prefetch_b(stage, tile + 2, 1);
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
                });
                sched_main<T>();
                __builtin_amdgcn_sched_barrier(0);

                // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
                // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
                // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
                // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
                // so it sits on a region boundary.
                // The two branches end in different (empty) asm, so the compiler cannot merge
                // their tails into one copy that selects between the sfa / sfb layouts (that
                // selection goes through scratch, whose vmcnt wait drains every copy in flight).
                // odd tile: sfa of the panel starting at tile + 3; past the last panel it only
                // overwrites the buffer of a consumed panel (harmless), so no branch.
                if ((tile & 7) == 1 && tile + 7 < loops) {     // t = 4P + 1: sfa of panel P + 2
                    prefetch_sfa(tile + 7, sf_wr);
                    asm volatile("; panel copy: sfa" ::: "memory");
                }
                __builtin_amdgcn_sched_barrier(0);

                // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
                load_a(rd_a, number<0>{});
                transpose_sf(number<0>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
                });
                sched_c11<T>();
                __builtin_amdgcn_sched_barrier(0);

                // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
                load_a(rd_a, number<1>{});
                load_b(rd_b, number<1>{});
                transpose_sf(number<1>{});
                opus::static_for<frags>([&](auto i) {
                    constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                    mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
                });
                sched_c00<T>();
                __builtin_amdgcn_sched_barrier(0);
            });
        }
    }
    // At most one more steady iteration (odd number of them).
#pragma clang loop unroll(disable)
    for (int once = 0; (!TRANS || once < 1) && tile + 2 < loops; ++once, ++tile) {      // transposed: at most once
        const int stage = tile & 1;
        const int next_stage = stage ^ 1;
        const int sf_wr = sf_rd == 0 ? T::SCALE_PANEL_BUFFERS - 1 : sf_rd - 1;   // panel P + 2 (buffer of P - 1)
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            // B_a: every wave is done reading stage(t) (A1 / B1 of tile t in C00), so
            // tile t+2 may be copied into it.
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C10(t) + A0, B0 of tile t+2 -> stage(t), one buffer_load per 4 MFMAs.
            prefetch_a(stage, tile + 2, 0);
            prefetch_b(stage, tile + 2, 0);
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
            });
            sched_c10<T>();
            __builtin_amdgcn_sched_barrier(0);
            // B_b: tile t+1 (and its scale panel) landed; the 8 copies just issued may still fly.
            s_waitcnt_vmcnt(number<T::a_copy_insts + T::b_copy_insts>{});
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C01(t) + scales and B0 of tile t+1 + A1, B1 of tile t+2.
            load_sfa(rd_sfa, sf_rd, (tile + 1) & 7);
            load_sfb(rd_sfb, sf_rd, (tile + 1) & 7);
            load_b(rd_b, number<0>{});
            prefetch_a(stage, tile + 2, 1);
            prefetch_b(stage, tile + 2, 1);
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
            });
            sched_main<T>();
            __builtin_amdgcn_sched_barrier(0);

            // Scale panel copies, split over two tiles (one panel instruction touches 64 cache
            // lines, ~5x an A/B copy on the TA): odd t -> sfa, even t -> sfb of the panel
            // starting at t + 3 / t + 2.  With 3 panel buffers it overwrites the panel three
            // back (read long ago) and lands before the B_b that reads it.  A runtime branch,
            // so it sits on a region boundary.
            // The two branches end in different (empty) asm, so the compiler cannot merge
            // their tails into one copy that selects between the sfa / sfb layouts (that
            // selection goes through scratch, whose vmcnt wait drains every copy in flight).
            if ((tile & 7) == 1) {
                if (tile + 7 < loops) prefetch_sfa(tile + 7, sf_wr);
                asm volatile("; panel copy: sfa" ::: "memory");
            } else if ((tile & 7) == 0) {
                if (tile + 8 < loops) prefetch_sfb(tile + 8, sf_wr);
                asm volatile("; panel copy: sfb" ::: "memory");
            }
            __builtin_amdgcn_sched_barrier(0);

            // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
            load_a(rd_a, number<0>{});
            transpose_sf(number<0>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
            });
            sched_c11<T>();
            __builtin_amdgcn_sched_barrier(0);

            // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, number<1>{});
            transpose_sf(number<1>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
            });
            sched_c00<T>();
            __builtin_amdgcn_sched_barrier(0);
        });
        if ((tile & 7) == 6) sf_rd = sf_rd == T::SCALE_PANEL_BUFFERS - 1 ? 0 : sf_rd + 1;
    }

    // ===== Epilogue =====
    // 1. Second-to-last tile: C10, C01, C11 of it and C00 of the last tile; no more copies.
    if (TRANS || tile + 1 < loops) {   // transposed: always (loops is even and >= 2, here tile == loops - 2)
        const int next_stage = (tile & 1) ^ 1;
        views(s_a.ptr + sa_offset(next_stage, 0), s_b.ptr + sb_offset(next_stage, 0), s_sfa.ptr, s_sfb.ptr,
              [&](auto& rd_a, auto& rd_b, auto& rd_sfa, auto& rd_sfb) {
            // C10(t).
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
            });
            sched_plain<T>();
            __builtin_amdgcn_sched_barrier(0);
            // B_b: tile t+1 (and its scale panel) landed.
            s_waitcnt_vmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // C01(t) + scales and B0 of tile t+1.
            load_sfa(rd_sfa, sf_rd, (tile + 1) & 7);
            load_sfb(rd_sfb, sf_rd, (tile + 1) & 7);
            load_b(rd_b, number<0>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
            });
            sched_last<T>();
            __builtin_amdgcn_sched_barrier(0);

            // C11(t) + A0 of tile t+1 + scale transposes of half 0 of tile t+1.
            load_a(rd_a, number<0>{});
            transpose_sf(number<0>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
            });
            sched_c11<T>();
            __builtin_amdgcn_sched_barrier(0);

            // C00(t+1) + A1, B1 of tile t+1 + scale transposes of half 1 of tile t+1.
            load_a(rd_a, number<1>{});
            load_b(rd_b, number<1>{});
            transpose_sf(number<1>{});
            opus::static_for<frags>([&](auto i) {
                constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
                mma_scale_one<T, m, n, c_reg<T, 0, 0, m, n>()>(mma, v_a[0][m], v_b[0][n], v_c[0][0][m * T::E_N + n], v_sfa[0][0], v_sfb[0][0]);
            });
            sched_c00<T>();
            __builtin_amdgcn_sched_barrier(0);
        });
        ++tile;
    }
    // 2. Last tile: C10, C01, C11.
    {
        // C10(t).
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, m, n, c_reg<T, 1, 0, m, n>()>(mma, v_a[1][m], v_b[0][n], v_c[1][0][m * T::E_N + n], v_sfa[1][0], v_sfb[0][0]);
        });
        sched_plain<T>();
        __builtin_amdgcn_sched_barrier(0);

        // C01(t).
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, m, n, c_reg<T, 0, 1, m, n>()>(mma, v_a[0][m], v_b[1][n], v_c[0][1][m * T::E_N + n], v_sfa[0][0], v_sfb[1][0]);
        });
        sched_plain<T>();
        __builtin_amdgcn_sched_barrier(0);

        // C11(t).
        opus::static_for<frags>([&](auto i) {
            constexpr int m = decltype(i)::value / T::E_N, n = decltype(i)::value % T::E_N;
            mma_scale_one<T, m, n, c_reg<T, 1, 1, m, n>()>(mma, v_a[1][m], v_b[1][n], v_c[1][1][m * T::E_N + n], v_sfa[1][0], v_sfb[1][0]);
        });
        sched_plain<T>();
        __builtin_amdgcn_sched_barrier(0);
    }

    // 3. Write back C (bf16) through LDS, as opus_gemm_pipeline_a8w8_blockscale_mxscale_
    //    bpreshuffle_4wave_pin_agpr_gfx950 (PR 6696, Output::lds).  A wave owns only 16
    //    columns of a row per fragment (the other wave_n holds the neighbours), so direct
    //    fragment stores leave every 128-byte line partially written.  The whole 256 x 256
    //    tile is staged as one bf16 image in the idle A+B LDS, rows padded to B_N + 8
    //    elements (the 16 rows of a fragment then step by 16 bytes across the banks);
    //    each quadrant gets its own region, so staging needs no buffer reuse.  Per
    //    quadrant: stage -> sync -> 8 copies of 16 B per thread (one buffer_store = 4
    //    rows x 256 B), non-temporal (aux 2).
    constexpr int c_lds_stride = T::B_N + 8;
    constexpr int VEC_OUT = 16 / sizeof(D_C);                                            // 8 bf16 = 16 B
    static_assert(T::B_M * c_lds_stride * sizeof(D_C) <= smem_all_bytes);
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_ab));
    auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(opus::number<c_lds_stride>{}, 1_I), p_coord_c);
    auto sc_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_stride + half_tile_n * T::HALF_B_N; };
    const int output_thread_id = static_cast<int>(thread_id_x());
    // bf16 C: LDS -> global, one quadrant.
    // BOUNDARY: columns past N go to an element offset past the tile's buffer range (byte offset 2^31).  When every copy of a
    // thread has the same column (BLOCK_SIZE * 8 a multiple of HALF_B_N) the offset is computed once per quadrant.
    constexpr int C_OOB = 0x40000000;
    constexpr bool C_COL_FIXED = (T::BLOCK_SIZE * VEC_OUT) % T::HALF_B_N == 0;
    auto copy_output_bf16 = [&](int half_tile_m, int half_tile_n) {
        const int oob_q = T::BOUNDARY && C_COL_FIXED &&
                          (output_thread_id * VEC_OUT) % T::HALF_B_N + half_tile_n * T::HALF_B_N >= cols_in ? C_OOB : 0;
        opus::static_for<T::HALF_B_M * T::HALF_B_N / (T::BLOCK_SIZE * VEC_OUT)>([&](auto i) {   // 8
            const int linear = output_thread_id * VEC_OUT + decltype(i)::value * T::BLOCK_SIZE * VEC_OUT;
            const int r = linear / T::HALF_B_N + half_tile_m * T::HALF_B_M;
            const int c = linear % T::HALF_B_N + half_tile_n * T::HALF_B_N;
            const auto value = load<VEC_OUT>(s_c, r * c_lds_stride + c);
            const int oob = C_COL_FIXED ? oob_q : (T::BOUNDARY && c >= cols_in ? C_OOB : 0);
            store<VEC_OUT>(g_c, value, r * kargs.stride_c + c + oob, 0, opus::number<2>{});
        });
    };
    // fp32 accumulators -> bf16 -> LDS, then sync and copy that quadrant out.
    auto store_quadrant = [&](const auto& acc, auto half_tile_m_, auto half_tile_n_) {
        constexpr int hm = decltype(half_tile_m_)::value, hn = decltype(half_tile_n_)::value;
        if constexpr (TRANS) {
            // fragment by fragment (C is in AGPRs: converting the whole 64-float quadrant at once needs ~96 VGPRs)
            static_assert(T::VEC_C == MMA::mma_c_len);
            const auto sc_offs = opus::layout_to_offsets<T::VEC_C>(u_sc);
            opus::static_for<T::E_M * T::E_N>([&](auto f) {
                store<T::VEC_C>(s_c, cast<D_C>(acc[f.value]), sc_offs[f.value] + sc_offset(hm, hn));
            });
        } else {
            typename MMA::vtype_c quad;
            opus::static_for<T::E_M * T::E_N>([&](auto f) {
                opus::set_slice(quad, acc[f.value], opus::number<f.value * MMA::mma_c_len>{}, opus::number<(f.value + 1) * MMA::mma_c_len>{});
            });
            store<T::VEC_C>(s_c, cast<D_C>(quad), u_sc + sc_offset(hm, hn));
        }
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (!(hm == 1 && hn == 0)) s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        copy_output_bf16(hm, hn);
    };
    // Split-K, S > 1: store this segment's fp32 partial tile (row-major, row stride padded_N) to the workspace
    // [S][batch][padded_M][padded_N]; splitk_reduce_kernel (splitk_reduce_gfx950.cuh) sums the S partials into C.
    if constexpr (T::SPLITK) {
        if (split_k > 1) {
            const int padded_m = opus_mxfp8_ceil_div(kargs.m, T::B_M) * T::B_M, padded_n = num_tiles_n * T::B_N;
            float* const ws_tile = reinterpret_cast<float*>(kargs.ptr_ws) +
                ((static_cast<long>(k_split) * kargs.batch + batch_id) * padded_m + row) * padded_n + col;
            auto g_ws = make_gmem(ws_tile, static_cast<unsigned int>(T::B_M) * static_cast<unsigned int>(padded_n) * 4u);
            auto u_wc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(padded_n, 1_I), p_coord_c);
            const auto wc_offs = opus::layout_to_offsets<T::VEC_C>(u_wc);
            static_assert(MMA::mma_c_len == T::VEC_C);
            opus::static_for<2>([&](auto hm) { opus::static_for<2>([&](auto hn) { opus::static_for<T::E_M * T::E_N>([&](auto f) {
                store<T::VEC_C>(g_ws, v_c[hm.value][hn.value][f.value],
                                wc_offs[f.value] + hm.value * T::HALF_B_M * padded_n + hn.value * T::HALF_B_N);
            }); }); });
            return;
        }
    }

    // All waves have finished their last matrix LDS reads.
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);
    store_quadrant(v_c[0][0], number<0>{}, number<0>{});
    store_quadrant(v_c[1][0], number<1>{}, number<0>{});
    store_quadrant(v_c[0][1], number<0>{}, number<1>{});
    store_quadrant(v_c[1][1], number<1>{}, number<1>{});
    if constexpr (T::PERSIST) {
        // The next tile's prologue copies overwrite the staged C: every wave must have finished reading it.
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    };   // run_tile
    if constexpr (T::PERSIST) {
#pragma clang loop unroll(disable)
        for (int tile_id = static_cast<int>(block_id_x()); tile_id < tile_end; tile_id += kargs.grid_blocks) run_tile(tile_id);
    } else if constexpr (T::SPLITK) {
        (void)tile_end;
        run_tile(static_cast<int>(block_id_x()) / split_k);
    } else {
        run_tile(0);
    }
#endif   // __gfx950__
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, 1)
void gemm_a8w8_mxfp8_kernel(opus_gemm_mxfp8_kargs_gfx950 kargs) {
    gemm_a8w8_mxfp8_body<Traits>(kargs);
}
