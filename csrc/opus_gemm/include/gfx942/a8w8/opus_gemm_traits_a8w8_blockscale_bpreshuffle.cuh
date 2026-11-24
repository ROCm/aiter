// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits and kargs for gfx942 a8w8 blockscale bpreshuffle pipeline.
// Tile/wave mapping is supplied by generated traits. 5-tuple DTYPE with GROUP.
#pragma once

#include "../../opus_gemm_utils.cuh"

template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename GROUP_,
        typename TILE_,
        typename WAVE_>
struct opus_gemm_a8w8_blockscale_bpreshuffle_traits_gfx942 {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using GROUP = opus::remove_cvref_t<GROUP_>;
    using TILE  = opus::remove_cvref_t<TILE_>;
    using WAVE  = opus::remove_cvref_t<WAVE_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A   = opus::tuple_element_t<0, DTYPE>;
    using D_B   = opus::tuple_element_t<1, DTYPE>;
    using D_C   = opus::tuple_element_t<2, DTYPE>;
    using D_ACC = opus::tuple_element_t<3, DTYPE>;
    using D_SF  = opus::tuple_element_t<4, DTYPE>;
    static_assert(std::is_same<D_A, D_B>::value);

    static constexpr int T_M = opus::get<0>(TILE{});
    static constexpr int T_N = opus::get<1>(TILE{});
    static constexpr int T_K = opus::get<2>(TILE{});

    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
    static_assert(T_K == 1);

    static constexpr int W_M = opus::get<0>(WAVE{});
    static constexpr int W_N = opus::get<1>(WAVE{});
    static constexpr int W_K = opus::get<2>(WAVE{});

    static constexpr int HALF_B_M = B_M / 2;
    static constexpr int HALF_B_N = B_N / 2;

    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % (W_K * T_K) == 0);

    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / (W_K * T_K);
    static_assert(E_M == 1 && E_N == 2);

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});
    static_assert(VEC_A == VEC_B);
    static_assert(VEC_A == 16);
    static_assert(VEC_C == 4);

    static constexpr int GROUP_M = opus::get<0>(GROUP{});
    static constexpr int GROUP_N = opus::get<1>(GROUP{});
    static constexpr int GROUP_K = opus::get<2>(GROUP{});
    static_assert(B_K == GROUP_K);

    static constexpr int smem_linear_wave = opus::get_warp_size() * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int threads_k_a = B_K / VEC_A;
    static constexpr int threads_m_per_block = BLOCK_SIZE / threads_k_a;
    static constexpr int smem_m_rows =
        ceil_div_constexpr(HALF_B_M, threads_m_per_block) * threads_m_per_block;
    static constexpr int threads_k_b = B_K / VEC_B;
    static constexpr int threads_n_per_block = BLOCK_SIZE / threads_k_b;
    static constexpr int smem_n_rows =
        ceil_div_constexpr(HALF_B_N, threads_n_per_block) * threads_n_per_block;
    static constexpr int smem_m_rep = ceil_div_constexpr(smem_m_rows, smem_sub);
    static constexpr int smem_n_rep = ceil_div_constexpr(smem_n_rows, smem_sub);

    static constexpr int a_buffer_load_insts = ceil_div_constexpr(HALF_B_M * B_K, BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = ceil_div_constexpr(HALF_B_N * B_K, BLOCK_SIZE * VEC_B);
};

struct opus_gemm_a8w8_blockscale_bpreshuffle_kargs_gfx942 {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    int m;
    int k;
    int stride_a;
    int stride_b;
    int stride_c;

    const void* __restrict__ ptr_sfa;
    const void* __restrict__ ptr_sfb;
    int stride_sfb;
};
