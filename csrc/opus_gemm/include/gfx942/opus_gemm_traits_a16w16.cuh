// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Traits for a16w16 (bf16) pipelines.
#pragma once

#include "../opus_gemm_utils.cuh"

// Split-barrier a16w16 traits

// Unified gfx942 a16w16 traits.
// LDS_DEPTH_ selects half-tile ping-pong (2) or full-tile single LDS (1).
template<int BLOCK_SIZE_,
        typename BLOCK_,
        typename DTYPE_,
        typename VEC_,
        typename TILE_,
        typename WAVE_,
        int LDS_DEPTH_ = 2>
struct opus_gemm_a16w16_traits {
    using BLOCK = opus::remove_cvref_t<BLOCK_>;
    using DTYPE = opus::remove_cvref_t<DTYPE_>;
    using VEC   = opus::remove_cvref_t<VEC_>;
    using TILE  = opus::remove_cvref_t<TILE_>;
    using WAVE  = opus::remove_cvref_t<WAVE_>;

    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr int LDS_DEPTH = LDS_DEPTH_;

    static constexpr int B_M = opus::get<0>(BLOCK{});
    static constexpr int B_N = opus::get<1>(BLOCK{});
    static constexpr int B_K = opus::get<2>(BLOCK{});

    using D_A   = opus::tuple_element_t<0, DTYPE>;
    using D_B   = opus::tuple_element_t<1, DTYPE>;
    using D_C   = opus::tuple_element_t<2, DTYPE>;
    using D_ACC = opus::tuple_element_t<3, DTYPE>;
    static_assert(std::is_same<D_A, D_B>::value);

    static constexpr int T_M = opus::get<0>(TILE{});
    static constexpr int T_N = opus::get<1>(TILE{});
    static constexpr int T_K = opus::get<2>(TILE{});

    static_assert(BLOCK_SIZE / opus::get_warp_size() == T_M * T_N * T_K);
    // T_K > 1 is used by the wave-K-coop pipeline; splitK/SB paths use T_K=1.

    static constexpr int W_M = opus::get<0>(WAVE{});
    static constexpr int W_N = opus::get<1>(WAVE{});
    static constexpr int W_K = opus::get<2>(WAVE{});

    // Effective per-buffer tile = full tile / LDS_DEPTH.
    static constexpr int HALF_B_M = B_M / LDS_DEPTH;
    static constexpr int HALF_B_N = B_N / LDS_DEPTH;

    static_assert(HALF_B_M % (W_M * T_M) == 0);
    static_assert(HALF_B_N % (W_N * T_N) == 0);
    static_assert(B_K % W_K == 0);

    static constexpr int E_M = HALF_B_M / (W_M * T_M);
    static constexpr int E_N = HALF_B_N / (W_N * T_N);
    static constexpr int E_K = B_K / W_K;

    static constexpr int VEC_A = opus::get<0>(VEC{});
    static constexpr int VEC_B = opus::get<1>(VEC{});
    static constexpr int VEC_C = opus::get<2>(VEC{});

    static_assert(VEC_A == 16 / sizeof(D_A));
    static constexpr int smem_linear_wave = opus::get_warp_size() * VEC_A;
    static constexpr int smem_sub = smem_linear_wave / B_K;
    static constexpr int smem_m_rep = HALF_B_M / smem_sub;
    static constexpr int smem_n_rep = HALF_B_N / smem_sub;

    static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
    static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
    static constexpr int a_ds_read_insts = (E_M * E_K * W_M * W_K) / (opus::get_warp_size() * VEC_A);
    static constexpr int b_ds_read_insts = (E_N * E_K * W_N * W_K) / (opus::get_warp_size() * VEC_B);
};

#ifndef OPUS_GEMM_NOSCALE_KARGS_DEFINED
#define OPUS_GEMM_NOSCALE_KARGS_DEFINED
// Shared kargs struct between a16w16 nosplit launchers (kbuf1_large_tile / kbuf1 / kbuf2v /
// kbuf2v_bk128) and a8w8 noscal...
struct opus_gemm_noscale_kargs {
    const void* __restrict__ ptr_a;
    const void* __restrict__ ptr_b;
    void* __restrict__ ptr_c;
    const void* __restrict__ ptr_bias;
    int m;
    int n;
    int k;
    int batch;
    int stride_a;
    int stride_b;
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_c_batch;
    int stride_bias_batch;
};
#endif

#ifndef OPUS_GEMM_SPLITK_KARGS_GFX942_DEFINED
#define OPUS_GEMM_SPLITK_KARGS_GFX942_DEFINED
#ifndef OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
#define OPUS_GEMM_SPLITK_WS_HANDLE_DEFINED
struct opus_splitk_ws_handle {
    void*         ptr;
    unsigned long bytes;
};
#endif

// Shared kargs for splitK pipelines (splitk / splitk_p1 / splitk_legacy / splitk_p1_bk128).
struct opus_gemm_splitk_kargs {
    const void* __restrict__ ptr_a;         // bf16 [B, M, K]
    const void* __restrict__ ptr_b;         // bf16 [B, N, K] (pre-transposed)
    const opus_splitk_ws_handle* __restrict__ ws_handle; // deref at kernel entry
    void*       __restrict__ ptr_c;         // bf16 [B, M, N] final output (reduce kernel writes)
    const void* __restrict__ ptr_bias;      // unused (reserved)
    int m;
    int n;
    int k;
    int batch;
    int split_k;
    int stride_a;
    int stride_b;
    int stride_ws;          // = padded_N
    int stride_c;
    int stride_a_batch;
    int stride_b_batch;
    int stride_ws_batch;    // = padded_M * padded_N
    int stride_c_batch;
    int stride_bias_batch;
};
#endif
