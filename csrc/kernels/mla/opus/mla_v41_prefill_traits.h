#pragma once

#include <cstdint>
#include <type_traits>

#include <opus/dtypes.hpp>

#include "mla_v41_prefill_kargs.h"

using bf16_t = opus::dtypes::bf16;
using fp8_t  = _BitInt(8);

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = fp8_t,
         typename D_PV_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_v41_prefill_a8w8_16mx4_64nx1_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_HEAD_SIZE = 512;
    static constexpr int D_SCALE_BLOCK = 32;
    static constexpr int D_SCALE_SIZE = D_HEAD_SIZE / D_SCALE_BLOCK;

    using D_ATTN = D_ATTN_;
    using D_PV   = D_PV_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_QK = 128;
    static constexpr int W_K_PV = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_HEAD_SIZE / W_K_QK;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_HEAD_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_PV;

    static constexpr int VEC_Q     = 16;
    static constexpr int VEC_KV    = 16;
    static constexpr int VEC_SCALE = 4;
    static constexpr int VEC_DEQ   = 8;
    static constexpr int VEC_TR_V  = 4;
    static constexpr int VEC_O     = 4;

    static constexpr int dwordx4_size = 16;
    static constexpr int D_128B_SIZE = 128 / sizeof(D_ATTN);
    static constexpr int smem_linear_wave = WARP_SIZE * dwordx4_size / sizeof(D_ATTN);
    static constexpr int smem_n_per_wave = smem_linear_wave / D_128B_SIZE;
    static constexpr int smem_n_rpt = KV_TILE_SIZE / smem_n_per_wave;
    static constexpr int smem_d_rpt = D_HEAD_SIZE / D_128B_SIZE;
    static constexpr int smem_padding_32B = 32 / sizeof(D_ATTN);
    static constexpr int smem_brick = smem_linear_wave + smem_padding_32B;
    static constexpr int smem_n_sub_tile = smem_n_per_wave * NUM_WARPS;
    static constexpr int smem_n_sub_tile_rpt = KV_TILE_SIZE / smem_n_sub_tile;
    static constexpr int kv_async_load_insts = smem_n_sub_tile_rpt * smem_d_rpt;
    static constexpr int smem_kv_bytes = smem_n_rpt * smem_d_rpt * smem_brick * sizeof(D_ATTN);

    // Scale rows ride the K lanes (8 per row), so each row takes a 32 B slot holding its 16 B twice.
    static constexpr int smem_scale_row = D_128B_SIZE / VEC_KV * VEC_SCALE;
    static constexpr int smem_scale_padding = 16;
    static constexpr int smem_scale_brick = smem_n_per_wave * smem_scale_row + smem_scale_padding;
    static constexpr int scale_async_load_insts = smem_n_sub_tile_rpt;
    static constexpr int smem_scale_bytes = smem_n_rpt * smem_scale_brick;

    static constexpr int D_128B_V_SIZE = 128 / sizeof(D_PV);
    static constexpr int smem_v_padding = 32 / sizeof(D_PV);
    static constexpr int smem_v_row = D_HEAD_SIZE + smem_v_padding;
    static constexpr int smem_v_bytes = KV_TILE_SIZE * smem_v_row * sizeof(D_PV);

    static constexpr int deq_rows_per_inst = WARP_SIZE * VEC_DEQ / D_128B_SIZE;
    static constexpr int deq_insts = smem_n_sub_tile_rpt * smem_d_rpt * smem_n_per_wave / deq_rows_per_inst;

    static constexpr int smem_p_bytes   = NUM_WARPS * Q_TILE_SIZE * KV_TILE_SIZE * sizeof(D_PV);
    static constexpr int smem_fac_bytes = NUM_WARPS * Q_TILE_SIZE * sizeof(D_ACC);

    static constexpr int smem_bytes() {
        return 2 * smem_kv_bytes + 2 * smem_scale_bytes + smem_v_bytes + smem_p_bytes + smem_fac_bytes;
    }
};

// 16mx1_16nx4: the four waves split the KV tile and share the 16-head Q tile, for H <= 16.
template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = fp8_t,
         typename D_PV_ = bf16_t,
         typename D_OUT_ = bf16_t,
         bool SPLIT_ = false>
struct opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr bool SPLIT = SPLIT_;
    using kargs_type = std::conditional_t<SPLIT_, opus_mla_prefill_split_kargs<opus_mla_v41_prefill_kargs>,
                                          opus_mla_v41_prefill_kargs>;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_HEAD_SIZE = 512;
    static constexpr int D_SCALE_BLOCK = 32;
    static constexpr int D_SCALE_SIZE = D_HEAD_SIZE / D_SCALE_BLOCK;

    using D_ATTN = D_ATTN_;
    using D_PV   = D_PV_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_QK = 128;
    static constexpr int W_K_PV = 32;

    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = KV_TILE_SIZE / (W_N * T_N);
    static constexpr int GEMM0_E_K = D_HEAD_SIZE / W_K_QK;

    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = D_HEAD_SIZE / (W_N * T_N);
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_PV;

    static constexpr int VEC_Q    = 16;
    static constexpr int VEC_KV   = 16;
    static constexpr int VEC_TR_V = 4;
    static constexpr int VEC_O    = 4;

    // bf16 V tile staged row per key.
    static constexpr int SMEM_KV_PAD = 16;
    static constexpr int SMEM_KV_ROW = D_HEAD_SIZE + SMEM_KV_PAD;

    static constexpr int smem_bytes() {
        return KV_TILE_SIZE * SMEM_KV_ROW * sizeof(D_PV) + T_N * W_M * sizeof(D_ACC) + T_N * W_M * W_N * sizeof(D_PV);
    }
};
