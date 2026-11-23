// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 (wave32, WMMA, TDM) MLA v4.1 sparse prefill traits.

#pragma once

#include <cstddef>

#include <opus/dtypes.hpp>

#include "../mla_prefill_kargs.h"

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

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_HEAD_SIZE = 512;
    static constexpr int D_SCALE_BLOCK = 32;
    static constexpr int D_SCALE_SIZE = D_HEAD_SIZE / D_SCALE_BLOCK;

    using D_ATTN = D_ATTN_;
    using D_PV   = D_PV_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // QK splits the heads across waves, PV splits the head dim.
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_QK = 128;
    static constexpr int W_K_PV = 32;

    static constexpr int GEMM0_STAGE_N = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_E_K = D_HEAD_SIZE / W_K_QK;

    static constexpr int PV_GROUPS = NUM_WARPS;
    static constexpr int PV_DV = D_HEAD_SIZE / NUM_WARPS;
    static constexpr int PV_DV_TILES = PV_DV / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_PV;

    static constexpr int VEC_KV    = 16;
    static constexpr int VEC_SCALE = D_SCALE_SIZE;
    static constexpr int VEC_PV    = 8;
    static constexpr int VEC_O     = 8;

    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;

    static constexpr int K_ROW_LDS_BYTES = D_HEAD_SIZE * sizeof(D_ATTN) + 16;
    static constexpr int K_ROW_LDS_ELEMS = K_ROW_LDS_BYTES / sizeof(D_ATTN);
    static constexpr int V_ROW_LDS_BYTES = PV_DV * sizeof(D_PV) + 16;
    static constexpr int V_ROW_LDS_ELEMS = V_ROW_LDS_BYTES / sizeof(D_PV);

    // Five 64KB partitions; what the two SIMD pairs read at once sits in different ones. Partition
    // 2 * slot + h: K rows [32h, +32) of the slot and their scales, the V slices {h, h + 2}, the Q
    // tile of wave 2 * slot + h. Partition 4: the P and rescale exchange.
    static constexpr int PARTITION_BYTES  = 64 * 1024;
    static constexpr int NUM_KV_BUFS      = 2;
    static constexpr int ROWS_PER_HALF    = KV_TILE_SIZE / 2;
    static constexpr int K_HALF_BYTES     = ROWS_PER_HALF * K_ROW_LDS_BYTES;
    static constexpr int K_SCALE_OFF      = K_HALF_BYTES;
    static constexpr int K_SCALE_BYTES    = ROWS_PER_HALF * D_SCALE_SIZE;
    static constexpr int V_OFF            = K_SCALE_OFF + K_SCALE_BYTES;
    static constexpr int V_SLICE_BYTES    = KV_TILE_SIZE * V_ROW_LDS_BYTES;
    static constexpr int V_PAIR_BYTES     = 2 * V_SLICE_BYTES;
    static constexpr int Q_OFF            = V_OFF + V_PAIR_BYTES;
    static constexpr int Q_TILE_BYTES     = Q_TILE_SIZE * K_ROW_LDS_BYTES;
    static constexpr int SLOT_BYTES       = 2 * PARTITION_BYTES;
    static_assert(Q_OFF + Q_TILE_BYTES <= PARTITION_BYTES);
    static_assert(NUM_WARPS == 2 * NUM_KV_BUFS);

    // P as [producer wave][kv group][lane][VEC_PV], then the rescale factors as [head][wave].
    static constexpr int XCHG_OFF      = NUM_KV_BUFS * SLOT_BYTES;
    static constexpr int P_CHUNK_BYTES = WARP_SIZE * VEC_PV * sizeof(D_PV);
    static constexpr int P_OFF         = XCHG_OFF;
    static constexpr int P_BYTES       = NUM_WARPS * GEMM0_STAGE_N * P_CHUNK_BYTES;
    static constexpr int FAC_OFF       = P_OFF + P_BYTES;
    static constexpr int FAC_BYTES     = W_M * NUM_WARPS * sizeof(D_ACC);

    static constexpr int O_ROW_LDS_ELEMS = D_HEAD_SIZE + 16 / sizeof(D_OUT);
    static constexpr int O_ROW_LDS_BYTES = O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static_assert(NUM_WARPS * Q_TILE_SIZE * O_ROW_LDS_BYTES <= XCHG_OFF, "O staging reuses partitions 0 and 1");
    static constexpr int PO_ROW_LDS_ELEMS = D_HEAD_SIZE + 16 / sizeof(D_ACC);
    static constexpr int PO_ROW_LDS_BYTES = PO_ROW_LDS_ELEMS * sizeof(D_ACC);
    static_assert(NUM_WARPS * Q_TILE_SIZE * PO_ROW_LDS_BYTES <= XCHG_OFF, "fp32 O staging must not overlap the exchange");

    static constexpr size_t smem_size_bytes() { return (size_t)(FAC_OFF + FAC_BYTES); }
    static_assert(FAC_OFF + FAC_BYTES <= 5 * PARTITION_BYTES);
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = fp8_t,
         typename D_PV_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_v41_prefill_a8w8_16mx1_16nx4_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int D_HEAD_SIZE = 512;
    static constexpr int D_SCALE_BLOCK = 32;
    static constexpr int D_SCALE_SIZE = D_HEAD_SIZE / D_SCALE_BLOCK;

    using D_ATTN = D_ATTN_;
    using D_PV   = D_PV_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // Every wave holds all heads: QK splits the KV rows across waves, PV splits the head dim.
    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_QK = 128;
    static constexpr int W_K_PV = 32;

    static constexpr int ROWS_PER_WAVE = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int GEMM0_E_K = D_HEAD_SIZE / W_K_QK;
    static_assert(Q_TILE_SIZE == W_M && ROWS_PER_WAVE == W_N, "one 16x16 S tile per wave");

    static constexpr int PV_GROUPS = 1;
    static constexpr int PV_DV = D_HEAD_SIZE / NUM_WARPS;
    static constexpr int PV_DV_TILES = PV_DV / W_N;
    static constexpr int GEMM1_E_K = KV_TILE_SIZE / W_K_PV;

    static constexpr int VEC_KV    = 16;
    static constexpr int VEC_SCALE = D_SCALE_SIZE;
    static constexpr int VEC_PV    = 8;
    static constexpr int VEC_O     = 8;

    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;

    static constexpr int K_ROW_LDS_BYTES = D_HEAD_SIZE * sizeof(D_ATTN) + 16;
    static constexpr int K_ROW_LDS_ELEMS = K_ROW_LDS_BYTES / sizeof(D_ATTN);
    static constexpr int V_ROW_LDS_BYTES = PV_DV * sizeof(D_PV) + 16;
    static constexpr int V_ROW_LDS_ELEMS = V_ROW_LDS_BYTES / sizeof(D_PV);

    // K/scale double buffer, one V tile as NUM_WARPS dim slices, P and the row maxes double buffered, Q.
    static constexpr int NUM_K_BUFS    = 2;
    static constexpr int K_SCALE_OFF   = KV_TILE_SIZE * K_ROW_LDS_BYTES;
    static constexpr int K_BUF_BYTES   = K_SCALE_OFF + KV_TILE_SIZE * D_SCALE_SIZE;
    static constexpr int V_OFF         = NUM_K_BUFS * K_BUF_BYTES;
    static constexpr int V_SLICE_BYTES = KV_TILE_SIZE * V_ROW_LDS_BYTES;
    static constexpr int XCHG_OFF      = V_OFF + NUM_WARPS * V_SLICE_BYTES;
    static constexpr int P_CHUNK_BYTES = WARP_SIZE * VEC_PV * sizeof(D_PV);
    static constexpr int MP_OFF        = NUM_WARPS * P_CHUNK_BYTES;            // [head][wave] within a buffer
    static constexpr int XCHG_BYTES    = MP_OFF + W_M * NUM_WARPS * sizeof(D_ACC);
    static constexpr int L_OFF         = XCHG_OFF + 2 * XCHG_BYTES;
    static constexpr int Q_OFF         = L_OFF + W_M * NUM_WARPS * sizeof(D_ACC);
    static constexpr int Q_TILE_BYTES  = Q_TILE_SIZE * K_ROW_LDS_BYTES;
    static constexpr int QO_ROWS_PER_WAVE = Q_TILE_SIZE / NUM_WARPS;

    static constexpr int O_ROW_LDS_ELEMS = D_HEAD_SIZE + 16 / sizeof(D_OUT);
    static constexpr int O_ROW_LDS_BYTES = O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static_assert(Q_TILE_SIZE * O_ROW_LDS_BYTES <= V_OFF, "O staging reuses the K buffers");
    static constexpr int PO_ROW_LDS_ELEMS = D_HEAD_SIZE + 16 / sizeof(D_ACC);
    static constexpr int PO_ROW_LDS_BYTES = PO_ROW_LDS_ELEMS * sizeof(D_ACC);
    static_assert(Q_TILE_SIZE * PO_ROW_LDS_BYTES <= V_OFF, "fp32 O staging reuses the K buffers");

    static constexpr size_t smem_size_bytes() { return (size_t)(Q_OFF + Q_TILE_BYTES); }
    static_assert(Q_OFF + Q_TILE_BYTES <= 160 * 1024, "two workgroups per CU");
};
