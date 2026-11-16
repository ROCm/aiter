// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 (wave32, WMMA, TDM) MLA v4 sparse prefill traits.

#pragma once

#include <cstddef>

#include <opus/dtypes.hpp>

#include "../mla_prefill_kargs.h"

using bf16_t = opus::dtypes::bf16;
using fp8_t  = _BitInt(8);

__host__ __device__ constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 4,
         int CLUSTER_Y_ = 1,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_v4_prefill_a16w16_16mx4_64nx1_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int D_HEAD_SIZE = D_TILE_SIZE;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    
    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int CLUSTER_Y = CLUSTER_Y_;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // Wave grid
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    // WMMA base tile
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;

    // GEMM0: S = Q @ K^T
    static constexpr int GEMM0_STAGE_N = 4;
    static constexpr int GEMM0_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM0_E_N = (KV_TILE_SIZE / GEMM0_STAGE_N) / W_N;
    static constexpr int GEMM0_E_K = D_TILE_SIZE / W_K;

    // GEMM1: O = P @ V
    static constexpr int GEMM1_STAGE_N = 2;
    static constexpr int GEMM1_STAGE_K = 2;
    static constexpr int GEMM1_E_M = Q_TILE_SIZE / W_M;
    static constexpr int GEMM1_E_N = (D_TILE_SIZE / GEMM1_STAGE_N) / W_N;
    static constexpr int GEMM1_E_K = (KV_TILE_SIZE / GEMM1_STAGE_K) / W_K;

    static constexpr int VEC_Q  = 8;
    static constexpr int VEC_KV = 8;
    static constexpr int VEC_O  = 8;

    // ds_load instruction count per GEMM0 / GEMM1 stage
    static constexpr int k_ds_load_insts = (GEMM0_E_N * GEMM0_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);
    static constexpr int v_ds_load_insts = (GEMM1_E_N * GEMM1_E_K * W_N * W_K) / (WARP_SIZE * VEC_KV);

    // TDM gather KV load
    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;                                     // 32-bit gather cap
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;
    static constexpr int KV_ROW_LDS_BYTES   = D_TILE_SIZE * sizeof(D_ATTN) + 16;
    static constexpr int KV_ROW_PAD_SIZE    = 16 / sizeof(D_ATTN);
    static constexpr int KV_ROW_LDS_ELEMS   = D_TILE_SIZE + KV_ROW_PAD_SIZE;

    // LDS: a tile's two row halves sit SEG_BYTES apart, slots pack back to back inside each half
    static constexpr int WAVES_PER_SEG      = 2;
    static constexpr int SEGS_PER_BUF       = NUM_WARPS / WAVES_PER_SEG;
    static constexpr int ROWS_PER_SEG       = WAVES_PER_SEG * ROWS_PER_WAVE;
    static constexpr int WAVE_LDS_BYTES     = ROWS_PER_WAVE * KV_ROW_LDS_BYTES;
    static constexpr int SEG_BYTES          = 160 * 1024;
    static constexpr int KV_BUF_BYTES       = ROWS_PER_SEG * KV_ROW_LDS_BYTES;       // slot stride
    static constexpr int NUM_KV_BUFS        = 4;                                     // QK(t) gathers t+2, so t-1, t, t+1, t+2 coexist
    static constexpr int KV_STAGE_BYTES     = SEG_BYTES + NUM_KV_BUFS * KV_BUF_BYTES;

    static constexpr int Q_ROW_LDS_ELEMS    = D_TILE_SIZE + 16 / sizeof(D_ATTN);
    static constexpr int O_ROW_LDS_ELEMS    = D_TILE_SIZE + 16 / sizeof(D_OUT);
    static constexpr int QO_SEG_BYTES       = 64 * 1024;
    static constexpr int QO_STAGE_BYTES     = NUM_WARPS * QO_SEG_BYTES;

    static constexpr size_t smem_size_bytes() { return (size_t)(KV_STAGE_BYTES > QO_STAGE_BYTES ? KV_STAGE_BYTES : QO_STAGE_BYTES); }
};

template<int Q_TILE_SIZE_ = 32,
         int KV_TILE_SIZE_ = 64,
         int D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_v4_prefill_a16w16_32mx1_16nx4_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int D_HEAD_SIZE = D_TILE_SIZE;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // Every wave holds all heads as M_BLOCKS 16-head blocks; QK splits the KV rows across waves, PV the head dim.
    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;
    static constexpr int M_BLOCKS = Q_TILE_SIZE / W_M;

    static constexpr int ROWS_PER_WAVE = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int GEMM0_E_K     = D_TILE_SIZE / W_K;
    static_assert(M_BLOCKS == 2 && ROWS_PER_WAVE == W_N, "two 16x16 S tiles per wave");

    static constexpr int PV_DV       = D_TILE_SIZE / NUM_WARPS;
    static constexpr int PV_DV_TILES = PV_DV / W_N;
    static constexpr int GEMM1_E_K   = KV_TILE_SIZE / W_K;

    static constexpr int VEC_KV = 8;
    static constexpr int VEC_PV = 8;
    static constexpr int VEC_O  = 8;

    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;
    static constexpr int KV_ROW_LDS_BYTES   = D_TILE_SIZE * sizeof(D_ATTN) + 16;
    static constexpr int KV_ROW_LDS_ELEMS   = KV_ROW_LDS_BYTES / sizeof(D_ATTN);

    // A ring of KV tiles with V^T read in place, so a slot is regathered two tiles after its PV; then P and the row
    // maxes as [m-block][head][wave], double buffered. Q is staged in the last slot, first gathered after the first
    // exchange.
    static constexpr int NUM_KV_BUFS    = 4;
    static constexpr int KV_BUF_BYTES   = KV_TILE_SIZE * KV_ROW_LDS_BYTES;
    static constexpr int XCHG_OFF       = NUM_KV_BUFS * KV_BUF_BYTES;
    static constexpr int P_CHUNK_BYTES  = WARP_SIZE * VEC_PV * sizeof(D_ATTN);
    static constexpr int P_BLOCK_BYTES  = NUM_WARPS * P_CHUNK_BYTES;
    static constexpr int MP_OFF         = M_BLOCKS * P_BLOCK_BYTES;
    static constexpr int ML_BLOCK_BYTES = W_M * NUM_WARPS * sizeof(D_ACC);
    static constexpr int XCHG_BYTES     = MP_OFF + M_BLOCKS * ML_BLOCK_BYTES;
    static constexpr int L_OFF          = XCHG_OFF + 2 * XCHG_BYTES;
    static constexpr int Q_OFF          = (NUM_KV_BUFS - 1) * KV_BUF_BYTES;
    static constexpr int QO_ROWS_PER_WAVE = Q_TILE_SIZE / NUM_WARPS;
    static_assert(Q_TILE_SIZE * KV_ROW_LDS_BYTES <= KV_BUF_BYTES);

    static constexpr int O_ROW_LDS_ELEMS  = D_TILE_SIZE + 16 / sizeof(D_OUT);
    static constexpr int O_ROW_LDS_BYTES  = O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static constexpr int PO_ROW_LDS_ELEMS = D_TILE_SIZE + 16 / sizeof(D_ACC);
    static constexpr int PO_ROW_LDS_BYTES = PO_ROW_LDS_ELEMS * sizeof(D_ACC);
    static_assert(Q_TILE_SIZE * PO_ROW_LDS_BYTES <= XCHG_OFF, "fp32 O staging reuses the KV ring");

    static constexpr size_t smem_size_bytes() { return (size_t)(L_OFF + M_BLOCKS * ML_BLOCK_BYTES); }
    static_assert(L_OFF + M_BLOCKS * ML_BLOCK_BYTES <= 320 * 1024, "gfx1250 LDS is 320KB");
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int D_TILE_SIZE_ = 512,
         int NUM_WARPS_ = 4,
         typename D_ATTN_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_v4_prefill_a16w16_16mx1_16nx4_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int D_TILE_SIZE = D_TILE_SIZE_;
    static constexpr int D_HEAD_SIZE = D_TILE_SIZE;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    using D_ATTN = D_ATTN_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // Every wave holds all heads: QK splits the KV rows across waves, PV the head dim.
    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K = 32;
    static constexpr int M_BLOCKS = Q_TILE_SIZE / W_M;

    static constexpr int ROWS_PER_WAVE = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int GEMM0_E_K     = D_TILE_SIZE / W_K;
    static_assert(M_BLOCKS == 1 && ROWS_PER_WAVE == W_N, "one 16x16 S tile per wave");

    static constexpr int PV_DV       = D_TILE_SIZE / NUM_WARPS;
    static constexpr int PV_DV_TILES = PV_DV / W_N;
    static constexpr int GEMM1_E_K   = KV_TILE_SIZE / W_K;

    static constexpr int VEC_KV = 8;
    static constexpr int VEC_PV = 8;
    static constexpr int VEC_O  = 8;

    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;
    static constexpr int KV_ROW_LDS_BYTES   = D_TILE_SIZE * sizeof(D_ATTN) + 16;
    static constexpr int KV_ROW_LDS_ELEMS   = KV_ROW_LDS_BYTES / sizeof(D_ATTN);

    // KV double buffer with V^T read in place, so a slot is regathered once every wave has read its V^T; then P and the
    // row maxes as [head][wave], read before that release so one buffer does. Q is staged in slot 1, first gathered
    // after the first release.
    static constexpr int NUM_KV_BUFS    = 2;
    static constexpr int KV_BUF_BYTES   = KV_TILE_SIZE * KV_ROW_LDS_BYTES;
    static constexpr int XCHG_OFF       = NUM_KV_BUFS * KV_BUF_BYTES;
    static constexpr int P_CHUNK_BYTES  = WARP_SIZE * VEC_PV * sizeof(D_ATTN);
    static constexpr int P_BLOCK_BYTES  = NUM_WARPS * P_CHUNK_BYTES;
    static constexpr int MP_OFF         = M_BLOCKS * P_BLOCK_BYTES;
    static constexpr int ML_BLOCK_BYTES = W_M * NUM_WARPS * sizeof(D_ACC);
    static constexpr int L_OFF          = XCHG_OFF + MP_OFF + M_BLOCKS * ML_BLOCK_BYTES;
    static constexpr int Q_OFF          = KV_BUF_BYTES;
    static constexpr int QO_ROWS_PER_WAVE = Q_TILE_SIZE / NUM_WARPS;

    static constexpr int O_ROW_LDS_ELEMS  = D_TILE_SIZE + 16 / sizeof(D_OUT);
    static constexpr int O_ROW_LDS_BYTES  = O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static constexpr int PO_ROW_LDS_ELEMS = D_TILE_SIZE + 16 / sizeof(D_ACC);
    static constexpr int PO_ROW_LDS_BYTES = PO_ROW_LDS_ELEMS * sizeof(D_ACC);
    static_assert(Q_TILE_SIZE * PO_ROW_LDS_BYTES <= XCHG_OFF, "fp32 O staging reuses the KV slots");

    static constexpr size_t smem_size_bytes() { return (size_t)(L_OFF + M_BLOCKS * ML_BLOCK_BYTES); }
    static_assert(L_OFF + M_BLOCKS * ML_BLOCK_BYTES <= 160 * 1024, "two workgroups per CU");
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         int CLUSTER_Y_ = 1,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_v4_prefill_a8w8_16mx4_64nx1_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int CLUSTER_Y = CLUSTER_Y_;
    static constexpr int MXSCL_BLOCK_SIZE = 32;

    // Packed DSA hdim split
    static constexpr int D_NOPE_SIZE = 448;
    static constexpr int D_NOPE_PADDED_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_PV   = D_ROPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // QK splits the heads across waves, PV splits the head dim.
    static constexpr int T_M = NUM_WARPS;
    static constexpr int T_N = 1;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;
    static constexpr int W_K_PV = 32;

    static constexpr int GEMM0_STAGE_N  = KV_TILE_SIZE / W_N;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;

    static constexpr int PV_GROUPS   = NUM_WARPS;
    static constexpr int PV_DV       = D_HEAD_SIZE / NUM_WARPS;
    static constexpr int PV_DV_TILES = PV_DV / W_N;
    static constexpr int GEMM1_E_K   = KV_TILE_SIZE / W_K_PV;

    static constexpr int VEC_NOPE  = 16;
    static constexpr int VEC_ROPE  = 8;
    static constexpr int VEC_SCALE = 16;
    static constexpr int VEC_PV    = 8;
    static constexpr int VEC_O     = 8;

    static constexpr int ROWS_PER_WAVE      = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;

    static constexpr int K_NOPE_ROW_LDS_BYTES = D_NOPE_PADDED_SIZE * sizeof(D_NOPE) + 16;
    static constexpr int K_NOPE_ROW_LDS_ELEMS = K_NOPE_ROW_LDS_BYTES / sizeof(D_NOPE);
    static constexpr int K_ROPE_ROW_LDS_BYTES = D_ROPE_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int K_ROPE_ROW_LDS_ELEMS = K_ROPE_ROW_LDS_BYTES / sizeof(D_ROPE);
    static constexpr int V_ROW_LDS_BYTES      = PV_DV * sizeof(D_PV) + 16;
    static constexpr int V_ROW_LDS_ELEMS      = V_ROW_LDS_BYTES / sizeof(D_PV);

    // Five 64KB partitions; what the two SIMD pairs read at once sits in different ones. Partition
    // 2 * slot + h: NoPE and RoPE K rows [32h, +32) of the slot, the V slices {h, h + 2}. Partition 4: the P and
    // rescale exchange, then the Q tiles, which are read once.
    static constexpr int PARTITION_BYTES = 64 * 1024;
    static constexpr int NUM_KV_BUFS     = 2;
    static constexpr int ROWS_PER_HALF   = KV_TILE_SIZE / 2;
    static constexpr int K_ROPE_OFF      = ROWS_PER_HALF * K_NOPE_ROW_LDS_BYTES;
    static constexpr int V_OFF           = K_ROPE_OFF + ROWS_PER_HALF * K_ROPE_ROW_LDS_BYTES;
    static constexpr int V_SLICE_BYTES   = KV_TILE_SIZE * V_ROW_LDS_BYTES;
    static constexpr int SLOT_BYTES      = 2 * PARTITION_BYTES;
    static_assert(V_OFF + 2 * V_SLICE_BYTES <= PARTITION_BYTES);
    static_assert(NUM_WARPS == 2 * NUM_KV_BUFS);

    // P as [producer wave][kv group][lane][VEC_PV], then the rescale factors as [head][wave].
    static constexpr int XCHG_OFF      = NUM_KV_BUFS * SLOT_BYTES;
    static constexpr int P_CHUNK_BYTES = WARP_SIZE * VEC_PV * sizeof(D_PV);
    static constexpr int P_OFF         = XCHG_OFF;
    static constexpr int P_BYTES       = NUM_WARPS * GEMM0_STAGE_N * P_CHUNK_BYTES;
    static constexpr int FAC_OFF       = P_OFF + P_BYTES;
    static constexpr int FAC_BYTES     = W_M * NUM_WARPS * sizeof(D_ACC);

    static constexpr int Q_OFF          = (FAC_OFF + FAC_BYTES + 255) / 256 * 256;
    static constexpr int Q_ROPE_OFF     = Q_TILE_SIZE * K_NOPE_ROW_LDS_BYTES;
    static constexpr int Q_WAVE_BYTES   = Q_ROPE_OFF + Q_TILE_SIZE * K_ROPE_ROW_LDS_BYTES;

    static constexpr int O_ROW_LDS_ELEMS  = D_HEAD_SIZE + 16 / sizeof(D_OUT);
    static constexpr int O_ROW_LDS_BYTES  = O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static constexpr int PO_ROW_LDS_ELEMS = D_HEAD_SIZE + 16 / sizeof(D_ACC);
    static constexpr int PO_ROW_LDS_BYTES = PO_ROW_LDS_ELEMS * sizeof(D_ACC);
    static_assert(NUM_WARPS * Q_TILE_SIZE * PO_ROW_LDS_BYTES <= XCHG_OFF, "fp32 O staging must not overlap the exchange");

    static constexpr size_t smem_size_bytes() { return (size_t)(Q_OFF + NUM_WARPS * Q_WAVE_BYTES); }
    static_assert(Q_OFF + NUM_WARPS * Q_WAVE_BYTES <= 5 * PARTITION_BYTES, "gfx1250 LDS is 320KB");
};

template<int Q_TILE_SIZE_ = 32,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_v4_prefill_a8w8_32mx1_16nx4_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int MXSCL_BLOCK_SIZE = 32;

    static constexpr int D_NOPE_SIZE = 448;
    static constexpr int D_NOPE_PADDED_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_PV   = D_ROPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // Every wave holds all heads as M_BLOCKS 16-head blocks; QK splits the KV rows across waves, PV the head dim.
    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;
    static constexpr int W_K_PV = 32;
    static constexpr int M_BLOCKS   = Q_TILE_SIZE / W_M;
    static constexpr int WARP_SIZE  = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;

    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int ROWS_PER_WAVE  = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;
    static_assert(M_BLOCKS == 2 && ROWS_PER_WAVE == W_N, "two 16x16 S tiles per wave");

    static constexpr int PV_DV       = D_HEAD_SIZE / NUM_WARPS;
    static constexpr int PV_DV_TILES = PV_DV / W_N;
    static constexpr int GEMM1_E_K   = KV_TILE_SIZE / W_K_PV;

    static constexpr int VEC_NOPE  = 16;
    static constexpr int VEC_ROPE  = 8;
    static constexpr int VEC_SCALE = 16;
    static constexpr int VEC_PV    = 8;
    static constexpr int VEC_O     = 8;

    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;

    static constexpr int K_NOPE_ROW_LDS_BYTES = D_NOPE_PADDED_SIZE * sizeof(D_NOPE) + 16;
    static constexpr int K_NOPE_ROW_LDS_ELEMS = K_NOPE_ROW_LDS_BYTES / sizeof(D_NOPE);
    static constexpr int K_ROPE_ROW_LDS_BYTES = D_ROPE_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int K_ROPE_ROW_LDS_ELEMS = K_ROPE_ROW_LDS_BYTES / sizeof(D_ROPE);
    static constexpr int V_ROW_LDS_BYTES      = PV_DV * sizeof(D_PV) + 16;
    static constexpr int V_ROW_LDS_ELEMS      = V_ROW_LDS_BYTES / sizeof(D_PV);

    // K double buffer, one V tile as NUM_WARPS dim slices, then P and the row maxes as [m-block][head][wave], read
    // before the V tile is released so one buffer does. Q is staged in the V tile, read before its first write.
    static constexpr int NUM_K_BUFS    = 2;
    static constexpr int K_ROPE_OFF    = KV_TILE_SIZE * K_NOPE_ROW_LDS_BYTES;
    static constexpr int K_BUF_BYTES   = K_ROPE_OFF + KV_TILE_SIZE * K_ROPE_ROW_LDS_BYTES;
    static constexpr int V_OFF         = NUM_K_BUFS * K_BUF_BYTES;
    static constexpr int V_SLICE_BYTES = KV_TILE_SIZE * V_ROW_LDS_BYTES;
    static constexpr int XCHG_OFF      = V_OFF + NUM_WARPS * V_SLICE_BYTES;
    static constexpr int P_CHUNK_BYTES = WARP_SIZE * VEC_PV * sizeof(D_PV);
    static constexpr int P_BLOCK_BYTES = NUM_WARPS * P_CHUNK_BYTES;
    static constexpr int MP_OFF        = M_BLOCKS * P_BLOCK_BYTES;
    static constexpr int ML_BLOCK_BYTES = W_M * NUM_WARPS * sizeof(D_ACC);
    static constexpr int L_OFF         = XCHG_OFF + MP_OFF + M_BLOCKS * ML_BLOCK_BYTES;
    static constexpr int Q_OFF         = V_OFF;
    static constexpr int Q_ROPE_OFF    = Q_TILE_SIZE * K_NOPE_ROW_LDS_BYTES;
    static constexpr int QO_ROWS_PER_WAVE = Q_TILE_SIZE / NUM_WARPS;
    static_assert(Q_ROPE_OFF + Q_TILE_SIZE * K_ROPE_ROW_LDS_BYTES <= NUM_WARPS * V_SLICE_BYTES);

    static constexpr int O_ROW_LDS_ELEMS  = D_HEAD_SIZE + 16 / sizeof(D_OUT);
    static constexpr int O_ROW_LDS_BYTES  = O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static_assert(Q_TILE_SIZE * O_ROW_LDS_BYTES <= V_OFF, "O staging reuses the K buffers");
    static constexpr int PO_ROW_LDS_ELEMS = D_HEAD_SIZE + 16 / sizeof(D_ACC);
    static constexpr int PO_ROW_LDS_BYTES = PO_ROW_LDS_ELEMS * sizeof(D_ACC);
    static_assert(Q_TILE_SIZE * PO_ROW_LDS_BYTES <= V_OFF, "fp32 O staging reuses the K buffers");

    static constexpr size_t smem_size_bytes() { return (size_t)(L_OFF + M_BLOCKS * ML_BLOCK_BYTES); }
    static_assert(L_OFF + M_BLOCKS * ML_BLOCK_BYTES <= 160 * 1024, "two workgroups per CU");
};

template<int Q_TILE_SIZE_ = 16,
         int KV_TILE_SIZE_ = 64,
         int NUM_WARPS_ = 4,
         typename D_NOPE_ = fp8_t,
         typename D_ROPE_ = bf16_t,
         typename D_OUT_ = bf16_t>
struct opus_mla_v4_prefill_a8w8_16mx1_16nx4_traits {
    static constexpr int Q_TILE_SIZE = Q_TILE_SIZE_;
    static constexpr int KV_TILE_SIZE = KV_TILE_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;

    static constexpr int WARP_SIZE = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS * WARP_SIZE;
    static constexpr int MXSCL_BLOCK_SIZE = 32;

    static constexpr int D_NOPE_SIZE = 448;
    static constexpr int D_NOPE_PADDED_SIZE = 512;
    static constexpr int D_ROPE_SIZE = 64;
    static constexpr int D_HEAD_SIZE = D_NOPE_SIZE + D_ROPE_SIZE;

    using D_NOPE = D_NOPE_;
    using D_ROPE = D_ROPE_;
    using D_ATTN = D_NOPE_;
    using D_PV   = D_ROPE_;
    using D_OUT  = D_OUT_;
    using D_ACC  = float;

    // Every wave holds all heads: QK splits the KV rows across waves, PV the head dim.
    static constexpr int T_M = 1;
    static constexpr int T_N = NUM_WARPS;
    static constexpr int T_K = 1;

    static constexpr int W_M = 16;
    static constexpr int W_N = 16;
    static constexpr int W_K_NOPE = 128;
    static constexpr int W_K_ROPE = 32;
    static constexpr int W_K_PV = 32;

    static constexpr int ROWS_PER_WAVE  = KV_TILE_SIZE / NUM_WARPS;
    static constexpr int GEMM0_NOPE_E_K = D_NOPE_PADDED_SIZE / W_K_NOPE;
    static constexpr int GEMM0_ROPE_E_K = D_ROPE_SIZE / W_K_ROPE;
    static_assert(Q_TILE_SIZE == W_M && ROWS_PER_WAVE == W_N, "one 16x16 S tile per wave");

    static constexpr int PV_DV       = D_HEAD_SIZE / NUM_WARPS;
    static constexpr int PV_DV_TILES = PV_DV / W_N;
    static constexpr int GEMM1_E_K   = KV_TILE_SIZE / W_K_PV;

    static constexpr int VEC_NOPE  = 16;
    static constexpr int VEC_ROPE  = 8;
    static constexpr int VEC_SCALE = 16;
    static constexpr int VEC_PV    = 8;
    static constexpr int VEC_O     = 8;

    static constexpr int INDICES_PER_TDM    = 8;
    static constexpr int TDM_LOADS_PER_WAVE = ROWS_PER_WAVE / INDICES_PER_TDM;

    static constexpr int K_NOPE_ROW_LDS_BYTES = D_NOPE_PADDED_SIZE * sizeof(D_NOPE) + 16;
    static constexpr int K_NOPE_ROW_LDS_ELEMS = K_NOPE_ROW_LDS_BYTES / sizeof(D_NOPE);
    static constexpr int K_ROPE_ROW_LDS_BYTES = D_ROPE_SIZE * sizeof(D_ROPE) + 16;
    static constexpr int K_ROPE_ROW_LDS_ELEMS = K_ROPE_ROW_LDS_BYTES / sizeof(D_ROPE);
    static constexpr int V_ROW_LDS_BYTES      = PV_DV * sizeof(D_PV) + 16;
    static constexpr int V_ROW_LDS_ELEMS      = V_ROW_LDS_BYTES / sizeof(D_PV);

    // K double buffer, one V tile as NUM_WARPS dim slices, P and the row maxes double buffered. Q is staged in the V
    // tile, read before its first write.
    static constexpr int NUM_K_BUFS    = 2;
    static constexpr int K_ROPE_OFF    = KV_TILE_SIZE * K_NOPE_ROW_LDS_BYTES;
    static constexpr int K_BUF_BYTES   = K_ROPE_OFF + KV_TILE_SIZE * K_ROPE_ROW_LDS_BYTES;
    static constexpr int V_OFF         = NUM_K_BUFS * K_BUF_BYTES;
    static constexpr int V_SLICE_BYTES = KV_TILE_SIZE * V_ROW_LDS_BYTES;
    static constexpr int XCHG_OFF      = V_OFF + NUM_WARPS * V_SLICE_BYTES;
    static constexpr int P_CHUNK_BYTES = WARP_SIZE * VEC_PV * sizeof(D_PV);
    static constexpr int MP_OFF        = NUM_WARPS * P_CHUNK_BYTES;            // [head][wave] within a buffer
    static constexpr int XCHG_BYTES    = MP_OFF + W_M * NUM_WARPS * sizeof(D_ACC);
    static constexpr int L_OFF         = XCHG_OFF + 2 * XCHG_BYTES;
    static constexpr int Q_OFF         = V_OFF;
    static constexpr int Q_ROPE_OFF    = Q_TILE_SIZE * K_NOPE_ROW_LDS_BYTES;
    static constexpr int QO_ROWS_PER_WAVE = Q_TILE_SIZE / NUM_WARPS;
    static_assert(Q_ROPE_OFF + Q_TILE_SIZE * K_ROPE_ROW_LDS_BYTES <= NUM_WARPS * V_SLICE_BYTES);

    static constexpr int O_ROW_LDS_ELEMS  = D_HEAD_SIZE + 16 / sizeof(D_OUT);
    static constexpr int O_ROW_LDS_BYTES  = O_ROW_LDS_ELEMS * sizeof(D_OUT);
    static_assert(Q_TILE_SIZE * O_ROW_LDS_BYTES <= V_OFF, "O staging reuses the K buffers");
    static constexpr int PO_ROW_LDS_ELEMS = D_HEAD_SIZE + 16 / sizeof(D_ACC);
    static constexpr int PO_ROW_LDS_BYTES = PO_ROW_LDS_ELEMS * sizeof(D_ACC);
    static_assert(Q_TILE_SIZE * PO_ROW_LDS_BYTES <= V_OFF, "fp32 O staging reuses the K buffers");

    static constexpr size_t smem_size_bytes() { return (size_t)(L_OFF + W_M * NUM_WARPS * sizeof(D_ACC)); }
    static_assert(L_OFF + W_M * NUM_WARPS * sizeof(D_ACC) <= 160 * 1024, "two workgroups per CU");
};
