// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Opus PA decode kernel — shared types, kargs, traits.
//
// `pa_opus_fwd` fuses paged-attention decode (BF16 Q, FP8 KV, GQA8) into one
// sp3-structural HIP kernel:
//   prologue -> core_loop(cl_p) per wave-group -> tail -> R_div_L -> write_out
//
// Target: gfx942 (MI300X) / gfx950 (MI350), MFMA fp8 16×16×32.
#pragma once

#include <cstdint>

#ifdef __HIP_DEVICE_COMPILE__
using bf16_t = __bf16;
#else
using bf16_t = unsigned short;
#endif

// --------------------------------------------------------------------------
// Kernel arguments (matches poc_kl mi300/pa_asm/pa.cpp packed struct).
// Pointer fields are 8 bytes on device; padding mirrors asm kernarg layout.
// --------------------------------------------------------------------------
struct pa_decode_kargs {
    void* ptr_O;           // 0x00  R / output
    uint32_t _pad_o[2];    // 0x08
    void* ptr_Q;           // 0x10
    uint32_t _pad_q[2];    // 0x18
    void* ptr_K;           // 0x20
    uint32_t _pad_k[2];    // 0x28
    void* ptr_V;           // 0x30
    uint32_t _pad_v[2];    // 0x38
    void* ptr_BT;          // 0x40  block table
    uint32_t _pad_bt[2];   // 0x48
    void* ptr_CL;          // 0x50  context lengths
    uint32_t _pad_cl[2];   // 0x58
    void* ptr_KQ;          // 0x60  K quant scales (per-token FP8)
    uint32_t _pad_kq[2];    // 0x68
    void* ptr_VQ;          // 0x70  V quant scales
    uint32_t _pad_vq[2];    // 0x78
    float scale_log2e;     // 0x80  (1/sqrt(head_dim)) * log2(e)
    uint32_t _pad_s[3];    // 0x84
    uint32_t max_blks;     // 0x90  max blocks per sequence
    uint32_t _pad_m[3];    // 0x94
    uint32_t kv_nheads;    // 0xA0
    uint32_t _pad_h[3];    // 0xA4
    uint32_t stride_Q;     // 0xB0  bytes per kv-head slice of Q
    uint32_t _pad_qs[3];   // 0xB4
    uint32_t stride_blk;   // 0xC0  bytes per KV page/block
    uint32_t _pad_bs[3];   // 0xC4
    uint32_t stride_kvhead;// 0xD0  bytes per kv-head within one block
    uint32_t _pad_kvs[3];  // 0xD4
    uint32_t mtp;          // 0xE0
    uint32_t _pad_mtp[3];  // 0xE4
    uint32_t gqa_ratio;    // 0xF0
    uint32_t _pad_gqa[3];  // 0xF4
    void* ptr_QTP;         // 0x100 optional Q token positions (MTP)
    uint32_t _pad_qtp[2];  // 0x108
    void* ptr_DBG;         // 0x118 debug buffer (optional, PA_DEBUG_SCORES)
    uint32_t _pad_dbg[2];  // 0x120
};

// --------------------------------------------------------------------------
// Traits — PA_A16W8_Q8_2TG_4W_16mx1_64nx4 (head=128, block=16, GQA=8)
//
// Grid: (num_kv_heads, batch, 1)  Block: 256 (4 warps × 64)
// MFMA: fp8 16×16×32 on gfx942/gfx950
// --------------------------------------------------------------------------
template<int GQA_RATIO_ = 8,
         int SUB_Q_ = 8,
         int SUB_KV_ = 256,
         int HEAD_DIM_ = 128,
         int BLOCK_SIZE_ = 16,
         int NUM_WARPS_ = 4,
         int NUM_TG_ = 2,
         bool KV_FP8_ = true>
struct pa_a16w8_gqa8_2tg_traits {
    static constexpr int GQA_RATIO = GQA_RATIO_;
    static constexpr int SUB_Q = SUB_Q_;
    static constexpr int SUB_KV = SUB_KV_;
    static constexpr int HEAD_DIM = HEAD_DIM_;
    static constexpr int BLOCK_SIZE = BLOCK_SIZE_;
    static constexpr int NUM_WARPS = NUM_WARPS_;
    static constexpr int NUM_TG = NUM_TG_;
    static constexpr int QUERIES_PER_TG = GQA_RATIO_ / NUM_TG_;
    static constexpr bool KV_FP8 = KV_FP8_;

    static constexpr int WARP_SIZE = 64;
    static constexpr int BLOCK_THREADS = NUM_WARPS * WARP_SIZE;

    using D_Q = bf16_t;
    using D_ACC = float;

    static constexpr int Q_TILE = SUB_Q;
    static constexpr int KV_TILE = SUB_KV;
    static constexpr int MFMA_M = 16;
    static constexpr int MFMA_N = 64;

    static constexpr int Q_LDS_ROWS = 16;
    static constexpr int Q_LDS_ROW_ELEMS = HEAD_DIM_ + 4;

    static constexpr int KV_LOAD_INSTS = 8;
    static constexpr int KV_REG_DWORDS = 32;
    static constexpr int KV_IMM_STRIDE = 1024;

    static constexpr int GEMM0_KV_SLICE = 64;
    static constexpr int GEMM0_K_DIM_MFMA_STEPS = HEAD_DIM_ / 32;
    static constexpr int Q_REG_DWORDS = 8;
    static constexpr int P_REG_DWORDS = 16;
    static constexpr int R_REG_MFMA_SLICES = HEAD_DIM_ / 64;
    static constexpr int S_REG_MFMA_SLICES = SUB_KV_ / 64;
};

using pa_default_traits = pa_a16w8_gqa8_2tg_traits<>;
using pa_opus_default_traits = pa_default_traits;
