#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch
// headers.
#include "aiter_hip_common.h"
#include "fmha_bwd.hpp"
#include <variant>

namespace aiter {

struct mha_bwd_args {
    // aiter args
    int mask_type; // 0: no mask   1: top_left_causal   2: bottom_right_causal   3: sliding_window
    bool use_asm_v3;
    bool v3_atomic_fp32;
    int v3_bf16_cvt;
    bool v3_api_check;

    // From ck fmha_bwd_traits
    int hdim_q;
    int hdim_v;
    std::string data_type;
    bool is_group_mode;
    int ck_mask_type;
    int bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_dbias;
    bool has_dropout;
    bool is_store_randval;
    bool is_deterministic;

    // From ck fmha_bwd_args
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr;
    const void* o_ptr;
    const void* lse_ptr;
    const void* do_ptr;
    void* d_ptr;
    void* rand_val_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* dbias_ptr;
    void* dq_acc_ptr;
    // Usage notes for sequence length pointer parameters:
    //
    // [Note: Define "Group mode" vs "Batch mode" here if possible, e.g., "Group mode handles
    // MQA/GQA..."]
    //
    // With padding:
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical (including padding) sequence
    //     lengths. [array size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr: Records logical (excluding padding) length for each
    //     sequence. [array size: batch]
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqlen_q_ptr (per-sequence) and cu_seqlen_q_ptr (cumulative logical) are mutually
    //     exclusive. Use one set, not both.
    //
    //   Batch mode:
    //     - cu_seqlen_q_ptr/cu_seqlen_k_ptr: Records cumulative logical (excluding padding)
    //     sequence lengths. [array size: batch + 1]
    //     - seqstart_* and seqlen_* pointers must be nullptr.
    //
    // Without padding:
    //   (Note: Physical length equals logical length)
    //
    //   Group mode:
    //     - seqstart_q_ptr, seqstart_k_ptr: Record cumulative physical sequence lengths. [array
    //     size: batch + 1]
    //     - seqlen_q_ptr/seqlen_k_ptr and cu_seqlen_q_ptr/cu_seqlen_k_ptr must be nullptr.
    //
    //   Batch mode:
    //     - All sequence length pointers (seqstart_*, seqlen_*, cu_seqlen_*) must be nullptr.
    //
    const void* seqstart_q_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqstart_k_ptr =
        nullptr; // Cumulative physical sequence length array [batch + 1]. (Used in Group mode)
    const void* seqlen_q_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* seqlen_k_ptr = nullptr;    // Per-sequence logical (excluding padding) length array
                                           // [batch]. (Used in Group mode with padding)
    const void* cu_seqlen_q_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    const void* cu_seqlen_k_ptr = nullptr; // Cumulative logical (excluding padding) sequence length
                                           // array [batch + 1]. (Used with padding)
    int seqlen_q;
    int seqlen_k;
    int batch;
    int max_seqlen_q;
    int max_seqlen_k;
    int nhead_q;
    int nhead_k;
    float scale;
    int stride_q;
    int stride_k;
    int stride_v;
    int stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    int stride_o;
    int stride_randval;
    int stride_do;
    int stride_dq_acc;
    int stride_dq;
    int stride_dk;
    int stride_dv;
    int stride_dbias;
    int nhead_stride_q;
    int nhead_stride_k;
    int nhead_stride_v;
    int nhead_stride_bias;
    int nhead_stride_o;
    int nhead_stride_randval;
    int nhead_stride_do;
    int nhead_stride_lsed;
    int nhead_stride_dq_acc;
    int nhead_stride_dq;
    int nhead_stride_dk;
    int nhead_stride_dv;
    int nhead_stride_dbias;
    int batch_stride_q;
    int batch_stride_k;
    int batch_stride_v;
    int batch_stride_bias;
    int batch_stride_o;
    int batch_stride_randval;
    int batch_stride_do;
    int batch_stride_lsed;
    int batch_stride_dq_acc;
    int batch_stride_dq;
    int batch_stride_dk;
    int batch_stride_dv;
    int batch_stride_dbias;
    int split_stride_dq_acc;
    int window_size_left;
    int window_size_right;
    float p_drop;
    float p_undrop;
    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

struct __attribute__((packed)) fmha_bwd_dqdkdv_args
{
    void* ptr_dq;                   // 0x00: dq or dq_acc
    p2 _p0;
    void* ptr_dk;                   // 0x10
    p2 _p1;
    void* ptr_dv;                   // 0x20
    p2 _p2;
    const void* ptr_q;              // 0x30
    p2 _p3;
    const void* ptr_k;              // 0x40
    p2 _p4;
    const void* ptr_v;              // 0x50
    p2 _p5;
    const void* ptr_do;             // 0x60
    p2 _p6;
    const void* ptr_lse;            // 0x70
    p2 _p7;
    const void* ptr_d;              // 0x80
    p2 _p8;
    float scalar;                   // 0x90
    p3 _p9;
    float log2e;                    // 0xa0
    p3 _p10;
    unsigned int seqlen_q;          // 0xb0: s_seq_len_q
    p3 _p11;
    unsigned int Ts;                // 0xc0: s_Seqs_k*sub_K
    p3 _p12;
    unsigned int Hs_q;              // 0xd0: s_Hs_q
    p3 _p13;
    unsigned int BAs_q;             // 0xe0: s_BAs_q
    p3 _p14;
    unsigned int Seqs_q;            // 0xf0: s_Seqs_q
    p3 _p15;
    unsigned int ratio;             // 0x100
    p3 _p16;
    unsigned int Hs_k;              // 0x110: s_Hs_k
    p3 _p17;
    unsigned int BAs_k;             // 0x120: s_BAs_k
    p3 _p18;
    unsigned int Seqs_k;            // 0x130: s_Seqs_k
    p3 _p19;
    unsigned int Seqs_dk;           // 0x140: s_Seqs_dk
    p3 _p20;
    unsigned int seqlen_k;          // 0x150: batch mode
    p3 _p21;
    unsigned int head_dim_q;        // 0x160: batch&group mode for headdim padding
    p3 _p22;
    unsigned int head_dim_v;        // 0x170: batch&group mode for headdim padding
    p3 _p23;
    unsigned int nhead_q;           // 0x180: batch mode lsed([B,H,S]) addr = batch_idx * nhead_q * seqlen_q * 4 + head_idx * seqlen_q * 4
    p3 _p24;
    unsigned int Hs_v;              // 0x190: batch&group mode
    p3 _p25;
    unsigned int BAs_v;             // 0x1a0: batch mode
    p3 _p26;
    unsigned int Seqs_v;            // 0x1b0: batch&group mode
    p3 _p27;
    unsigned int Hs_do;             // 0x1c0: batch&group mode
    p3 _p28;
    unsigned int BAs_do;            // 0x1d0: group mode
    p3 _p29;
    unsigned int Seqs_do;           // 0x1e0: batch&group mode
    p3 _p30;
    unsigned int Hs_dk;             // 0x1f0: batch&group mode
    p3 _p31;
    unsigned int BAs_dk;            // 0x200: group mode
    p3 _p32;
    unsigned int Hs_dv;             // 0x210: batch&group mode
    p3 _p33;
    unsigned int BAs_dv;            // 0x220: group mode
    p3 _p34;
    unsigned int Seqs_dv;           // 0x230: batch&group mode
    p3 _p35;
    unsigned int Hs_lsed;           // 0x240: group mode lsed([H,TotalValid_Q(90)]) addr = seqstart_q[batch_idx] * 4 + head_idx * nhead_stride_lsed(s_Hs_lsed)
    p3 _p36;
    const void* ptr_qseq;           // 0x250: group mode seqstart_q [0, 20, 50, 90]
    p2 _p37;
    const void* ptr_kseq;           // 0x260: group mode seqstart_k [0, 50, 110, 180]
    p2 _p38;
    const void* ptr_qseq_padded;    // 0x270: group mode seqstart_q_padded [0, 30(20+10), 70(20+10+30+10), 120(20+10+30+10+40+10)] if 10 is padded after each seqlen[30(20+10), 40(30+10), 50(40+10)]
    p2 _p39;
    const void* ptr_kseq_padded;    // 0x280: group mode seqstart_k_padded [0, 60(50+10), 130(50+10+60+10), 200(50+10+60+10+70+10)] if 10 is padded after each seqlen[60(50+10), 70(60+10), 80(70+10)]
    p2 _p40;
    unsigned int max_seqlen_dq;     // 0x290: gorup mode max seqlen q for a16 dq_acc store, padding to 16x
    p3 _p41;
    int mask_x;                     // 0x2a0
    p3 _p42;
    int mask_y;                     // 0x2b0
    p3 _p43;
};

struct __attribute__((packed)) fmha_bwd_odo_args
{
    const void* ptr_o;
    p2 _p0;
    const void* ptr_do;
    p2 _p1;
    void* ptr_d;
    p2 _p2;
    unsigned int Hs_odo;
    p3 _p3;
    unsigned int BAs_odo;
    p3 _p4;
    unsigned int Seqs_odo;
    p3 _p5;
    unsigned int Hs_d;
    p3 _p6;
    unsigned int BAs_d;
    p3 _p7;
    unsigned int Seqs_d;
    p3 _p8;
    unsigned int seqlen_q;
    p3 _p9;
    unsigned int head_dim;
    p3 _p10;
    const void* ptr_qseq;
    p2 _p11;
    const void* ptr_qseq_padded;
    p2 _p12;
};

// dq_shuffle & dq_convert post process kernel args
struct __attribute__((packed)) fmha_bwd_post_kernel_args
{
    void* ptr_dq_acc;
    p2 _p0;
    void* ptr_dq;
    p2 _p1;
    unsigned int Hs_dq_acc;
    p3 _p2;
    unsigned int BAs_dq_acc;
    p3 _p3;
    unsigned int Seqs_dq_acc;
    p3 _p4;
    unsigned int Hs_dq;
    p3 _p5;
    unsigned int BAs_dq;
    p3 _p6;
    unsigned int Seqs_dq;
    p3 _p7;
    unsigned int seqlen_q;
    p3 _p8;
    unsigned int head_dim;
    p3 _p9;
    const void* ptr_qseq;
    p2 _p10;
    const void* ptr_qseq_padded;
    p2 _p11;
};

__attribute__((visibility("default"))) float mha_bwd(mha_bwd_args, const ck_tile::stream_config&);

float fmha_v3_bwd(mha_bwd_args, const ck_tile::stream_config&);

} // namespace aiter
