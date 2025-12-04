#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch
// headers.
#include "aiter_hip_common.h"
#include "fmha_bwd.hpp"
#include "mask.hpp"

namespace aiter {
struct mha_bwd_traits : public fmha_bwd_traits
{
    mha_bwd_traits(int head_size_q,
                   int head_size_v,
                   std::string dtype,
                   bool is_group_mode,
                   mask_enum mask_type,
                   bias_enum bias_type,
                   bool has_dbias,
                   bool has_dropout,
                   bool is_store_randval,
                   bool deterministic,
                   bool use_ext_asm,
                   bool is_v3_atomic_fp32,
                   int how_v3_bf16_cvt)
        : fmha_bwd_traits{head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          mask_type,
                          bias_type,
                          has_dbias,
                          has_dropout,
                          is_store_randval,
                          deterministic},
          use_ext_asm(use_ext_asm),
          is_v3_atomic_fp32(is_v3_atomic_fp32),
          how_v3_bf16_cvt(how_v3_bf16_cvt)
    {
    }
    bool use_ext_asm;
    bool is_v3_atomic_fp32;
    int how_v3_bf16_cvt;
};

using mha_bwd_args = fmha_bwd_args;

// FIXME: use aiter mha_args
__attribute__((visibility("default"))) float mha_bwd(mha_bwd_args args,
                                                     const ck_tile::stream_config& stream_config,
                                                     std::string q_dtype_str,
                                                     bool is_group_mode,
                                                     mask_enum mask_type,
                                                     bias_enum bias_type,
                                                     bool has_dbias,
                                                     bool is_store_randval,
                                                     bool deterministic,
                                                     bool use_ext_asm,
                                                     bool is_v3_atomic_fp32,
                                                     int how_v3_bf16_cvt,
                                                     const void* seqlen_q_padded = nullptr,
                                                     const void* seqlen_k_padded = nullptr,
                                                     bool is_v3_api_check        = false);

struct __attribute__((packed)) fmha_bwd_v3_args
{
    void* ptr_dq;
    p2 _p0;
    void* ptr_dk;
    p2 _p1;
    void* ptr_dv;
    p2 _p2;
    const void* ptr_q;
    p2 _p3;
    const void* ptr_k;
    p2 _p4;
    const void* ptr_v;
    p2 _p5;
    const void* ptr_do;
    p2 _p6;
    const void* ptr_lse;
    p2 _p7;
    const void* ptr_d;
    p2 _p8;
    float scalar;
    p3 _p9;
    float log2e;
    p3 _p10;
    unsigned int seq_len;
    p3 _p11;
    unsigned int Ts;
    p3 _p12;
    unsigned int Hs;
    p3 _p13;
    unsigned int BAs;
    p3 _p14;
    unsigned int Seqs;
    p3 _p15;
    unsigned int ratio;
    p3 _p16;
    unsigned int Hs_kv;
    p3 _p17;
    unsigned int BAs_kv;
    p3 _p18;
    unsigned int Seqs_kv;
    p3 _p19;
    unsigned int Seqs_dkv;
    p3 _p20;
};

struct __attribute__((packed)) fmha_bwd_v3_gen_args
{
    void* ptr_dq;
    p2 _p0;
    void* ptr_dk;
    p2 _p1;
    void* ptr_dv;
    p2 _p2;
    const void* ptr_q;
    p2 _p3;
    const void* ptr_k;
    p2 _p4;
    const void* ptr_v;
    p2 _p5;
    const void* ptr_do;
    p2 _p6;
    const void* ptr_lse;
    p2 _p7;
    const void* ptr_d;
    p2 _p8;
    float scalar;
    p3 _p9;
    float log2e;
    p3 _p10;
    unsigned int seq_len;
    p3 _p11;
    unsigned int Ts;
    p3 _p12;
    unsigned int Hs;
    p3 _p13;
    unsigned int BAs;
    p3 _p14;
    unsigned int Seqs;
    p3 _p15;
    unsigned int ratio;
    p3 _p16;
    unsigned int Hs_kv;
    p3 _p17;
    unsigned int BAs_kv;
    p3 _p18;
    unsigned int Seqs_kv;
    p3 _p19;
    unsigned int Seqs_dkv;
    p3 _p20;
    unsigned int head_dim;
    p3 _p21;
};

struct __attribute__((packed)) fmha_bwd_v3_genl_args
{
    void* ptr_dq;
    void* ptr_dk;
    void* ptr_dv;
    const void* ptr_q;
    const void* ptr_k;
    const void* ptr_v;
    const void* ptr_do;
    const void* ptr_lse;
    const void* ptr_d;
    float scalar;
    p1 _p0;
    float log2e;
    p1 _p1;
    unsigned int ratio;
    p1 _p2;
    unsigned int seqlen_q;
    p1 _p3;
    unsigned int seqlen_k;
    p1 _p4;
    unsigned int head_dim;
    p1 _p5;
    unsigned int nhead_q;
    p1 _p6;
    unsigned int Hs_q;
    p1 _p7;
    unsigned int BAs_q;
    p1 _p8;
    unsigned int Seqs_q;
    p1 _p9;
    unsigned int Hs_k;
    p1 _p10;
    unsigned int BAs_k;
    p1 _p11;
    unsigned int Seqs_k;
    p1 _p12;
    unsigned int Hs_v;
    p1 _p13;
    unsigned int BAs_v;
    p1 _p14;
    unsigned int Seqs_v;
    p1 _p15;
    unsigned int Hs_do;
    p1 _p16;
    unsigned int BAs_do;
    p1 _p17;
    unsigned int Seqs_do;
    p1 _p18;
    unsigned int Hs_dk;
    p1 _p19;
    unsigned int BAs_dk;
    p1 _p20;
    unsigned int Seqs_dk;
    p1 _p21;
    unsigned int Hs_dv;
    p1 _p22;
    unsigned int BAs_dv;
    p1 _p23;
    unsigned int Seqs_dv;
    p1 _p24;
};

struct __attribute__((packed)) fmha_bwd_v3_group_args
{
    void* ptr_dq;
    void* ptr_dk;
    void* ptr_dv;
    const void* ptr_q;
    const void* ptr_k;
    const void* ptr_v;
    const void* ptr_do;
    const void* ptr_lse;
    const void* ptr_d;
    const void* ptr_qseq;
    const void* ptr_kseq;
    const void* ptr_qseq_padded;
    const void* ptr_kseq_padded;
    float scalar;
    p1 _p0;
    float log2e;
    p1 _p1;
    unsigned int ratio;
    p1 _p2;
    unsigned int Hs_lsed;
    p1 _p3;
    unsigned int seqlen_k; // total length of k sequences
    p1 _p4;
    unsigned int Hs_q;
    p1 _p5;
    unsigned int Seqs_q;
    p1 _p6;
    unsigned int Hs_k;
    p1 _p7;
    unsigned int Seqs_k;
    p1 _p8;
    unsigned int Hs_v;
    p1 _p9;
    unsigned int Seqs_v;
    p1 _p10;
    unsigned int Hs_do;
    p1 _p11;
    unsigned int Seqs_do;
    p1 _p12;
    unsigned int Hs_dk;
    p1 _p13;
    unsigned int Seqs_dk;
    p1 _p14;
    unsigned int Hs_dv;
    p1 _p15;
    unsigned int Seqs_dv;
    p1 _p16;
    unsigned int head_dim;
    p1 _p17;
};

struct __attribute__((packed)) fmha_bwd_v3_swa_genl_args
{
    void* ptr_dq;
    void* ptr_dk;
    void* ptr_dv;
    const void* ptr_q;
    const void* ptr_k;
    const void* ptr_v;
    const void* ptr_do;
    const void* ptr_lse;
    const void* ptr_d;
    float scalar;
    p1 _p0;
    float log2e;
    p1 _p1;
    unsigned int ratio;
    p1 _p2;
    unsigned int seqlen_q;
    p1 _p3;
    unsigned int seqlen_k;
    p1 _p4;
    unsigned int head_dim;
    p1 _p5;
    unsigned int nhead_q;
    p1 _p6;
    unsigned int Hs_q;
    p1 _p7;
    unsigned int BAs_q;
    p1 _p8;
    unsigned int Seqs_q;
    p1 _p9;
    unsigned int Hs_k;
    p1 _p10;
    unsigned int BAs_k;
    p1 _p11;
    unsigned int Seqs_k;
    p1 _p12;
    unsigned int Hs_v;
    p1 _p13;
    unsigned int BAs_v;
    p1 _p14;
    unsigned int Seqs_v;
    p1 _p15;
    unsigned int Hs_do;
    p1 _p16;
    unsigned int BAs_do;
    p1 _p17;
    unsigned int Seqs_do;
    p1 _p18;
    unsigned int Hs_dk;
    p1 _p19;
    unsigned int BAs_dk;
    p1 _p20;
    unsigned int Seqs_dk;
    p1 _p21;
    unsigned int Hs_dv;
    p1 _p22;
    unsigned int BAs_dv;
    p1 _p23;
    unsigned int Seqs_dv;
    p1 _p24;
    int mask_x;
    p1 _p25;
    int mask_y;
    p1 _p26;
};

struct __attribute__((packed)) fmha_bwd_v3_args_universal
{
    void *ptr_dq;                   // 0x00: dq or dq_acc
    p2 _p0;
    void *ptr_dk;                   // 0x10
    p2 _p1;
    void *ptr_dv;                   // 0x20
    p2 _p2;
    const void *ptr_q;              // 0x30
    p2 _p3;
    const void *ptr_k;              // 0x40
    p2 _p4;
    const void *ptr_v;              // 0x50
    p2 _p5;
    const void *ptr_do;             // 0x60
    p2 _p6;
    const void *ptr_lse;            // 0x70
    p2 _p7;
    const void *ptr_d;              // 0x80
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
    const void *ptr_qseq;                 // 0x250: group mode seqstart_q [0, 20, 50, 90]
    p2 _p37;
    const void *ptr_kseq;                 // 0x260: group mode seqstart_k [0, 50, 110, 180]
    p2 _p38;
    const void *ptr_qseq_padded;          // 0x270: group mode seqstart_q_padded [0, 30(20+10), 70(20+10+30+10), 120(20+10+30+10+40+10)] if 10 is padded after each seqlen[30(20+10), 40(30+10), 50(40+10)]
    p2 _p39;
    const void *ptr_kseq_padded;          // 0x280: group mode seqstart_k_padded [0, 60(50+10), 130(50+10+60+10), 200(50+10+60+10+70+10)] if 10 is padded after each seqlen[60(50+10), 70(60+10), 80(70+10)]
    p2 _p40;
    unsigned int max_seqlen_dq;    // 0x290: gorup mode max seqlen q for a16 dq_acc store, padding to 16x
    p3 _p41;
    int mask_x;                     // 0x2a0
    p3 _p42;
    int mask_y;                     // 0x2b0
    p3 _p43;
};

struct __attribute__((packed)) fmha_bwd_odo_args
{
    const void *ptr_o;
    p2 _p0;
    const void *ptr_do;
    p2 _p1;
    void *ptr_d;
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
    const void *ptr_qseq;
    p2 _p11;
    const void *ptr_qseq_padded;
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

struct fmha_bwd_v3_traits
{
    int b;
    int h;
    int sq;
    int sk;
    int d;

    int mask;
    int ts_qo;
    int ts_kv;

    int ts_pre_kernel = 128;
    int ts_post_kernel = 64;
};

template <ck_tile::index_t HDim_q_,
          ck_tile::index_t HDim_v_,
          typename DataType_,
          int mask_type_,
          bool kIsAtomic32_,
          ck_tile::index_t BF16Cvt_,
          bool kIsSEQPad_,
          bool kIsHDPad_,
          bool kIsGroupMode_,
          GPUArch GPUArch_>
struct fmha_bwd_dq_dk_dv_v3_traits_
{
    static constexpr ck_tile::index_t HDim_q  = HDim_q_;
    static constexpr ck_tile::index_t HDim_v  = HDim_v_;
    using DataType                            = ck_tile::remove_cvref_t<DataType_>;
    static constexpr int mask_type            = mask_type_;
    static constexpr bool kIsAtomic32         = kIsAtomic32_;
    static constexpr ck_tile::index_t BF16Cvt = BF16Cvt_;
    static constexpr bool kIsSEQPad           = kIsSEQPad_;
    static constexpr bool kIsHDPad            = kIsHDPad_;
    static constexpr bool kIsGroupMode        = kIsGroupMode_;
};

template <typename fmha_bwd_dq_dk_dv_v3_traits_>
struct FmhaBwdV3Name;
template <typename fmha_bwd_dq_dk_dv_v3_traits_>
struct FmhaBwdV3Buf;
template <typename fmha_bwd_dq_dk_dv_v3_traits_>
struct FmhaBwdV3Ts;

namespace gfx942 {
float fmha_bwd_v3(mha_bwd_traits t,
                  mha_bwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqlen_q_padded = nullptr,
                  const void* seqlen_k_padded = nullptr,
                  bool is_v3_api_check        = false);
}

namespace gfx950 {
float fmha_bwd_v3(mha_bwd_traits t,
                  mha_bwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqlen_q_padded = nullptr,
                  const void* seqlen_k_padded = nullptr,
                  bool is_v3_api_check        = false);
}
} // namespace aiter
