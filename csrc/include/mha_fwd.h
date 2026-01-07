#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include "aiter_hip_common.h"
#include "fmha_fwd.hpp"
#include "mask.hpp"

namespace aiter {
struct mha_fwd_traits : public fmha_fwd_traits
{
    mha_fwd_traits(int head_size_q,
                   int head_size_v,
                   std::string dtype,
                   bool is_group_mode,
                   bool has_logits_soft_cap,
                   mask_enum mask_type,
                   bias_enum bias_type,
                   bool has_lse,
                   bool has_dropout,
                   quant_scale_enum qscale_type,
                   bool use_ext_asm,
                   int how_v3_bf16_cvt,
                   bool skip_min_seqlen_q,
                   bool has_sink)
        : fmha_fwd_traits{head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          true, // is_v_rowmajor
                          has_logits_soft_cap,
                          mask_type,
                          bias_type,
                          has_lse,
                          has_dropout,
                          qscale_type,
                          skip_min_seqlen_q,
                          has_sink},
          use_ext_asm(use_ext_asm),
          how_v3_bf16_cvt(how_v3_bf16_cvt)
    {
    }
    bool use_ext_asm;
    int how_v3_bf16_cvt;
};

struct mha_fwd_splitkv_traits : public fmha_fwd_splitkv_traits
{
    mha_fwd_splitkv_traits(int head_size_q,
                           int head_size_v,
                           std::string dtype,
                           bool is_group_mode,
                           bool has_logits_soft_cap,
                           mask_enum mask_type,
                           bias_enum bias_type,
                           bool has_lse,
                           bool has_sink)
        : fmha_fwd_splitkv_traits{head_size_q,
                                  head_size_v,
                                  dtype,
                                  is_group_mode,
                                  true, // is_v_rowmajor
                                  has_logits_soft_cap,
                                  mask_type,
                                  bias_type,
                                  has_lse,
                                  false, // do_fp8_static_quant
                                  has_sink}
    {
    }
};

struct mha_fwd_args
{
    // aiter
    bool use_asm_v3;
    bool v3_api_check;
    int how_v3_bf16_cvt;

    // from ck fmha_fwd_traits
    std::string data_type;
    bool is_group_mode;
    int bias_type; // 0:no bias, 1:elementwise bias, 2:alibi. sync with BlockAttentionBiasEnum
    bool has_lse;
    int qscale_type;
    bool has_sink = false;

    // from ck fmha_fwd_args
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* q_descale_ptr;
    const void* k_descale_ptr;
    const void* v_descale_ptr;
    void* rand_val_ptr;
    void* lse_ptr;
    void* o_ptr;

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

    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;

    float scale_s;
    float logits_soft_cap;

    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_o;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_lse;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_lse;
    ck_tile::index_t batch_stride_o;

    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t sink_size;
    ck_tile::index_t
        mask_type; // 0: no mask   1: top_left_causal   2: bottom_right_causal   3: window_generic
    ck_tile::index_t min_seqlen_q;

    float p_drop;
    bool s_randval;

    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

using mha_fwd_splitkv_args   = fmha_fwd_splitkv_args;
using mha_batch_prefill_args = fmha_batch_prefill_args;

__attribute__((visibility("default"))) float
mha_fwd_splitkv(mha_fwd_splitkv_args args,
                const ck_tile::stream_config& stream_config,
                std::string q_dtype_str,
                bool is_group_mode,
                mask_enum mask_type,
                bias_enum bias_type,
                bool has_lse,
                bool has_sink = false);

__attribute__((visibility("default"))) float
mha_batch_prefill(mha_batch_prefill_args args,
                  const ck_tile::stream_config& stream_config,
                  std::string q_dtype_str,
                  bool is_group_mode,
                  mask_enum mask_type,
                  bias_enum bias_type,
                  bool has_lse,
                  quant_scale_enum qscale_type,
                  bool use_ext_asm);

struct __attribute__((packed)) fmha_fwd_v3_args
{
    void* ptr_o;
    p2 _p0;
    const void* ptr_q;
    p2 _p1;
    const void* ptr_k;
    p2 _p2;
    const void* ptr_v;
    p2 _p3;
    void* ptr_lse;
    p2 _p4;
    float scalar;
    p3 _p5;
    unsigned int s_seq_len;
    p3 _p6;
    unsigned int s_Seqs;
    p3 _p7;
    unsigned int s_Ts;
    p3 _p8;
    unsigned int s_Hs;
    p3 _p9;
    unsigned int s_Bs;
    p3 _p10;
    unsigned int s_gqa;
    p3 _p11;
    unsigned int s_k_Seqs;
    p3 _p12;
    unsigned int s_k_Hs;
    p3 _p13;
    unsigned int s_k_Bs;
    p3 _p14;
    unsigned int s_opt;
    p3 _p15;
    unsigned int s_lse;
    p3 _p16;
    unsigned int s_kv_seq_len;
    p3 _p17;
    unsigned int s_qk_head_dim;
    p3 _p18;
    unsigned int s_v_head_dim;
    p3 _p19;
    unsigned int s_q_head_num;
    p3 _p20;
    unsigned int s_v_Seqs;
    p3 _p21;
    unsigned int s_v_Hs;
    p3 _p22;
    unsigned int s_v_Bs;
    p3 _p23;
    unsigned int s_o_Seqs;
    p3 _p24;
    unsigned int s_o_Hs;
    p3 _p25;
    unsigned int s_o_Bs;
    p3 _p26;
    const void* ptr_qseq;
    p2 _p27;
    const void* ptr_kseq;
    p2 _p28;
    unsigned int s_lse_Hs;
    p3 _p29;
    const void* ptr_qseq_padding;
    p2 _p30;
    const void* ptr_kseq_padding;
    p2 _p31;
};

__attribute__((visibility("default"))) float mha_fwd(mha_fwd_args args,
                                                     const ck_tile::stream_config& s);

float fmha_fwd_v3(mha_fwd_args a, const ck_tile::stream_config& s);
} // namespace aiter
