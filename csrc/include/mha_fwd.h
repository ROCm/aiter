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

using mha_fwd_args           = fmha_fwd_args;
using mha_fwd_splitkv_args   = fmha_fwd_splitkv_args;
using mha_batch_prefill_args = fmha_batch_prefill_args;

__attribute__((visibility("default"))) float mha_fwd(mha_fwd_args args,
                                                     const ck_tile::stream_config& stream_config,
                                                     std::string q_dtype_str,
                                                     bool is_group_mode,
                                                     mask_enum mask_type,
                                                     bias_enum bias_type,
                                                     bool has_lse,
                                                     quant_scale_enum qscale_type,
                                                     bool use_ext_asm,
                                                     bool has_sink = false,
                                                     int how_v3_bf16_cvt                = 1,
                                                     const void* seqstart_q_padding_ptr = nullptr,
                                                     const void* seqstart_k_padding_ptr = nullptr,
                                                     bool is_v3_api_check = false);

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

struct fmha_fwd_v3_traits
{
    int b;
    int h;
    int s;
    int d;

    int mask;
    int ts_qo;
    int ts_kv;
};

template <typename DataType_,
          ck_tile::index_t HDim_,
          ck_tile::index_t MaskType_,
          bool kIsSEQPad_,
          bool kIsHDPad_,
          int kStoreLSE_,
          GPUArch GPUArch_,
          ck_tile::index_t BF16Cvt_ = 1,
          bool kIsGroupMode_        = false>
struct fmha_fwd_kernel_selector
{
    using DataType                             = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t HDim     = HDim_;
    static constexpr ck_tile::index_t MaskType = MaskType_;
    static constexpr bool kIsSEQPad            = kIsSEQPad_;
    static constexpr bool kIsHDPad             = kIsHDPad_;
    static constexpr int kStoreLSE =
        kStoreLSE_; // kStoreLSE_ won't affect kernel selection, but will pass in kernel args
    static constexpr ck_tile::index_t BF16Cvt = BF16Cvt_;
    static constexpr bool kIsGroupMode        = kIsGroupMode_;
};

template <typename fmha_fwd_kernel_selector>
struct FmhaFwdV3Name;
template <typename fmha_fwd_kernel_selector>
struct FmhaFwdV3Buf;
template <typename fmha_fwd_kernel_selector>
struct FmhaFwdV3Ts;

namespace gfx942 {
float fmha_fwd_v3(mha_fwd_traits t,
                  mha_fwd_args a,
                  const ck_tile::stream_config& s,
                  bool is_v3_api_check = false);
}

namespace gfx950 {
float fmha_fwd_v3(mha_fwd_traits t,
                  mha_fwd_args a,
                  const ck_tile::stream_config& s,
                  bool is_v3_api_check = false);
}
} // namespace aiter
