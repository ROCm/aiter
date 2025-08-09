#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

// Include these 2 headers instead of torch/extension.h since we don't need all
// of the torch headers.
#include "aiter_hip_common.h"
#include "fmha_fwd.hpp"
#include "mask.hpp"

namespace aiter {
struct mha_fwd_traits : public fmha_fwd_traits {
  mha_fwd_traits(int head_size_q, int head_size_v, std::string dtype,
                 bool is_group_mode, bool has_logits_soft_cap,
                 mask_enum mask_type, bias_enum bias_type, bool has_lse,
                 bool has_dropout, bool use_ext_asm,
                 bool skip_min_seqlen_q)
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
                        false, // do_fp8_static_quant
                        skip_min_seqlen_q},
        use_ext_asm(use_ext_asm) {}
  bool use_ext_asm;
};

struct mha_fwd_splitkv_traits : public fmha_fwd_splitkv_traits {
  mha_fwd_splitkv_traits(int head_size_q, int head_size_v, std::string dtype,
                         bool is_group_mode, bool has_logits_soft_cap,
                         mask_enum mask_type, bias_enum bias_type, bool has_lse)
      : fmha_fwd_splitkv_traits{head_size_q,
                                head_size_v,
                                dtype,
                                is_group_mode,
                                true, // is_v_rowmajor
                                has_logits_soft_cap,
                                mask_type,
                                bias_type,
                                has_lse,
                                false} // do_fp8_static_quant
  {}
};

using mha_fwd_args = fmha_fwd_args;
using mha_fwd_splitkv_args = fmha_fwd_splitkv_args;
using mha_batch_prefill_args = fmha_batch_prefill_args;

__attribute__((visibility("default")))
float mha_fwd(mha_fwd_args args, const ck_tile::stream_config &stream_config,
              std::string q_dtype_str, bool is_group_mode, mask_enum mask_type,
              bias_enum bias_type, bool has_lse, bool use_ext_asm);

__attribute__((visibility("default")))
float mha_fwd_splitkv(mha_fwd_splitkv_args args,
                      const ck_tile::stream_config &stream_config,
                      std::string q_dtype_str, bool is_group_mode,
                      mask_enum mask_type, bias_enum bias_type, bool has_lse);

__attribute__((visibility("default")))
float mha_batch_prefill(mha_batch_prefill_args args,
                        const ck_tile::stream_config &stream_config,
                        std::string q_dtype_str, bool is_group_mode,
                        mask_enum mask_type, bias_enum bias_type, bool has_lse,
                        bool use_ext_asm);

struct __attribute__((packed)) fmha_fwd_v3_args
{
    void *ptr_o;
    p2 _p0;
    const void *ptr_q;
    p2 _p1;
    const void *ptr_k;
    p2 _p2;
    const void *ptr_v;
    p2 _p3;
    void *ptr_lse;
    p2 _p4;
    float scalar;
    p3 _p5;
    unsigned int seq_len;
    p3 _p6;
    unsigned int Seqs;
    p3 _p7;
    unsigned int Ts;
    p3 _p8;
    unsigned int Hs;
    p3 _p9;
    unsigned int BAs;
    p3 _p10;
    unsigned int gqa;
    p3 _p11;
    unsigned int Seqs_kv;
    p3 _p12;
    unsigned int Hs_kv;
    p3 _p13;
    unsigned int BAs_kv;
    p3 _p14;
    unsigned int opt;
    p3 _p15;
    unsigned int s_lse;
    p3 _p16;
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
          GPUArch GPUArch_>
struct fmha_fwd_kernel_selector
{
    using DataType                              = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t HDim      = HDim_;
    static constexpr ck_tile::index_t MaskType  = MaskType_;
    static constexpr bool kIsSEQPad             = kIsSEQPad_;
    static constexpr bool kIsHDPad              = kIsHDPad_;
    static constexpr int kStoreLSE              = kStoreLSE_; // kStoreLSE_ won't affect kernel selection, but will pass in kernel args
};

template <typename fmha_fwd_kernel_selector> struct FmhaFwdV3Name;
template <typename fmha_fwd_kernel_selector> struct FmhaFwdV3Buf;
template <typename fmha_fwd_kernel_selector> struct FmhaFwdV3Ts;

float fmha_fwd_v3(mha_fwd_traits t, fmha_fwd_args a, const ck_tile::stream_config &s, GPUArch arch);

} // namespace aiter
