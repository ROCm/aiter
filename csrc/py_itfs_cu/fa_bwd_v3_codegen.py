# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import copy
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Optional
import argparse

BOOL_MAP = {
    "t" : "true",
    "f" : "false"
}

BWD_V3_HDIM_MAP = {
    "64": "64",
    "128": "128"
}

BWD_DTYPE_MAP = {
    "fp16": "FmhaBwdFp16",
    "bf16": "FmhaBwdBf16"
}

BF16_CVT_MAP = {
    0 : "rtne",
    1 : "rtna",
    2 : "rtz",
}

BWD_V3_MASK_MAP = {
    "t": "((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0)))",
    "f": "(t.mask_type == mask_enum::no_mask)"
}

BWD_V3_ATOMIC32_MAP = {
    "t": "((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/))",
    "f": "(t.is_v3_atomic_fp32 == false)"
}

BWD_V3_HDIM_CASE_MAP = {
    0: "(a.hdim_q == 128)",
    1: "(a.hdim_q == 64)",
    2: "((a.hdim_q > 64) && (a.hdim_q < 128))"
}

BWD_V3_HDIM_CASE_CHECK_MAP = {
    0: 128,
    1: 64,
    2: 128
}

BWD_V3_PADDING_CHECK_MAP = {
    0: "false",
    1: "false",
    2: "true"
}

FMHA_BWD_API_FILENAME="asm_fmha_bwd_v3.cpp"

FMHA_BWD_KERNEL_HEADER = """// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.\n
// auto generated by generate.py
#include "fmha_bwd.hpp"
"""

FMHA_BWD_V3_TEMPLATE="""template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<{F_hdim}, {F_dtype}, {F_is_causal}, {F_is_atomic}, {F_bf16_cvt}, {F_hdpad}>> {{ static constexpr const char * bwd_v3_name = "bwd_v3{F_hdim_name}{F_dtype_name}{F_causal_name}{F_atomic_name}{F_bf16_cvt_name}{F_hdpad_name}"; }};
template<> struct FmhaBwdV3hsaco<fmha_bwd_dq_dk_dv_v3_traits_<{F_hdim}, {F_dtype}, {F_is_causal}, {F_is_atomic}, {F_bf16_cvt}, {F_hdpad}>> {{ static constexpr const char * bwd_v3_hsaco = "bwd{F_hdim_name}{F_dtype_name}{F_causal_name}{F_atomic_name}{F_bf16_cvt_name}{F_hdpad_name}.co"; }};
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<{F_hdim}, {F_dtype}, {F_is_causal}, {F_is_atomic}, {F_bf16_cvt}, {F_hdpad}>> {{ static constexpr int ts_qo = {F_Ts_qo}; static constexpr int ts_kv = 192; }};
"""

FMHA_BWD_API = """
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "aiter_hip_common.h"
#include <iostream>
#include "hsaco/fmha_hsaco.hpp"

#define HSA_KERNEL "kernel_func"

struct __attribute__((packed)) fmha_bwd_xqa_v3_args
{{
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
}};

struct __attribute__((packed)) fmha_bwd_xqa_v3_dp_args
{{
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
}};

struct fmha_bwd_v3_traits
{{
    int b;
    int h;
    int s;
    int d;

    int mask;
    int ts_qo;
    int ts_kv;
}};

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsCausal_,
          bool kIsAtomic32_,
          ck_tile::index_t BF16Cvt_,
          bool kIsHDPad_>
struct fmha_bwd_dq_dk_dv_v3_traits_
{{
    static constexpr ck_tile::index_t HDim    = HDim_;
    using DataType                            = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsCausal           = kIsCausal_;
    static constexpr bool kIsAtomic32         = kIsAtomic32_;
    static constexpr ck_tile::index_t BF16Cvt = BF16Cvt_;
    static constexpr bool kIsHDPad            = kIsHDPad_;
}};

template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3Name;
template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3hsaco;
template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3Ts;

{F_template}

class fmha_bwd_v3_kernel
{{
    public:
    fmha_bwd_v3_kernel(const std::string& name, const char *hsaco)
    {{
        std::cout << "hipModuleLoad: " << (std::string(AITER_ASM_DIR) + hsaco).c_str() << " GetFunction: " << name;
        HIP_CALL(hipModuleLoad(&module, (std::string(AITER_ASM_DIR) + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name.c_str()));
        std::cout << " Success" << std::endl;
    }}

    template <typename fmha_bwd_v3_args>
    void launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_args args, const ck_tile::stream_config& s) const
    {{
        size_t arg_size = sizeof(args);
        void* config[]  = {{HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END}};

        int bdx = 256;
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {{
            int num_tg = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }}
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       gdy,
                                       gdz,
                                       bdx,
                                       1,
                                       1,
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }}

    private:
    hipModule_t module;
    hipFunction_t kernel_func;
}};

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_>
float fmha_bwd_v3_xqa_(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << std::flush;
    fmha_bwd_xqa_v3_args args;
    args.ptr_dq  = a.dq_ptr;
    args.ptr_dk  = a.dk_ptr;
    args.ptr_dv  = a.dv_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_do  = a.do_ptr;
    args.ptr_lse = a.lse_ptr;
    args.ptr_d   = a.d_ptr;
    args.scalar  = a.scale;
    args.log2e   = ck_tile::log2e_v<float>;
    args.seq_len = a.seqlen_q;

    args.Ts   = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    auto traits = fmha_bwd_v3_traits{{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv}};
    static fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); }},
        [=](const ck_tile::stream_config& s_){{ impl.launch_kernel(traits, args, s_); }}
    );
}}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_>
float fmha_bwd_v3_hdp_xqa_(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << std::flush;
    fmha_bwd_xqa_v3_dp_args args;
    args.ptr_dq  = a.dq_ptr;
    args.ptr_dk  = a.dk_ptr;
    args.ptr_dv  = a.dv_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_do  = a.do_ptr;
    args.ptr_lse = a.lse_ptr;
    args.ptr_d   = a.d_ptr;
    args.scalar  = a.scale;
    args.log2e   = ck_tile::log2e_v<float>;
    args.seq_len = a.seqlen_q;

    args.Ts   = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    args.head_dim = a.hdim_q;
    auto traits = fmha_bwd_v3_traits{{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv}};
    static fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); }},
        [=](const ck_tile::stream_config& s_){{ impl.launch_kernel(traits, args, s_); }}
    );
}}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_xqa_(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_xqa_v3_args args;
    args.ptr_dq  = a.dq_acc_ptr;
    args.ptr_dk  = a.dk_ptr;
    args.ptr_dv  = a.dv_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_do  = a.do_ptr;
    args.ptr_lse = a.lse_ptr;
    args.ptr_d   = a.d_ptr;
    args.scalar  = a.scale;
    args.log2e   = ck_tile::log2e_v<float>;
    args.seq_len = a.seqlen_q;

    args.Ts   = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    auto traits = fmha_bwd_v3_traits{{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv}};
    static fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); }},
        [=](const ck_tile::stream_config& s_){{ impl.launch_kernel(traits, args, s_); }},
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }}
    );
}}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_hdp_xqa_(const ck_tile::stream_config& s, fmha_bwd_args a)
{{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_xqa_v3_dp_args args;
    args.ptr_dq  = a.dq_acc_ptr;
    args.ptr_dk  = a.dk_ptr;
    args.ptr_dv  = a.dv_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_do  = a.do_ptr;
    args.ptr_lse = a.lse_ptr;
    args.ptr_d   = a.d_ptr;
    args.scalar  = a.scale;
    args.log2e   = ck_tile::log2e_v<float>;
    args.seq_len = a.seqlen_q;

    args.Ts   = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    args.head_dim = a.hdim_q;
    auto traits = fmha_bwd_v3_traits{{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv}};
    static fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); }},
        [=](const ck_tile::stream_config& s_){{ impl.launch_kernel(traits, args, s_); }},
        [=](const ck_tile::stream_config& s_){{ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }}
    );
}}

float fmha_bwd_v3(fmha_bwd_traits t, fmha_bwd_args a, const ck_tile::stream_config& s){{
    float r = -1;

    if (t.uses_bwd_v3 == true){{
        if ((t.is_group_mode == false) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) && (t.is_deterministic == false) && (a.hdim_q == a.hdim_v) &&
                    (a.seqlen_q == a.seqlen_k) && (a.nhead_q % a.nhead_k == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))) {{
            if (((a.hdim_q >= 64) && (a.hdim_q <= 128) && (a.hdim_q % 8 == 0) && (a.seqlen_k % 64 == 0))) {{
{F_v3_dispatch}
            }}
        }}
    }}
    return r;
}}
"""

FMHA_BWD_V3_ATOMIC32_INNER_DISPATCH="""                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<{F_hdim}, {F_dtype}, false, false, {F_padding}>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<{F_hdim}, {F_dtype}, {F_is_causal}, {F_is_atomic32}, {F_how_v3_bf16_cvt}, {F_padding}>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<{F_hdim}, {F_dtype}, false, false, {F_padding}, false>;
                                    r = fmha_bwd_v3{F_padding_suffix}_xqa_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;"""

FMHA_BWD_V3_ATOMIC16_INNER_DISPATCH="""                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<{F_hdim}, {F_dtype}, false, false, {F_padding}>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<{F_hdim}, {F_dtype}, {F_is_causal}, {F_is_atomic32}, {F_how_v3_bf16_cvt}, {F_padding}>;
                                    r = fmha_bwd_v3{F_padding_suffix}_xqa_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;"""

FMHA_BWD_V3_PER_DTYPE_CASE="""                {F_if} (t.data_type.compare(\"{F_dtype}\") == 0) {{
{per_bf16_cvt_dispatch}
                }}
"""

FMHA_BWD_V3_PER_BF16_CVT_CASE="""                    {F_if} (t.how_v3_bf16_cvt == {F_bf16_cvt}) {{
{per_mask_dispatch}
                    }}
"""

FMHA_BWD_V3_PER_MASK_CASE="""                        {F_if} {F_mask_expression}{{
{per_atomic_dispatch}
                        }}
"""

FMHA_BWD_V3_PER_ATOMIC_CASE="""                            {F_if} {F_atomic_expression}{{
{per_hdim_dispatch}
                            }}
"""

FMHA_BWD_V3_PER_HDIM_CASE="""                                {F_if} {F_hdim_expression}{{
{inner_dispatch}
                                }}
"""

@dataclass
class FmhaBwdV3DQDKDVApiTrait:
    hdim            : str
    dtype           : str  # data type
    is_causal       : str
    is_atomic       : str
    bf16_cvt        : int
    is_hdpad        : str

    def remap_hdim(self):
        hdim_int = int(self.hdim)
        if hdim_int > 64:
            self.hdim = 128
        hdim_int = (hdim_int + 64 - 1) / 64 * 64

class FmhaBwdApiPool:
    def __init__(self):
        self.dq_dk_dv_v3_pool = dict()

    def register_dq_dk_dv_v3_traits(self, trait : FmhaBwdV3DQDKDVApiTrait) -> None:
        # TODO: do we need to check duplication?
        if trait.dtype not in self.dq_dk_dv_v3_pool.keys():
            self.dq_dk_dv_v3_pool[trait.dtype] = dict()
        if trait.hdim not in self.dq_dk_dv_v3_pool[trait.dtype].keys():
            self.dq_dk_dv_v3_pool[trait.dtype][trait.hdim] = list()

        self.dq_dk_dv_v3_pool[trait.dtype][trait.hdim].append(copy.copy(trait))

    @property
    def api(self) -> str:
        gen_template = str()
        for i, dtype in enumerate(self.dq_dk_dv_v3_pool.keys()):
            for j, hdim in enumerate(BWD_V3_HDIM_MAP.keys()):
                traits = self.dq_dk_dv_v3_pool[dtype][hdim]
                hdim = int(hdim)
                Ts_qo = 32 if hdim == 64 else 16
                for k, trait in enumerate(traits):
                    if hdim == 64 and trait.is_hdpad == "t":
                        continue
                    hdim_name = "_hd64" if hdim == 64 else ""
                    dtype_name = "_{}".format(dtype)
                    causal_name = "_causal" if trait.is_causal == "t" else ""
                    atomic_name = "_a32" if trait.is_atomic == "t" else "_a16"
                    bf16_cvt_name = "_{}".format(BF16_CVT_MAP[trait.bf16_cvt])
                    bf16_cvt_name = bf16_cvt_name if dtype == "bf16" else ""
                    hdpad_name = "_pddv" if trait.is_hdpad == "t" else ""
                    gen_template = gen_template + FMHA_BWD_V3_TEMPLATE.format(F_hdim=hdim, F_dtype=BWD_DTYPE_MAP[dtype], F_is_atomic=BOOL_MAP[trait.is_atomic],
                                    F_is_causal=BOOL_MAP[trait.is_causal], F_bf16_cvt=trait.bf16_cvt, F_hdpad=BOOL_MAP[trait.is_hdpad], F_Ts_qo = Ts_qo, F_hdim_name=hdim_name,
                                    F_dtype_name=dtype_name, F_causal_name=causal_name, F_atomic_name=atomic_name, F_bf16_cvt_name=bf16_cvt_name, F_hdpad_name=hdpad_name)
                    
        v3_code = str()
        for i, dtype in enumerate(self.dq_dk_dv_v3_pool.keys()):
            per_bf16_cvt = str()
            for j, bf16_cvt in enumerate([0, 1, 2]):
                per_mask = str()
                for k, is_causal in enumerate(["t", "f"]):
                    per_atomic = str()
                    for l, is_atomic in enumerate(["t", "f"]):
                        per_hdim = str()
                        for m, hdim in enumerate(BWD_V3_HDIM_CASE_CHECK_MAP.values()):
                            if_m = 'if' if m == 0 else 'else if'
                            inners = str()
                            bf16_cvt_tmp = 0 if dtype == "fp16" else bf16_cvt
                            padding_suffix = "_hdp" if BWD_V3_PADDING_CHECK_MAP[m] == "true" else ""
                            if is_atomic == "t":
                                inners = FMHA_BWD_V3_ATOMIC32_INNER_DISPATCH.format(F_hdim=hdim, F_dtype=BWD_DTYPE_MAP[dtype], F_is_causal=BOOL_MAP[is_causal], F_is_atomic32=BOOL_MAP[is_atomic], F_how_v3_bf16_cvt=bf16_cvt_tmp, F_padding=BWD_V3_PADDING_CHECK_MAP[m], F_padding_suffix=padding_suffix)
                            else:
                                inners = FMHA_BWD_V3_ATOMIC16_INNER_DISPATCH.format(F_hdim=hdim, F_dtype=BWD_DTYPE_MAP[dtype], F_is_causal=BOOL_MAP[is_causal], F_is_atomic32=BOOL_MAP[is_atomic], F_how_v3_bf16_cvt=bf16_cvt_tmp, F_padding=BWD_V3_PADDING_CHECK_MAP[m], F_padding_suffix=padding_suffix)
                            per_hdim = per_hdim + FMHA_BWD_V3_PER_HDIM_CASE.format(F_if=if_m, F_hdim_expression=BWD_V3_HDIM_CASE_MAP[m], inner_dispatch=inners)

                        if_l = 'if' if l == 0 else 'else if'
                        per_atomic = per_atomic + FMHA_BWD_V3_PER_ATOMIC_CASE.format(F_if=if_l, F_atomic_expression=BWD_V3_ATOMIC32_MAP[is_atomic], per_hdim_dispatch=per_hdim)
                    if_k = 'if' if k == 0 else 'else if'
                    per_mask = per_mask + FMHA_BWD_V3_PER_MASK_CASE.format(F_if=if_k, F_mask_expression=BWD_V3_MASK_MAP[is_causal], per_atomic_dispatch=per_atomic)
                if_j = 'if' if j == 0 else 'else if'
                per_bf16_cvt = per_bf16_cvt + FMHA_BWD_V3_PER_BF16_CVT_CASE.format(F_if=if_j, F_bf16_cvt=bf16_cvt, per_mask_dispatch=per_mask)
            if_i = 'if' if i == 0 else 'else if'
            v3_code = v3_code + FMHA_BWD_V3_PER_DTYPE_CASE.format(F_if=if_i, F_dtype=dtype, per_bf16_cvt_dispatch=per_bf16_cvt)
                    
        return FMHA_BWD_KERNEL_HEADER + FMHA_BWD_API.format(F_template = gen_template, F_v3_dispatch = v3_code)


@dataclass
class FmhaBwdV3DQDKDVKernel:
    F_hdim          : int  # hdim
    F_dtype         : str  # data type
    F_is_causal     : str
    F_is_atomic     : str
    F_bf16_cvt      : int
    F_is_hdpad      : str

    def v3_api_trait(self) -> FmhaBwdV3DQDKDVApiTrait:
        return FmhaBwdV3DQDKDVApiTrait(hdim=str(self.F_hdim),
                dtype=self.F_dtype,
                is_causal=self.F_is_causal,
                is_atomic=self.F_is_atomic,
                bf16_cvt=self.F_bf16_cvt,
                is_hdpad=self.F_is_hdpad
                )

def get_bwd_dq_dk_dv_blobs(receipt) -> FmhaBwdApiPool:
    # TODO: we don't support tuning yet, so pick up one value for pad
    #       support this in future
    api_pool = FmhaBwdApiPool()

    for dtype in BWD_DTYPE_MAP.keys():
        for hdim_str, is_causal, is_atomic, bf16_cvt, is_hdpad in itertools.product(['32', '64', '128', '256'], ["t", "f"], ["t", "f"], [0, 1, 2], ["t", "f"]):
            hdim = int(hdim_str)
            k = FmhaBwdV3DQDKDVKernel(F_hdim=hdim, F_dtype=dtype, F_is_causal=is_causal, F_is_atomic=is_atomic, F_bf16_cvt=bf16_cvt, F_is_hdpad=is_hdpad)
            if receipt == 3:
                    cond = (dtype == 'fp16') and (bf16_cvt in [1, 2])
                    if cond:
                        # print(dtype, bf16_cvt)
                        continue
            api_pool.register_dq_dk_dv_v3_traits(k.v3_api_trait())

    return api_pool

def write_blobs(output_dir : Path, receipt) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / ""
    api_pool = get_bwd_dq_dk_dv_blobs(receipt)
    (output_dir / FMHA_BWD_API_FILENAME).write_text(api_pool.api)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory"
    )

    parser.add_argument(
        "-r",
        "--receipt",
        default=0,
        required=False,
        help="codegen receipt. 0: generate only 8xhdim coverage\n"  + \
             "  1: generate more instance to cover all hdim\n"  + \
             "  2: Only generate instance for Flash attention integration"
    )
    args = parser.parse_args()
    write_blobs(args.output_dir, int(args.receipt))
