# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from pathlib import Path
from typing import Optional

GEN_DIR = ""  # in Cmake, have to generate files in same folder

FMHA_BWD_API_FILENAME = "asm_fmha_bwd_v3_gfx950.cpp"

FMHA_BWD_KERNEL_HEADER = """// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.\n
"""

FMHA_BWD_API = """#include <hip/hip_fp16.h>
#include "mha_bwd.h"

namespace aiter {

struct __attribute__((packed)) fmha_bwd_v3_args_gfx950
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
    int mask_x;                     // 0x290
    p3 _p42;
    int mask_y;                     // 0x2a0
    p3 _p43;
};


template <uint32_t HDim_, bool kIsGroupMode_>
struct dq_shuffle_traits_
{
private:
    static std::string GetName()
    {
        return std::string("hd") + std::to_string(HDim_) +
               (kIsGroupMode_ ? "_dq_shuffle_group" : "_dq_shuffle");
    }
public:
    static const char* kernel_name()
    {
        static std::string name = std::string("fmha_bwd_") + GetName();
        return name.c_str();
    }

    static const char* file_name()
    {
        static std::string name = std::string("bwd_") + GetName() + ".co";
        return name.c_str();
    }

    static constexpr int ts_dq = 64;
};

// ########################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|kIsGroupMode|         GPUArch|
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      0,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a16_rtne_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      1,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a16_rtna_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      2,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a16_rtz_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a32_rtne_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a32_rtna_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a32_rtz_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      0,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a16_rtne_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      1,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a16_rtna_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      2,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a16_rtz_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a32_rtne_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a32_rtna_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      0,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_br_a32_rtne_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      1,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_br_a32_rtna_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      2,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_br_a32_rtz_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,      false,      0,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_fp16_a16_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_fp16_a32_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,      false,      0,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_fp16_causal_a16_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_fp16_causal_a32_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        3,       true,      0,     true,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_fp16_causal_br_a32_pssk_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_a16"; }; // native gfx950, currently not used
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_a32"; }; // native gfx950, currently not used
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_a16"; }; // native gfx950, currently not used
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,    false,   false,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_a32"; }; // native gfx950, currently not used
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_a16_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_a32_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,      false,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_br_a16_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_a16_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_br_a32_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_a32_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_a16_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_a32_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,      false,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_causal_br_a16_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_causal_a16_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_causal_br_a32_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_causal_a32_psskddv"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_a32_rtne_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_a32_rtna_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_a32_rtz_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_a32_rtne_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_a32_rtna_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_a32_rtz_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_br_a32_rtne_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      1,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_br_a32_rtna_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      2,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_br_a32_rtz_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_fp16_a32_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_fp16_causal_a32_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        3,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_fp16_causal_br_a32_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        2,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_swa_a32_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      0,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_swa_a32_rtne_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      1,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_swa_a32_rtna_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      2,     true,    true,       false, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_swa_a32_rtz_psskddv_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a32_rtne_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a32_rtna_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_a32_rtz_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a32_rtne_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a32_rtna_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      0,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_br_a32_rtne_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      1,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_br_a32_rtna_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      2,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_bf16_causal_br_a32_rtz_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_fp16_a32_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_fp16_causal_a32_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        3,       true,      0,     true,   false,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd64_fp16_causal_br_a32_pssk_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_a16_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_a32_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,      false,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_br_a16_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_a16_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_br_a32_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_bf16_causal_a32_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_a16_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_a32_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,      false,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_causal_br_a16_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_causal_a16_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_causal_br_a32_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd128_fp16_causal_a32_psskddv_group"; }; // native gfx950
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_a32_rtne_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_a32_rtna_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_a32_rtz_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_a32_rtne_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_a32_rtna_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_a32_rtz_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_br_a32_rtne_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      1,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_br_a32_rtna_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      2,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_bf16_causal_br_a32_rtz_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_fp16_a32_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_fp16_causal_a32_psskddv_group_recompile"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        3,       true,      0,     true,    true,        true, GPUArch::gfx950>> { static constexpr const char * kernel_name = "fmha_bwd_hd192_fp16_causal_br_a32_psskddv_group_recompile"; };

// ########################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|kIsGroupMode|         GPUArch|
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      0,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      1,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      2,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a32_rtne_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a32_rtna_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a32_rtz_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      0,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      1,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      2,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a32_rtne_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a32_rtna_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a32_rtz_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      0,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_br_a32_rtne_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      1,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_br_a32_rtna_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      2,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_br_a32_rtz_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,      false,      0,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_fp16_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_fp16_a32_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,      false,      0,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_fp16_causal_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_fp16_causal_a32_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        3,       true,      0,     true,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_fp16_causal_br_a32_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_a16.co"; };  // native gfx950, currently not used
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_a32.co"; };  // native gfx950, currently not used
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_a16.co"; };  // native gfx950, currently not used
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,    false,   false,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_a32.co"; };  // native gfx950, currently not used
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_a16_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_a32_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,      false,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_br_a16_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_a16_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_br_a32_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_a32_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_a16_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_a32_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,      false,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_causal_br_a16_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_causal_a16_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_causal_br_a32_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_causal_a32_psskddv.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_br_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      1,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_br_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      2,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_br_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_fp16_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_fp16_causal_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        3,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_fp16_causal_br_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        2,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_swa_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      0,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      1,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      2,     true,    true,        false,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a32_rtne_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a32_rtna_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_a32_rtz_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a32_rtne_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a32_rtna_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_a32_rtz_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      0,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_br_a32_rtne_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      1,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_br_a32_rtna_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      2,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_bf16_causal_br_a32_rtz_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_fp16_a32_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_fp16_causal_a32_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        3,       true,      0,     true,   false,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd64_fp16_causal_br_a32_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_a16_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_a32_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,      false,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_causal_br_a16_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_causal_a16_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_causal_br_a32_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_fp16_causal_a32_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_a16_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_a32_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,      false,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_br_a16_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_a16_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_br_a32_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd128_bf16_causal_a32_psskddv_group.co"; }; // native gfx950
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_a32_rtne_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_a32_rtna_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_a32_rtz_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_a32_rtne_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_a32_rtna_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_a32_rtz_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_br_a32_rtne_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      1,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_br_a32_rtna_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      2,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_bf16_causal_br_a32_rtz_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_fp16_a32_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_fp16_causal_a32_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        3,       true,      0,     true,    true,         true,   GPUArch::gfx950>> { static constexpr const char * file_name = "bwd_hd192_fp16_causal_br_a32_psskddv_group.co"; };

// ########################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|kIsGroupMode|         GPUArch|
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      0,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      1,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      2,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      0,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      1,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      2,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      0,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      1,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      2,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,      false,      0,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,      false,      0,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        3,       true,      0,     true,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950, currently not used
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950, currently not used
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950, currently not used
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,    false,   false,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950, currently not used
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,      false,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,      false,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      1,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      2,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        3,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        2,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      0,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      1,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      2,     true,    true,         false, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      0,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      1,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        3,       true,      2,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        3,       true,      0,     true,   false,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,      false,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        3,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,      false,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        3,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 256; };  // native gfx950
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      1,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        3,       true,      2,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        3,       true,      0,     true,    true,          true, GPUArch::gfx950>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };

namespace gfx950{
class fmha_dq_shuffle_kernel
{
    public:
    fmha_dq_shuffle_kernel(const char *name, const char *hsaco)
    {
        int length = strlen(name);
        std::string kernel_func_name = "_ZN5aiter" + std::to_string(length) + name + "E";
        std::string AITER_ASM_DIR = std::string(std::getenv("AITER_ASM_DIR")) + "fmha_v3_bwd/";
        HIP_CALL(hipModuleLoad(&module, (AITER_ASM_DIR + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_func_name.c_str()));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_dq_shuffle_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.sq + fmha_v3_traits.ts_dq - 1) / fmha_v3_traits.ts_dq;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;

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
    }
    private:
    hipModule_t module;
    hipFunction_t kernel_func;
};

class fmha_bwd_v3_kernel
{
    public:
    fmha_bwd_v3_kernel(const char *name, const char *hsaco)
    {
        int length = strlen(name);
        std::string kernel_func_name = "_ZN5aiter" + std::to_string(length) + name + "E";
        std::string AITER_ASM_DIR = std::string(std::getenv("AITER_ASM_DIR")) + "fmha_v3_bwd/";
        HIP_CALL(hipModuleLoad(&module, (AITER_ASM_DIR + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_func_name.c_str()));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.sq + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.sq + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }
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
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_gen_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.sq + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.sq + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }
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
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_genl_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }
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
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_group_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int gdx = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        if(fmha_v3_traits.mask > 0)
        {
            gdx = (gdx % 2) ? (gdx / 2 + 1) : (gdx / 2);
        }
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       fmha_v3_traits.h, /*gdy*/
                                       fmha_v3_traits.b, /*gdz*/
                                       256, /*bdx*/
                                       1, /*bdy*/
                                       1, /*bdz*/
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_swa_genl_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;

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
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_args_gfx950 args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.sk + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }
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
    }
    private:
    hipModule_t module;
    hipFunction_t kernel_func;
};

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_>
float fmha_bwd_v3_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << std::flush;
    fmha_bwd_v3_args args;
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
    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};

    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.

    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_>
float fmha_bwd_v3_gen_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << std::flush;
    fmha_bwd_v3_gen_args args;
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
    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_v3_args args;
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
    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_gen_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_v3_gen_args args;
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
    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_genl_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_v3_genl_args args;
    args.ptr_dq   = a.dq_acc_ptr;
    args.ptr_dk   = a.dk_ptr;
    args.ptr_dv   = a.dv_ptr;
    args.ptr_q    = a.q_ptr;
    args.ptr_k    = a.k_ptr;
    args.ptr_v    = a.v_ptr;
    args.ptr_do   = a.do_ptr;
    args.ptr_lse  = a.lse_ptr;
    args.ptr_d    = a.d_ptr;
    args.scalar   = a.scale;
    args.log2e    = ck_tile::log2e_v<float>;
    args.ratio    = a.nhead_q / a.nhead_k;
    args.seqlen_q = a.seqlen_q;
    args.seqlen_k = a.seqlen_k;
    args.head_dim = a.hdim_q;
    args.nhead_q  = a.nhead_q;
    args.Hs_q     = a.nhead_stride_q * 2;
    args.BAs_q    = a.batch_stride_q * 2;
    args.Seqs_q   = a.stride_q * 2;
    args.Hs_k     = a.nhead_stride_k * 2;
    args.BAs_k    = a.batch_stride_k * 2;
    args.Seqs_k   = a.stride_k * 2;
    args.Hs_v     = a.nhead_stride_v * 2;
    args.BAs_v    = a.batch_stride_v * 2;
    args.Seqs_v   = a.stride_v * 2;
    args.Hs_do    = a.nhead_stride_do * 2;
    args.BAs_do   = a.batch_stride_do * 2;
    args.Seqs_do  = a.stride_do * 2;
    args.Hs_dk    = a.nhead_stride_dk * 2;
    args.BAs_dk   = a.batch_stride_dk * 2;
    args.Seqs_dk  = a.stride_dk * 2;
    args.Hs_dv    = a.nhead_stride_dv * 2;
    args.BAs_dv   = a.batch_stride_dv * 2;
    args.Seqs_dv  = a.stride_dv * 2;

    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_group_(const ck_tile::stream_config& s, fmha_bwd_args a, const void* seqlen_q_unpadded = nullptr, const void* seqlen_k_unpadded = nullptr)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;

    fmha_bwd_v3_group_args args;
    auto seqstart_k = reinterpret_cast<const int32_t*>(a.seqstart_k_ptr);
    args.ptr_dq             = a.dq_acc_ptr;
    args.ptr_dk             = a.dk_ptr;
    args.ptr_dv             = a.dv_ptr;
    args.ptr_q              = a.q_ptr;
    args.ptr_k              = a.k_ptr;
    args.ptr_v              = a.v_ptr;
    args.ptr_do             = a.do_ptr;
    args.ptr_lse            = a.lse_ptr;
    args.ptr_d              = a.d_ptr;

    if (a.seqlen_k_ptr && a.seqstart_k_ptr) {
        args.ptr_kseq_padded    = a.seqstart_k_ptr;
        args.ptr_kseq           = seqlen_k_unpadded;
    } else {
        args.ptr_kseq           = a.seqstart_k_ptr;
        args.ptr_kseq_padded    = a.seqstart_k_ptr;
    }

    if (a.seqlen_q_ptr && a.seqstart_q_ptr) {
        args.ptr_qseq_padded    = a.seqstart_q_ptr;
        args.ptr_qseq           = seqlen_q_unpadded;
    } else {
        args.ptr_qseq           = a.seqstart_q_ptr;
        args.ptr_qseq_padded    = a.seqstart_q_ptr;
    }

    args.scalar             = a.scale;
    args.log2e              = ck_tile::log2e_v<float>;
    args.ratio              = a.nhead_q / a.nhead_k;
    args.Hs_lsed            = a.nhead_stride_lsed * 4;
    args.seqlen_k           = seqstart_k[a.batch];
    args.Hs_q               = a.nhead_stride_q * 2;
    args.Seqs_q             = a.stride_q * 2;
    args.Hs_k               = a.nhead_stride_k * 2;
    args.Seqs_k             = a.stride_k * 2;
    args.Hs_v               = a.nhead_stride_v * 2;
    args.Seqs_v             = a.stride_v * 2;
    args.Hs_do              = a.nhead_stride_do * 2;
    args.Seqs_do            = a.stride_do * 2;
    args.Hs_dk              = a.nhead_stride_dk * 2;
    args.Seqs_dk            = a.stride_dk * 2;
    args.Hs_dv              = a.nhead_stride_dv * 2;
    args.Seqs_dv            = a.stride_dv * 2;
    args.head_dim           = a.hdim_q;

    auto traits = fmha_bwd_v3_traits{ a.batch,
                                       a.nhead_q,
                                       a.max_seqlen_q,
                                       a.max_seqlen_k,
                                       a.hdim_q,
                                       a.mask_type,
                                       FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                       FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv };
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

// SWA supposes to include following circumstances:
// 1. FA style SWA: t/b: mask_left > 0 or mask_right > 0
// 2. xformer style SWA: xt / xb: window_size > 0
// 3. generic style SWA: g: x, y (TODO: ck doesn't support generic style)
// after preprocessing, 1 & 2 can be unioned into:
// mask_type == mask_top_left or mask_bottom_right
// left > 0 or right > 0
template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_swa_genl_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_v3_swa_genl_args args;
    args.ptr_dq   = a.dq_acc_ptr;
    args.ptr_dk   = a.dk_ptr;
    args.ptr_dv   = a.dv_ptr;
    args.ptr_q    = a.q_ptr;
    args.ptr_k    = a.k_ptr;
    args.ptr_v    = a.v_ptr;
    args.ptr_do   = a.do_ptr;
    args.ptr_lse  = a.lse_ptr;
    args.ptr_d    = a.d_ptr;
    args.scalar   = a.scale;
    args.log2e    = ck_tile::log2e_v<float>;
    args.ratio    = a.nhead_q / a.nhead_k;
    args.seqlen_q = a.seqlen_q;
    args.seqlen_k = a.seqlen_k;
    args.head_dim = a.hdim_q;
    args.nhead_q = a.nhead_q;
    args.Hs_q     = a.nhead_stride_q * 2;
    args.BAs_q    = a.batch_stride_q * 2;
    args.Seqs_q   = a.stride_q * 2;
    args.Hs_k     = a.nhead_stride_k * 2;
    args.BAs_k    = a.batch_stride_k * 2;
    args.Seqs_k   = a.stride_k * 2;
    args.Hs_v     = a.nhead_stride_v * 2;
    args.BAs_v    = a.batch_stride_v * 2;
    args.Seqs_v   = a.stride_v * 2;
    args.Hs_do    = a.nhead_stride_do * 2;
    args.BAs_do   = a.batch_stride_do * 2;
    args.Seqs_do  = a.stride_do * 2;
    args.Hs_dk    = a.nhead_stride_dk * 2;
    args.BAs_dk   = a.batch_stride_dk * 2;
    args.Seqs_dk  = a.stride_dk * 2;
    args.Hs_dv    = a.nhead_stride_dv * 2;
    args.BAs_dv   = a.batch_stride_dv * 2;
    args.Seqs_dv  = a.stride_dv * 2;

    // convert l/r to x/y HERE
    auto generic_mask = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(a.window_size_left, a.window_size_right, a.seqlen_q, a.seqlen_k, (a.mask_type == static_cast<ck_tile::index_t>(mask_enum::mask_top_left) || a.mask_type == static_cast<ck_tile::index_t>(mask_enum::window_generic)));
    args.mask_y = generic_mask.at(ck_tile::number<0>{});
    args.mask_x = generic_mask.at(ck_tile::number<1>{});

    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_genl_gfx950(const ck_tile::stream_config& s, fmha_bwd_args a, bool is_v3_api_check, const void* seqlen_q_unpadded = nullptr, const void* seqlen_k_unpadded = nullptr)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    if (is_v3_api_check) return 1;

    fmha_bwd_v3_args_gfx950 args;
    args.ptr_dq         = a.dq_acc_ptr;
    args.ptr_dk         = a.dk_ptr;
    args.ptr_dv         = a.dv_ptr;
    args.ptr_q          = a.q_ptr;
    args.ptr_k          = a.k_ptr;
    args.ptr_v          = a.v_ptr;
    args.ptr_do         = a.do_ptr;
    args.ptr_lse        = a.lse_ptr;
    args.ptr_d          = a.d_ptr;
    args.scalar         = a.scale;
    args.log2e          = ck_tile::log2e_v<float>;;
    args.ratio          = a.nhead_q / a.nhead_k;
    args.seqlen_q       = a.seqlen_q;
    args.seqlen_k       = a.seqlen_k;
    args.head_dim_q     = a.hdim_q;
    args.nhead_q        = a.nhead_q;
    args.Ts             = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs_q           = a.nhead_stride_q * 2;
    args.BAs_q          = a.batch_stride_q * 2;
    args.Seqs_q         = a.stride_q * 2;
    args.Hs_k           = a.nhead_stride_k * 2;
    args.BAs_k          = a.batch_stride_k * 2;
    args.Seqs_k         = a.stride_k * 2;
    args.Hs_v           = a.nhead_stride_v * 2;
    args.BAs_v          = a.batch_stride_v * 2;
    args.Seqs_v         = a.stride_v * 2;
    args.Hs_do          = a.nhead_stride_do * 2;
    args.BAs_do         = a.batch_stride_do * 2;
    args.Seqs_do        = a.stride_do * 2;
    args.Hs_dk          = a.nhead_stride_dk * 2;
    args.BAs_dk         = a.batch_stride_dk * 2;
    args.Seqs_dk        = a.stride_dk * 2;
    args.Hs_dv          = a.nhead_stride_dv * 2;
    args.BAs_dv         = a.batch_stride_dv * 2;
    args.Seqs_dv        = a.stride_dv * 2;
    args.Hs_lsed            = a.nhead_stride_lsed * 4;

    if (a.seqlen_k_ptr && a.seqstart_k_ptr) {
        args.ptr_kseq_padded    = a.seqstart_k_ptr;
        args.ptr_kseq           = seqlen_k_unpadded;
    } else {
        args.ptr_kseq           = a.seqstart_k_ptr;
        args.ptr_kseq_padded    = a.seqstart_k_ptr;
    }

    if (a.seqlen_q_ptr && a.seqstart_q_ptr) {
        args.ptr_qseq_padded    = a.seqstart_q_ptr;
        args.ptr_qseq           = seqlen_q_unpadded;
    } else {
        args.ptr_qseq           = a.seqstart_q_ptr;
        args.ptr_qseq_padded    = a.seqstart_q_ptr;
    }
    args.max_seqlen_dq     = a.max_seqlen_q;

    auto traits = fmha_bwd_v3_traits{a.batch,
                                     a.nhead_q,
                                     a.seqlen_q,
                                     a.seqlen_k,
                                     a.hdim_q,
                                     a.mask_type,
                                     FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                     FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};

    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_>
float fmha_bwd_v3_genl_gfx950(const ck_tile::stream_config& s, fmha_bwd_args a, bool is_v3_api_check, const void* seqlen_q_unpadded = nullptr, const void* seqlen_k_unpadded = nullptr)
{
    using dq_shuffle_traits = dq_shuffle_traits_<dq_dk_dv_v3_traits_::HDim, dq_dk_dv_v3_traits_::kIsGroupMode>;

    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name << ", " << dq_shuffle_traits::kernel_name() << std::flush;
    if (is_v3_api_check) return 1;

    fmha_bwd_v3_args_gfx950 args;
    args.ptr_dq             = a.dq_acc_ptr;
    args.ptr_dk             = a.dk_ptr;
    args.ptr_dv             = a.dv_ptr;
    args.ptr_q              = a.q_ptr;
    args.ptr_k              = a.k_ptr;
    args.ptr_v              = a.v_ptr;
    args.ptr_do             = a.do_ptr;
    args.ptr_lse            = a.lse_ptr;
    args.ptr_d              = a.d_ptr;
    args.scalar             = a.scale;
    args.log2e              = ck_tile::log2e_v<float>;;
    args.ratio              = a.nhead_q / a.nhead_k;
    args.seqlen_q           = a.seqlen_q;
    args.seqlen_k           = a.seqlen_k;
    args.head_dim_q         = a.hdim_q;
    args.nhead_q            = a.nhead_q;
    args.Ts                 = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs_q               = a.nhead_stride_q * 2;
    args.BAs_q              = a.batch_stride_q * 2;
    args.Seqs_q             = a.stride_q * 2;
    args.Hs_k               = a.nhead_stride_k * 2;
    args.BAs_k              = a.batch_stride_k * 2;
    args.Seqs_k             = a.stride_k * 2;
    args.Hs_v               = a.nhead_stride_v * 2;
    args.BAs_v              = a.batch_stride_v * 2;
    args.Seqs_v             = a.stride_v * 2;
    args.Hs_do              = a.nhead_stride_do * 2;
    args.BAs_do             = a.batch_stride_do * 2;
    args.Seqs_do            = a.stride_do * 2;
    args.Hs_dk              = a.nhead_stride_dk * 2;
    args.BAs_dk             = a.batch_stride_dk * 2;
    args.Seqs_dk            = a.stride_dk * 2;
    args.Hs_dv              = a.nhead_stride_dv * 2;
    args.BAs_dv             = a.batch_stride_dv * 2;
    args.Seqs_dv            = a.stride_dv * 2;
    args.Hs_lsed            = a.nhead_stride_lsed * 4;

    if (a.seqlen_k_ptr && a.seqstart_k_ptr) {
        args.ptr_kseq_padded    = a.seqstart_k_ptr;
        args.ptr_kseq           = seqlen_k_unpadded;
    } else {
        args.ptr_kseq           = a.seqstart_k_ptr;
        args.ptr_kseq_padded    = a.seqstart_k_ptr;
    }

    if (a.seqlen_q_ptr && a.seqstart_q_ptr) {
        args.ptr_qseq_padded    = a.seqstart_q_ptr;
        args.ptr_qseq           = seqlen_q_unpadded;
    } else {
        args.ptr_qseq           = a.seqstart_q_ptr;
        args.ptr_qseq_padded    = a.seqstart_q_ptr;
    }
    args.max_seqlen_dq     = (a.max_seqlen_q + 15) / 16 * 16;

    fmha_bwd_dq_shuffle_args dq_shuffule_args;
    dq_shuffule_args.ptr_dq_acc         = a.dq_acc_ptr;
    dq_shuffule_args.ptr_dq             = a.dq_ptr;
    dq_shuffule_args.Ts                 = 64 * a.stride_dq * 2;
    dq_shuffule_args.Hs_dq_acc          = a.nhead_stride_dq_acc * 2;
    dq_shuffule_args.BAs_dq_acc         = a.batch_stride_dq_acc * 2;
    dq_shuffule_args.Seqs_dq_acc        = a.stride_dq_acc * 2;
    dq_shuffule_args.Hs_dq              = a.nhead_stride_dq * 2;
    dq_shuffule_args.BAs_dq             = a.batch_stride_dq * 2;
    dq_shuffule_args.Seqs_dq            = a.stride_dq * 2;
    dq_shuffule_args.seqlen_q           = a.seqlen_q;
    dq_shuffule_args.head_dim           = a.hdim_q;
    dq_shuffule_args.ptr_qseq           = a.seqstart_q_ptr;
    dq_shuffule_args.ptr_qseq_padded    = seqlen_q_unpadded == nullptr
                                        ? a.seqstart_q_ptr
                                        : seqlen_q_unpadded;
    dq_shuffule_args.max_seqlen_dq     = (a.max_seqlen_q + 15) / 16 * 16;

    auto traits = fmha_bwd_v3_traits{a.batch,
                                     a.nhead_q,
                                     a.max_seqlen_q, // when batch mode, max_seqlen equal to seqlen
                                     a.max_seqlen_k, // when batch mode, max_seqlen equal to seqlen
                                     a.hdim_q,
                                     a.mask_type,
                                     FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                     FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv,
                                     dq_shuffle_traits::ts_dq};

    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::kernel_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::file_name); // static here is for thread safety.
    static thread_local fmha_dq_shuffle_kernel impl_dq_shuffle(dq_shuffle_traits::kernel_name(), dq_shuffle_traits::file_name()); // static here is for thread safety.

    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ impl_dq_shuffle.launch_kernel(traits, dq_shuffule_args, s_); }
    );
}

float fmha_bwd_v3(mha_bwd_traits t,
                  fmha_bwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqlen_q_unpadded,
                  const void* seqlen_k_unpadded,
                  bool is_v3_api_check){
    float r = -1;

    if (t.use_ext_asm == true){
        if ((t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                    (t.is_deterministic == false) && (a.hdim_q == a.hdim_v) && (a.nhead_q % a.nhead_k == 0)) {
            if((a.hdim_q > 128) && (a.hdim_q <= 192) && (a.hdim_q % 8 == 0)){
                if(t.data_type.compare("fp16") == 0){
                    if((t.is_group_mode == false) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.mask_type == mask_enum::no_mask){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdFp16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16, false, true, 0, true, true, false, GPUArch::gfx950>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdFp16, false, true, true, false, 0>;
                            // const std::string kernel_name = "bwd_v3_hd192_fp16_a32_psskddv";
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((((t.mask_type != mask_enum::no_mask) && (a.seqlen_q == a.seqlen_k)) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))) &&
                                ((a.window_size_left == -1) && (a.window_size_right == 0))){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdFp16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16, true, true, 0, true, true, false, GPUArch::gfx950>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdFp16, false, true, true, false, 0>;
                            // const std::string kernel_name = "bwd_v3_hd192_fp16_causal_a32_psskddv";
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdFp16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdFp16, false, true, true, false, 0>;
                            // const std::string kernel_name = "bwd_v3_hd192_fp16_causal_br_a32_psskddv";
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                    else if((t.is_group_mode == true) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){//group mode
                        if(t.mask_type == mask_enum::no_mask){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdFp16, true, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16, false, true, 0, true, true, true, GPUArch::gfx950>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdFp16, true, true, true, false, 0>;
                            // const std::string kernel_name = "bwd_v3_hd192_fp16_a32_psskddv_group";
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                            return r;
                        }
                        else if(((a.window_size_left == -1) && (a.window_size_right == 0)) && (t.mask_type == mask_enum::mask_top_left)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdFp16, true, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16, true, true, 0, true, true, true, GPUArch::gfx950>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdFp16, true, true, true, false, 0>;
                            // const std::string kernel_name = "bwd_v3_hd192_fp16_causal_a32_psskddv_group";
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                            return r;
                        }
                        else if(((a.window_size_left == -1) && (a.window_size_right == 0)) && (t.mask_type == mask_enum::mask_bottom_right)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdFp16, true, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16, 3, true, 0, true, true, true, GPUArch::gfx950>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdFp16, true, true, true, false, 0>;
                            // const std::string kernel_name = "bwd_v3_hd192_fp16_causal_br_a32_psskddv_group";
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                            return r;
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if((t.is_group_mode == false) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.mask_type == mask_enum::no_mask){
                            if(t.how_v3_bf16_cvt == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_a32_rtne_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 1, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_a32_rtna_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 2, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_a32_rtz_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if((((t.mask_type != mask_enum::no_mask) && (a.seqlen_q == a.seqlen_k)) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))) &&
                                ((a.window_size_left == -1) && (a.window_size_right == 0))){
                            if(t.how_v3_bf16_cvt == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_a32_rtne_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 1, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_a32_rtna_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 2, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_a32_rtz_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if((t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                            if(t.how_v3_bf16_cvt == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_br_a32_rtne_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, 3, true, 1, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_br_a32_rtna_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, 3, true, 2, true, true, false, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_br_a32_rtz_psskddv";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){//group mode
                        if(t.mask_type == mask_enum::no_mask){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, true, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, true, true, true, false, 0>;
                            if(t.how_v3_bf16_cvt == 0){
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 0, true, true, true, GPUArch::gfx950>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_a32_rtne_psskddv_group";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 1, true, true, true, GPUArch::gfx950>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_a32_rtna_psskddv_group";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 2, true, true, true, GPUArch::gfx950>;
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_a32_rtz_psskddv_group";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }

                        }
                        else if(((a.window_size_left == -1) && (a.window_size_right == 0)) && (t.mask_type == mask_enum::mask_top_left)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, true, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, true, true, true, false, 0>;
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_a32_rtne_psskddv_group";
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 0, true, true, true, GPUArch::gfx950>;
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_a32_rtna_psskddv_group";
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 1, true, true, true, GPUArch::gfx950>;
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_a32_rtz_psskddv_group";
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 2, true, true, true, GPUArch::gfx950>;
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        }
                        else if(((a.window_size_left == -1) && (a.window_size_right == 0)) && (t.mask_type == mask_enum::mask_bottom_right)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, true, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, true, true, true, false, 0>;
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_br_a32_rtne_psskddv_group";
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, 3, true, 0, true, true, true, GPUArch::gfx950>;
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_br_a32_rtna_psskddv_group";
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, 3, true, 1, true, true, true, GPUArch::gfx950>;
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string kernel_name = "bwd_v3_hd192_bf16_causal_br_a32_rtz_psskddv_group";
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, 3, true, 2, true, true, true, GPUArch::gfx950>;
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        }
                    }
                }
            }
            else if ((a.hdim_q > 64) && (a.hdim_q <= 128) && (a.hdim_q % 8 == 0) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                if (t.data_type.compare("fp16") == 0){
                    if (t.is_group_mode == false){
                        if (t.mask_type == mask_enum::no_mask) {
                            if (t.is_v3_atomic_fp32 == true){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                            else if (t.is_v3_atomic_fp32 == false){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                        } else if ((t.mask_type == mask_enum::mask_top_left) && ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                            if (t.is_v3_atomic_fp32 == true){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            } else if (t.is_v3_atomic_fp32 == false){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                        } else if ((t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                            if (t.is_v3_atomic_fp32 == true){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_br_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_br_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_br_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_br_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            } else if (t.is_v3_atomic_fp32 == false){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_br_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_br_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_br_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_fp16_causal_br_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                        } else if (((t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left > 0) || (a.window_size_right > 0))) || (t.mask_type == mask_enum::window_generic)){
                            if(t.is_v3_atomic_fp32 == true){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 2, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false, 0>;
                                    // const std::string kernel_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 2, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 2, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false, 0>;
                                    // const std::string kernel_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv;
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 2, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false, 0>;
                                    // const std::string kernel_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                    }
                    else if (t.is_group_mode == true){
                        if (t.mask_type == mask_enum::no_mask) {
                            if (t.is_v3_atomic_fp32 == true){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true, true, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, true, true, true, false, 0>;
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a32_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            } else {
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, true, true, true, GPUArch::gfx950>;
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a16_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        } else if ((t.mask_type == mask_enum::mask_top_left) && ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                            if (t.is_v3_atomic_fp32 == true){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true, true, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, true, true, true, false, 0>;
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a32_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            } else {
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, true, true, true, GPUArch::gfx950>;
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a16_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        } else if ((t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                            if (t.is_v3_atomic_fp32 == true){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, true, 0, true, true, true, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, true, true, true, false, 0>;
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_br_a32_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            } else {
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 3, false, 0, true, true, true, GPUArch::gfx950>;
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_br_a16_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if (t.is_group_mode == false){
                        if (t.mask_type == mask_enum::no_mask) {
                            if (t.is_v3_atomic_fp32 == true){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                            else if (t.is_v3_atomic_fp32 == false){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                        } else if ((t.mask_type == mask_enum::mask_top_left) && ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                            if (t.is_v3_atomic_fp32 == true){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                            else if (t.is_v3_atomic_fp32 == false){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                        } else if ((t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                            if (t.is_v3_atomic_fp32 == true){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_br_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_br_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_br_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, true, 0, true, true, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false, 0>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_br_a32_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                            else if (t.is_v3_atomic_fp32 == false){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_br_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_br_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_br_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, false, 0, true, true, false, GPUArch::gfx950>;
                                    // const std::string kernel_name = "bwd_hd128_bf16_causal_br_a16_psskddv";
                                    r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check);
                                    return r;
                                }
                            }
                        } else if (((t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left > 0) || (a.window_size_right > 0))) || (t.mask_type == mask_enum::window_generic)){
                            if(t.is_v3_atomic_fp32 == true){
                                if(t.how_v3_bf16_cvt == 0){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 0, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 0, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 0, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 0, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 1, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 1, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 1, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 1, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 2){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 2, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 2, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 2, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 2, true, true, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false, 0>;
                                        // const std::string kernel_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                            }
                        }
                    }
                    else if (t.is_group_mode == true){
                        if (t.mask_type == mask_enum::no_mask) {
                            if (t.is_v3_atomic_fp32 == true){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true, true, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false, 0>;
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a32_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            } else {
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, true, true, true, GPUArch::gfx950>;
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a16_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        } else if ((t.mask_type == mask_enum::mask_top_left) && ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                            if (t.is_v3_atomic_fp32 == true){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true, true, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false, 0>;
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a32_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            } else {
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, true, true, true, GPUArch::gfx950>;
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a16_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        } else if ((t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                            if (t.is_v3_atomic_fp32 == true){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, true, 0, true, true, true, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false, 0>;
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_br_a32_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            } else {
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 3, false, 0, true, true, true, GPUArch::gfx950>;
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_br_a16_psskddv_group";
                                r = fmha_bwd_v3_genl_gfx950<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a, is_v3_api_check, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        }
                    }
                }
            }
            else if(a.hdim_q == 64){
                if(t.data_type.compare("fp16") == 0){
                    if(t.mask_type == mask_enum::no_mask){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.is_group_mode == false){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, true, 0, true, false, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, false, false, false, 0>;
                                    // const std::string kernel_name = "bwd_v3_hd64_fp16_a32_pssk";
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, true, 0, true, false, false, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_v3_hd64_fp16_a32_pssk";
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, true, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, true, 0, true, false, true, GPUArch::gfx950>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, true, true, false, false, 0>;
                                // const std::string kernel_name = "bwd_v3_hd64_fp16_a32_pssk_group";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, false, 0, false, false, false, GPUArch::gfx950>;
                            // const std::string kernel_name = "bwd_v3_hd64_fp16_a16";
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.is_group_mode == false){
                                if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                    if(a.seqlen_q % 64 == 0){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, true, 0, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, false, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else{
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, true, 0, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, true, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                                else if(t.mask_type == mask_enum::mask_bottom_right){
                                    if(a.seqlen_q % 64 == 0){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, 3, true, 0, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, false, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_fp16_causal_br_a32_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else{
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, 3, true, 0, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, true, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_fp16_causal_br_a32_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                            }
                            else if(t.is_group_mode == true){
                                if(t.mask_type == mask_enum::mask_top_left){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, true, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, true, 0, true, false, true, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, true, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_v3_hd64_fp16_causal_a32_pssk_group";
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                    return r;
                                }
                                else if(t.mask_type == mask_enum::mask_bottom_right){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, true, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, 3, true, 0, true, false, true, GPUArch::gfx950>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, true, true, false, false, 0>;
                                    // const std::string kernel_name = "bwd_v3_hd64_fp16_causal_br_a32_pssk_group";
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                    return r;
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, false, 0, false, false, false, GPUArch::gfx950>;
                            // const std::string kernel_name = "bwd_v3_hd64_fp16_causal_a16";
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if(t.mask_type == mask_enum::no_mask){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.is_group_mode == false){
                                if(t.how_v3_bf16_cvt == 0){
                                    if(a.seqlen_q % 64 == 0){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 0, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else{
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 0, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    if(a.seqlen_q % 64 == 0){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 1, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else{
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 1, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 2){
                                    if(a.seqlen_q % 64 == 0){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 2, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else{
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 2, true, false, false, GPUArch::gfx950>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                        // const std::string kernel_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, true, true, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, true, true, false, false, 0>;
                                if(t.how_v3_bf16_cvt == 0){
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 0, true, false, true, GPUArch::gfx950>;
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 1, true, false, true, GPUArch::gfx950>;
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                }
                                else{
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 2, true, false, true, GPUArch::gfx950>;
                                    if (is_v3_api_check) {
                                        return 1;
                                    }
                                    r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                }
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 0, false, false, false, GPUArch::gfx950>;
                                // const std::string kernel_name = "bwd_v3_hd64_bf16_a16_rtne";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 1, false, false, false, GPUArch::gfx950>;
                                // const std::string kernel_name = "bwd_v3_hd64_bf16_a16_rtna";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 2, false, false, false, GPUArch::gfx950>;
                                // const std::string kernel_name = "bwd_v3_hd64_bf16_a16_rtz";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.is_group_mode == false){
                                if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                    if(t.how_v3_bf16_cvt == 0){
                                        if(a.seqlen_q % 64 == 0){
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 0, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                        else{
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 0, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                    }
                                    else if(t.how_v3_bf16_cvt == 1){
                                        if(a.seqlen_q % 64 == 0){
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 1, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                        else{
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 1, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                    }
                                    else if(t.how_v3_bf16_cvt == 2){
                                        if(a.seqlen_q % 64 == 0){
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 2, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                        else{
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 2, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                    }
                                }
                                else if(t.mask_type == mask_enum::mask_bottom_right){
                                    if(t.how_v3_bf16_cvt == 0){
                                        if(a.seqlen_q % 64 == 0){
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 0, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_br_a32_rtne_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                        else{
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 0, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_br_a32_rtne_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                    }
                                    else if(t.how_v3_bf16_cvt == 1){
                                        if(a.seqlen_q % 64 == 0){
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 1, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_br_a32_rtna_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                        else{
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 1, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_br_a32_rtna_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                    }
                                    else if(t.how_v3_bf16_cvt == 2){
                                        if(a.seqlen_q % 64 == 0){
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 2, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_br_a32_rtz_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                        else{
                                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 2, true, false, false, GPUArch::gfx950>;
                                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false, 0>;
                                            // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_br_a32_rtz_pssk";
                                            if (is_v3_api_check) {
                                                return 1;
                                            }
                                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                            return r;
                                        }
                                    }
                                }
                            }
                            else if(t.is_group_mode == true){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, true, true, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, true, true, false, false, 0>;
                                if(t.mask_type == mask_enum::mask_top_left){
                                    if(t.how_v3_bf16_cvt == 0){
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 0, true, false, true, GPUArch::gfx950>;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                    }
                                    else if(t.how_v3_bf16_cvt == 1){
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 1, true, false, true, GPUArch::gfx950>;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                    }
                                    else{
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 2, true, false, true, GPUArch::gfx950>;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                    }
                                    return r;
                                }
                                else if(t.mask_type == mask_enum::mask_bottom_right){
                                    if(t.how_v3_bf16_cvt == 0){
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 0, true, false, true, GPUArch::gfx950>;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                    }
                                    else if(t.how_v3_bf16_cvt == 1){
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 1, true, false, true, GPUArch::gfx950>;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                    }
                                    else{
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, 3, true, 2, true, false, true, GPUArch::gfx950>;
                                        if (is_v3_api_check) {
                                            return 1;
                                        }
                                        r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a, seqlen_q_unpadded, seqlen_k_unpadded);
                                    }
                                    return r;
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 0, false, false, false, GPUArch::gfx950>;
                                const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a16_rtne";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 1, false, false, false, GPUArch::gfx950>;
                                // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a16_rtna";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 2, false, false, false, GPUArch::gfx950>;
                                // const std::string kernel_name = "bwd_v3_hd64_bf16_causal_a16_rtz";
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                    }
                }
            }
        }
    }

    return r;
}
}
} // namespace aiter
"""


def write_blobs(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / FMHA_BWD_API_FILENAME).write_text(
        FMHA_BWD_KERNEL_HEADER + FMHA_BWD_API
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory",
    )

    args = parser.parse_args()

    write_blobs(args.output_dir)
