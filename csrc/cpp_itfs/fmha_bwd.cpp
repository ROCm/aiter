#include <iostream>
#include "fmha_bwd.hpp"
#include "mask.hpp"
#include "aiter_hip_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define HSA_KERNEL "kernel_func"

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
    unsigned int Hs_q;
    p1 _p6;
    unsigned int BAs_q;
    p1 _p7;
    unsigned int Seqs_q;
    p1 _p8;
    unsigned int Hs_k;
    p1 _p9;
    unsigned int BAs_k;
    p1 _p10;
    unsigned int Seqs_k;
    p1 _p11;
    unsigned int Hs_v;
    p1 _p12;
    unsigned int BAs_v;
    p1 _p13;
    unsigned int Seqs_v;
    p1 _p14;
    unsigned int Hs_do;
    p1 _p15;
    unsigned int BAs_do;
    p1 _p16;
    unsigned int Seqs_do;
    p1 _p17;
    unsigned int Hs_dk;
    p1 _p18;
    unsigned int BAs_dk;
    p1 _p19;
    unsigned int Seqs_dk;
    p1 _p20;
    unsigned int Hs_dv;
    p1 _p21;
    unsigned int BAs_dv;
    p1 _p22;
    unsigned int Seqs_dv;
    p1 _p23;
};

struct fmha_bwd_v3_traits
{
    int b;
    int h;
    int s;
    int d;

    int mask;
    int ts_qo;
    int ts_kv;
};

template <ck_tile::index_t HDim_,
          typename DataType_,
          bool kIsCausal_,
          bool kIsAtomic32_,
          ck_tile::index_t BF16Cvt_,
          bool kIsSEQPad_,
          bool kIsHDPad_>
struct fmha_bwd_dq_dk_dv_v3_traits_
{
    static constexpr ck_tile::index_t HDim    = HDim_;
    using DataType                            = ck_tile::remove_cvref_t<DataType_>;
    static constexpr bool kIsCausal           = kIsCausal_;
    static constexpr bool kIsAtomic32         = kIsAtomic32_;
    static constexpr ck_tile::index_t BF16Cvt = BF16Cvt_;
    static constexpr bool kIsSEQPad           = kIsSEQPad_;
    static constexpr bool kIsHDPad            = kIsHDPad_;
};

template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3Name;
// ########################################################|HDim|    DataType|kIsCausal|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a16_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a16_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a16_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a32_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a32_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a32_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a16_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a16_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a16_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a32_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a32_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a32_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_fp16_a16"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_fp16_a32"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_fp16_causal_a16"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_fp16_causal_a32"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a16_rtne_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      1,    false,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a16_rtna_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      2,    false,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a16_rtz_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a32_rtne_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a32_rtna_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_a32_rtz_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a16_rtne_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      1,    false,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a16_rtna_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      2,    false,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a16_rtz_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a32_rtne_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a32_rtna_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_bf16_causal_a32_rtz_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_fp16_a16_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_fp16_a32_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_fp16_causal_a16_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "bwd_v3_fp16_causal_a32_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      1,     true,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      2,     true,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      1,     true,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      2,     true,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,    false,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_fp16_a16"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,    false,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,     true,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_fp16_causal_a16"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,     true,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk"; };

template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3Buf;
// #######################################################|HDim|    DataType|kIsCausal|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a32_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a32_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a32_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a32_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a32_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a32_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_fp16_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_fp16_a32.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_fp16_causal_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_fp16_causal_a32.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a16_rtne_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      1,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a16_rtna_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      2,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a16_rtz_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a16_rtne_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      1,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a16_rtna_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      2,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a16_rtz_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_bf16_causal_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_fp16_a16_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_fp16_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_fp16_causal_a16_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_fp16_causal_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtne_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      1,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtna_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      2,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtz_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtne_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      1,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtna_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      2,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtz_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,    false,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,    false,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_a32_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,     true,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_causal_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,     true,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_causal_a32_pssk.co"; };

template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3Ts;
// ######################################################|HDim|    DataType|kIsCausal|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      1,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      2,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      1,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      2,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      1,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      2,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      1,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      2,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,      false,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,       true,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,      false,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,       true,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      0,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      1,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,      false,      2,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      1,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,    false,       true,      2,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      0,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      1,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,      false,      2,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      1,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,     true,       true,      2,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,      false,      0,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,    false,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,      false,      0,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,     true,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      0,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      1,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,      false,      2,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      0,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      1,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,    false,       true,      2,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      0,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      1,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,      false,      2,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      0,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      1,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,     true,       true,      2,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,    false,      false,      0,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,    false,       true,      0,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,     true,      false,      0,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,     true,       true,      0,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };

class fmha_bwd_v3_kernel
{
    public:
    fmha_bwd_v3_kernel(const char *name, const char *hsaco)
    {
        const char *AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
        HIP_CALL(hipModuleLoad(&module, (std::string(AITER_ASM_DIR) + "fmha_v3_bwd/" + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, name));
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
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
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
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
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
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
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
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << std::flush;
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
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a);  return hipPeekAtLastError() == hipSuccess; },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_);  return hipPeekAtLastError() == hipSuccess; }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_>
float fmha_bwd_v3_gen_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << std::flush;
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
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a);  return hipPeekAtLastError() == hipSuccess; },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_);  return hipPeekAtLastError() == hipSuccess; }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
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
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a);  return hipPeekAtLastError() == hipSuccess; },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_);  return hipPeekAtLastError() == hipSuccess; },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a);  return hipPeekAtLastError() == hipSuccess; }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_gen_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
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
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a);  return hipPeekAtLastError() == hipSuccess; },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_);  return hipPeekAtLastError() == hipSuccess; },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a);  return hipPeekAtLastError() == hipSuccess; }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_genl_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
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
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(HSA_KERNEL, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a);  return hipPeekAtLastError() == hipSuccess; },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_);  return hipPeekAtLastError() == hipSuccess; },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a);  return hipPeekAtLastError() == hipSuccess; }
    );
}

float fmha_bwd_v3(fmha_bwd_traits_all t, fmha_bwd_args a, const ck_tile::stream_config& s){
    float r = -1;

    if (t.use_ext_asm == true){
        if ((t.is_group_mode == false) && (t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                    (t.is_deterministic == false) && (a.hdim_q == a.hdim_v) && (a.nhead_q % a.nhead_k == 0)) {
            if((a.hdim_q > 64) && (a.hdim_q <= 128) && (a.hdim_q % 8 == 0)){
                if(t.data_type.compare("fp16") == 0){
                    if(t.mask_type == mask_enum::no_mask){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, false, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_a32";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_a32_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_a32_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_a32_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_a32_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(a.hdim_q == 128){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_a16";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, false, true>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_a16_pddv";
                                r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, false, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_causal_a32";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_fp16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_fp16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_fp16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_fp16_causal_a32_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(a.hdim_q == 128){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_causal_a16";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, false, true>;
                                // const std::string bwd_v3_name = "bwd_v3_fp16_causal_a16_pddv";
                                r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if(t.mask_type == mask_enum::no_mask){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.how_v3_bf16_cvt == 0){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, false, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtne";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtne_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtne_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtne_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtne_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, false, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtna";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtna_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtna_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtna_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtna_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, false, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtz";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtz_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtz_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtz_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a32_rtz_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.hdim_q == 128){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a16_rtne";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, false, true>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a16_rtne_pddv";
                                    r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.hdim_q == 128){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 1, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a16_rtna";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 1, false, true>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a16_rtna_pddv";
                                    r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.hdim_q == 128){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 2, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a16_rtz";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 2, false, true>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_a16_rtz_pddv";
                                    r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                            }
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.how_v3_bf16_cvt == 0){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, false, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtne";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtne_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtne_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtne_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtne_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, false, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtna";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtna_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtna_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtna_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtna_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, false, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtz";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtz_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtz_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtz_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, true, true>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a32_rtz_psskddv";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.hdim_q == 128){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a16_rtne";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, false, true>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a16_rtne_pddv";
                                    r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.hdim_q == 128){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 1, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a16_rtna";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 1, false, true>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a16_rtna_pddv";
                                    r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.hdim_q == 128){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 2, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a16_rtz";
                                    r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 2, false, true>;
                                    // const std::string bwd_v3_name = "bwd_v3_bf16_causal_a16_rtz_pddv";
                                    r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                    return r;
                                }
                            }
                        }
                    }
                }
            }
            else if(a.hdim_q == 64){
                if(t.data_type.compare("fp16") == 0){
                    if(t.mask_type == mask_enum::no_mask){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(a.seqlen_q % 64 == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, true, 0, true, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, true, 0, true, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, false, 0, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a16";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, true, 0, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, true, 0, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, false, 0, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a16";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if(t.mask_type == mask_enum::no_mask){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 0, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 0, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 1, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 1, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 2, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 2, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 0, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtne";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 1, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtna";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 2, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtz";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if(t.how_v3_bf16_cvt == 0){
                                    if(a.seqlen_q % 64 == 0){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 0, true, false>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else{
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 0, true, false>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    if(a.seqlen_q % 64 == 0){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 1, true, false>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else{
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 1, true, false>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 2){
                                    if(a.seqlen_q % 64 == 0){
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 2, true, false>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                    else{
                                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 2, true, false>;
                                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                        return r;
                                    }
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 0, false, false>;
                                const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtne";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 1, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtna";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 2, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtz";
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

float fmha_bwd_aiter(const void* q_ptr,
        const void* k_ptr,
        const void* v_ptr,
        const void* bias_ptr, // alibi slopes
        const void* o_ptr,
        const void* lse_ptr,
        const void* do_ptr,
        void* d_ptr,
        // void* rand_val_ptr, nullptr
        void* dq_ptr,
        void* dk_ptr,
        void* dv_ptr,
        // void* dbias_ptr, // TODO: fix this
        void* dq_acc_ptr,
        const void* seqstart_q_ptr, // nullptr
        const void* seqstart_k_ptr, // nullptr
        const void* seqlen_k_ptr, // nullptr
        ck_tile::index_t seqlen_q,
        ck_tile::index_t seqlen_k,
        ck_tile::index_t batch,
        ck_tile::index_t max_seqlen_q,
        ck_tile::index_t max_seqlen_k,
        ck_tile::index_t hdim_q,
        ck_tile::index_t hdim_v,
        ck_tile::index_t nhead_q,
        ck_tile::index_t nhead_k,
        float scale, // softmax_scale
        ck_tile::index_t stride_q,
        ck_tile::index_t stride_k,
        ck_tile::index_t stride_v,
        ck_tile::index_t stride_bias, // if alibi, b*h need set this to h, 1*h need set this to 0
        ck_tile::index_t stride_o,
        // ck_tile::index_t stride_randval, 0
        ck_tile::index_t stride_do,
        ck_tile::index_t stride_dq_acc,
        ck_tile::index_t stride_dq,
        ck_tile::index_t stride_dk,
        ck_tile::index_t stride_dv,
        // ck_tile::index_t stride_dbias, // TODO: fix this
        ck_tile::index_t nhead_stride_q,
        ck_tile::index_t nhead_stride_k,
        ck_tile::index_t nhead_stride_v,
        // ck_tile::index_t nhead_stride_bias, // TODO: fix this
        ck_tile::index_t nhead_stride_o,
        // ck_tile::index_t nhead_stride_randval, 0
        ck_tile::index_t nhead_stride_do,
        ck_tile::index_t nhead_stride_lsed,
        ck_tile::index_t nhead_stride_dq_acc,
        ck_tile::index_t nhead_stride_dq,
        ck_tile::index_t nhead_stride_dk,
        ck_tile::index_t nhead_stride_dv,
        // ck_tile::index_t nhead_stride_dbias, // TODO: fix this
        ck_tile::index_t batch_stride_q,
        ck_tile::index_t batch_stride_k,
        ck_tile::index_t batch_stride_v,
        // ck_tile::index_t batch_stride_bias, // TODO: fix this
        ck_tile::index_t batch_stride_o,
        // ck_tile::index_t batch_stride_randval, 0
        ck_tile::index_t batch_stride_do,
        ck_tile::index_t batch_stride_lsed,
        ck_tile::index_t batch_stride_dq_acc,
        ck_tile::index_t batch_stride_dq,
        ck_tile::index_t batch_stride_dk,
        ck_tile::index_t batch_stride_dv,
        // ck_tile::index_t batch_stride_dbias, TODO: fix this
        ck_tile::index_t split_stride_dq_acc,
        float p_drop,
        // float p_undrop, // calculate here
        std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>> drop_seed_offset,
        mask_info mask,
        std::string q_dtype_str,
        // int head_size_q,  // hdim_q
        // int head_size_v,  // hdim_v
        // bool is_dropout, // calculate here
        bool enable_alibi,
        bool deterministic,
        bool is_v3_atomic_fp32,
        int how_v3_bf16_cvt)
{

}
