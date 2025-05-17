// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd.hpp"

#include <iostream>
#include "aiter_hip_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "mha_fwd.h"

namespace aiter {

struct __attribute__((packed)) fmha_fwd_v3_args
{
    void *ptr_d;
    p2 _p0;
    void *ptr_q;
    p2 _p1;
    void *ptr_k;
    p2 _p2;
    void *ptr_v;
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
    unsigned int kv_Seqs;
    p3 _p12;
    unsigned int kv_Hs;
    p3 _p13;
    unsigned int kv_BAs;
    p3 _p14;
    unsigned int opt;
    p3 _p15;
}

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
          ck_tile::index_t MaskType,
          bool kIsSEQPad_,
          bool kIsHDPad_>
struct fmha_fwd_kernel_selector
{
    using DataType                              = ck_tile::remove_cvref_t<DataType_>;
    static constexpr ck_tile::index_t HDim      = HDim_;
    static constexpr ck_tile::index_t MaskType  = mask_type_;
    static constexpr bool kIsSEQPad             = kIsSEQPad_;
    static constexpr bool kIsHDPad              = kIsHDPad_;
};


template <typename fmha_fwd_kernel_selector> struct FmhaFwdV3Name;
// ######################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_causal"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16_causal"; };

template <typename fmha_fwd_kernel_selector> struct FmhaFwdV3Buf;
// #####################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_causal.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16_causal.co"; };

template <typename fmha_fwd_kernel_selector> struct FmhaFwdV3Ts;
// ####################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };


class fmha_fwd_v3_kernel
{
    public:
    fmha_fwd_v3_kernel(const char *name, const char *hsaco)
    {
        int length = strlen(name);
        std::string kernel_func_name = "_ZN5aiter" + std::to_string(length) + name + "E";
        std::string AITER_ASM_DIR = "/workspace/aiter/hsa/gfx942/";
        HIP_CALL(hipModuleLoad(&module, (AITER_ASM_DIR + "fmha_v3_fwd/" + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_func_name.c_str()));
    }

    void
    launch_kernel(fmha_fwd_v3_traits fmha_v3_traits, fmha_fwd_v3_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int tg_div = (fmha_v3_traits.mask != 0) ? 2 : 1;

        int bdx = 512;
        int gdx = ((fmha_v3_traits.s + fmha_v3_traits.sub_qo - 1) / fmha_v3_traits.sub_qo + tg_div - 1) / tg_div;
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

template <typename fmha_fwd_kernel_selector>
float fmha_fwd_v3_dispatcher(const ck_tile::stream_config& s, fmha_fwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << FmhafwdV3Name<fmha_fwd_kernel_selector>::fwd_v3_name << std::flush;
    fmha_fwd_v3_args args;
    args.ptr_d   = a.d_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_lse = a.lse_ptr;

    args.scalar  = a.scale;
    args.seq_len = a.seqlen_q;
    args.Seqs    = a.stride_q * 2;
    args.Ts      = FmhafwdV3Ts<fmha_fwd_kernel_selector>::ts_qo * a.hdim_q * 2;
    args.Hs      = a.nhead_stride_q * 2;
    args.BAs     = a.batch_stride_q * 2;
    args.gqa      = a.nhead_q / a.nhead_k;
    args.Seqs_kv  = a.stride_k * 2;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.opt      = 1;

    auto traits = fmha_fwd_v3_traits{a.batch,
                                     a.nhead_q,
                                     a.seqlen_q,
                                     a.hdim_q,
                                     a.mask_type,
                                     FmhafwdV3Ts<fmha_fwd_kernel_selector>::ts_qo,
                                     FmhafwdV3Ts<fmha_fwd_kernel_selector>::ts_kv};

    static thread_local fmha_fwd_v3_kernel impl(FmhafwdV3Name<fmha_fwd_kernel_selector>::fwd_v3_name, FmhafwdV3Buf<fmha_fwd_kernel_selector>::fwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); }
    );
}

float fmha_fwd_v3(mha_fwd_traits t, fmha_fwd_args a, const ck_tile::stream_config& s){
    float r = -1;
    // TODO: 
    // 1.only support bhsd/bshd
    // 2.LSE is forced to bhsd
    if (t.use_ext_asm == true) {
        if (t.data_type.compare("bf16") == 0) {
            if ((t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) && 
                        (t.has_lse == true) && (a.seqlen_q == a.seqlen_k) && (a.seq_len_q % 256 == 0) &&
                        // TODO: need this two?
                        (t.is_deterministic == false) && (a.hdim_q == a.hdim_v)) {
                if (t.mask_type == mask_enum::no_mask) {
                    using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 0, false, false>;
                    r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a);
                }
                else if ((t.mask_type == mask_enum::mask_top_left) || (t.mask_type == mask_enum::mask_bottom_right)) {
                    using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 1, false, false>;
                    r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a);
                }
            }
        }
        else if (t.data_type.compare("fp16") == 0) {
            if ((t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) && 
                        (t.has_lse == true) && (a.seqlen_q == a.seqlen_k) && (a.seq_len_q % 256 == 0) &&
                        // TODO: need this two?
                        (t.is_deterministic == false) && (a.hdim_q == a.hdim_v)) {
                if (t.mask_type == mask_enum::no_mask) {
                    using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdFp16, 128, 0, false, false>;
                    r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a);
                }
                else if ((t.mask_type == mask_enum::mask_top_left) || (t.mask_type == mask_enum::mask_bottom_right)) {
                    using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdFp16, 128, 1, false, false>;
                    r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a);
                }
            }
        }
    }
    return r;
}
} // namespace aiter
