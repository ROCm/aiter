// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fmha_fwd.hpp"

#include <iostream>
#include "aiter_hip_common.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "mha_fwd.h"

namespace aiter {

struct __attribute__((packed))
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
    unsigned int s_seq_len;
    p3 _p6;
    unsigned int s_Seqs;
    p3 _p7;
    unsigned int s_Ts;
    p3 _p8;
    unsigned int s_Hs;
    p3 _p9;
    unsigned int s_BAs;
    p3 _p10;
    unsigned int s_gqa;
    p3 _p11;
    unsigned int s_kv_Seqs;
    p3 _p12;
    unsigned int s_kv_Hs;
    p3 _p13;
    unsigned int s_kv_BAs;
    p3 _p14;
    unsigned int s_opt;
    p3 _p15;
}

// TODO: fix here
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

// TODO: fix here
template <ck_tile::index_t HDim_,
          typename DataType_,
          int mask_type_,
          bool kIsSEQPad_,
          bool kIsHDPad_>
struct fmha_fwd_v3_traits_
{
    static constexpr ck_tile::index_t HDim    = HDim_;
    using DataType                            = ck_tile::remove_cvref_t<DataType_>;
    static constexpr int mask_type            = mask_type_;
    static constexpr bool kIsSEQPad           = kIsSEQPad_;
    static constexpr bool kIsHDPad            = kIsHDPad_;
};

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
        // TODO: FIX here
        int bdx = 512;
        int gdx = ((q_seq_lens+sub_Q-1)/sub_Q + tg_div - 1)/tg_div;
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
float fmha_fwd_v3_(const ck_tile::stream_config& s, fmha_fwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_fwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhafwdV3Name<dq_dk_dv_v3_traits_>::fwd_v3_name << std::flush;
    fmha_fwd_v3_args args;
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

    args.Ts   = FmhafwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    auto traits = fmha_fwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhafwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhafwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_fwd_v3_kernel impl(FmhafwdV3Name<dq_dk_dv_v3_traits_>::fwd_v3_name, FmhafwdV3Buf<dq_dk_dv_v3_traits_>::fwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_fwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); }
    );
}

// TODO: fix here
template <typename fmha_fwd_v3_traits_> struct FmhaFwdV3Name;
template <typename fmha_fwd_v3_traits_> struct FmhaFwdV3Buf;
template <typename fmha_fwd_v3_traits_> struct FmhaFwdV3Ts;


float fmha_fwd_v3(mha_fwd_traits t, fmha_fwd_args a, const ck_tile::stream_config& s){
    float r = -1;
/*  
1.only support bhsd/bshd
2.LSE is forced to bhsd
3.Qlen should be equal with KVlen
*/
    return r;
}
} // namespace aiter
