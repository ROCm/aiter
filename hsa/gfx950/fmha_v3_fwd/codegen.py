# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from pathlib import Path
from typing import Optional


GEN_DIR = ""  # in Cmake, have to generate files in same folder

FMHA_FWD_API_FILENAME = "asm_fmha_fwd_v3_gfx950.cpp"

FMHA_FWD_KERNEL_HEADER = """// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.\n
"""

FMHA_FWD_API = """#include <hip/hip_fp16.h>
#include "mha_fwd.h"

namespace aiter {

// ######################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_causal"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_causal"; };
// template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16"; };
// template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16"; };
// template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16_causal"; };
// template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_fp16_causal"; };

// ######################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch |        BF16Cvt | kIsGroupMode_ |
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950, 1,        true>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_group"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950, 1,        true>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_group"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950, 1,        true>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_causal_group"; };
template<> struct FmhaFwdV3Name<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950, 1,        true>> { static constexpr const char * fwd_v3_name = "fmha_fwd_hd128_bf16_causal_group"; };

// #####################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_causal.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_causal.co"; };
// template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16.co"; };
// template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16.co"; };
// template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16_causal.co"; };
// template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_fp16_causal.co"; };

// #####################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch |        BF16Cvt | kIsGroupMode_ |
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950, 1,        true>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_group.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950, 1,        true>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_group.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950, 1,        true>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_causal_group.co"; };
template<> struct FmhaFwdV3Buf<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950, 1,        true>> { static constexpr const char * fwd_v3_buf = "fwd_hd128_bf16_causal_group.co"; };

// ####################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
// template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     0,          GPUArch::gfx950>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
// template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      0,      false,      false,     1,          GPUArch::gfx950>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
// template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     0,          GPUArch::gfx950>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
// template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdFp16, 128,      1,      false,      false,     1,          GPUArch::gfx950>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };

// ####################################################| DataType | HDim | MaskType | kIsSEQPad | kIsHDPad | kStoreLSE | GPUArch |        BF16Cvt | kIsGroupMode_ |
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     0,          GPUArch::gfx950, 1,        true>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      0,      false,      false,     1,          GPUArch::gfx950, 1,        true>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     0,          GPUArch::gfx950, 1,        true>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };
template<> struct FmhaFwdV3Ts<fmha_fwd_kernel_selector<FmhaFwdBf16, 128,      1,      false,      false,     1,          GPUArch::gfx950, 1,        true>> { static constexpr int ts_qo = 256; static constexpr int ts_kv = 64; };

namespace gfx950{
class fmha_fwd_v3_kernel
{
    public:
    fmha_fwd_v3_kernel(const char *name, const char *hsaco)
    {
        int length = strlen(name);
        std::string kernel_func_name = "_ZN5aiter" + std::to_string(length) + name + "E";
        std::string AITER_ASM_DIR = std::string(std::getenv("AITER_ASM_DIR")) + "fmha_v3_fwd/";
        HIP_CALL(hipModuleLoad(&module, (AITER_ASM_DIR + hsaco).c_str()));
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
        int gdx = ((fmha_v3_traits.s + fmha_v3_traits.ts_qo - 1) / fmha_v3_traits.ts_qo + tg_div - 1) / tg_div;
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
float fmha_fwd_v3_dispatcher(const ck_tile::stream_config& s, mha_fwd_args a,
                             const void* seqstart_q_padding_ptr, const void* seqstart_k_padding_ptr)
{
    if(s.log_level_ > 0)
        std::cout << ", " << FmhaFwdV3Name<fmha_fwd_kernel_selector>::fwd_v3_name << std::flush;

    int tune_opt = 5;
    if (a.mask_type != 0 && ((a.nhead_q % 8 != 0) || (a.seqlen_q > 16384))) //if num_head is not 8N, or seqlen is bigger than 16K, downgrade to 2and3
    {
        tune_opt -= 2;
    }

    fmha_fwd_v3_args args;
    args.ptr_o   = a.o_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_lse = a.lse_ptr;

    args.scalar  = a.scale_s;
    args.s_seq_len = a.seqlen_q;
    args.s_Seqs    = a.stride_q * 2;
    args.s_Ts      = FmhaFwdV3Ts<fmha_fwd_kernel_selector>::ts_qo * a.stride_q * 2;
    args.s_Hs      = a.nhead_stride_q * 2;
    args.s_Bs     = a.batch_stride_q * 2;
    args.s_gqa      = a.nhead_q / a.nhead_k;
    args.s_k_Seqs  = a.stride_k * 2;
    args.s_k_Hs    = a.nhead_stride_k * 2;
    args.s_k_Bs   = a.batch_stride_k * 2;
    args.s_opt      = tune_opt;
    args.s_lse    = fmha_fwd_kernel_selector::kStoreLSE;
    args.s_kv_seq_len = a.seqlen_k;
    args.s_qk_head_dim = a.hdim_q;
    args.s_v_head_dim = a.hdim_v;
    args.s_q_head_num = a.nhead_q;
    args.s_v_Seqs = a.stride_v * 2;
    args.s_v_Hs = a.nhead_stride_v * 2;
    args.s_v_Bs = a.batch_stride_v * 2;
    args.s_o_Seqs = a.stride_o * 2;
    args.s_o_Hs = a.nhead_stride_o * 2;
    args.s_o_Bs = a.batch_stride_o * 2;

    args.s_lse_Hs = a.nhead_stride_lse * 4;
    args.ptr_qseq = a.seqstart_q_ptr;
    args.ptr_kseq = a.seqstart_k_ptr;
    args.ptr_qseq_padding = seqstart_q_padding_ptr == nullptr ? a.seqstart_q_ptr : seqstart_q_padding_ptr;
    args.ptr_kseq_padding = seqstart_k_padding_ptr == nullptr ? a.seqstart_k_ptr : seqstart_k_padding_ptr;

    auto traits = fmha_fwd_v3_traits{a.batch,
                                     a.nhead_q,
                                     a.seqlen_q,
                                     a.hdim_q,
                                     a.mask_type,
                                     FmhaFwdV3Ts<fmha_fwd_kernel_selector>::ts_qo,
                                     FmhaFwdV3Ts<fmha_fwd_kernel_selector>::ts_kv};

    static thread_local fmha_fwd_v3_kernel impl(FmhaFwdV3Name<fmha_fwd_kernel_selector>::fwd_v3_name, FmhaFwdV3Buf<fmha_fwd_kernel_selector>::fwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); }
    );
}

float fmha_fwd_v3(mha_fwd_traits t, mha_fwd_args a, const ck_tile::stream_config& s, const void* seqstart_q_padding_ptr, const void* seqstart_k_padding_ptr,
                  bool is_v3_api_check) {
    float r = -1;
    if (t.use_ext_asm == true) {
        if (t.data_type.compare("bf16") == 0) {
            if ((t.bias_type == bias_enum::no_bias) && (t.has_dropout == false) &&
                    (a.hdim_q == 128) && (a.hdim_q == a.hdim_v)) {
                if (t.is_group_mode == false) {
                    if ((t.mask_type == mask_enum::mask_bottom_right || (a.seqlen_q == a.seqlen_k && t.mask_type == mask_enum::mask_top_left)) &&
                            ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                        if (t.has_lse == false) {
                            using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 1, false, false, 0, GPUArch::gfx950>;
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a, seqstart_q_padding_ptr, seqstart_k_padding_ptr);
                        }
                        else {
                            if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 1, false, false, 1, GPUArch::gfx950>;
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a, seqstart_q_padding_ptr, seqstart_k_padding_ptr);
                            }
                        }
                    }
                    else if (t.mask_type == mask_enum::no_mask) {
                        if (t.has_lse == false) {
                            using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 0, false, false, 0, GPUArch::gfx950>;
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a, seqstart_q_padding_ptr, seqstart_k_padding_ptr);
                        }
                        else {
                            if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 0, false, false, 1, GPUArch::gfx950>;
                                if (is_v3_api_check) {
                                    return 1;
                                }
                                r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a, seqstart_q_padding_ptr, seqstart_k_padding_ptr);
                            }
                        }
                    }
                }
                else {
                    if (t.mask_type == mask_enum::mask_bottom_right) {
                        if (t.has_lse == false) {
                            using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 1, false, false, 0, GPUArch::gfx950, 1, true>;
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a, seqstart_q_padding_ptr, seqstart_k_padding_ptr);
                        }
                        else {
                            using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 1, false, false, 1, GPUArch::gfx950, 1, true>;
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a, seqstart_q_padding_ptr, seqstart_k_padding_ptr);
                        }
                    }
                    else if (t.mask_type == mask_enum::no_mask) {
                        if (t.has_lse == false) {
                            using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 0, false, false, 0, GPUArch::gfx950, 1, true>;
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a, seqstart_q_padding_ptr, seqstart_k_padding_ptr);
                        }
                        else {
                            using fmha_fwd_kernel = fmha_fwd_kernel_selector<FmhaFwdBf16, 128, 0, false, false, 1, GPUArch::gfx950, 1, true>;
                            if (is_v3_api_check) {
                                return 1;
                            }
                            r = fmha_fwd_v3_dispatcher<fmha_fwd_kernel>(s, a, seqstart_q_padding_ptr, seqstart_k_padding_ptr);
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
    (output_dir / FMHA_FWD_API_FILENAME).write_text(
        FMHA_FWD_KERNEL_HEADER + FMHA_FWD_API
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen fmha fwd asm kernel API",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory",
    )

    args = parser.parse_args()
    write_blobs(args.output_dir)
