// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "mha_fwd_param_v3_check.h"

using namespace aiter;

bool mha_fwd_v3_check(const arch_enum gfx_version,
                    mha_fwd_args args,
                    const ck_tile::stream_config& stream_config,
                    std::string q_dtype_str,
                    bool is_group_mode,
                    mask_enum mask_type,
                    bias_enum bias_type,
                    bool has_lse,
                    bool use_ext_asm,
                    int how_v3_bf16_cvt,
                    const void* seqstart_q_padding_ptr,
                    const void* seqstart_k_padding_ptr)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool has_dropout = args.p_drop > 0.f;
    mha_fwd_traits traits(head_size_q,
                        head_size_v,
                        q_dtype_str,
                        is_group_mode,
                        args.logits_soft_cap > 0.f,
                        mask_type,
                        bias_type,
                        has_lse,
                        has_dropout,
                        use_ext_asm,
                        how_v3_bf16_cvt,
                        args.min_seqlen_q != 0);
    float r = -1;
    switch (gfx_version) {
        case arch_enum::gfx942:
            r = fmha_fwd_v3_gfx942_check(traits, args, stream_config);
            break;
        case arch_enum::gfx950:
            r = fmha_fwd_v3_gfx950_check(traits, args, stream_config);
            break;
    }
    return r != -1;
}

float fmha_fwd_v3_gfx942_check(mha_fwd_traits t, mha_fwd_args a, const ck_tile::stream_config& s, const void* seqstart_q_padding_ptr, const void* seqstart_k_padding_ptr) {
    float r = -1;
    if (t.use_ext_asm == true) {
        if (t.data_type.compare("bf16") == 0) {
            if ((t.bias_type == bias_enum::no_bias) && (t.has_dropout == false) &&
                    (a.hdim_q == 128) && (a.hdim_q == a.hdim_v)) {
                if (t.is_group_mode == false) {
                    if ((t.mask_type == mask_enum::mask_bottom_right || (a.seqlen_q == a.seqlen_k && t.mask_type == mask_enum::mask_top_left)) &&
                            ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                        if (t.how_v3_bf16_cvt == 0) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                    }
                    else if (t.mask_type == mask_enum::no_mask) {
                        if (t.how_v3_bf16_cvt == 0) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                    }
                }
                else {
                    if (t.mask_type == mask_enum::mask_bottom_right) {
                        if (t.how_v3_bf16_cvt == 0) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                    }
                    else if (t.mask_type == mask_enum::no_mask) {
                        if (t.how_v3_bf16_cvt == 0) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2) {
                            if (t.has_lse == false) {
                                r = 1;
                            }
                            else {
                                if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                    r = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return r;
}
float fmha_fwd_v3_gfx950_check(mha_fwd_traits t, mha_fwd_args a, const ck_tile::stream_config& s, const void* seqstart_q_padding_ptr, const void* seqstart_k_padding_ptr) {
    float r = -1;
    if (t.use_ext_asm == true) {
        if (t.data_type.compare("bf16") == 0) {
            if ((t.bias_type == bias_enum::no_bias) && (t.has_dropout == false) &&
                    (a.hdim_q == 128) && (a.hdim_q == a.hdim_v)) {
                if (t.is_group_mode == false) {
                    if ((t.mask_type == mask_enum::mask_bottom_right || (a.seqlen_q == a.seqlen_k && t.mask_type == mask_enum::mask_top_left)) &&
                            ((a.window_size_left == -1) && (a.window_size_right == 0))) {
                        if (t.has_lse == false) {
                            r = 1;
                        }
                        else {
                            if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                r = 1;
                            }
                        }
                    }
                    else if (t.mask_type == mask_enum::no_mask) {
                        if (t.has_lse == false) {
                            r = 1;
                        }
                        else {
                            if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                r = 1;
                            }
                        }
                    }
                }
                else {
                    if (t.mask_type == mask_enum::mask_bottom_right) {
                        if (t.has_lse == false) {
                            r = 1;
                        }
                        else {
                            if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                r = 1;
                            }
                        }
                    }
                    else if (t.mask_type == mask_enum::no_mask) {
                        if (t.has_lse == false) {
                            r = 1;
                        }
                        else {
                            if (a.batch_stride_lse >= a.nhead_stride_lse) {
                                r = 1;
                            }
                        }
                    }
                }
            }
        }
    }
    return r;
}
