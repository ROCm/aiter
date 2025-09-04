// SPDX-License-Identifier: MIT
// Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "mha_bwd_param_v3_check.h"

using namespace aiter;

bool mha_bwd_v3_check(const arch_enum gfx_version,
            mha_bwd_args args,
            const ck_tile::stream_config& stream_config,
            std::string q_dtype_str,
            bool is_group_mode,
            mask_enum mask_type,
            bias_enum bias_type,
            bool has_dbias,
            bool is_store_randval,
            bool deterministic,
            bool use_ext_asm,
            bool is_v3_atomic_fp32,
            int how_v3_bf16_cvt,
            const void* seqlen_q_padded,
            const void* seqlen_k_padded)
{

    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool has_dropout = args.p_drop > 0;
    // bool enable_ailib = args.alibi_slopes_ptr == nullptr;
    mha_bwd_traits traits(head_size_q,
                        head_size_v,
                        q_dtype_str,
                        is_group_mode,
                        mask_type,
                        bias_type,
                        has_dbias,
                        has_dropout,
                        is_store_randval,
                        deterministic,
                        use_ext_asm,
                        is_v3_atomic_fp32,
                        how_v3_bf16_cvt);
    float r = -1;
    switch (gfx_version) {
        case arch_enum::gfx942:
            r = fmha_bwd_v3_gfx942_check(traits, args, stream_config);
            break;
        case arch_enum::gfx950:
            r= fmha_bwd_v3_gfx950_check(traits, args, stream_config);
            break;
    }
    return r != -1;
}

float fmha_bwd_v3_gfx942_check(mha_bwd_traits t,
                  fmha_bwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqlen_q_padded,
                  const void* seqlen_k_padded){
    float r = -1;

    if (t.use_ext_asm == true){
        if ((t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                    (t.is_deterministic == false) && (a.hdim_q == a.hdim_v) && (a.nhead_q % a.nhead_k == 0)) {
            if((a.hdim_q > 128) && (a.hdim_q <= 192) && (a.hdim_q % 8 == 0)){
                if(t.data_type.compare("fp16") == 0){
                    if((t.is_group_mode == false) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.mask_type == mask_enum::no_mask){
                            // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_a32_psskddv";
                            r = 1;
                            return r;
                        }
                        else if((((t.mask_type != mask_enum::no_mask) && (a.seqlen_q == a.seqlen_k)) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))) &&
                                ((a.window_size_left == -1) && (a.window_size_right == 0))){
                            // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_causal_a32_psskddv";
                            r = 1;
                            return r;
                        }
                    }
                    else if((t.is_group_mode == true) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){//group mode
                        if(t.mask_type == mask_enum::no_mask){
                            // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_a32_psskddv_group";
                            r = 1;
                            return r;
                        }
                        else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0)) && (t.mask_type == mask_enum::mask_top_left)){
                            // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_causal_a32_psskddv_group";
                            r = 1;
                            return r;
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if((t.is_group_mode == false) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.mask_type == mask_enum::no_mask){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtne_psskddv";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtna_psskddv";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtz_psskddv";
                                r = 1;
                                return r;
                            }
                        }
                        else if((((t.mask_type != mask_enum::no_mask) && (a.seqlen_q == a.seqlen_k)) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))) &&
                                ((a.window_size_left == -1) && (a.window_size_right == 0))){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtne_psskddv";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtna_psskddv";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtz_psskddv";
                                r = 1;
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){//group mode
                        if(t.mask_type == mask_enum::no_mask){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtne_psskddv_group";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtna_psskddv_group";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtz_psskddv_group";
                                r = 1;
                                return r;
                            }

                        }
                        else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0)) && (t.mask_type == mask_enum::mask_top_left)){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtne_psskddv_group";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtna_psskddv_group";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtz_psskddv_group";
                                r = 1;
                                return r;
                            }
                        }
                    }
                }
            }
            else if((a.hdim_q > 64) && (a.hdim_q <= 128) && (a.hdim_q % 8 == 0)){
                if(t.data_type.compare("fp16") == 0){
                    if((t.is_group_mode == false) && (t.mask_type == mask_enum::no_mask)){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv";
                                r = 1;
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(a.hdim_q == 128){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a16";
                                r = 1;
                                return r;
                            }
                            else{
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a16_pddv";
                                r = 1;
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask)){//group mode
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(a.hdim_q == 128){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_pssk_group";
                                r = 1;
                                return r;
                            }
                            else{
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv_group";
                                r = 1;
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == false) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv";
                                    r = 1;
                                    return r;
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(a.hdim_q == 128){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a16";
                                r = 1;
                                return r;
                            }
                            else{
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a16_pddv";
                                r = 1;
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == false) && ((t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left > 0) || (a.window_size_right > 0))) || (t.mask_type == mask_enum::window_generic)){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv;
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                                r = 1;
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){//group mode
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/) && (t.mask_type == mask_enum::mask_top_left)){
                            if(a.hdim_q == 128){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_pssk_group";
                                r = 1;
                                return r;
                            }
                            else{
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv_group";
                                r = 1;
                                return r;
                            }
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if((t.is_group_mode == false) && (t.mask_type == mask_enum::no_mask)){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.how_v3_bf16_cvt == 0){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna_psskddv";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz_psskddv";
                                    r = 1;
                                    return r;
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.hdim_q == 128 && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtne";
                                    r = 1;
                                    return r;
                                }
                                else if(a.hdim_q != 128 && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtne_pddv";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.hdim_q == 128 && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtna";
                                    r = 1;
                                    return r;
                                }
                                else if(a.hdim_q != 128 && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtna_pddv";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.hdim_q == 128 && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtz";
                                    r = 1;
                                    return r;
                                }
                                else if(a.hdim_q != 128 && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtz_pddv";
                                    r = 1;
                                    return r;
                                }
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask)){ //group mode
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_pssk_group";
                                    r = 1;
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv_group";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtna_pssk_group";
                                    r = 1;
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtna_psskddv_group";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtz_pssk_group";
                                    r = 1;
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtz_psskddv_group";
                                    r = 1;
                                    return r;
                                }
                            }
                        }
                    }
                    else if((t.is_group_mode == false) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.how_v3_bf16_cvt == 0){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                            (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                            (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                    if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                    else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                        // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz_psskddv";
                                        r = 1;
                                        return r;
                                    }
                                }
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.hdim_q == 128  && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtne";
                                    r = 1;
                                    return r;
                                }
                                else if(a.hdim_q != 128  && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtne_pddv";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.hdim_q == 128  && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtna";
                                    r = 1;
                                    return r;
                                }
                                else if(a.hdim_q != 128  && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtna_pddv";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.hdim_q == 128  && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtz";
                                    r = 1;
                                    return r;
                                }
                                else if(a.hdim_q != 128  && (a.seqlen_k % 64 == 0)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtz_pddv";
                                    r = 1;
                                    return r;
                                }
                            }
                        }
                    }
                    else if((t.is_group_mode == false) && ((t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left > 0) || (a.window_size_right > 0))) || (t.mask_type == mask_enum::window_generic)){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if(t.how_v3_bf16_cvt == 0){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtne_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtne_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtne_psskddv;
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtne_psskddv";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtna_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtna_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtna_psskddv;
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtna_psskddv";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtz_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtz_psskddv";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtz_psskddv;
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtz_psskddv";
                                    r = 1;
                                    return r;
                                }
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){//group mode
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/) && (t.mask_type == mask_enum::mask_top_left)){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_pssk_group";
                                    r = 1;
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv_group";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtna_pssk_group";
                                    r = 1;
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtna_psskddv_group";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtz_pssk_group";
                                    r = 1;
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtz_psskddv_group";
                                    r = 1;
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
                            if(t.is_group_mode == false){
                                if(a.seqlen_q % 64 == 0){
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk";
                                    r = 1;
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk";
                                    r = 1;
                                    return r;
                                }
                            }
                            else{
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk_group";
                                r = 1;
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a16";
                            r = 1;
                            return r;
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((t.is_group_mode == false) && ((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left)))){
                                if(a.seqlen_q % 64 == 0){
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                    r = 1;
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                    r = 1;
                                    return r;
                                }
                            }
                            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::mask_top_left)){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk_group";
                                r = 1;
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a16";
                            r = 1;
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
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                        r = 1;
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                        r = 1;
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                        r = 1;
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                        r = 1;
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 2){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                        r = 1;
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                        r = 1;
                                        return r;
                                    }
                                }
                            }
                            else{
                                if(t.how_v3_bf16_cvt == 0){
                                    r = 1;
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    r = 1;
                                }
                                else{
                                    r = 1;
                                }
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtne";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtna";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtz";
                                r = 1;
                                return r;
                            }
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((t.is_group_mode == false) && ((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left)))){
                                if(t.how_v3_bf16_cvt == 0){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                        r = 1;
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                        r = 1;
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                        r = 1;
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                        r = 1;
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 2){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                        r = 1;
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                        r = 1;
                                        return r;
                                    }
                                }
                            }
                            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::mask_top_left)){
                                if(t.how_v3_bf16_cvt == 0){
                                    r = 1;
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    r = 1;
                                }
                                else{
                                    r = 1;
                                }
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtne";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtna";
                                r = 1;
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtz";
                                r = 1;
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

float fmha_bwd_v3_gfx950_check(mha_bwd_traits t,
                  fmha_bwd_args a,
                  const ck_tile::stream_config& s,
                  const void* seqlen_q_padded,
                  const void* seqlen_k_padded){
    float r = -1;

    if (t.use_ext_asm == true){
        if ((t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                    (t.is_deterministic == false) && (a.hdim_q == a.hdim_v) && (a.nhead_q % a.nhead_k == 0)) {
            if((a.hdim_q > 128) && (a.hdim_q <= 192) && (a.hdim_q % 8 == 0)){
                if(t.data_type.compare("fp16") == 0){
                    if((t.is_group_mode == false) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.mask_type == mask_enum::no_mask){
                            // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_a32_psskddv";
                            return r;
                        }
                        else if((((t.mask_type != mask_enum::no_mask) && (a.seqlen_q == a.seqlen_k)) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))) &&
                                ((a.window_size_left == -1) && (a.window_size_right == 0))){
                            // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_causal_a32_psskddv";
                            return r;
                        }
                    }
                    else if((t.is_group_mode == true) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){//group mode
                        if(t.mask_type == mask_enum::no_mask){
                            // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_a32_psskddv_group";
                            return r;
                        }
                        else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0)) && (t.mask_type == mask_enum::mask_top_left)){
                            // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_causal_a32_psskddv_group";
                            return r;
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if((t.is_group_mode == false) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.mask_type == mask_enum::no_mask){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtne_psskddv";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtna_psskddv";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtz_psskddv";
                                return r;
                            }
                        }
                        else if((((t.mask_type != mask_enum::no_mask) && (a.seqlen_q == a.seqlen_k)) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))) &&
                                ((a.window_size_left == -1) && (a.window_size_right == 0))){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtne_psskddv";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtna_psskddv";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtz_psskddv";
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){//group mode
                        if(t.mask_type == mask_enum::no_mask){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtne_psskddv_group";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtna_psskddv_group";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtz_psskddv_group";
                                return r;
                            }

                        }
                        else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0)) && (t.mask_type == mask_enum::mask_top_left)){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtne_psskddv_group";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtna_psskddv_group";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtz_psskddv_group";
                                return r;
                            }
                        }
                    }
                }
            }
            else if((a.hdim_q > 64) && (a.hdim_q <= 128) && (a.hdim_q % 8 == 0) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                if(t.data_type.compare("fp16") == 0){
                    if((t.is_group_mode == false) && (t.mask_type == mask_enum::no_mask)){
                        if(t.is_v3_atomic_fp32 == true){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 256 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a32_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a32_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a32_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a32_psskddv";
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a32_psskddv";
                                return r;
                            }
                        }
                        else if(t.is_v3_atomic_fp32 == false){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 256 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a16_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a16_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a16_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a16_psskddv";
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_a16_psskddv";
                                // return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask)){//group mode
                        if(t.is_v3_atomic_fp32 == true){
                            if(a.hdim_q == 128){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_pssk_group";
                                return r;
                            }
                            else{
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv_group";
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == false) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if(t.is_v3_atomic_fp32 == true){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 256 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a32_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a32_pssk";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a32_pssk";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a32_psskddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a32_psskddv";
                                    return r;
                                }
                            }
                        }
                        else if(t.is_v3_atomic_fp32 == false){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 256 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a16_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a16_pssk";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a16_pssk";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a16_pddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // // const std::string bwd_v3_name = "bwd_hd128_fp16_causal_a16_psskddv";
                                    // return r;
                                }
                            }
                        }
                    }
                    // TODO: recompiled swa kernel has random Nan issue.
                    else if((t.is_group_mode == false) && ((t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left > 0) || (a.window_size_right > 0))) || (t.mask_type == mask_enum::window_generic)){
                        if(t.is_v3_atomic_fp32 == true){
                            if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_swa_a32_psskddv";
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_swa_a32_psskddv";
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_swa_a32_psskddv;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_fp16_swa_a32_psskddv";
                                return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){//group mode
                        if((t.is_v3_atomic_fp32 == true) && (t.mask_type == mask_enum::mask_top_left)){
                            if(a.hdim_q == 128){
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_pssk_group";
                                return r;
                            }
                            else{
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv_group";
                                return r;
                            }
                        }
                    }
                }
                else if(t.data_type.compare("bf16") == 0){
                    if((t.is_group_mode == false) && (t.mask_type == mask_enum::no_mask)){
                        if(t.is_v3_atomic_fp32 == true){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 256 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a32_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a32_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a32_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a32_psskddv";
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a32_psskddv";
                                return r;
                            }
                        }
                        else if(t.is_v3_atomic_fp32 == false){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 256 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a16_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a16_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a16_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a16_psskddv";
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_a16_psskddv";
                                // return r;
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask)){ //group mode
                        if(t.is_v3_atomic_fp32 == true){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_pssk_group";
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv_group";
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtna_pssk_group";
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtna_psskddv_group";
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtz_pssk_group";
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtz_psskddv_group";
                                    return r;
                                }
                            }
                        }
                    }
                    else if((t.is_group_mode == false) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if(t.is_v3_atomic_fp32 == true){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 256 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a32_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a32_pssk";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a32_pssk";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a32_psskddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a32_psskddv";
                                    return r;
                                }
                            }
                        }
                        else if(t.is_v3_atomic_fp32 == false){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 256 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a16_pssk";
                                r = 1;
                                return r;
                            }
                            else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a16_pssk";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a16_pssk";
                                    r = 1;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a16_pddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // // const std::string bwd_v3_name = "bwd_hd128_bf16_causal_a16_psskddv";
                                    // return r;
                                }
                            }
                        }
                    }
                    else if((t.is_group_mode == false) && ((t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left > 0) || (a.window_size_right > 0))) || (t.mask_type == mask_enum::window_generic)){
                        if(t.is_v3_atomic_fp32 == true){
                            if(t.how_v3_bf16_cvt == 0){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtne_psskddv";
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtna_psskddv";
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv";
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv;
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    // const std::string bwd_v3_name = "bwd_hd128_bf16_swa_a32_rtz_psskddv";
                                    return r;
                                }
                            }
                        }
                    }
                    else if((t.is_group_mode == true) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){//group mode
                        if((t.is_v3_atomic_fp32 == true) && (t.mask_type == mask_enum::mask_top_left)){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_pssk_group";
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv_group";
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtna_pssk_group";
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtna_psskddv_group";
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.hdim_q == 128){
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtz_pssk_group";
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtz_psskddv_group";
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
                            if(t.is_group_mode == false){
                                if(a.seqlen_q % 64 == 0){
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk";
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk";
                                    return r;
                                }
                            }
                            else{
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk_group";
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a16";
                            return r;
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((t.is_group_mode == false) && ((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left)))){
                                if(a.seqlen_q % 64 == 0){
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                    return r;
                                }
                                else{
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                    return r;
                                }
                            }
                            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::mask_top_left)){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk_group";
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a16";
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
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 2){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                        return r;
                                    }
                                }
                            }
                            else{
                                if(t.how_v3_bf16_cvt == 0){
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                }
                                else{
                                }
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtne";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtna";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtz";
                                return r;
                            }
                        }
                    }
                    else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                            if((t.is_group_mode == false) && ((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left)))){
                                if(t.how_v3_bf16_cvt == 0){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                        return r;
                                    }
                                }
                                else if(t.how_v3_bf16_cvt == 2){
                                    if(a.seqlen_q % 64 == 0){
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                        return r;
                                    }
                                    else{
                                        // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                        return r;
                                    }
                                }
                            }
                            else if((t.is_group_mode == true) && (t.mask_type == mask_enum::mask_top_left)){
                                if(t.how_v3_bf16_cvt == 0){
                                }
                                else if(t.how_v3_bf16_cvt == 1){
                                }
                                else{
                                }
                                return r;
                            }
                        }
                        else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            if(t.how_v3_bf16_cvt == 0){
                                const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtne";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtna";
                                return r;
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtz";
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
