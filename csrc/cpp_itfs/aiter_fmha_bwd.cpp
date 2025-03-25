#include <iostream>
#include "aiter_fmha_bwd.h"

fmha_bwd_traits_all get_ck_fmha_bwd_traits_all(const mask_info &mask,
    std::string dtype,
    int head_size_q,
    int head_size_v,
    bool has_dropout,
    bool is_group_mode,
    bool enable_alibi,
    bool deterministic,
    bool use_ext_asm,
    bool is_v3_atomic_fp32,
    int how_v3_bf16_cvt)
{
    return fmha_bwd_traits_all(mask,
            dtype,
            head_size_q,
            head_size_v,
            has_dropout,
            is_group_mode,
            enable_alibi,
            deterministic,
            use_ext_asm,
            is_v3_atomic_fp32,
            how_v3_bf16_cvt);
}

// fmha v3 api
float fmha_bwd_aiter(fmha_bwd_args args,
        const ck_tile::stream_config& stream_config,
        mask_info mask,
        std::string q_dtype_str,
        bool enable_alibi,
        bool deterministic,
        bool use_ext_asm,
        bool is_v3_atomic_fp32,
        int how_v3_bf16_cvt)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool is_dropout = args.p_drop > 0;
    // bool enable_ailib = args.alibi_slopes_ptr == nullptr;
    auto traits = get_ck_fmha_bwd_traits_all(mask, q_dtype_str, head_size_q, head_size_v, is_dropout, enable_alibi, deterministic, use_ext_asm, is_v3_atomic_fp32, how_v3_bf16_cvt);
    float t = -1;
    t = fmha_bwd_v3(traits, args, stream_config);
    return t;
}

// fmha v2 api
float fmha_bwd_aiter(fmha_bwd_args args,
    const ck_tile::stream_config& stream_config,
    mask_info mask,
    std::string q_dtype_str,
    bool enable_alibi,
    bool deterministic)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool is_dropout = args.p_drop > 0;
    // bool enable_ailib = args.alibi_slopes_ptr == nullptr;
    auto traits = get_ck_fmha_bwd_traits_all(mask, q_dtype_str, head_size_q, head_size_v, is_dropout, enable_alibi, deterministic, false, false, 0);
    float t = -1;
    t = fmha_bwd(traits, args, stream_config);
    return t;
}
