#include "fmha_bwd.hpp"
#include "mask.hpp"

struct fmha_bwd_traits_all: public fmha_bwd_traits
{
    fmha_bwd_traits_all(const mask_info &mask,
        std::string dtype,
        int head_size_q,
        int head_size_v,
        bool has_dropout,
        bool is_group_mode,
        bool enable_alibi,
        bool deterministic,
        bool use_ext_asm,
        bool is_v3_atomic_fp32,
        int how_v3_bf16_cvt): fmha_bwd_traits{head_size_q,
            head_size_v,
            dtype,
            is_group_mode,
            mask.type,
            enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
            false,    // has_dbias
            has_dropout,
            false, // s_randval
            deterministic}, 
            use_ext_asm(use_ext_asm),
            is_v3_atomic_fp32(is_v3_atomic_fp32),
            how_v3_bf16_cvt(how_v3_bf16_cvt) {}
    bool use_ext_asm;
    bool is_v3_atomic_fp32;
    int how_v3_bf16_cvt;
};

float fmha_bwd_aiter(fmha_bwd_args args,
    const ck_tile::stream_config& stream_config,
    mask_info mask,
    std::string q_dtype_str,
    bool is_group_mode,
    bool enable_alibi,
    bool deterministic,
    bool use_ext_asm,
    bool is_v3_atomic_fp32,
    int how_v3_bf16_cvt);

float fmha_bwd_v3(fmha_bwd_traits_all t, fmha_bwd_args a, const ck_tile::stream_config& s);