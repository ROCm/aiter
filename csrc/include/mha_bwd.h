#include "fmha_bwd.hpp"
#include "mask.hpp"

namespace aiter {
struct mha_bwd_traits : public fmha_bwd_traits
{
    mha_bwd_traits(const mask_info& mask,
                   std::string dtype,
                   int head_size_q,
                   int head_size_v,
                   bool has_dropout,
                   bool is_group_mode,
                   bias_enum bias_type,
                   bool deterministic,
                   bool has_dbias,
                   bool use_ext_asm,
                   bool is_v3_atomic_fp32,
                   int how_v3_bf16_cvt)
        : fmha_bwd_traits{head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          mask.type,
                          bias_type,
                          has_dbias,
                          has_dropout,
                          false, // s_randval
                          deterministic},
          use_ext_asm(use_ext_asm),
          is_v3_atomic_fp32(is_v3_atomic_fp32),
          how_v3_bf16_cvt(how_v3_bf16_cvt)
    {
    }
    bool use_ext_asm;
    bool is_v3_atomic_fp32;
    int how_v3_bf16_cvt;
};

using mha_bwd_args = fmha_bwd_args;

float mha_bwd(mha_bwd_args args,
              const ck_tile::stream_config& stream_config,
              mask_info mask,
              std::string q_dtype_str,
              bool is_group_mode,
              bias_enum bias_type,
              bool deterministic,
              bool has_dbias,
              bool use_ext_asm,
              bool is_v3_atomic_fp32,
              int how_v3_bf16_cvt);

float fmha_bwd_v3(mha_bwd_traits t, mha_bwd_args a, const ck_tile::stream_config& s);

} // namespace aiter
