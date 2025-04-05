#include "fmha_fwd.hpp"
#include "mask.hpp"

namespace aiter {
struct mha_fwd_traits : public fmha_fwd_traits
{
    mha_fwd_traits(int head_size_q,
                   int head_size_v,
                   std::string dtype,
                   bool is_group_mode,
                   const mask_info& mask,
                   bias_enum bias_type,
                   bool has_lse,
                   bool has_dropout)
        : fmha_fwd_traits{head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          true, // is_v_rowmajor
                          mask.type,
                          bias_type,
                          has_lse,
                          has_dropout,
                          false} // do_fp8_static_quant
    {
    }
};

struct mha_fwd_splitkv_traits : public fmha_fwd_splitkv_traits
{
    mha_fwd_splitkv_traits(int head_size_q,
                           int head_size_v,
                           std::string dtype,
                           bool is_group_mode,
                           const mask_info& mask,
                           bias_enum bias_type,
                           bool has_lse)
        : fmha_fwd_splitkv_traits{head_size_q,
                                  head_size_v,
                                  dtype,
                                  is_group_mode,
                                  true, // is_v_rowmajor
                                  mask.type,
                                  bias_type,
                                  has_lse,
                                  false} // do_fp8_static_quant
    {
    }
};

using mha_fwd_args = fmha_fwd_args;
using mha_fwd_splitkv_args = fmha_fwd_splitkv_args;

float mha_fwd(mha_fwd_args args,
              const ck_tile::stream_config& stream_config,
              mask_info mask,
              std::string q_dtype_str,
              bool is_group_mode,
              bias_enum bias_type,
              bool has_lse);
              
float mha_fwd_splitkv(mha_fwd_splitkv_args args,
                      const ck_tile::stream_config& stream_config,
                      mask_info mask,
                      std::string q_dtype_str,
                      bool is_group_mode,
                      bias_enum bias_type,
                      bool has_lse);
              
} // namespace aiter
