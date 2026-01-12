#include "mha_fwd.h"
#include <string>

namespace aiter {
mha_fwd_splitkv_traits get_mha_fwd_splitkv_traits(int head_size_q,
                                                  int head_size_v,
                                                  std::string dtype,
                                                  bool is_group_mode,
                                                  bool has_logits_soft_cap,
                                                  mask_enum mask_type,
                                                  bias_enum bias_type,
                                                  bool has_lse,
                                                  bool has_sink)
{
    return mha_fwd_splitkv_traits(head_size_q,
                                  head_size_v,
                                  dtype,
                                  is_group_mode,
                                  has_logits_soft_cap,
                                  mask_type,
                                  bias_type,
                                  has_lse,
                                  has_sink);
}

float mha_fwd_splitkv(mha_fwd_splitkv_args args,
                      const ck_tile::stream_config& stream_config,
                      std::string q_dtype_str,
                      bool is_group_mode,
                      mask_enum mask_type,
                      bias_enum bias_type,
                      bool has_lse,
                      bool has_sink)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    auto traits     = get_mha_fwd_splitkv_traits(head_size_q,
                                             head_size_v,
                                             q_dtype_str,
                                             is_group_mode,
                                             args.logits_soft_cap > 0.f,
                                             mask_type,
                                             bias_type,
                                             has_lse,
                                             has_sink);
    return fmha_fwd_splitkv(traits, args, stream_config);
}

} // namespace aiter
