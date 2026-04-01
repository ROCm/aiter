#include "mha_fwd.h"
#include <string>

namespace aiter {

float mha_fwd_pagedkv(mha_fwd_pagedkv_args args,
                      const ck_tile::stream_config& stream_config,
                      std::string q_dtype_str,
                      bool is_group_mode,
                      mask_enum mask_type,
                      bias_enum bias_type,
                      bool has_lse,
                      bool has_sink)
{
    mha_fwd_pagedkv_traits traits(args.hdim_q,
                                  args.hdim_v,
                                  q_dtype_str,
                                  is_group_mode,
                                  args.logits_soft_cap > 0.f,
                                  mask_type,
                                  bias_type,
                                  has_lse,
                                  has_sink);
    return fmha_fwd_pagedkv(traits, args, stream_config);
}

} // namespace aiter
