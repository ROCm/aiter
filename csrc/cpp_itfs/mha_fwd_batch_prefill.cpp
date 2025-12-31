#include "mha_fwd.h"
#include <string>

namespace aiter {
mha_fwd_traits get_mha_fwd_traits(int head_size_q,
                                  int head_size_v,
                                  std::string dtype,
                                  bool is_group_mode,
                                  bool has_logits_soft_cap,
                                  mask_enum mask_type,
                                  bias_enum bias_type,
                                  bool has_lse,
                                  bool has_dropout,
                                  quant_scale_enum qscale_type,
                                  bool use_ext_asm,
                                  bool has_sink          = false,
                                  int how_v3_bf16_cvt    = 1,
                                  bool skip_min_seqlen_q = false)
{
    return mha_fwd_traits(head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          has_logits_soft_cap,
                          mask_type,
                          bias_type,
                          has_lse,
                          has_dropout,
                          qscale_type,
                          use_ext_asm,
                          how_v3_bf16_cvt,
                          skip_min_seqlen_q,
                          has_sink);
}

float mha_batch_prefill(mha_batch_prefill_args args,
                        const ck_tile::stream_config& stream_config,
                        std::string q_dtype_str,
                        bool is_group_mode,
                        mask_enum mask_type,
                        bias_enum bias_type,
                        bool has_lse,
                        quant_scale_enum qscale_type,
                        bool use_ext_asm)
{
    int head_size_q  = args.hdim_q;
    int head_size_v  = args.hdim_v;
    bool has_dropout = args.p_drop > 0.f;
    auto traits      = get_mha_fwd_traits(head_size_q,
                                     head_size_v,
                                     q_dtype_str,
                                     is_group_mode,
                                     args.logits_soft_cap > 0.f,
                                     mask_type,
                                     bias_type,
                                     has_lse,
                                     has_dropout,
                                     qscale_type,
                                     use_ext_asm);
    return fmha_batch_prefill(traits, args, stream_config);
}

} // namespace aiter
