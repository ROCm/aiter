#include "mha_fwd.h"
#include <cstdio>
#include <cstdlib>
#include <string>

namespace aiter {
namespace {
std::string get_env_or_empty(const char* key)
{
    const char* v = std::getenv(key);
    if(v == nullptr || *v == '\0')
    {
        return "";
    }
    return std::string(v);
}

std::string get_forced_splitkv_kernel_filter()
{
    std::string filter = get_env_or_empty("AITER_FORCE_FMHA_SPLITKV_KERNEL");
    if(filter.empty())
    {
        filter = get_env_or_empty("AITER_FORCE_FMHA_KERNEL");
    }
    if(!filter.empty() && filter.find("splitkv") == std::string::npos)
    {
        return "";
    }
    return filter;
}

bool should_debug_splitkv_dispatch()
{
    const std::string v = get_env_or_empty("AITER_DEBUG_SPLITKV_DISPATCH");
    return !v.empty() && v != "0";
}

bool should_list_splitkv_kernels_on_fail()
{
    const std::string v = get_env_or_empty("AITER_DEBUG_SPLITKV_LIST_ON_FAIL");
    return !v.empty() && v != "0";
}

void dump_splitkv_dispatch_state(const fmha_fwd_splitkv_traits& traits,
                                const fmha_fwd_splitkv_args& args,
                                const std::string& forced_kernel_filter)
{
    std::fprintf(stderr,
                 "[aiter] splitkv traits dtype=%s hq=%d hv=%d group=%d vrow=%d logits=%d mask=%d bias=%d has_lse=%d has_sink=%d fp8_static=%d filter=%s\n",
                 traits.data_type.c_str(),
                 traits.hdim_q,
                 traits.hdim_v,
                 traits.is_group_mode ? 1 : 0,
                 traits.is_v_rowmajor ? 1 : 0,
                 traits.has_logits_soft_cap ? 1 : 0,
                 static_cast<int>(traits.mask_type),
                 static_cast<int>(traits.bias_type),
                 traits.has_lse ? 1 : 0,
                 traits.has_sink ? 1 : 0,
                 traits.do_fp8_static_quant ? 1 : 0,
                 forced_kernel_filter.empty() ? "<none>" : forced_kernel_filter.c_str());

    std::fprintf(stderr,
                 "[aiter] splitkv args batch=%d max_sq=%d hq=%d hk=%d dq=%d dv=%d num_splits=%d page_block=%d block_table=%d seqstart_q=%d seqstart_k=%d seqlen_k=%d sink_ptr=%d\n",
                 static_cast<int>(args.batch),
                 static_cast<int>(args.max_seqlen_q),
                 static_cast<int>(args.nhead_q),
                 static_cast<int>(args.nhead_k),
                 static_cast<int>(args.hdim_q),
                 static_cast<int>(args.hdim_v),
                 static_cast<int>(args.num_splits),
                 static_cast<int>(args.page_block_size),
                 args.block_table_ptr != nullptr ? 1 : 0,
                 args.seqstart_q_ptr != nullptr ? 1 : 0,
                 args.seqstart_k_ptr != nullptr ? 1 : 0,
                 args.seqlen_k_ptr != nullptr ? 1 : 0,
                 args.sink_ptr != nullptr ? 1 : 0);
}
} // namespace

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
    const bool effective_has_sink = has_sink && (args.sink_ptr != nullptr);
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
                                             effective_has_sink);
    const std::string forced_kernel_filter = get_forced_splitkv_kernel_filter();
    if(!forced_kernel_filter.empty())
    {
        traits.kernel_filter = forced_kernel_filter;
    }
    static bool printed_path_once = false;
    if(!printed_path_once)
    {
        std::fprintf(stderr,
                     "[aiter] fmha dispatch path=splitkv_ck filter=%s\n",
                     forced_kernel_filter.empty() ? "<none>" : forced_kernel_filter.c_str());
        printed_path_once = true;
    }
    if(should_debug_splitkv_dispatch())
    {
        dump_splitkv_dispatch_state(traits, args, forced_kernel_filter);
    }

    const float result = fmha_fwd_splitkv(traits, args, stream_config);
    if(result < 0.f && should_list_splitkv_kernels_on_fail())
    {
        std::fprintf(stderr,
                     "[aiter] splitkv dispatch failed; listing matching kernels (filter cleared) ...\n");
        auto debug_traits           = traits;
        debug_traits.list_kernels   = true;
        debug_traits.kernel_filter  = "";
        (void)fmha_fwd_splitkv(debug_traits, args, stream_config);
    }
    return result;
}

} // namespace aiter
