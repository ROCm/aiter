#include "mha_fwd.h"
#include <string>

namespace aiter {
mha_batch_prefill_traits
get_mha_batch_prefill_traits(int head_size_q,
                             int head_size_v,
                             std::string dtype,
                             bool is_group_mode,
                             bool has_logits_soft_cap,
                             mask_enum mask_type,
                             bias_enum bias_type,
                             bool has_lse,
                             bool has_dropout,
                             quant_scale_enum qscale_type,
                             ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout,
                             ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table,
                             int page_size,
                             bool skip_min_seqlen_q = false,
                             bool use_64bit_load    = false)
{
    return mha_batch_prefill_traits(head_size_q,
                                    head_size_v,
                                    dtype,
                                    is_group_mode,
                                    has_logits_soft_cap,
                                    mask_type,
                                    bias_type,
                                    has_lse,
                                    has_dropout,
                                    qscale_type,
                                    skip_min_seqlen_q,
                                    kv_memory_layout,
                                    kv_lookup_table,
                                    page_size,
                                    use_64bit_load);
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

    // Determine element size for KV cache overflow check
    int element_size = 2; // default: fp16/bf16
    if(q_dtype_str == "fp8" || q_dtype_str == "fp8bf16")
        element_size = 1;

    // Check if KV cache exceeds 4GB (INT32_MAX byte offset).
    // Only relevant when page_block_size < kN0 (tile N dimension), because:
    //   - page_size >= kN0: SRD is rebased per-page using 64-bit pointer arithmetic,
    //     so within-page offsets are always small. No overflow possible.
    //   - page_size < kN0: full offsets (page * stride + within_page) are used as
    //     32-bit buffer_load voffset, which overflows at >2GB.
    // kN0 = 128 across all batch prefill tile configurations (bn0 in codegen).
    constexpr int kN0_min = 128;
    bool use_64bit_load   = false;
    if(args.page_block_size < kN0_min && args.num_total_pages > 1)
    {
        int64_t max_page_byte_offset = static_cast<int64_t>(args.num_total_pages - 1) *
                                       static_cast<int64_t>(args.batch_stride_k) *
                                       element_size;
        use_64bit_load = (max_page_byte_offset > INT32_MAX);
    }

    auto traits = get_mha_batch_prefill_traits(head_size_q,
                                               head_size_v,
                                               q_dtype_str,
                                               is_group_mode,
                                               args.logits_soft_cap > 0.f,
                                               mask_type,
                                               bias_type,
                                               has_lse,
                                               has_dropout,
                                               qscale_type,
                                               args.kv_memory_layout,
                                               args.kv_lookup_table,
                                               args.page_block_size,
                                               false, // skip_min_seqlen_q
                                               use_64bit_load);
    return fmha_batch_prefill(traits, args, stream_config);
}

} // namespace aiter
