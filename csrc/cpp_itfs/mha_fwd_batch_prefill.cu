#include "mha_fwd.h"
#include "aiter_hip_common.h"
#include <cstdint>
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
                             bool has_sink         = false)
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
                                    has_sink,
                                    kv_memory_layout,
                                    kv_lookup_table,
                                    page_size);
}

// 688-byte PAGED_VARLEN kernarg. fmha_fwd_v3_args is 656 bytes.
struct __attribute__((packed)) fmha_fwd_v3_paged_varlen_args
{
    fmha_fwd_v3_args base;
    const void* ptr_cu_seqlens_q;
    p2 _p41;
    const void* ptr_seqlens_kvcache;
    p2 _p42;
};

static bool is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

// gfx950 hd256 FP8 page_size=64 LINEAR paged-varlen asm. Packed Q/O;
// K/V [N, 64, H, D]. Returns -1 if ineligible.
static float fmha_batch_prefill_v3(const mha_batch_prefill_args& a,
                                   const ck_tile::stream_config& s,
                                   const std::string& q_dtype_str,
                                   mask_enum mask_type,
                                   bias_enum bias_type,
                                   bool has_lse,
                                   quant_scale_enum qscale_type,
                                   bool use_ext_asm)
{
    if(!use_ext_asm)
        return -1;

    if(get_gpu_arch() != "gfx950" || q_dtype_str != "fp8bf16" || a.hdim_q != 256 ||
       a.hdim_v != 256 || a.page_block_size != 64 ||
       a.kv_memory_layout != ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT ||
       a.kv_lookup_table != ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D ||
       qscale_type != quant_scale_enum::pertensor || a.q_descale_ptr == nullptr ||
       a.k_descale_ptr == nullptr || a.v_descale_ptr == nullptr || a.p_drop > 0.f ||
       bias_type != bias_enum::no_bias || a.logits_soft_cap > 0.f || a.sink_size > 0 ||
       a.sink_ptr != nullptr || a.nhead_k <= 0 || a.nhead_q % a.nhead_k != 0 ||
       a.kv_indptr == nullptr || a.kv_page_indices == nullptr || a.seqstart_q_ptr == nullptr ||
       a.seqlen_k_ptr == nullptr)
        return -1;

    const int gqa = a.nhead_q / a.nhead_k;
    if(!is_pow2(gqa))
        return -1;

    const bool causal = mask_type == mask_enum::mask_bottom_right && a.window_size_left < 0 &&
                        a.window_size_right == 0;
    const bool no_mask = mask_type == mask_enum::no_mask;
    if(!causal && !no_mask)
        return -1;

    const char* knl_name = causal ? "_ZN5aiter45fmha_fwd_hd256_fp8_causal_paged_varlen_gfx950E"
                                  : "_ZN5aiter38fmha_fwd_hd256_fp8_paged_varlen_gfx950E";
    const char* co_name  = causal ? "fmha_v3_fwd/fwd_hd256_fp8_causal_paged_varlen.co"
                                  : "fmha_v3_fwd/fwd_hd256_fp8_paged_varlen.co";

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr =
        &impl_ptr_map.get_or_create(knl_name, [&]() { return AiterAsmKernel(knl_name, co_name); });

    constexpr int ts_qo   = 256;
    constexpr int in_bpe  = 1;
    constexpr int out_bpe = 2;

    fmha_fwd_v3_paged_varlen_args packed{};
    static_assert(sizeof(fmha_fwd_v3_args) == 656, "fmha_fwd_v3_args is 656 bytes");
    static_assert(sizeof(packed) == 688, "PAGED_VARLEN kernarg must be 688 bytes");
    auto& args = packed.base;

    int tune_opt = 5;
    if(causal && ((a.nhead_q % 8 != 0) || (a.max_seqlen_q > 16384)))
        tune_opt -= 2;

    args.ptr_o            = a.o_ptr;
    args.ptr_q            = a.q_ptr;
    args.ptr_k            = a.k_ptr;
    args.ptr_v            = a.v_ptr;
    args.ptr_lse          = a.lse_ptr;
    args.scalar           = a.scale_s;
    args.s_seq_len        = static_cast<unsigned int>(a.max_seqlen_q);
    args.s_Seqs           = static_cast<unsigned int>(a.stride_q * in_bpe);
    args.s_Ts             = static_cast<unsigned int>(ts_qo * a.stride_q * in_bpe);
    args.s_Hs             = static_cast<unsigned int>(a.nhead_stride_q * in_bpe);
    args.s_Bs             = 0;
    args.s_gqa            = static_cast<unsigned int>(gqa);
    args.s_k_Seqs         = static_cast<unsigned int>(a.stride_k * in_bpe);
    args.s_k_Hs           = static_cast<unsigned int>(a.nhead_stride_k * in_bpe);
    args.s_k_Bs           = 0;
    args.s_opt            = static_cast<unsigned int>(tune_opt);
    args.s_lse            = has_lse ? 1 : 0;
    args.s_kv_seq_len     = static_cast<unsigned int>(a.seqlen_k);
    args.s_qk_head_dim    = static_cast<unsigned int>(a.hdim_q);
    args.s_v_head_dim     = static_cast<unsigned int>(a.hdim_v);
    args.s_q_head_num     = static_cast<unsigned int>(a.nhead_q);
    args.s_v_Seqs         = static_cast<unsigned int>(a.stride_v * in_bpe);
    args.s_v_Hs           = static_cast<unsigned int>(a.nhead_stride_v * in_bpe);
    args.s_v_Bs           = 0;
    args.s_o_Seqs         = static_cast<unsigned int>(a.stride_o * out_bpe);
    args.s_o_Hs           = static_cast<unsigned int>(a.nhead_stride_o * out_bpe);
    args.s_o_Bs           = 0;
    args.ptr_qseq         = a.kv_page_indices; // physical page IDs
    args.ptr_kseq         = a.kv_indptr;       // per-request page-range prefix
    args.s_lse_Hs         = static_cast<unsigned int>(a.nhead_stride_lse * 4);
    args.ptr_qseq_padding =
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a.batch_stride_k * in_bpe));
    args.ptr_kseq_padding =
        reinterpret_cast<const void*>(static_cast<uintptr_t>(a.batch_stride_v * in_bpe));
    args.ptr_q_descale    = a.q_descale_ptr;
    args.ptr_k_descale    = a.k_descale_ptr;
    args.ptr_v_descale    = a.v_descale_ptr;
    packed.ptr_cu_seqlens_q    = a.seqstart_q_ptr;
    packed.ptr_seqlens_kvcache = a.seqlen_k_ptr;

    const int tg_div = causal ? 2 : 1;
    const int gdx =
        ((a.max_seqlen_q + ts_qo - 1) / ts_qo + tg_div - 1) / tg_div;
    const int gdy = a.nhead_q;
    const int gdz = a.batch;
    const int bdx = 512;
    size_t arg_size = sizeof(packed);

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        void* args_ptr       = &packed;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz, bdx, 1, 1, s_.stream_id_});
    });
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
    bool has_sink    = args.sink_size > 0 || args.sink_ptr != nullptr;

    float t = fmha_batch_prefill_v3(args,
                                    stream_config,
                                    q_dtype_str,
                                    mask_type,
                                    bias_type,
                                    has_lse,
                                    qscale_type,
                                    use_ext_asm);
    if(t >= 0)
        return t;

    // The kUseGlobalLoad decision (>2GB KV cache → use `global_load_lds_*`
    // instead of SRD `buffer_load_*`) is made per-arm inside the auto-generated
    // dispatcher in fmha_batch_prefill_api.cpp, where each arm knows its own
    // compile-time bn0 and dtype element size. The wrapper just forwards args;
    // no runtime trait field for it.
    auto traits      = get_mha_batch_prefill_traits(head_size_q,
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
                                               /*skip_min_seqlen_q=*/false,
                                               has_sink);
    return fmha_batch_prefill(traits, args, stream_config);
}

} // namespace aiter
