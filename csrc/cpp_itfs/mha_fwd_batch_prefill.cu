#include "mha_fwd.h"
#include "aiter_hip_common.h"
#include "asm_fmha_v3_fwd_configs.hpp"
#include <cstddef>
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

static_assert(sizeof(fmha_fwd_v3_args) == 656, "fmha_fwd_v3_args must remain 656 bytes");
static_assert(offsetof(fmha_fwd_v3_args, ptr_qseq) == 0x1b0, "ptr_qseq ABI offset");
static_assert(offsetof(fmha_fwd_v3_args, ptr_kseq) == 0x1c0, "ptr_kseq ABI offset");
static_assert(offsetof(fmha_fwd_v3_args, ptr_qseq_padding) == 0x1e0,
              "ptr_qseq_padding ABI offset");
static_assert(offsetof(fmha_fwd_v3_args, ptr_kseq_padding) == 0x1f0,
              "ptr_kseq_padding ABI offset");
static_assert(offsetof(fmha_fwd_v3_args, ptr_q_descale) == 0x200,
              "ptr_q_descale ABI offset");

static bool is_pow2(int x) { return x > 0 && (x & (x - 1)) == 0; }

static const char* get_kv_layout_name(
    ck_tile::BlockAttentionKVCacheMemoryLayoutEnum kv_memory_layout)
{
    switch(kv_memory_layout)
    {
    case ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT: return "linear";
    case ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::VECTORIZED_LAYOUT: return "vectorized";
    }
    return "unknown";
}

static const char* get_kv_lookup_table_name(
    ck_tile::BlockAttentionKVCacheLookupTableEnum kv_lookup_table)
{
    switch(kv_lookup_table)
    {
    case ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D: return "sglang";
    case ck_tile::BlockAttentionKVCacheLookupTableEnum::VLLM_BLOCK_TABLE_2D: return "vllm";
    }
    return "unknown";
}

static const char* get_qscale_name(quant_scale_enum qscale_type)
{
    switch(qscale_type)
    {
    case quant_scale_enum::no_scale: return "no";
    case quant_scale_enum::pertensor: return "pertensor";
    case quant_scale_enum::blockscale: return "blockscale";
    case quant_scale_enum::kv_blockscale: return "kv_blockscale";
    case quant_scale_enum::mx: return "mx";
    }
    return "unknown";
}

static int get_input_element_bytes(const std::string& dtype)
{
    if(dtype == "fp8bf16")
        return 1;
    if(dtype == "bf16" || dtype == "fp16")
        return 2;
    return 0;
}

static int get_output_element_bytes(const std::string& dtype)
{
    if(dtype == "fp8bf16" || dtype == "bf16" || dtype == "fp16")
        return 2;
    return 0;
}

static const CFG::mapped_type*
find_mha_batch_prefill_asm_config(const mha_batch_prefill_args& a,
                                  const std::string& q_dtype_str,
                                  mask_enum mask_type,
                                  quant_scale_enum qscale_type)
{
    const auto arch_id       = get_gpu_arch();
    const auto* kv_layout    = get_kv_layout_name(a.kv_memory_layout);
    const auto* lookup_table = get_kv_lookup_table_name(a.kv_lookup_table);
    const auto* qscale       = get_qscale_name(qscale_type);
    const CFG::mapped_type* result = nullptr;

    for(const auto& entry : cfg_fmha_batch_prefill)
    {
        const auto& cfg = entry.second;
        if(cfg.arch != arch_id || cfg.dtype != q_dtype_str || cfg.hdim_q != a.hdim_q ||
           cfg.hdim_v != a.hdim_v || cfg.mask != static_cast<int>(mask_type) ||
           cfg.page_size != a.page_block_size || cfg.kv_layout != kv_layout ||
           cfg.lookup_table != lookup_table || cfg.qscale != qscale ||
           (cfg.min_seqlen_q > 0 && a.max_seqlen_q < cfg.min_seqlen_q) ||
           (cfg.max_seqlen_q > 0 && a.max_seqlen_q > cfg.max_seqlen_q))
            continue;

        AITER_CHECK(result == nullptr,
                    __func__,
                    ": ambiguous assembly manifest rows for arch=", arch_id,
                    ", dtype=", q_dtype_str,
                    ", hdim_q=", a.hdim_q,
                    ", hdim_v=", a.hdim_v,
                    ", mask=", static_cast<int>(mask_type),
                    ", page_size=", a.page_block_size,
                    ", kv_layout=", kv_layout,
                    ", lookup_table=", lookup_table,
                    ", qscale=", qscale,
                    ", max_seqlen_q=", a.max_seqlen_q);
        result = &cfg;
    }
    return result;
}

// Paged-varlen assembly path. The manifest selects the statically compiled
// page size, ABI family, and launch configuration.
// Returns -1 when no exact manifest row is applicable so CK may be attempted.
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

    const bool causal = mask_type == mask_enum::mask_bottom_right && a.window_size_left < 0 &&
                        a.window_size_right == 0;
    const bool no_mask = mask_type == mask_enum::no_mask;
    if(!causal && !no_mask)
        return -1;

    const auto* cfg =
        find_mha_batch_prefill_asm_config(a, q_dtype_str, mask_type, qscale_type);
    if(cfg == nullptr)
        return -1;

    const bool uses_pertensor_scale = qscale_type == quant_scale_enum::pertensor;
    if((uses_pertensor_scale && (a.q_descale_ptr == nullptr || a.k_descale_ptr == nullptr ||
                                a.v_descale_ptr == nullptr)) ||
       (qscale_type != quant_scale_enum::no_scale && !uses_pertensor_scale) || a.p_drop > 0.f ||
       bias_type != bias_enum::no_bias || a.logits_soft_cap > 0.f || a.sink_size > 0 ||
       a.sink_ptr != nullptr || a.nhead_k <= 0 || a.nhead_q % a.nhead_k != 0 ||
       a.kv_indptr == nullptr || a.kv_page_indices == nullptr || a.seqstart_q_ptr == nullptr ||
       a.seqlen_k_ptr == nullptr)
        return -1;

    const int gqa = a.nhead_q / a.nhead_k;
    if(!is_pow2(gqa))
        return -1;

    const char* knl_name = cfg->knl_name.c_str();
    const char* co_name  = cfg->co_name.c_str();

    const int in_bpe  = get_input_element_bytes(q_dtype_str);
    const int out_bpe = get_output_element_bytes(q_dtype_str);
    if(in_bpe == 0 || out_bpe == 0)
        return -1;

    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;
    AiterAsmKernel* impl_ptr =
        &impl_ptr_map.get_or_create(knl_name, [&]() { return AiterAsmKernel(knl_name, co_name); });

    const int ts_qo = cfg->ts_qo;

    fmha_fwd_v3_args base{};
    // The BF16 reused-field ABI was validated with remapping disabled. The
    // extended FP8 ABI retains the v3 tune-option policy.
    int tune_opt = cfg->abi == "paged_varlen_v3_reuse" ? 0 : 5;
    if(tune_opt != 0 && causal && ((a.nhead_q % 8 != 0) || (a.max_seqlen_q > 16384)))
        tune_opt -= 2;

    base.ptr_o            = a.o_ptr;
    base.ptr_q            = a.q_ptr;
    base.ptr_k            = a.k_ptr;
    base.ptr_v            = a.v_ptr;
    base.ptr_lse          = a.lse_ptr;
    base.scalar           = a.scale_s;
    base.s_seq_len        = static_cast<unsigned int>(a.max_seqlen_q);
    base.s_Seqs           = static_cast<unsigned int>(a.stride_q * in_bpe);
    base.s_Ts             = static_cast<unsigned int>(ts_qo * a.stride_q * in_bpe);
    base.s_Hs             = static_cast<unsigned int>(a.nhead_stride_q * in_bpe);
    base.s_Bs             = static_cast<unsigned int>(a.batch_stride_q * in_bpe);
    base.s_gqa            = static_cast<unsigned int>(gqa);
    base.s_k_Seqs         = static_cast<unsigned int>(a.stride_k * in_bpe);
    base.s_k_Hs           = static_cast<unsigned int>(a.nhead_stride_k * in_bpe);
    base.s_k_Bs           = static_cast<unsigned int>(a.batch_stride_k * in_bpe);
    base.s_opt            = static_cast<unsigned int>(tune_opt);
    base.s_lse            = has_lse ? 1 : 0;
    base.s_kv_seq_len     = static_cast<unsigned int>(a.seqlen_k);
    base.s_qk_head_dim    = static_cast<unsigned int>(a.hdim_q);
    base.s_v_head_dim     = static_cast<unsigned int>(a.hdim_v);
    base.s_q_head_num     = static_cast<unsigned int>(a.nhead_q);
    base.s_v_Seqs         = static_cast<unsigned int>(a.stride_v * in_bpe);
    base.s_v_Hs           = static_cast<unsigned int>(a.nhead_stride_v * in_bpe);
    base.s_v_Bs           = static_cast<unsigned int>(a.batch_stride_v * in_bpe);
    base.s_o_Seqs         = static_cast<unsigned int>(a.stride_o * out_bpe);
    base.s_o_Hs           = static_cast<unsigned int>(a.nhead_stride_o * out_bpe);
    base.s_o_Bs           = static_cast<unsigned int>(a.batch_stride_o * out_bpe);
    base.s_lse_Hs         = static_cast<unsigned int>(a.nhead_stride_lse * 4);
    base.ptr_q_descale    = a.q_descale_ptr;
    base.ptr_k_descale    = a.k_descale_ptr;
    base.ptr_v_descale    = a.v_descale_ptr;

    const int tg_div = causal ? 2 : 1;
    const int qtiles =
        ((a.max_seqlen_q + ts_qo - 1) / ts_qo + tg_div - 1) / tg_div;
    int gdx;
    int gdy;
    int gdz;
    if(cfg->grid_layout == "qtiles_heads_batch")
    {
        gdx = qtiles;
        gdy = a.nhead_q;
        gdz = a.batch;
    }
    else if(cfg->grid_layout == "heads_batch_qtiles")
    {
        gdx = a.nhead_q;
        gdy = a.batch;
        gdz = qtiles;
    }
    else
    {
        AITER_CHECK(false, __func__, ": unsupported grid_layout ", cfg->grid_layout);
    }
    const int bdx = cfg->bdx;

    if(cfg->abi == "paged_varlen_v3_ext")
    {
        fmha_fwd_v3_paged_varlen_args packed{};
        static_assert(sizeof(packed) == 688,
                      "PAGED_VARLEN extended kernarg must be 688 bytes");
        packed.base                 = base;
        packed.base.s_k_Bs          = 0;
        packed.base.s_v_Bs          = 0;
        packed.base.ptr_qseq        = a.kv_page_indices;
        packed.base.ptr_kseq        = a.kv_indptr;
        packed.base.ptr_qseq_padding =
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a.batch_stride_k * in_bpe));
        packed.base.ptr_kseq_padding =
            reinterpret_cast<const void*>(static_cast<uintptr_t>(a.batch_stride_v * in_bpe));
        packed.ptr_cu_seqlens_q    = a.seqstart_q_ptr;
        packed.ptr_seqlens_kvcache = a.seqlen_k_ptr;
        size_t arg_size            = sizeof(packed);
        return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
            void* args_ptr       = &packed;
            size_t* arg_size_ptr = &arg_size;
            impl_ptr->launch_kernel(
                {args_ptr, arg_size_ptr, gdx, gdy, gdz, bdx, 1, 1, s_.stream_id_});
        });
    }

    if(cfg->abi == "paged_varlen_v3_reuse")
    {
        if(a.kv_last_page_lens == nullptr)
            return -1;
        base.ptr_qseq         = a.seqstart_q_ptr;
        base.ptr_kseq         = a.kv_indptr;
        base.ptr_qseq_padding = a.seqstart_q_ptr;
        base.ptr_kseq_padding = a.kv_last_page_lens;
        base.ptr_q_descale    = a.kv_page_indices;
        size_t arg_size       = sizeof(base);
        return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
            void* args_ptr       = &base;
            size_t* arg_size_ptr = &arg_size;
            impl_ptr->launch_kernel(
                {args_ptr, arg_size_ptr, gdx, gdy, gdz, bdx, 1, 1, s_.stream_id_});
        });
    }

    AITER_CHECK(false, __func__, ": unsupported paged-prefill ABI ", cfg->abi);
    return -1;
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
