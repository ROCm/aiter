#include "mha_fwd.h"
#include "aiter_hip_common.h"
#if FAV3_ON
#include "asm_fmha_v3_fwd_configs.hpp"
#endif
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

namespace aiter {
#if FAV3_ON

// THD rows reuse the v1 kernarg layout but read ptr_qseq_padding's slot as a KV-split descriptor.
static_assert(offsetof(fmha_fwd_v3_args, ptr_qseq_padding) == 0x1E0,
              "THD rows read their KV split from the ptr_qseq_padding slot");
static_assert(sizeof(fmha_fwd_v3_args) == 656, "fmha_fwd_v3_args must stay 656 bytes");

// The THD rows' KV-split descriptor, in ptr_qseq_padding's 16-byte slot (pointer and padding).
// Slice i writes its normalized FP32 O at ptr_o + i * o_stride and its LSE at
// ptr_lse + i * lse_stride, both in bytes; count 0 walks the keys once and writes in place.
// cu_budget is read by the causal row only: the CUs a pair's kv_splits + 1 work-groups must fit,
// 0 to run every pair on kv_splits work-groups.
struct FmhaFwdV3ThdKvSplit
{
    uint32_t count;
    uint32_t o_stride;
    uint32_t lse_stride;
    uint32_t cu_budget;
};
static_assert(sizeof(FmhaFwdV3ThdKvSplit) == 16, "THD KV-split descriptor is 16 bytes");
static_assert(offsetof(fmha_fwd_v3_args, ptr_kseq_padding) - offsetof(fmha_fwd_v3_args,
                                                                       ptr_qseq_padding) ==
                  sizeof(FmhaFwdV3ThdKvSplit),
              "THD KV-split descriptor fills ptr_qseq_padding's slot");

// The THD objects address every token through a 64-bit product, so a pack may pass 4 GiB, but
// they take each stride as a 32-bit byte count. Strides are computed in 64 bits and a call whose
// stride does not fit stays off them.
constexpr int64_t kThdMaxStrideBytes = UINT32_MAX;

uint32_t thd_stride_bytes(int64_t elements, int64_t bytes_per_element)
{
    return static_cast<uint32_t>(elements * bytes_per_element);
}

// mha_v4(kv_splits="auto") takes at most this many slices: past it the merge outweighs the
// work-groups it adds.
constexpr int64_t kMaxThdKvSplits = 4;

// Values of the "mode" column in hsa/<arch>/fmha_v3_fwd/fmha_fwd.csv.
enum class FmhaFwdV3Mode : int
{
    Batch = 0, // fixed seqlen_q / seqlen_k for every batch
    Group = 1, // varlen: per-batch seqstart_q / seqstart_k, optionally padded
    Thd   = 2, // varlen self-attention: equal per-document Q and K lengths, no padding
};

int get_cfg_mask_type(const mha_fwd_args& a)
{
    if(a.mask_type == 0)
    {
        return 0;
    }
    if((a.mask_type == 2 || (a.mask_type == 1 && a.seqlen_q == a.seqlen_k)) &&
       a.window_size_left == -1 && a.window_size_right == 0)
    {
        return 2;
    }
    return -1;
}

std::string get_kernel_name_key(const std::string& arch_id,
                                const std::string& data_type,
                                int hdim_q,
                                int hdim_v,
                                int mask_type,
                                int bf16_cvt,
                                FmhaFwdV3Mode mode,
                                const CFG* cfgs)
{
    std::string kernel_name_key{};
    for(const auto& el : *cfgs)
    {
        const auto& cfg = el.second;
        if(cfg.arch != arch_id)
        {
            continue;
        }

        if(cfg.dtype == data_type && cfg.hdim_q == hdim_q && cfg.hdim_v == hdim_v &&
           cfg.mask == mask_type && cfg.mode == static_cast<int>(mode))
        {
            if(arch_id == "gfx950")
            {
                kernel_name_key = el.first;
                break;
            }
            else if(arch_id == "gfx942" && cfg.bf16_cvt == bf16_cvt)
            {
                kernel_name_key = el.first;
                break;
            }
        }
    }

    return kernel_name_key;
}

std::string get_kernel_co_name(const std::string& cfg_co_name, const std::string& arch_id)
{
    std::string co_name = cfg_co_name;
    if(arch_id == "gfx942")
    {
        auto pos = cfg_co_name.rfind('/');
        if(is_mi308_device())
        {
            co_name = cfg_co_name.substr(0, pos + 1) + "MI308/" + cfg_co_name.substr(pos + 1);
        }
        else
        {
            co_name = cfg_co_name.substr(0, pos + 1) + "MI300/" + cfg_co_name.substr(pos + 1);
        }
    }
    return co_name;
}

// gfx950 BF16 hd256 THD rows (the fmha_v4 THD objects): varlen self-attention, every document
// has the same length for Q and K. The objects read only ptr_qseq and take each document's K/V
// rows from Q's offsets, so cu_seqlens_k must equal cu_seqlens_q. Equal totals and maxima do not
// imply that, and comparing contents here would synchronize every launch, so Q and K must share
// one seqstart array: flash_attn_varlen_func compares a separate cu_seqlens_k and passes
// cu_seqlens_q for both when they are equal.
// AITER_FMHA_V3_FWD_THD=0 keeps these calls on the hd256 group rows (MHA only).
bool is_fmha_fwd_v3_thd(const mha_fwd_args& a, const std::string& arch_id)
{
    if(arch_id != "gfx950" || !a.is_group_mode || a.data_type != "bf16" || a.hdim_q != 256 ||
       a.hdim_v != 256)
        return false;
    if(const char* env = std::getenv("AITER_FMHA_V3_FWD_THD"); env && std::strcmp(env, "0") == 0)
        return false;
    if(a.seqstart_q_ptr == nullptr || a.seqstart_q_ptr != a.seqstart_k_ptr)
        return false;
    if(a.cu_seqlen_q_ptr || a.cu_seqlen_k_ptr || a.seqlen_q_ptr || a.seqlen_k_ptr)
        return false;
    if(a.seqlen_q != a.seqlen_k || a.max_seqlen_q <= 0)
        return false;
    if(a.nhead_k <= 0 || a.nhead_q % a.nhead_k != 0)
        return false;
    if(a.bias_type != 0 || a.p_drop > 0.f || a.logits_soft_cap > 0.f || a.has_sink ||
       a.qscale_type != 0)
        return false;
    // The output's strides are checked at FP32 width, which KV-slice partials are written in.
    for(const int64_t bytes : {int64_t(a.stride_q) * 2,
                               int64_t(a.nhead_stride_q) * 2,
                               int64_t(a.stride_k) * 2,
                               int64_t(a.nhead_stride_k) * 2,
                               int64_t(a.stride_v) * 2,
                               int64_t(a.nhead_stride_v) * 2,
                               int64_t(a.stride_o) * 4,
                               int64_t(a.nhead_stride_o) * 4,
                               a.has_lse ? int64_t(a.nhead_stride_lse) * 4 : int64_t(0)})
    {
        if(bytes < 0 || bytes > kThdMaxStrideBytes)
            return false;
    }
    return true;
}

int fmha_fwd_v3_thd_kv_splits(const mha_fwd_args& a)
{
    const std::string arch_id = get_gpu_arch();
    if(!a.use_asm_v3 || !is_fmha_fwd_v3_thd(a, arch_id))
        return 1;
    const int mask_type = get_cfg_mask_type(a);
    if(mask_type < 0)
        return 1;
    const auto it = cfg_fmha_fwd.find(get_kernel_name_key(arch_id,
                                                          a.data_type,
                                                          a.hdim_q,
                                                          a.hdim_v,
                                                          mask_type,
                                                          a.how_v3_bf16_cvt,
                                                          FmhaFwdV3Mode::Thd,
                                                          &cfg_fmha_fwd));
    if(it == cfg_fmha_fwd.end())
        return 1;

    // A causal work-group owns Q tile i and its mirror n - 1 - i, which together walk n + 1 tiles
    // of keys; a non-causal one owns one Q tile and walks all n. A slice keeps at least one Q tile
    // of that walk.
    const bool causal      = mask_type != 0;
    const int64_t ts_qo    = it->second.ts_qo;
    const int64_t q_tiles  = (a.max_seqlen_q + ts_qo - 1) / ts_qo;
    const int64_t q_groups = causal ? (q_tiles + 1) / 2 : q_tiles;
    const int64_t most     = causal ? q_tiles + 1 : q_tiles;

    int64_t splits  = 1;
    const char* env = std::getenv("AITER_FMHA_V3_FWD_THD_KV_SPLITS");
    if(env == nullptr || *env == '\0' || std::strcmp(env, "auto") == 0)
    {
        // mha_v4's rule: the most slices whose work-groups still fit the CUs once, since a second
        // slice on a CU reloads Q.
        const int64_t work_groups =
            std::max<int64_t>(static_cast<int64_t>(a.batch) * a.nhead_q * q_groups, 1);
        splits = std::min<int64_t>(
            {kMaxThdKvSplits, static_cast<int64_t>(get_num_cu_func()) / work_groups, most});
    }
    else
    {
        splits = std::min<int64_t>(std::atoll(env), most);
    }
    if(splits <= 1)
        return 1;
    // The kernel steps between slices through 32-bit byte strides, and the merge launches one
    // block per (token, head) row.
    const int64_t rows        = static_cast<int64_t>(a.seqlen_q) * a.nhead_q;
    const int64_t slice_bytes = rows * a.hdim_v * 4;
    if(slice_bytes > kThdMaxStrideBytes || rows > std::numeric_limits<int32_t>::max())
        return 1;
    return static_cast<int>(splits);
}

// Mirrors fmha_v4_fwd_thd. With one KV slice the descriptor at 0x1E0 stays zero and the kernel
// writes BF16 O and the LSE in place; with more, a.o_ptr and a.lse_ptr are FP32 partials.
void init_fmha_fwd_v3_thd_args(fmha_fwd_v3_args& args, const mha_fwd_args& a, uint32_t cu_budget)
{
    std::memset(&args, 0, sizeof(args));
    constexpr int bpe     = 2;
    const int out_bpe     = a.kv_splits > 1 ? 4 : 2;
    args.ptr_o         = a.o_ptr;
    args.ptr_q         = a.q_ptr;
    args.ptr_k         = a.k_ptr;
    args.ptr_v         = a.v_ptr;
    args.scalar        = a.scale_s;
    args.s_seq_len     = a.max_seqlen_q;
    args.s_kv_seq_len  = a.max_seqlen_q;
    args.s_Seqs        = thd_stride_bytes(a.stride_q, bpe);
    args.s_Hs          = thd_stride_bytes(a.nhead_stride_q, bpe);
    args.s_gqa         = a.nhead_q / a.nhead_k;
    args.s_k_Seqs      = thd_stride_bytes(a.stride_k, bpe);
    args.s_k_Hs        = thd_stride_bytes(a.nhead_stride_k, bpe);
    args.s_qk_head_dim = a.hdim_q;
    args.s_v_head_dim  = a.hdim_v;
    args.s_q_head_num  = a.nhead_q;
    args.s_v_Seqs      = thd_stride_bytes(a.stride_v, bpe);
    args.s_v_Hs        = thd_stride_bytes(a.nhead_stride_v, bpe);
    args.s_o_Seqs      = thd_stride_bytes(a.stride_o, out_bpe);
    args.s_o_Hs        = thd_stride_bytes(a.nhead_stride_o, out_bpe);
    args.ptr_qseq      = a.seqstart_q_ptr;
    args.ptr_kseq      = a.seqstart_k_ptr;
    if(a.has_lse)
    {
        // [nhead_q, total_tokens] float32 per slice; head h's rows start at ptr_lse + h * s_lse_Hs.
        args.ptr_lse  = a.lse_ptr;
        args.s_lse    = 1;
        args.s_lse_Hs = thd_stride_bytes(a.nhead_stride_lse, 4);
    }
    if(a.kv_splits > 1)
    {
        const FmhaFwdV3ThdKvSplit split{static_cast<uint32_t>(a.kv_splits),
                                        thd_stride_bytes(a.kv_split_stride_o, 4),
                                        thd_stride_bytes(a.kv_split_stride_lse, 4),
                                        cu_budget};
        std::memcpy(reinterpret_cast<char*>(&args) + offsetof(fmha_fwd_v3_args, ptr_qseq_padding),
                    &split,
                    sizeof(split));
    }
}

void init_fmha_fwd_v3_args(fmha_fwd_v3_args& args,
                           const mha_fwd_args& a,
                           int ts_qo,
                           const std::string& arch_id)
{
    int tune_opt = 5;
    // if num_head is not 8N, or seqlen is bigger than 16K, downgrade to 2and3
    if(a.mask_type != 0 && ((a.nhead_q % 8 != 0) || (a.seqlen_q > 16384)))
    {
        tune_opt -= 2;
    }
    if(a.hdim_q == 192 && a.hdim_v == 128 && arch_id == "gfx942")
    {
        tune_opt = 0;
    }
    args.ptr_o            = a.o_ptr;
    args.ptr_q            = a.q_ptr;
    args.ptr_k            = a.k_ptr;
    args.ptr_v            = a.v_ptr;
    args.ptr_lse          = a.lse_ptr;
    args.ptr_qseq         = nullptr;
    args.ptr_kseq         = nullptr;
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;
    args.ptr_q_descale    = nullptr;
    args.ptr_k_descale    = nullptr;
    args.ptr_v_descale    = nullptr;
    args.s_descale_q_Bs   = 0;
    args.s_descale_q_Hs   = 0;
    args.s_descale_k_Bs   = 0;
    args.s_descale_k_Hs   = 0;
    args.s_descale_v_Bs   = 0;
    args.s_descale_v_Hs   = 0;

    int in_bpe = 2;
    int out_bpe = 2;
    if(a.data_type == "fp8bf16")
    {
        in_bpe = 1;
        args.ptr_q_descale = a.q_descale_ptr;
        args.ptr_k_descale = a.k_descale_ptr;
        args.ptr_v_descale = a.v_descale_ptr;
        args.s_descale_q_Bs = a.batch_stride_q_descale * 4;
        args.s_descale_q_Hs = a.nhead_stride_q_descale * 4;
        args.s_descale_k_Bs = a.batch_stride_k_descale * 4;
        args.s_descale_k_Hs = a.nhead_stride_k_descale * 4;
        args.s_descale_v_Bs = a.batch_stride_v_descale * 4;
        args.s_descale_v_Hs = a.nhead_stride_v_descale * 4;
    }

    args.scalar           = a.scale_s;
    args.s_seq_len        = a.seqlen_q;
    args.s_Seqs           = a.stride_q * in_bpe;
    args.s_Ts             = ts_qo * a.stride_q * in_bpe;
    args.s_Hs             = a.nhead_stride_q * in_bpe;
    args.s_Bs             = a.batch_stride_q * in_bpe;
    args.s_gqa            = a.nhead_q / a.nhead_k;
    args.s_k_Seqs         = a.stride_k * in_bpe;
    args.s_k_Hs           = a.nhead_stride_k * in_bpe;
    args.s_k_Bs           = a.batch_stride_k * in_bpe;
    args.s_opt            = tune_opt;
    args.s_lse            = a.has_lse ? 1 : 0;
    args.s_kv_seq_len     = a.seqlen_k;
    args.s_qk_head_dim    = a.hdim_q;
    args.s_v_head_dim     = a.hdim_v;
    args.s_q_head_num     = a.nhead_q;
    args.s_v_Seqs         = a.stride_v * in_bpe;
    args.s_v_Hs           = a.nhead_stride_v * in_bpe;
    args.s_v_Bs           = a.batch_stride_v * in_bpe;
    args.s_o_Seqs         = a.stride_o * out_bpe;
    args.s_o_Hs           = a.nhead_stride_o * out_bpe;
    args.s_o_Bs           = a.batch_stride_o * out_bpe;
    args.s_lse_Hs         = a.nhead_stride_lse * 4;
    // batch mode does not support padded
    if(a.is_group_mode)
    {
        args.ptr_kseq_padding = a.seqstart_k_ptr;
        if(a.cu_seqlen_k_ptr && a.seqstart_k_ptr)
        {
            args.ptr_kseq = a.cu_seqlen_k_ptr;
        }
        else
        {
            args.ptr_kseq = a.seqstart_k_ptr;
        }
        args.ptr_qseq_padding = a.seqstart_q_ptr;
        if(a.cu_seqlen_q_ptr && a.seqstart_q_ptr)
        {
            args.ptr_qseq = a.cu_seqlen_q_ptr;
        }
        else
        {
            args.ptr_qseq = a.seqstart_q_ptr;
        }
    }
}

std::tuple<int, int, int> get_grid_dim(const mha_fwd_args& a, int ts_qo, const std::string& arch_id)
{

    int tg_div = (a.mask_type != 0) ? 2 : 1;
    if(arch_id == "gfx942" && a.is_group_mode && a.hdim_q == 192 && a.hdim_v == 128)
    {
        tg_div = 1; // do not merge the head and tail in seqlen_q direction
    }
    if(arch_id == "gfx950" && a.hdim_q == 256 && a.data_type == "bf16")
    {
        tg_div = 1; // hd256 bf16 kernel does not implement tile merging
    }
    // batch
    int gdx = ((a.seqlen_q + ts_qo - 1) / ts_qo + tg_div - 1) / tg_div;
    int gdy = a.nhead_q;
    int gdz = a.batch;
    if(arch_id == "gfx942" && a.hdim_q == 192 && a.hdim_v == 128)
    {
        gdx = a.nhead_q;
        gdy = (a.seqlen_q + ts_qo - 1) /
              ts_qo; // do not merge the head and tail in seqlen_q direction
        gdz = a.batch;
    }
    // group
    if(a.is_group_mode)
    {
        gdx = a.nhead_q;
        gdy = a.batch;
        gdz = ((a.seqlen_q + ts_qo - 1) / ts_qo + tg_div - 1) / tg_div;
    }

    return std::make_tuple(gdx, gdy, gdz);
}

float fmha_fwd_v3(mha_fwd_args a, const ck_tile::stream_config& s)
{
    if(!a.use_asm_v3)
        return -1;

    std::string arch_id = get_gpu_arch();

    bool hdim_ok = (a.hdim_q == 128 && a.hdim_v == 128) ||
                   (a.hdim_q == 192 && a.hdim_v == 128) ||
                   (a.hdim_q == 256 && a.hdim_v == 256);
    if(!hdim_ok ||
       (a.data_type != "bf16" && a.data_type != "fp8bf16") || (a.bias_type != 0) || (a.p_drop > 0.f) ||
       ((arch_id != "gfx942") && (arch_id != "gfx950")))
    {
        AITER_LOG_WARNING("unsupported condition in fwd_v3!!! data type: " << a.data_type);
        return -1;
    }
    const bool thd = is_fmha_fwd_v3_thd(a, arch_id);
    // gfx950 bf16 hd256 group rows: no GQA/MQA (kernel indexes K/V by h_q directly). The THD
    // rows take any nhead_q % nhead_k == 0.
    if(arch_id == "gfx950" && a.hdim_q == 256 && a.data_type == "bf16" &&
       a.nhead_q != a.nhead_k && !thd)
    {
        return -1;
    }
    // FP32 KV-slice partials exist only on the THD rows.
    if(a.kv_splits > 1 &&
       (!thd || !a.has_lse || a.kv_split_stride_o * 4 > kThdMaxStrideBytes ||
        a.kv_split_stride_lse * 4 > kThdMaxStrideBytes))
    {
        return -1;
    }

    auto fwd_cfgs               = &cfg_fmha_fwd;
    int cfg_mask_type           = get_cfg_mask_type(a);
    const FmhaFwdV3Mode mode    = thd               ? FmhaFwdV3Mode::Thd
                                  : a.is_group_mode ? FmhaFwdV3Mode::Group
                                                    : FmhaFwdV3Mode::Batch;
    std::string kernel_name_key = get_kernel_name_key(arch_id,
                                                      a.data_type,
                                                      a.hdim_q,
                                                      a.hdim_v,
                                                      cfg_mask_type,
                                                      a.how_v3_bf16_cvt,
                                                      mode,
                                                      fwd_cfgs);
    auto it                     = fwd_cfgs->find(kernel_name_key);
    if(it == fwd_cfgs->end())
    {
        return -1;
    };

    if(a.v3_api_check)
    {
        return 1;
    };

    AiterAsmKernel* impl_ptr = nullptr;
    static SynchronizedCache<std::string_view, AiterAsmKernel> impl_ptr_map;

    const auto& cfg     = it->second;
    const char* name    = cfg.knl_name.c_str();
    std::string co_name = get_kernel_co_name(cfg.co_name, arch_id);

    impl_ptr =
        &impl_ptr_map.get_or_create(name, [&]() { return AiterAsmKernel(name, co_name.c_str()); });

    fmha_fwd_v3_args args;
    size_t arg_size = sizeof(args);

    int bdx = 512;
    int gdx = 0, gdy = 0, gdz = 0;
    if(thd)
    {
        // The causal object (pair_split in hsa/gfx950/fmha_v4_fwd) runs a pair's 2-5 slices on
        // kv_splits + 1 work-groups when its document has 4+ Q tiles and
        // nhead_q * pairs * (kv_splits + 1) <= cu_budget; it decides this per document. Work-groups
        // past a document's need exit at once, so x is sized by max_seqlen whenever any document,
        // down to the smallest such pair count, 2, could qualify.
        const bool causal         = cfg_mask_type != 0;
        const int64_t splits      = a.kv_splits;
        const uint32_t cu_budget  = causal && splits > 1 ? get_num_cu_func() : 0;
        init_fmha_fwd_v3_thd_args(args, a, cu_budget);
        // 32 query rows per wave. The grid is (Q tile groups of the longest document times the
        // work-groups per group, nhead_q, documents); a causal work-group owns Q tile i and its
        // mirror n - 1 - i.
        bdx                     = cfg.ts_qo / 32 * 64;
        const int64_t q_tiles   = (a.max_seqlen_q + cfg.ts_qo - 1) / cfg.ts_qo;
        const int64_t q_groups  = causal ? (q_tiles + 1) / 2 : q_tiles;
        const bool pair_extra   = cu_budget != 0 && splits <= 5 && q_tiles >= 4 &&
                                int64_t(a.nhead_q) * 2 * (splits + 1) <= cu_budget;
        gdx = static_cast<int>(q_groups * (pair_extra ? splits + 1 : splits));
        gdy = a.nhead_q;
        gdz = a.batch;
    }
    else
    {
        init_fmha_fwd_v3_args(args, a, cfg.ts_qo, arch_id);
        if(a.hdim_q == 192 && a.hdim_v == 128)
            bdx = 256;
        if(arch_id == "gfx950" && a.hdim_q == 256 && a.data_type == "bf16" && a.is_group_mode)
            bdx = 256;
        std::tie(gdx, gdy, gdz) = get_grid_dim(a, cfg.ts_qo, arch_id);
    }

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        // Explicit assignment forces evaluation order and prevents compiler from
        // reordering operations that could lead to accessing uninitialized args
        void* args_ptr     = &args;
        size_t* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz, bdx, 1, 1, s_.stream_id_});
    });
}
#endif

#if FAV2_ON
float fmha_fwd_ck(mha_fwd_args a, const ck_tile::stream_config& s)
{
    fmha_fwd_traits traits{.hdim_q              = a.hdim_q,
                           .hdim_v              = a.hdim_v,
                           .data_type           = a.data_type,
                           .is_group_mode       = a.is_group_mode,
                           .is_v_rowmajor       = true,
                           .has_logits_soft_cap = a.logits_soft_cap > 0.f,
                           .mask_type           = static_cast<mask_enum>(a.mask_type),
                           .bias_type           = static_cast<bias_enum>(a.bias_type),
                           .has_lse             = a.has_lse,
                           .has_dropout         = a.p_drop > 0.f,
                           .qscale_type         = static_cast<quant_scale_enum>(a.qscale_type),
                           .skip_min_seqlen_q   = a.min_seqlen_q != 0,
                           .has_sink            = a.has_sink};

    fmha_fwd_args args{.q_ptr                      = a.q_ptr,
                       .k_ptr                      = a.k_ptr,
                       .v_ptr                      = a.v_ptr,
                       .bias_ptr                   = a.bias_ptr,
                       .q_descale_ptr              = a.q_descale_ptr,
                       .k_descale_ptr              = a.k_descale_ptr,
                       .v_descale_ptr              = a.v_descale_ptr,
                       .rand_val_ptr               = a.rand_val_ptr,
                       .lse_ptr                    = a.lse_ptr,
                       .o_ptr                      = a.o_ptr,
                       .seqstart_q_ptr             = a.seqstart_q_ptr,
                       .seqstart_k_ptr             = a.seqstart_k_ptr,
                       .seqlen_q_ptr               = a.seqlen_q_ptr,
                       .seqlen_k_ptr               = a.seqlen_k_ptr,
                       .cu_seqlen_q_ptr            = a.cu_seqlen_q_ptr,
                       .cu_seqlen_k_ptr            = a.cu_seqlen_k_ptr,
                       .block_scale_seqstart_q_ptr = a.block_scale_seqstart_q_ptr,
                       .block_scale_seqstart_k_ptr = a.block_scale_seqstart_k_ptr,
                       .seqstart_v_scale_ptr       = nullptr,
                       .sink_ptr                   = a.sink_ptr,
                       .seqlen_q                   = a.seqlen_q,
                       .seqlen_k                   = a.seqlen_k,
                       .batch                      = a.batch,
                       .max_seqlen_q               = a.max_seqlen_q,
                       .hdim_q                     = a.hdim_q,
                       .hdim_v                     = a.hdim_v,
                       .nhead_q                    = a.nhead_q,
                       .nhead_k                    = a.nhead_k,
                       .num_head_q_total           = 0,
                       .head_start                 = 0,
                       .scale_s                    = a.scale_s,
                       .logits_soft_cap            = a.logits_soft_cap,
                       .stride_q                   = a.stride_q,
                       .stride_k                   = a.stride_k,
                       .stride_v                   = a.stride_v,
                       .stride_bias                = a.stride_bias,
                       .stride_randval             = a.stride_randval,
                       .stride_o                   = a.stride_o,
                       .stride_q_descale           = 0,
                       .stride_k_descale           = 0,
                       .stride_v_descale           = 0,
                       .nhead_stride_q             = a.nhead_stride_q,
                       .nhead_stride_k             = a.nhead_stride_k,
                       .nhead_stride_v             = a.nhead_stride_v,
                       .nhead_stride_bias          = a.nhead_stride_bias,
                       .nhead_stride_randval       = a.nhead_stride_randval,
                       .nhead_stride_lse           = a.nhead_stride_lse,
                       .nhead_stride_o             = a.nhead_stride_o,
                       .nhead_stride_q_descale     = a.nhead_stride_q_descale,
                       .nhead_stride_k_descale     = a.nhead_stride_k_descale,
                       .nhead_stride_v_descale     = a.nhead_stride_v_descale,
                       .batch_stride_q             = a.batch_stride_q,
                       .batch_stride_k             = a.batch_stride_k,
                       .batch_stride_v             = a.batch_stride_v,
                       .batch_stride_bias          = a.batch_stride_bias,
                       .batch_stride_randval       = a.batch_stride_randval,
                       .batch_stride_lse           = a.batch_stride_lse,
                       .batch_stride_o             = a.batch_stride_o,
                       .batch_stride_q_descale     = a.batch_stride_q_descale,
                       .batch_stride_k_descale     = a.batch_stride_k_descale,
                       .batch_stride_v_descale     = a.batch_stride_v_descale,
                       .window_size_left           = a.window_size_left,
                       .window_size_right          = a.window_size_right,
                       .sink_size                  = a.sink_size,
                       .mask_type                  = a.mask_type,
                       .min_seqlen_q               = a.min_seqlen_q,
                       .p_drop                     = a.p_drop,
                       .s_randval                  = a.s_randval,
                       .drop_seed_offset           = a.drop_seed_offset,
                       .block_scale_size_q         = a.block_scale_size_q,
                       .block_scale_size_kv        = a.block_scale_size_kv};

    return fmha_fwd(traits, args, s);
}
#endif

float mha_fwd(mha_fwd_args args, const ck_tile::stream_config& s)
{
    float ret = -1;

#if FAV3_ON
    ret = fmha_fwd_v3(args, s);
#endif

#if FAV2_ON
    // KV-slice partials are FP32 buffers only the v3 THD rows write; CK must not get them.
    if(ret == -1 && !args.v3_api_check && args.kv_splits <= 1)
    {
        ret = fmha_fwd_ck(args, s);
    }
#endif
    return ret;
}

} // namespace aiter
