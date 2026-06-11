#include "mha_fwd.h"
#include "aiter_hip_common.h"
#if !ENABLE_CK
#define FAV2_ON 0
#endif
#if FAV3_ON
#include "asm_fmha_v3_fwd_configs.hpp"
#endif
#if FAV1250_ON
// aiter_tensor.h is already pulled in by aiter_hip_common.h but spelled out
// here for clarity.
#include "aiter_tensor.h"
// NOTE: asm_fmha_fwd_bf16_varlen_configs.hpp is intentionally NOT included here.
// Both asm_fmha_v3_fwd_configs.hpp (#if FAV3_ON) and asm_fmha_fwd_bf16*_configs.hpp
// define `using CFG = std::unordered_map<std::string, {module}Config>` at global
// scope; including both in the same TU causes a CFG type re-definition error.
// Instead, fmha_fwd_gfx1250_varlen() calls fmha_fwd_with_sink_varlen_asm_raw()
// (defined in py_itfs_cu/asm_fmha_fwd_with_sink_varlen.cu, which includes the
// varlen config and keeps kernel selection + launch co-located with the config).
//
// C-ABI entry points defined in py_itfs_cu/asm_fmha_fwd_with_sink*.cu and
// linked into the same libmha_fwd_asm.so.
extern "C" int fmha_fwd_with_sink_asm(
    aiter_tensor_t* q,
    aiter_tensor_t* k,
    aiter_tensor_t* v,
    aiter_tensor_t* out,
    aiter_tensor_t* lse,
    aiter_tensor_t* sink,
    float           softmax_scale,
    int             is_causal,
    int             return_lse,
    hipStream_t     stream);

//g_aiter_last_error
#include "aiter_ctypes_error.h"
AITER_CTYPES_ERROR_DEF

#endif // FAV1250_ON
#include <memory>
#include <string>

namespace aiter {
#if FAV3_ON

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
                                bool mode,
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
           cfg.mask == mask_type && cfg.mode == mode)
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

    if((a.hdim_q != 192 && a.hdim_q != 128) || (a.hdim_v != 128) ||
       (a.data_type != "bf16" && a.data_type != "fp8bf16") || (a.bias_type != 0) || (a.p_drop > 0.f) ||
       ((arch_id != "gfx942") && (arch_id != "gfx950")))
    {
        AITER_LOG_WARNING("unsupported condition in fwd_v3!!! data type: " << a.data_type);
        return -1;
    }

    auto fwd_cfgs               = &cfg_fmha_fwd;
    int cfg_mask_type           = get_cfg_mask_type(a);
    std::string kernel_name_key = get_kernel_name_key(arch_id,
                                                      a.data_type,
                                                      a.hdim_q,
                                                      a.hdim_v,
                                                      cfg_mask_type,
                                                      a.how_v3_bf16_cvt,
                                                      a.is_group_mode,
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
    init_fmha_fwd_v3_args(args, a, cfg.ts_qo, arch_id);

    int bdx              = (a.hdim_q == 192 && a.hdim_v == 128) ? 256 : 512;
    auto [gdx, gdy, gdz] = get_grid_dim(a, cfg.ts_qo, arch_id);

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
    fmha_fwd_traits traits{a.hdim_q,
                           a.hdim_v,
                           a.data_type,
                           a.is_group_mode,
                           true, // is_v_rowmajor
                           a.logits_soft_cap > 0.f,
                           static_cast<mask_enum>(a.mask_type),
                           static_cast<bias_enum>(a.bias_type),
                           a.has_lse,
                           a.p_drop > 0.f,
                           static_cast<quant_scale_enum>(a.qscale_type),
                           a.min_seqlen_q != 0,
                           a.has_sink};

    fmha_fwd_args args{a.q_ptr,
                       a.k_ptr,
                       a.v_ptr,
                       a.bias_ptr,
                       a.q_descale_ptr,
                       a.k_descale_ptr,
                       a.v_descale_ptr,
                       a.rand_val_ptr,
                       a.lse_ptr,
                       a.o_ptr,
                       a.seqstart_q_ptr,
                       a.seqstart_k_ptr,
                       a.seqlen_q_ptr,
                       a.seqlen_k_ptr,
                       a.cu_seqlen_q_ptr,
                       a.cu_seqlen_k_ptr,
                       a.block_scale_seqstart_q_ptr,
                       a.block_scale_seqstart_k_ptr,
                       nullptr, // seqstart_v_scale_ptr
                       a.sink_ptr,
                       a.seqlen_q,
                       a.seqlen_k,
                       a.batch,
                       a.max_seqlen_q,
                       a.hdim_q,
                       a.hdim_v,
                       a.nhead_q,
                       a.nhead_k,
                       0, // num_head_q_total
                       0, // head_start
                       a.scale_s,
                       a.logits_soft_cap,
                       a.stride_q,
                       a.stride_k,
                       a.stride_v,
                       a.stride_bias,
                       a.stride_randval,
                       a.stride_o,
                       0, // stride_q_descale
                       0, // stride_k_descale
                       0, // stride_v_descale
                       a.nhead_stride_q,
                       a.nhead_stride_k,
                       a.nhead_stride_v,
                       a.nhead_stride_bias,
                       a.nhead_stride_randval,
                       a.nhead_stride_lse,
                       a.nhead_stride_o,
                       a.nhead_stride_q_descale,
                       a.nhead_stride_k_descale,
                       a.nhead_stride_v_descale,
                       a.batch_stride_q,
                       a.batch_stride_k,
                       a.batch_stride_v,
                       a.batch_stride_bias,
                       a.batch_stride_randval,
                       a.batch_stride_lse,
                       a.batch_stride_o,
                       a.batch_stride_q_descale,
                       a.batch_stride_k_descale,
                       a.batch_stride_v_descale,
                       a.window_size_left,
                       a.window_size_right,
                       a.sink_size,
                       a.mask_type,
                       a.min_seqlen_q,
                       a.p_drop,
                       a.s_randval,
                       a.drop_seed_offset,
                       a.block_scale_size_q,
                       a.block_scale_size_kv};

    return fmha_fwd(traits, args, s);
}
#endif

#if FAV1250_ON
// ---------------------------------------------------------------------------
// fmha_fwd_gfx1250_batched: thin bridge from mha_fwd_args → aiter_tensor_t*
// for the gfx1250 BF16 ASM sink path (fmha_fwd_with_sink_asm).
//
// Supported: batched mode (!is_group_mode), bf16, hdim_q ∈ {64,128}, hdim_v
// == hdim_q, no bias/dropout/swa/descale.  Returns 0.f on success, -1.f
// when conditions are not met (caller falls through to the next path).
//
// mha_fwd_args stride fields (stride_q, nhead_stride_q, batch_stride_q …)
// are in *elements* — the same unit that aiter_tensor_t.strides[] stores —
// so they can be used directly without scaling.
// ---------------------------------------------------------------------------
static float fmha_fwd_gfx1250_batched(const mha_fwd_args& a,
                                       const ck_tile::stream_config& s)
{
    // ---- guard conditions --------------------------------------------------
    if(get_gpu_arch() != "gfx1250")                           return -1.f;
    if(a.data_type != "bf16")                                 return -1.f;
    if(a.is_group_mode)                                       return -1.f;  // varlen: not supported here
    if(a.hdim_q != 64 && a.hdim_q != 128)                    return -1.f;
    if(a.hdim_v != a.hdim_q)                                  return -1.f;
    if(a.bias_type != 0)                                      return -1.f;
    if(a.p_drop > 0.f)                                        return -1.f;
    if(a.nhead_q % a.nhead_k != 0)                           return -1.f;
    if(a.window_size_left != -1 || a.window_size_right != 0) return -1.f;  // no SWA
    if(a.q_descale_ptr || a.k_descale_ptr || a.v_descale_ptr) return -1.f;
    // D128 binary (ENABLE_SINK=0) ignores sink; D64 binary (ENABLE_SINK=1)
    // unconditionally reads it — require explicit sink for D64, forbid for D128.
    if(a.hdim_q == 64  && !a.sink_ptr)                       return -1.f;
    if(a.hdim_q == 128 &&  a.sink_ptr)                       return -1.f;

    // ---- build aiter_tensor_t descriptors from mha_fwd_args ---------------
    // Strides are in elements (matching aiter_tensor_t convention).
    int device_id = 0;
    (void)hipGetDevice(&device_id);

    // Helper: zero-initialise and fill a 4-D bshd descriptor.
    auto make4 = [&](const void* ptr, AiterDtype dt,
                     int64_t b, int64_t s_, int64_t h, int64_t d,
                     int64_t sb, int64_t ss, int64_t sh) -> aiter_tensor_t {
        aiter_tensor_t t{};
        t.ptr        = const_cast<void*>(ptr);
        t.dtype_     = dt;
        t.device_id  = device_id;
        t.ndim       = 4;
        t.shape[0]   = b;  t.shape[1]   = s_;  t.shape[2]   = h;  t.shape[3]   = d;
        t.strides[0] = sb; t.strides[1] = ss;  t.strides[2] = sh; t.strides[3] = 1;
        t.numel_     = static_cast<std::size_t>(b * s_ * h * d);
        return t;
    };
    // Helper: 3-D descriptor (batch, head, seq) for LSE.
    auto make3 = [&](void* ptr, AiterDtype dt,
                     int64_t d0, int64_t d1, int64_t d2,
                     int64_t s0, int64_t s1) -> aiter_tensor_t {
        aiter_tensor_t t{};
        t.ptr        = ptr;
        t.dtype_     = dt;
        t.device_id  = device_id;
        t.ndim       = 3;
        t.shape[0]   = d0;  t.shape[1]   = d1;  t.shape[2]   = d2;
        t.strides[0] = s0;  t.strides[1] = s1;  t.strides[2] = 1;
        t.numel_     = static_cast<std::size_t>(d0 * d1 * d2);
        return t;
    };
    // Helper: 1-D descriptor for sink.
    auto make1 = [&](const void* ptr, AiterDtype dt, int64_t n) -> aiter_tensor_t {
        aiter_tensor_t t{};
        t.ptr        = const_cast<void*>(ptr);
        t.dtype_     = dt;
        t.device_id  = device_id;
        t.ndim       = 1;
        t.shape[0]   = n;
        t.strides[0] = 1;
        t.numel_     = static_cast<std::size_t>(n);
        return t;
    };

    // q / k / v / out : bshd = [batch, seqlen, nhead, hdim]
    // mha_fwd_args strides: stride_q = seq stride (elem), nhead_stride_q = head stride (elem),
    //                        batch_stride_q = batch stride (elem)
    aiter_tensor_t q_t = make4(a.q_ptr, AITER_DTYPE_bf16,
        a.batch, a.seqlen_q, a.nhead_q, a.hdim_q,
        a.batch_stride_q, a.stride_q, a.nhead_stride_q);
    aiter_tensor_t k_t = make4(a.k_ptr, AITER_DTYPE_bf16,
        a.batch, a.seqlen_k, a.nhead_k, a.hdim_q,
        a.batch_stride_k, a.stride_k, a.nhead_stride_k);
    aiter_tensor_t v_t = make4(a.v_ptr, AITER_DTYPE_bf16,
        a.batch, a.seqlen_k, a.nhead_k, a.hdim_v,
        a.batch_stride_v, a.stride_v, a.nhead_stride_v);
    aiter_tensor_t o_t = make4(a.o_ptr, AITER_DTYPE_bf16,
        a.batch, a.seqlen_q, a.nhead_q, a.hdim_v,
        a.batch_stride_o, a.stride_o, a.nhead_stride_o);
    // lse : [batch, nhead_q, seqlen_q] fp32
    // nhead_stride_lse is stride in fp32 elements between heads within one batch.
    aiter_tensor_t lse_t = make3(a.lse_ptr, AITER_DTYPE_fp32,
        a.batch, a.nhead_q, a.seqlen_q,
        a.batch_stride_lse, a.nhead_stride_lse);
    // sink : [nhead_q] fp32 (optional; null for D128 / ENABLE_SINK=0 kernels)
    aiter_tensor_t  sink_storage{};
    aiter_tensor_t* sink_t = nullptr;
    if(a.sink_ptr)
    {
        sink_storage = make1(a.sink_ptr, AITER_DTYPE_fp32, a.nhead_q);
        sink_t       = &sink_storage;
    }

    // mask_type: 0=no mask, 1=top-left causal, 2=bottom-right causal
    const int is_causal = (a.mask_type == 1 || a.mask_type == 2) ? 1 : 0;

    aiter_clear_last_error();
    int ret = fmha_fwd_with_sink_asm(
        &q_t, &k_t, &v_t, &o_t, &lse_t, sink_t,
        a.scale_s, is_causal, a.has_lse ? 1 : 0,
        s.stream_id_);
    if(ret != 0)
    {
        AITER_LOG_WARNING("fmha_fwd_gfx1250_batched: fmha_fwd_with_sink_asm failed: "
                          << (aiter_get_last_error() ? aiter_get_last_error() : "(no error)"));
        return -1.f;
    }
    return 0.f; // success; timing not measured through this path
}
#endif // FAV1250_ON (fmha_fwd_gfx1250_batched)

float mha_fwd(mha_fwd_args args, const ck_tile::stream_config& s)
{
    float ret = -1;

#if FAV1250_ON
    ret = fmha_fwd_gfx1250_batched(args, s);
    if(ret != -1.f)
        return ret;
    ret = -1; // reset for next path
#endif

#if FAV3_ON
    ret = fmha_fwd_v3(args, s);
#endif

#if FAV2_ON
    if(ret == -1 && !args.v3_api_check)
    {
        ret = fmha_fwd_ck(args, s);
    }
#endif
    return ret;
}

} // namespace aiter
